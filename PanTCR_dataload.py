# MHCpeptide_dataload.py
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, List, Tuple, Optional, Callable

from model_config import PMHCPairConfig, TCRPairConfig  # separated configs


# ============================ Cleaning utils (RETAIN) ============================ #
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # canonical 20 only

def _clean_seq(x: Any) -> str:
    """Normalize to uppercase and keep only canonical 20 AAs."""
    if x is None:
        return ""
    # IMPORTANT: handle pandas NaN safely
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, (np.floating, np.integer)) and pd.isna(x):
        return ""
    s = str(x).strip().upper()
    return "".join(ch for ch in s if ch in VALID_AA)

def _read_csv_clean(path: str) -> pd.DataFrame:
    """Read CSV and convert empty strings to NaN so dropna works properly."""
    df = pd.read_csv(path)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df

def _filter_nonempty(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Clean specified columns and keep rows where all fields are non-empty."""
    df = df.copy()
    for c in columns:
        df[c] = df[c].map(_clean_seq)
    mask = np.ones(len(df), dtype=bool)
    for c in columns:
        mask &= df[c].str.len() > 0
    return df[mask].reset_index(drop=True)


# ============================ Extra helpers ============================ #
def _truncate(seq: str, max_len: int) -> str:
    """Truncate to max_len (no padding)."""
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")
    if not isinstance(seq, str):
        seq = str(seq)
    return seq[:max_len]

def _fix_len(seq: str, target_len: int, pad_char: str = "X") -> str:
    """
    Pad (right) or truncate to target_len.
    NOTE: VALID_AA excludes 'X', but we only pad AFTER cleaning, and we do not re-clean.
    The model's one-hot maps unknowns to X internally, so this is safe.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) < target_len:
        return seq + (pad_char * (target_len - len(seq)))
    return seq

LabelFn = Callable[[Any], float]

def _default_label_fn(x: Any) -> float:
    return float(x)


# =============================== pMHC Dataset =============================== #
class MHCpeptideDataset(Dataset):
    """
    IEDB_retrain_extraction_MHC_final.csv
    Expected columns: Antigen (peptide), MHC_sequence, Label

    Output:
      ("MHC_fixed:PEP_fixed", y_float_tensor)

    Fixed-length policy (MHC-I):
    - peptide length fixed to pair_cfg.pep_len
        * if peptide length > pep_len: DROP ROW
        * if peptide length < pep_len: PAD with 'X'
    - MHC length fixed to pair_cfg.mhc_len
        * if MHC length > mhc_len: TRUNCATE
        * if MHC length < mhc_len: PAD with 'X'
    """
    def __init__(
        self,
        csv_path: str,
        pair_cfg: PMHCPairConfig,
        *,
        sep: str = ":",
        antigen_col: str = "Antigen",
        mhc_col: str = "MHC_sequence",
        label_col: str = "Label",
        label_fn: Optional[LabelFn] = None,
    ):
        super().__init__()
        self.sep = sep
        self.pair_cfg = pair_cfg

        self.antigen_col = antigen_col
        self.mhc_col = mhc_col
        self.label_col = label_col

        self.mhc_len = int(getattr(pair_cfg, "mhc_len", 34))
        self.pep_len = int(getattr(pair_cfg, "pep_len", 11))
        self.fixed_len = int(getattr(pair_cfg, "fixed_len", self.mhc_len + self.pep_len))

        self.label_fn: LabelFn = label_fn if label_fn is not None else _default_label_fn

        df = _read_csv_clean(csv_path)

        for c in (self.antigen_col, self.mhc_col, self.label_col):
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {csv_path}. Columns: {list(df.columns)}")

        df = df.dropna(subset=[self.antigen_col, self.mhc_col, self.label_col]).reset_index(drop=True)

        # clean + enforce non-empty sequences
        df = _filter_nonempty(df, [self.antigen_col, self.mhc_col])

        # drop peptides longer than pep_len
        keep = df[self.antigen_col].str.len().to_numpy() <= self.pep_len
        df = df[keep].reset_index(drop=True)

        # apply label_fn robustly; drop rows where it fails / returns non-finite
        ys: List[float] = []
        keep_rows = np.ones(len(df), dtype=bool)
        for i, raw in enumerate(df[self.label_col].tolist()):
            try:
                y = float(self.label_fn(raw))
                if not np.isfinite(y):
                    raise ValueError("label_fn returned non-finite value")
                ys.append(y)
            except Exception:
                keep_rows[i] = False

        df = df[keep_rows].reset_index(drop=True)
        df["_y"] = np.asarray(ys, dtype=np.float32)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        r = self.df.iloc[idx]

        pep = str(r[self.antigen_col])
        mhc = str(r[self.mhc_col])

        # peptide fixed (longer already dropped)
        pep = _fix_len(pep, self.pep_len, pad_char="X")
        # mhc fixed
        mhc = _fix_len(mhc, self.mhc_len, pad_char="X")

        concat = f"{mhc}{self.sep}{pep}"

        if (len(mhc) + len(pep)) != self.fixed_len:
            raise RuntimeError(
                f"Fixed-length mismatch: got len(mhc)+len(pep)={len(mhc)+len(pep)}, "
                f"expected {self.fixed_len} (mhc_len={self.mhc_len}, pep_len={self.pep_len})."
            )

        y = torch.tensor(float(r["_y"]), dtype=torch.float32)
        return concat, y


# =============================== Collate =============================== #
def collate_concat_regression(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    """returns (list_of_concat_strings, y_tensor[B,1])"""
    seqs = [x[0] for x in batch]
    ys = torch.stack([x[1] for x in batch], dim=0).view(-1, 1)
    return seqs, ys


# =============================== Example label_fn =============================== #
def make_log10_ic50_label_fn(ic50_min: float = 1e-12) -> LabelFn:
    """raw IC50 -> log10(IC50) with clamping."""
    ic50_min = float(ic50_min)
    def _fn(raw: Any) -> float:
        x = float(raw)
        x = max(x, ic50_min)
        return float(np.log10(x))
    return _fn


# =============================== Integrated TCR+pMHC Dataset =============================== #
class IntegratedTCRDataset(Dataset):
    """
    Expected columns:
      Antigen, TCR_alpha, (optional) TCR_beta, MHC_sequence, Label

    Output per item:
      (pep_fixed, mhc_fixed, tcra_trunc, tcrb_trunc_or_None, y_tensor)

    Length policy:
      - peptide: drop if longer than pmhc_cfg.pep_len, else pad to pep_len
      - mhc: pad/truncate to pmhc_cfg.mhc_len
      - tcra: truncate to tcr_cfg.tcr_a_max_len (no padding)
      - tcrb: if missing/empty -> None else truncate to tcr_cfg.tcr_b_max_len
    """
    def __init__(
        self,
        csv_path: str,
        pmhc_cfg: PMHCPairConfig,
        tcr_cfg: TCRPairConfig,
        *,
        antigen_col: str = "Antigen",
        mhc_col: str = "MHC_sequence",
        tcra_col: str = "TCR_alpha",
        tcrb_col: str = "TCR_beta",
        label_col: str = "Label",
        label_fn: Optional[LabelFn] = None,
    ):
        super().__init__()
        self.pmhc_cfg = pmhc_cfg
        self.tcr_cfg = tcr_cfg

        self.antigen_col = antigen_col
        self.mhc_col = mhc_col
        self.tcra_col = tcra_col
        self.tcrb_col = tcrb_col
        self.label_col = label_col

        self.pep_len = int(getattr(pmhc_cfg, "pep_len", 11))
        self.mhc_len = int(getattr(pmhc_cfg, "mhc_len", 34))
        self.tcr_a_max_len = int(getattr(tcr_cfg, "tcr_a_max_len", 70))
        self.tcr_b_max_len = int(getattr(tcr_cfg, "tcr_b_max_len", 70))

        self.label_fn: LabelFn = label_fn if label_fn is not None else _default_label_fn

        df = _read_csv_clean(csv_path)

        # Require minimal columns (TCR_beta optional)
        required = [self.antigen_col, self.mhc_col, self.tcra_col, self.label_col]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {csv_path}. Columns: {list(df.columns)}")
        if self.tcrb_col not in df.columns:
            df[self.tcrb_col] = np.nan

        # Drop rows missing required fields (tcrb allowed missing)
        df = df.dropna(subset=required).reset_index(drop=True)

        # Clean sequences (canonical 20 AAs)
        df = df.copy()
        df[self.antigen_col] = df[self.antigen_col].map(_clean_seq)
        df[self.mhc_col]     = df[self.mhc_col].map(_clean_seq)
        df[self.tcra_col]    = df[self.tcra_col].map(_clean_seq)
        # For optional tcrb, keep empty string for NaN/blank
        df[self.tcrb_col]    = df[self.tcrb_col].map(_clean_seq)

        # Enforce non-empty required sequence cols
        mask = (
            (df[self.antigen_col].str.len() > 0) &
            (df[self.mhc_col].str.len() > 0) &
            (df[self.tcra_col].str.len() > 0)
        )
        df = df[mask].reset_index(drop=True)

        # Drop peptides longer than pep_len
        keep = df[self.antigen_col].str.len().to_numpy() <= self.pep_len
        df = df[keep].reset_index(drop=True)

        # Labels
        ys: List[float] = []
        keep_rows = np.ones(len(df), dtype=bool)
        for i, raw in enumerate(df[self.label_col].tolist()):
            try:
                y = float(self.label_fn(raw))
                if not np.isfinite(y):
                    raise ValueError("label_fn returned non-finite value")
                ys.append(y)
            except Exception:
                keep_rows[i] = False

        df = df[keep_rows].reset_index(drop=True)
        df["_y"] = np.asarray(ys, dtype=np.float32)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, Optional[str], torch.Tensor]:
        r = self.df.iloc[idx]

        pep = str(r[self.antigen_col])
        mhc = str(r[self.mhc_col])
        tcra = str(r[self.tcra_col])

        # NOTE: tcrb stored as "" if missing after cleaning
        tcrb_raw = r[self.tcrb_col]
        tcrb_raw = "" if (isinstance(tcrb_raw, float) and pd.isna(tcrb_raw)) else str(tcrb_raw)

        # pMHC fixed-length
        pep = _fix_len(pep, self.pep_len, pad_char="X")
        mhc = _fix_len(mhc, self.mhc_len, pad_char="X")

        # TCR truncation (ragged OK; no padding here)
        tcra = _truncate(tcra, self.tcr_a_max_len)

        tcrb = tcrb_raw if (isinstance(tcrb_raw, str) and len(tcrb_raw) > 0) else None
        if tcrb is not None:
            tcrb = _truncate(tcrb, self.tcr_b_max_len)

        y = torch.tensor(float(r["_y"]), dtype=torch.float32)
        return pep, mhc, tcra, tcrb, y


# =============================== Collate for IntegratedTCRDataset =============================== #
def collate_tcr_pmhc(
    batch: List[Tuple[str, str, str, Optional[str], torch.Tensor]]
) -> Tuple[List[str], List[str], List[str], List[Optional[str]], torch.Tensor]:
    """
    Returns:
      pep_list, mhc_list, tcra_list, tcrb_list, y[B,1]
    """
    peps = [b[0] for b in batch]
    mhcs = [b[1] for b in batch]
    tcras = [b[2] for b in batch]
    tcrbs = [b[3] for b in batch]
    ys = torch.stack([b[4] for b in batch], dim=0).view(-1, 1)
    return peps, mhcs, tcras, tcrbs, ys


# ============================== (Optional) test ============================== #
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import os

    pair_cfg = PMHCPairConfig(mhc_len=34, pep_len=11)
    label_fn = make_log10_ic50_label_fn(1e-12)

    pmhc_csv = "data/IEDB_retrain_extraction_MHC_final.csv"
    tcr_csv = "data/integrated_TCR_data.csv"

    if os.path.exists(pmhc_csv):
        ds = MHCpeptideDataset(pmhc_csv, pair_cfg, label_fn=label_fn)
        dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_concat_regression)
        seqs, y = next(iter(dl))
        print("[pMHC] sample:", seqs[0], y.shape)

    if os.path.exists(tcr_csv):
        tcr_cfg = TCRPairConfig(tcr_a_max_len=70, tcr_b_max_len=70)
        ds = IntegratedTCRDataset(tcr_csv, pair_cfg, tcr_cfg)
        dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_tcr_pmhc)
        peps, mhcs, tcras, tcrbs, y = next(iter(dl))
        print("[TCR+pMHC] pep:", peps[0])
        print("[TCR+pMHC] mhc_len:", len(mhcs[0]))
        print("[TCR+pMHC] tcra_len:", len(tcras[0]), "tcrb:", None if tcrbs[0] is None else len(tcrbs[0]))
        print("[TCR+pMHC] y:", y[:3].view(-1).tolist())
