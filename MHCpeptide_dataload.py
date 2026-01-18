# MHCpeptide_dataload.py
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, List, Tuple, Optional, Callable

from model_config import PairConfig  # uses mhc_len, pep_len (=11), fixed_len


# ============================ Cleaning utils (RETAIN) ============================ #
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # canonical 20 only

def _clean_seq(x: Any) -> str:
    """Normalize to uppercase and keep only canonical 20 AAs."""
    if isinstance(x, float) and pd.isna(x):
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

def _fix_len(seq: str, target_len: int, pad_char: str = "X") -> str:
    """
    Pad (right) or truncate to target_len.
    NOTE: since VALID_AA excludes 'X', we represent padding using 'X' here anyway.
    This is OK because the model's one-hot encoder maps unknowns to 'X' internally.
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


# =============================== Dataset =============================== #

class MHCpeptideDataset(Dataset):
    """
    IEDB_retrain_extraction_MHC_final.csv
    Expected columns: Antigen (peptide), MHC_sequence, Label

    Output:
      ("MHC_fixed:PEP_fixed", y_float_tensor)

    Fixed-length policy (MHC-I):
    - peptide length fixed to pair_cfg.pep_len (default 11)
        * if peptide length > pep_len: DROP ROW (requested)
        * if peptide length < pep_len: PAD with 'X'
    - MHC length fixed to pair_cfg.mhc_len (default 34)
        * if MHC length > mhc_len: TRUNCATE
        * if MHC length < mhc_len: PAD with 'X'

    Label processing:
    - Provide label_fn(raw_label)->float to implement raw processing (e.g., log10 transform).
    - Default label_fn just casts to float.
    """
    def __init__(
        self,
        csv_path: str,
        pair_cfg: PairConfig,
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

        # require columns
        for c in (self.antigen_col, self.mhc_col, self.label_col):
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {csv_path}. Columns: {list(df.columns)}")

        # drop NaNs first
        df = df.dropna(subset=[self.antigen_col, self.mhc_col, self.label_col]).reset_index(drop=True)

        # clean + enforce non-empty sequences (RETAIN behavior)
        df = _filter_nonempty(df, [self.antigen_col, self.mhc_col])

        # drop peptides longer than fixed pep_len (requested)
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

        # peptide: pad if shorter (longer already dropped)
        pep = _fix_len(pep, self.pep_len, pad_char="X")
        # mhc: pad/truncate
        mhc = _fix_len(mhc, self.mhc_len, pad_char="X")

        concat = f"{mhc}{self.sep}{pep}"

        # sanity: fixed length excluding separator
        if (len(mhc) + len(pep)) != self.fixed_len:
            raise RuntimeError(
                f"Fixed-length mismatch: got len(mhc)+len(pep)={len(mhc)+len(pep)}, "
                f"expected {self.fixed_len} (mhc_len={self.mhc_len}, pep_len={self.pep_len})."
            )

        y = torch.tensor(float(r["_y"]), dtype=torch.float32)
        return concat, y


# =============================== Collate =============================== #

def collate_concat_regression(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    """
    Collate for regression:
      returns (list_of_concat_strings, y_tensor[B,1])
    """
    seqs = [x[0] for x in batch]
    ys = torch.stack([x[1] for x in batch], dim=0).view(-1, 1)
    return seqs, ys


# =============================== Example label_fn =============================== #

def make_log10_ic50_label_fn(ic50_min: float = 1e-12) -> LabelFn:
    """
    label_fn that converts raw IC50 (float-like) -> log10(IC50) with clamping.
    """
    ic50_min = float(ic50_min)
    def _fn(raw: Any) -> float:
        x = float(raw)
        x = max(x, ic50_min)
        return float(np.log10(x))
    return _fn



class IntegratedTCRDataset(Dataset):
    """integrated_TCR_data.csv
       Columns: Antigen, TCR_alpha, TCR_beta, MHC_sequence, Label
       Returns: ("TCR_alpha:TCR_beta:Antigen:MHC_sequence", label)
    """
    def __init__(self, csv_path: str, *, sep: str = ":"):
        self.sep = sep
        df = _read_csv_clean(csv_path)
        df = df.dropna(subset=["Antigen", "TCR_alpha", "TCR_beta", "MHC_sequence", "Label"]).reset_index(drop=True)
        df = _filter_nonempty(df, ["Antigen", "TCR_alpha", "TCR_beta", "MHC_sequence"])
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        r = self.df.iloc[idx]
        parts = [r["Antigen"], r["MHC_sequence"], r["TCR_alpha"], r["TCR_beta"]]
        concat = self.sep.join(parts)
        label = r["Label"]  # keep label as-is
        return concat, label


class IntegratedAntibodyDataset(Dataset):
    """integrated_antibody_data.csv
       Columns: Heavy_chain, Light_chain (optional), Antigen, Label
       Returns: ("Heavy[:Light]:Antigen", label)
    """
    def __init__(self, csv_path: str, *, require_light: bool = False, sep: str = ":"):
        self.sep = sep
        df = _read_csv_clean(csv_path)
        needed = ["Heavy_chain", "Antigen", "Label"] + (["Light_chain"] if require_light else [])
        df = df.dropna(subset=needed).reset_index(drop=True)

        df = df.copy()
        df["Heavy_chain"] = df["Heavy_chain"].map(_clean_seq)
        df["Antigen"] = df["Antigen"].map(_clean_seq)
        if "Light_chain" in df.columns:
            df["Light_chain"] = df["Light_chain"].map(_clean_seq)
        else:
            df["Light_chain"] = ""

        mask = (df["Heavy_chain"].str.len() > 0) & (df["Antigen"].str.len() > 0)
        if require_light:
            mask &= (df["Light_chain"].str.len() > 0)
        df = df[mask].reset_index(drop=True)
        self.df = df
        self.require_light = require_light

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        r = self.df.iloc[idx]
        parts = [r["Heavy_chain"]]
        if (not self.require_light and len(r.get("Light_chain", "")) > 0) or self.require_light:
            if len(r["Light_chain"]) > 0:
                parts.append(r["Light_chain"])
        parts.append(r["Antigen"])
        concat = self.sep.join(parts)
        label = r["Label"]  # keep label as-is
        return concat, label

# ============================== (Optional) test ============================== #
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import os

    paths = {
        "MHC": "data/IEDB_retrain_extraction_MHC_final.csv",
        "TCR": "data/integrated_TCR_data.csv",
        "AB":  "data/integrated_antibody_data.csv",
    }

    if os.path.exists(paths["TCR"]):
        ds = IntegratedTCRDataset(paths["TCR"])
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        seqs, labels = next(iter(dl))
        print("[TCR] seqs:", seqs)
        print("[TCR] labels:", labels.tolist() if hasattr(labels, 'tolist') else labels)

    if os.path.exists(paths["AB"]):
        ds = IntegratedAntibodyDataset(paths["AB"], require_light=False)
        dl = DataLoader(ds, batch_size=1000, shuffle=True)
        seqs, labels = next(iter(dl))
        #print("[AB] seqs:", seqs)
        print("[AB] labels:", labels.tolist() if hasattr(labels, 'tolist') else labels)

if __name__ == "__main__":
    # Example usage (needs real CSV):
    pair_cfg = PairConfig(mhc_len=34, pep_len=11)
    label_fn = make_log10_ic50_label_fn(1e-12)
    ds = MHCpeptideDataset(r"data\IEDB_retrain_extraction_MHC_final.csv", pair_cfg, label_fn=label_fn)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_concat_regression)
    seqs, y = next(iter(dl))
    print(seqs[0], y.shape, y[:3].view(-1).tolist())
    pass