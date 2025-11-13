# Panimmune_dataload.py
from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, List, Tuple
import random

# ============================ Cleaning utils ============================ #
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

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

# =============================== Datasets =============================== #
class IEDBRetrainMHCDataset(Dataset):
    """IEDB_retrain_extraction_MHC_final.csv
       Columns: Antigen, MHC_sequence, Label
       Returns: ("MHC_sequence:Antigen", label)
    """
    def __init__(self, csv_path: str, *, sep: str = ":"):
        self.sep = sep
        df = _read_csv_clean(csv_path)
        df = df.dropna(subset=["Antigen", "MHC_sequence", "Label"]).reset_index(drop=True)
        df = _filter_nonempty(df, ["Antigen", "MHC_sequence"])
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        r = self.df.iloc[idx]
        parts = [r["Antigen"], r["MHC_sequence"]]
        concat = self.sep.join(parts)
        label = r["Label"]  # keep label as-is (no thresholding/casting)
        return concat, label


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
        random.shuffle(parts)
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
        random.shuffle(parts)
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

    if os.path.exists(paths["MHC"]):
        ds = IEDBRetrainMHCDataset(paths["MHC"])
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        seqs, labels = next(iter(dl))
        print("[MHC] seqs:", seqs)
        print("[MHC] labels:", labels.tolist() if hasattr(labels, 'tolist') else labels)
        raise

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
