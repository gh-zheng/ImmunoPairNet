# Panimmnue_dataload.py
from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional

# ============================ Cleaning utils ============================ #
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def _clean_seq(x: Any) -> str:
    """
    Normalize to uppercase, keep only canonical 20 AAs.
    NaN/None -> ''.
    """
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip().upper()
    return "".join(ch for ch in s if ch in VALID_AA)

def _read_csv_clean(path: str) -> pd.DataFrame:
    """
    Read CSV and convert empty strings to NaN so dropna works as expected.
    """
    df = pd.read_csv(path)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df

def _binarize_label(x: Any, threshold: float = 0.5) -> int:
    """
    Convert numeric/float labels to {0,1} using threshold.
    If parsing fails, default to 0.
    """
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return int(v >= threshold)

# =============================== Datasets =============================== #
class IEDBRetrainMHCDataset(Dataset):
    """
    IEDB_retrain_extraction_MHC_final.csv
      Columns (required): Antigen, MHC_sequence, Label
    Returns:
      {"sequences": [MHC_sequence, Antigen], "label": int}
    """
    def __init__(self, csv_path: str, *, binarize: bool = True, threshold: float = 0.5):
        self.binarize = binarize
        self.threshold = threshold
        df = _read_csv_clean(csv_path)
        df = df.dropna(subset=["Antigen", "MHC_sequence", "Label"]).reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        seqs = [
            _clean_seq(r["MHC_sequence"]),
            _clean_seq(r["Antigen"]),
        ]
        # drop any empty strings (safety)
        seqs = [s for s in seqs if s]
        label = _binarize_label(r["Label"], self.threshold) if self.binarize else float(r["Label"])
        return {"sequences": seqs, "label": int(label)}

class IntegratedTCRDataset(Dataset):
    """
    integrated_TCR_data.csv
      Columns (required): Antigen, TCR_alpha, TCR_beta, MHC_sequence, Label
    Returns:
      {"sequences": [TCR_alpha, TCR_beta, Antigen, MHC_sequence], "label": int}
    """
    def __init__(self, csv_path: str, *, binarize: bool = True, threshold: float = 0.5):
        self.binarize = binarize
        self.threshold = threshold
        df = _read_csv_clean(csv_path)
        df = df.dropna(subset=["Antigen", "TCR_alpha", "TCR_beta", "MHC_sequence", "Label"]).reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        seqs = [
            _clean_seq(r["TCR_alpha"]),
            _clean_seq(r["TCR_beta"]),
            _clean_seq(r["Antigen"]),
            _clean_seq(r["MHC_sequence"]),
        ]
        seqs = [s for s in seqs if s]
        label = _binarize_label(r["Label"], self.threshold) if self.binarize else float(r["Label"])
        return {"sequences": seqs, "label": int(label)}

class IntegratedAntibodyDataset(Dataset):
    """
    integrated_antibody_data.csv
      Columns (required): Heavy_chain, Antigen, Label
      Optional       : Light_chain
    Returns:
      {"sequences": [Heavy_chain, Light_chain?, Antigen], "label": int}
      (Light_chain is included only if non-empty)
    """
    def __init__(
        self,
        csv_path: str,
        *,
        require_light: bool = False,
        binarize: bool = True,
        threshold: float = 0.5,
    ):
        self.binarize = binarize
        self.threshold = threshold
        df = _read_csv_clean(csv_path)
        needed = ["Heavy_chain", "Antigen", "Label"] + (["Light_chain"] if require_light else [])
        df = df.dropna(subset=needed).reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        seqs = [
            _clean_seq(r["Heavy_chain"]),
            _clean_seq(r.get("Light_chain", "")),
            _clean_seq(r["Antigen"]),
        ]
        seqs = [s for s in seqs if s]
        label = _binarize_label(r["Label"], self.threshold) if self.binarize else float(r["Label"])
        return {"sequences": seqs, "label": int(label)}

# ============================ Collate helpers ============================ #
def _collate_common_panimmune(batch: List[Dict[str, Any]]):
    """
    Convert a list of dicts from the Dataset into:
      batch_samples: List[List[str]]  (variable-length string lists)
      labels:        LongTensor [B]
      metas:         List[dict] (empty dicts kept for compatibility)
    """
    batch_samples: List[List[str]] = []
    labels: List[int] = []
    metas: List[Dict[str, Any]] = []

    for item in batch:
        seqs = [s for s in item.get("sequences", []) if isinstance(s, str) and len(s) > 0]
        batch_samples.append(seqs)
        labels.append(int(item["label"]))
        metas.append(item.get("meta", {}))  # optional

    labels = torch.tensor(labels, dtype=torch.long)
    return batch_samples, labels, metas

def collate_mhc_panimmune(batch: List[Dict[str, Any]]):
    return _collate_common_panimmune(batch)

def collate_tcr_panimmune(batch: List[Dict[str, Any]]):
    return _collate_common_panimmune(batch)

def collate_antibody_panimmune(batch: List[Dict[str, Any]]):
    return _collate_common_panimmune(batch)

# ============================== (Optional) test ============================== #
if __name__ == "__main__":
    # Quick smoke tests if files are present in the working directory.
    import os
    from torch.utils.data import DataLoader

    paths = {
        "MHC": "data\IEDB_retrain_extraction_MHC_final.csv",
        "TCR": "data\integrated_TCR_data.csv",
        "AB":  "data\integrated_antibody_data.csv",
    }

    if os.path.exists(paths["MHC"]):
        ds = IEDBRetrainMHCDataset(paths["MHC"], binarize=True)
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_mhc_panimmune)
        batch_samples, labels, metas = next(iter(dl))
        print("[MHC] batch_samples[0]:", batch_samples[0])
        print("[MHC] labels:", labels.tolist())

    if os.path.exists(paths["TCR"]):
        ds = IntegratedTCRDataset(paths["TCR"], binarize=True)
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_tcr_panimmune)
        batch_samples, labels, metas = next(iter(dl))
        print("[TCR] batch_samples[0]:", batch_samples[0])
        print("[TCR] labels:", labels.tolist())

    if os.path.exists(paths["AB"]):
        ds = IntegratedAntibodyDataset(paths["AB"], require_light=False, binarize=True)
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_antibody_panimmune)
        batch_samples, labels, metas = next(iter(dl))
        print("[AB]  batch_samples[0]:", batch_samples[0])
        print("[AB]  labels:", labels.tolist())
