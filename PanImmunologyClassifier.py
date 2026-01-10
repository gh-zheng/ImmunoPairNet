# PanImmunologyClassifier.py
"""
PanImmunologyRegressor (fixed-length, Z-grid conv + flatten; NO pooling)

Use case:
- MHC-I peptide binding regression
- Input: batch_seqs (list like "MHC:PEPTIDE")
- Output: by default in (0,1) using sigmoid activation (normalized targets)

Architecture:
  PanimmuneEmbedderPairs (one-hot internal encoder)
    -> z [B, L, L, C]
    -> 2D Conv head over [B, C, L, L]
    -> Flatten -> MLP -> Linear output -> optional activation
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model_config import PairConfig, ZClassifierConfig
from PanimmuneEmbedderPairs import PanimmuneEmbedderPairs


class ZGridConvFlattenHead(nn.Module):
    """
    Fixed-size contact-map head (no pooling):
      z: [B, L, L, C] -> conv2d over [B, C, L, L] -> flatten -> MLP -> yhat
    """
    def __init__(
        self,
        pair_dim: int,
        grid_len: int,
        hidden_dim: int,
        out_dim: int = 1,
        dropout: float = 0.1,
        n_convs: int = 2,
    ):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.grid_len = int(grid_len)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.n_convs = int(n_convs)

        if self.grid_len <= 0:
            raise ValueError(f"grid_len must be > 0, got {self.grid_len}.")
        if self.pair_dim <= 0:
            raise ValueError(f"pair_dim must be > 0, got {self.pair_dim}.")
        if self.n_convs < 1:
            raise ValueError(f"n_convs must be >= 1, got {self.n_convs}.")

        layers = []
        C = self.pair_dim
        for _ in range(self.n_convs):
            layers.extend([
                nn.Conv2d(C, C, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
        self.conv = nn.Sequential(*layers)

        flat_dim = C * self.grid_len * self.grid_len
        self.fc1 = nn.Linear(flat_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(f"Expected z [B, L, L, C], got {tuple(z.shape)}.")
        _, L1, L2, C = z.shape
        if L1 != L2:
            raise ValueError(f"Expected square grid [L, L], got {L1}x{L2}.")
        if L1 != self.grid_len:
            raise ValueError(f"Expected grid_len={self.grid_len}, got L={L1}.")
        if C != self.pair_dim:
            raise ValueError(f"Expected pair_dim={self.pair_dim}, got C={C}.")

        x = z.permute(0, 3, 1, 2).contiguous()  # [B, C, L, L]
        x = self.conv(x)
        feats = x.flatten(1)
        x = self.fc1(feats)
        x = self.drop(self.act(x))
        yhat = self.fc2(x)  # [B, out_dim]
        return yhat


class PanImmunologyRegressor(nn.Module):
    """
    Regression model:
      batch_seqs -> z-grid -> conv+flatten -> output activation (config-driven)

    If cfg.output_activation == "sigmoid": output in (0,1).
    If cfg.output_activation == "none": output unbounded.
    """
    def __init__(
        self,
        pair_cfg: PairConfig,
        clf_cfg: Optional[ZClassifierConfig] = None,
        grid_len: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_to_label_range: bool = True,   # safety clamp after activation
    ):
        super().__init__()
        self.device = torch.device(device)

        self.pair_cfg = pair_cfg
        self.cfg = clf_cfg if clf_cfg is not None else ZClassifierConfig()

        # Embedder (one-hot)
        self.embeder = PanimmuneEmbedderPairs(pair_cfg, device=self.device)

        # Fixed L required
        if grid_len is None:
            grid_len = getattr(pair_cfg, "fixed_len", None)
        if grid_len is None:
            raise ValueError("grid_len must be provided, or pair_cfg.fixed_len must exist.")
        self.grid_len = int(grid_len)

        # Output dim must be 1 for regression
        out_dim = int(getattr(self.cfg, "num_classes", 1))
        if out_dim != 1:
            raise ValueError(f"For regression, set cfg.num_classes=1, got {out_dim}.")

        self.head = ZGridConvFlattenHead(
            pair_dim=int(pair_cfg.pair_dim),
            grid_len=self.grid_len,
            hidden_dim=int(self.cfg.hidden_dim),
            out_dim=1,
            dropout=float(self.cfg.dropout),
            n_convs=int(getattr(self.cfg, "n_convs", 2)),
        )

        # Activation from config (preferred)
        act = str(getattr(self.cfg, "output_activation", "sigmoid")).lower().strip()
        if act not in ("sigmoid", "none"):
            raise ValueError("cfg.output_activation must be 'sigmoid' or 'none'.")
        self.out_act = nn.Sigmoid() if act == "sigmoid" else nn.Identity()

        # Optional clamp (useful with AMP/fp16)
        self.clamp_to_label_range = bool(clamp_to_label_range)
        self.label_range: Optional[Tuple[float, float]] = getattr(self.cfg, "label_range", None)

        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        pair_cfg: PairConfig,
        clf_cfg: ZClassifierConfig,
        grid_len: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "PanImmunologyRegressor":
        return cls(pair_cfg=pair_cfg, clf_cfg=clf_cfg, grid_len=grid_len, device=device)

    def forward(self, batch_seqs: List[str]) -> torch.Tensor:
        z = self.embeder(batch_seqs)  # [B, L, L, C]
        yhat = self.head(z)           # [B, 1] (logit-like if sigmoid)
        yhat = self.out_act(yhat)     # apply activation

        if self.clamp_to_label_range and self.label_range is not None:
            lo, hi = float(self.label_range[0]), float(self.label_range[1])
            yhat = torch.clamp(yhat, lo, hi)

        return yhat


if __name__ == "__main__":
    pair_cfg = PairConfig()
    cfg = ZClassifierConfig(num_classes=1)  # and cfg.output_activation="sigmoid" in config

    grid_len = getattr(pair_cfg, "fixed_len", 45)

    model = PanImmunologyRegressor.from_config(
        pair_cfg, cfg, grid_len=grid_len, device="cpu"
    )

    seqs = [
        "A" * 34 + ":" + "C" * 11,
        "V" * 34 + ":" + "D" * 11,
    ]

    y = model(seqs)
    print("output shape:", y.shape)
    print("output range:", float(y.min()), float(y.max()))
    print("output values:", y)
