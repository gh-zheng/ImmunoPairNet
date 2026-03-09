# TCRmhcEmbeddingClassifier.py
"""
TCR+pMHC classifier/regressor (fixed-length, Z-grid conv + flatten; NO pooling)

Revised to use an explicit sigmoid function (torch.sigmoid) on logits
instead of nn.Sigmoid module. This matches "sigmod function" intent.

- If cfg.output_activation == "sigmoid": yhat = torch.sigmoid(logits)
- If cfg.output_activation == "none":    yhat = logits (unbounded)

Everything else stays the same (embedder -> head -> activation -> optional clamp).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.model_config import PMHCPairConfig, TCRPairConfig, FullGridPairConfig, TCRClassifierConfig
from src.tcrMHCpeptideEmbedding import TCRpMHCFullPairEmbedderMaxTotal


class ZGridConvFlattenHead(nn.Module):
    """
    Fixed-size contact-map head (no pooling):
      z: [B, L, L, C] -> conv2d over [B, C, L, L] -> flatten -> MLP -> logits
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
        logits = self.fc2(x)  # [B, out_dim]
        return logits


class TCRpMHCClassifier(nn.Module):
    """
    TCR+pMHC model:
      (peps,mhcs,tcras,tcrbs) -> z-grid -> conv+flatten -> activation (config-driven)

    If cfg.output_activation == "sigmoid": output in (0,1) via torch.sigmoid(logits).
    If cfg.output_activation == "none": output unbounded logits.
    """
    def __init__(
        self,
        pmhc_cfg: PMHCPairConfig,
        tcr_cfg: TCRPairConfig,
        full_cfg: FullGridPairConfig,
        clf_cfg: Optional[TCRClassifierConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_to_label_range: bool = True,
        apply_mask_in_embedder: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.pmhc_cfg = pmhc_cfg
        self.tcr_cfg = tcr_cfg
        self.full_cfg = full_cfg
        self.cfg = clf_cfg if clf_cfg is not None else TCRClassifierConfig()
        self.apply_mask_in_embedder = bool(apply_mask_in_embedder)

        # Embedder returns z [B, L, L, C] with L = full_cfg.max_len_total
        self.embedder = TCRpMHCFullPairEmbedderMaxTotal(
            pmhc_cfg=pmhc_cfg,
            tcr_cfg=tcr_cfg,
            full_cfg=full_cfg,
            device=self.device,
        )

        # Fixed L and pair_dim come from full_cfg (source-of-truth)
        self.grid_len = int(getattr(full_cfg, "max_len_total"))
        pair_dim = int(getattr(full_cfg, "pair_dim"))

        out_dim = int(getattr(self.cfg, "num_classes", 1))
        if out_dim != 1:
            raise ValueError(f"Set cfg.num_classes=1 for binary/regression, got {out_dim}.")

        self.head = ZGridConvFlattenHead(
            pair_dim=pair_dim,
            grid_len=self.grid_len,
            hidden_dim=int(self.cfg.hidden_dim),
            out_dim=1,
            dropout=float(self.cfg.dropout),
            n_convs=int(getattr(self.cfg, "n_convs", 2)),
        )

        # Activation mode
        act = str(getattr(self.cfg, "output_activation", "none")).lower().strip()
        if act not in ("sigmoid", "none"):
            raise ValueError("cfg.output_activation must be 'sigmoid' or 'none'.")
        self.output_activation = act  # store string instead of module

        # Optional clamp
        self.clamp_to_label_range = bool(clamp_to_label_range)
        self.label_range: Optional[Tuple[float, float]] = getattr(self.cfg, "label_range", None)

    @classmethod
    def from_config(
        cls,
        pmhc_cfg: PMHCPairConfig,
        tcr_cfg: TCRPairConfig,
        full_cfg: FullGridPairConfig,
        clf_cfg: TCRClassifierConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_to_label_range: bool = True,
        apply_mask_in_embedder: bool = True,
    ) -> "TCRpMHCClassifier":
        return cls(
            pmhc_cfg=pmhc_cfg,
            tcr_cfg=tcr_cfg,
            full_cfg=full_cfg,
            clf_cfg=clf_cfg,
            device=device,
            clamp_to_label_range=clamp_to_label_range,
            apply_mask_in_embedder=apply_mask_in_embedder,
        )

    def forward(
        self,
        peps: List[str],
        mhcs: List[str],
        tcras: List[str],
        tcrbs: List[Optional[str]],
    ) -> torch.Tensor:
        z = self.embedder(
            peptide_list=peps,
            mhc_list=mhcs,
            tcra_list=tcras,
            tcrb_list=tcrbs,
            apply_mask=self.apply_mask_in_embedder,
        )  # [B, L, L, C]

        logits = self.head(z)  # [B, 1]
        if self.output_activation == "sigmoid":
            yhat = torch.sigmoid(logits)     # <- explicit sigmoid function
        else:
            yhat = logits                    # unbounded
        return yhat


if __name__ == "__main__":
    # smoke test (requires tcrMHCpeptideEmbedding.py to implement the embedder)
    pmhc_cfg = PMHCPairConfig()
    tcr_cfg = TCRPairConfig()
    full_cfg = FullGridPairConfig(max_len_total=360)  # you can override max_len_tcr too
    clf_cfg = TCRClassifierConfig(num_classes=1, output_activation="sigmoid")

    model = TCRpMHCClassifier.from_config(
        pmhc_cfg=pmhc_cfg,
        tcr_cfg=tcr_cfg,
        full_cfg=full_cfg,
        clf_cfg=clf_cfg,
        device="cpu",
    )

    # Dummy sequences (embedder cleans/truncates/pads; these just satisfy types)
    peps  = ["A" * pmhc_cfg.pep_len, "C" * pmhc_cfg.pep_len]
    mhcs  = ["V" * pmhc_cfg.mhc_len, "D" * pmhc_cfg.mhc_len]
    tcras = ["G" * tcr_cfg.tcr_a_max_len, "T" * tcr_cfg.tcr_a_max_len]
    tcrbs = ["S" * tcr_cfg.tcr_b_max_len, None]

    y = model(peps, mhcs, tcras, tcrbs)
    print("output shape:", y.shape)
    print("output range:", float(y.min()), float(y.max()))
    print("output values:", y[:5].view(-1))
