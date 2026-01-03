# PanImmunologyClassifier.py
"""
PanImmunologyClassifier (Z-only)
- Input: batch_seqs (list of pre-concatenated sequences like 'A:B')
- Output: positive-valued tensor [B, num_classes] in (0, +inf)
- Architecture: PanimmuneEmbedderPairs (internal embeder)
  → z [B, L, L, C] → ZOnlyPooling → Linear (hidden)
  → GELU + Dropout → Linear (logits) → Softplus (positive output)
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from model_config import ZClassifierConfig
from PanimmuneEmbedderPairs import PanimmuneEmbedderPairs


class ZOnlyPooling(nn.Module):
    """Pool only the pair grid z: [B, L, L, C].

    Output feature dimension:
        C if use_max=False; 2C if use_max=True (mean ⊕ max).
    """
    def __init__(self, pair_dim: int, use_max: bool = True):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.use_max = bool(use_max)
        self.out_dim = self.pair_dim * (2 if self.use_max else 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_mean = z.mean(dim=(1, 2))  # [B, C]
        if self.use_max:
            z_max = z.amax(dim=(1, 2))  # [B, C]
            return torch.cat([z_mean, z_max], dim=-1)
        return z_mean


class PanImmunologyClassifier(nn.Module):
    """Pan-Immunology classifier using PanimmuneEmbedderPairs directly.

    Input:  batch_seqs -> ["CHAINA:CHAINB", ...]
    Output: positive scores tensor [B, num_classes]
    """
    def __init__(
        self,
        esm_cfg,
        pair_cfg,
        pair_dim: int = 256,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_max_pool: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.embeder = PanimmuneEmbedderPairs(
            esm_cfg, pair_cfg, device=torch.device(device)
        )

        self.pool = ZOnlyPooling(pair_dim, use_max=use_max_pool)
        self.fc1 = nn.Linear(self.pool.out_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Positive, unbounded output
        self.softplus = nn.Softplus()

    @classmethod
    def from_config(
        cls,
        cfg: ZClassifierConfig,
        esm_cfg,
        pair_cfg,
    ) -> "PanImmunologyClassifier":
        return cls(
            esm_cfg=esm_cfg,
            pair_cfg=pair_cfg,
            pair_dim=cfg.pair_dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_max_pool=cfg.use_max_pool,
        )

    def forward(self, batch_seqs: List[str]) -> torch.Tensor:
        z = self.embeder(batch_seqs)      # [B, L, L, C]
        feats = self.pool(z)              # [B, F]
        logits = self.fc2(
            self.drop(self.act(self.fc1(feats)))
        )                                 # [B, num_classes]
        return self.softplus(logits)      # (0, +inf)


if __name__ == "__main__":
    from model_config import ESMConfig, PairConfig

    esm_cfg = ESMConfig()
    pair_cfg = PairConfig()
    cfg = ZClassifierConfig()

    model = PanImmunologyClassifier.from_config(cfg, esm_cfg, pair_cfg)
    seqs = ["ACDEFG:LMNPQR", "MKTFF:GGGGG:GGGGGGG", "VVVVV:DDDDDDDDD"]

    y = model(seqs)
    print("output shape:", y.shape)
    print("output values:", y)
