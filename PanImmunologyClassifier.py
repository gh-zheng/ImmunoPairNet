# PanImmunologyClassifier.py
"""
PanImmunologyClassifier (Z-only, ESM-free)
- Input: batch_seqs (list of pre-concatenated sequences like 'A:B')
- Output: logits tensor [B, num_classes]
- Architecture: PanimmuneEmbedderPairs (internal embeder) → z [B, L, L, C]
               → ZOnlyPooling → Linear (hidden) → GELU+Dropout → Linear (logits)
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model_config import ZClassifierConfig, PairConfig
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
        # z is [B, L, L, C]
        z_mean = z.mean(dim=(1, 2))  # [B, C]
        if self.use_max:
            z_max = z.amax(dim=(1, 2))  # [B, C]
            return torch.cat([z_mean, z_max], dim=-1)
        return z_mean


class PanImmunologyClassifier(nn.Module):
    """Pan-Immunology classifier using PanimmuneEmbedderPairs directly for embedding generation.

    Input:  batch_seqs -> ["CHAINA:CHAINB", ...]
    Output: logits tensor [B, num_classes]
    """
    def __init__(
        self,
        classifier_cfg: ZClassifierConfig,
        pair_cfg: PairConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internal embedder (ESM-free, one-hot front end)
        self.embeder = PanimmuneEmbedderPairs.from_config(pair_cfg, device=self.device)

        # Pooling + MLP head
        self.pool = ZOnlyPooling(pair_dim=classifier_cfg.pair_dim, use_max=classifier_cfg.use_max_pool)
        self.fc1 = nn.Linear(self.pool.out_dim, classifier_cfg.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(classifier_cfg.dropout)
        self.fc2 = nn.Linear(classifier_cfg.hidden_dim, classifier_cfg.num_classes)

    # ---- Backward-compatible factory ----
    @classmethod
    def from_config(
        cls,
        classifier_cfg: ZClassifierConfig,
        *cfgs,
        device: Optional[torch.device] = None,
    ) -> "PanImmunologyClassifier":
        """
        Accepts either:
          - (classifier_cfg, pair_cfg)                 # new, ESM-free
          - (classifier_cfg, esm_cfg, pair_cfg)        # legacy; esm_cfg is ignored
        """
        if len(cfgs) == 1:
            pair_cfg = cfgs[0]
        elif len(cfgs) == 2:
            # legacy call form: (esm_cfg, pair_cfg) -> ignore esm_cfg
            _, pair_cfg = cfgs
        else:
            raise TypeError("from_config expects (classifier_cfg, pair_cfg) or (classifier_cfg, esm_cfg, pair_cfg)")
        return cls(classifier_cfg=classifier_cfg, pair_cfg=pair_cfg, device=device)

    def forward(self, batch_seqs: List[str]) -> torch.Tensor:
        # z: [B, L, L, C]
        z = self.embeder(batch_seqs)
        feats = self.pool(z)  # [B, F]
        logits = self.fc2(self.drop(self.act(self.fc1(feats))))
        return logits


if __name__ == "__main__":
    # ESM-free smoke test
    from model_config import load_default_config

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_default_config()
    model = PanImmunologyClassifier.from_config(cfg.classifier, cfg.pair, device=dev).to(dev)

    seqs = ["ACDEFG:LMNPQR", "MKTFF:GGGGG:GGGGGGG", "VVVVV:DDDDDDDDD"]
    with torch.no_grad():
        logits = model(seqs)
    print("logits shape:", tuple(logits.shape))  # [B, num_classes]
