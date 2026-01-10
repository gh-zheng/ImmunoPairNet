# model_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


# ========================= Pair / Embedder Config ========================= #
@dataclass
class PairConfig:
    """
    Configuration for PanimmuneEmbedderPairs (one-hot + U-Net + axial bottleneck).

    This config ASSUMES fixed-length MHC-I inputs:
      - sequence format: "MHC:PEPTIDE"
      - peptide length is fixed and enforced upstream
    """

    # ---- single (one-hot -> proj) ----
    proj_dim: int = 16
    aa_vocab: str = "ACDEFGHIKLMNPQRSTVWYX"  # 20 canonical + X (unknown/pad)

    # ---- chain embedding ----
    # 0 = PAD, 1 = MHC, 2 = peptide (extra chains clamp to chain_vocab-1)
    chain_vocab: int = 8

    # ---- pair / U-Net / axial bottleneck ----
    pair_dim: int = 32
    mha_heads: int = 4
    dropout: float = 0.1

    unet_depth: int = 3
    unet_base_channels: int = 16

    chunk_rows: int = 0
    n_transformers: int = 4

    # ---- fixed-length MHC-I binding setup ----
    mhc_len: int = 34          # MHC pseudo-sequence length
    pep_len: int = 15          # fixed peptide length

    @property
    def fixed_len(self) -> int:
        """Total fixed concatenated length L = mhc_len + pep_len."""
        return int(self.mhc_len + self.pep_len)


# ========================= Head / Task Config ========================= #
@dataclass
class ZClassifierConfig:
    """
    Configuration for the Z-grid conv + flatten head.

    Default setup:
      - regression
      - output in [0,1] via sigmoid
    """

    hidden_dim: int = 512
    num_classes: int = 1
    dropout: float = 0.1

    # Conv head
    n_convs: int = 2

    # Output behavior
    task_type: Literal["regression"] = "regression"
    output_activation: Literal["sigmoid", "none"] = "sigmoid"
    label_range: tuple[float, float] = (0.0, 1.0)


# ========================= Unified Model Config ========================= #
@dataclass
class ModelConfig:
    """
    Unified configuration container.
    """
    pair: PairConfig
    classifier: ZClassifierConfig


def load_default_config() -> ModelConfig:
    """
    Default configuration for MHC-I peptide binding regression:
      - fixed-length (34 + 11)
      - conv2d + flatten head
      - sigmoid output in [0,1]
    """
    pair_cfg = PairConfig()
    cls_cfg = ZClassifierConfig()
    return ModelConfig(pair=pair_cfg, classifier=cls_cfg)
