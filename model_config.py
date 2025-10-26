# model_config.py
from dataclasses import dataclass
from typing import Optional

# ========================== Pair (U-Net + Transformer) ========================== #
@dataclass
class PairConfig:
    # Single-embedding projection size (one-hot → proj_dim)
    proj_dim: int = 128
    # Pair grid channels (input to U-Net)
    pair_dim: int = 128
    # U-Net backbone
    unet_base_channels: int = 64
    unet_depth: int = 3
    dropout: float = 0.1
    # Axial attention bottleneck
    n_transformers: int = 4
    mha_heads: int = 8
    # Memory knob for axial attention (rows/cols chunking)
    chunk_rows: int = 512
    # Early spatial downsampling factor (1 = off)
    pre_pool_factor: int = 2


# ============================= Z-only Classifier ================================ #
@dataclass
class ZClassifierConfig:
    pair_dim: int = 128        # C = channels of pair embedding (last dim of z)
    hidden_dim: int = 512      # hidden size of the MLP head
    num_classes: int = 1       # number of output classes
    dropout: float = 0.1       # dropout in the MLP head
    use_max_pool: bool = True  # concat(mean(z), max(z)) if True; else mean(z)


# =========================== Unified Model Config ============================== #
@dataclass
class ModelConfig:
    """Unified container for model-related configurations."""
    pair: PairConfig
    classifier: ZClassifierConfig


def load_default_config() -> ModelConfig:
    """
    Create a unified default configuration combining Pair and Classifier configs.
    Returns:
        ModelConfig object containing:
          - pair (PairConfig)
          - classifier (ZClassifierConfig)
    """
    pair_cfg = PairConfig()
    cls_cfg = ZClassifierConfig()
    return ModelConfig(pair=pair_cfg, classifier=cls_cfg)
