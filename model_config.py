# model_config.py
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
# ... keep the rest of your imports

@dataclass
class ESMConfig:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    layer: Optional[int] = None     # None -> last hidden
    max_tokens: int = 1024          # per window (incl. special tokens)
    stride: int = 896               # window overlap (< max_tokens)
    sep_token: str = ":"            # you concatenate chains upstream
    pad_side: str = "right"
    freeze: bool = True             # freeze ESM parameters or not


@dataclass
class PairConfig:
    proj_dim: int = 128             # projector output dim for single embeddings
    pair_dim: int = 128             # pair channel dim (U-Net channels start here)
    mha_heads: int = 8
    dropout: float = 0.1
    # U-Net depth & base channels (first conv after pair init)
    unet_depth: int = 4
    unet_base_channels: int = 128
    # Axial attention memory knobs
    chunk_rows: int = 0             # >0 to chunk B*L rows/cols in attention
    # Number of axial Transformer blocks in bottleneck
    n_transformers: int = 16

@dataclass
class ZClassifierConfig:
    pair_dim: int = 128        # C = channels of pair embedding (last dim of z)
    hidden_dim: int = 512      # hidden size of the MLP head
    num_classes: int = 1       # number of output classes
    dropout: float = 0.1       # dropout in the MLP head
    use_max_pool: bool = True  # if True, concat(mean(z), max(z)); else mean(z) only
@dataclass
class ModelConfig:
    """Unified container for model-related configurations."""
    esm: ESMConfig
    pair: PairConfig
    classifier: ZClassifierConfig


def load_default_config() -> ModelConfig:
    """
    Create a unified default configuration combining ESM, Pair, and Classifier configs.
    Returns:
        ModelConfig object containing:
          - esm (ESMConfig)
          - pair (PairConfig)
          - classifier (ZClassifierConfig)
    """
    esm_cfg = ESMConfig()
    pair_cfg = PairConfig()
    cls_cfg = ZClassifierConfig()
    return ModelConfig(esm=esm_cfg, pair=pair_cfg, classifier=cls_cfg)