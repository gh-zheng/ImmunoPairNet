# model_config.py
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
# ... keep the rest of your imports

from dataclasses import dataclass
from typing import Optional

@dataclass
class ESMConfig:
    # Stronger default model
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    # Use the last layer by default; set -2 or -3 for more general features
    layer: Optional[int] = None          # None -> last hidden; or use -2 / -3
    # Sliding-window limits
    max_tokens: int = 1024               # per window (incl. special tokens)
    stride: int = 256                    # window overlap (< max_tokens); smaller saves VRAM
    # Input formatting
    sep_token: str = ":"                 # upstream chain separator; replaced by 'X' internally
    pad_side: str = "right"
    # Freezing / device
    freeze: bool = True                  # freeze ESM params; projector & downstream still train
    run_esm_device: str = "same"         # "same" | "cpu" | "cuda" (CPU offload reduces VRAM)

@dataclass
class PairConfig:
    # Single-embedding projection size
    proj_dim: int = 128
    # Pair grid channels (first conv input channels)
    pair_dim: int = 64
    # U-Net backbone
    unet_base_channels: int = 64
    unet_depth: int = 3
    dropout: float = 0.1
    # Axial attention bottleneck
    n_transformers: int = 4
    mha_heads: int = 8
    # Memory knob for axial attention (rows/cols chunking)
    chunk_rows: int = 512               # >0 to chunk B*L rows/cols in attention
    # Early spatial downsampling for the pair grid; final output restored to L×L
    pre_pool_factor: int = 2            # 1 (off), 2, or 4


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