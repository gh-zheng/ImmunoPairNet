from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
# ... keep the rest of your imports

@dataclass
class EmbedderConfig:
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    pairformer_blocks: int = 12
    pair_c: int = 128
    allow_inter_chain: bool = False
    attn_from_last_layer_only: bool = True
    torch_dtype: Optional[str] = None
    freeze_esm: bool = True
    pad_to_multiple_of: Optional[int] = 8
    enable_cache: Optional[bool] = True
    cache_size: Optional[int] = 200_000
    use_amp: Optional[bool] = True
    max_length: Optional[int] = None
    precomputed_pkl_gz: str="esm2_token_embeddings_650M_long.pkl.gz"
    precomputed_shards_dir: str = None  # or None
    strict_model_match: bool=True
    seq_normalize:bool=False

@dataclass
class ClassifierConfig:
    num_classes: int = 2
    pool_hidden: int = 256
    mlp_hidden: int = 256
    use_layernorm: bool = True

@dataclass
class ModelConfig:
    # ✅ use default_factory so each ModelConfig gets a fresh nested object
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

def load_default_config() -> ModelConfig:
    return ModelConfig()