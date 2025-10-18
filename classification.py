# modelconfig.py
from __future__ import annotations
from model_config import EmbedderConfig, ClassifierConfig, ModelConfig
from typing import Optional, Any, Dict, List
import inspect
import torch
import torch.nn as nn

# Your embedder (keep the import path you already use)
from PanimmuneEmbedderPairs import PanimmuneEmbedderPairs


def load_default_config() -> ModelConfig:
    """Return a default config you can tweak in one place."""
    return ModelConfig()


# ====================== Pair-aware pooling & classifier ====================== #
class PairAwarePooling(nn.Module):
    def __init__(self, c_s: int, c_z: int, hidden: int = 256):
        super().__init__()
        self.proj_pair = nn.Linear(c_z, hidden, bias=False)
        self.proj_single = nn.Linear(c_s, hidden, bias=False)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, s, z, single_mask, pair_mask):
        # z: [B, L, L, Cz] → project then masked mean over j
        Pz = self.proj_pair(z)  # [B, L, L, H]
        valid_ij = single_mask.unsqueeze(1) & single_mask.unsqueeze(2) & pair_mask  # [B,L,L]
        valid_ij_f = valid_ij.unsqueeze(-1).to(Pz.dtype)
        denom = valid_ij_f.sum(dim=2).clamp_min(1.0)  # [B,L,1]
        m = (Pz * valid_ij_f).sum(dim=2) / denom      # [B, L, H]

        # fuse with single features and pool
        s_h = self.proj_single(s)                     # [B, L, H]
        h = torch.tanh(self.gate(torch.cat([s_h, m], dim=-1)))  # [B, L, H]
        logits = self.attn(h).squeeze(-1)             # [B, L]
        logits = logits.masked_fill(~single_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)       # [B, L]
        pooled = torch.bmm(weights.unsqueeze(1), s).squeeze(1)  # [B, C_s]
        return pooled


class PairAwareClassifier(nn.Module):
    def __init__(self, embedder: PanimmuneEmbedderPairs, cfg: ClassifierConfig):
        super().__init__()
        self.embedder = embedder
        c_s = embedder.esm.config.hidden_size
        c_z = embedder.project_z.out_features
        self.pool = PairAwarePooling(c_s, c_z, hidden=cfg.pool_hidden)

        head: List[nn.Module] = []
        if cfg.use_layernorm:
            head.append(nn.LayerNorm(c_s))
        head += [nn.Linear(c_s, cfg.mlp_hidden), nn.ReLU(), nn.Linear(cfg.mlp_hidden, cfg.num_classes)]
        self.classifier = nn.Sequential(*head)

    def forward(self, batch_samples: List[List[str]]):
        s, z, (single_mask, pair_mask), _ = self.embedder(batch_samples, return_metadata=True)
        pooled = self.pool(s, z, single_mask, pair_mask)
        return self.classifier(pooled)


# ============================ Factories / Builders ============================ #
def _maybe_cast_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    low = dtype_str.lower()
    if low in ("bf16", "bfloat16"):
        return torch.bfloat16
    if low in ("fp16", "float16", "half"):
        return torch.float16
    if low in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unrecognized dtype string: {dtype_str}")


def build_embedder(cfg: EmbedderConfig, device: Optional[torch.device] = None) -> PanimmuneEmbedderPairs:
    """
    Create PanimmuneEmbedderPairs and apply freeze/precision knobs.
    We only pass kwargs that the class actually supports (safe on original or optimized versions).
    """
    # Prepare kwargs
    kwargs: Dict[str, Any] = dict(
        esm_model_name=cfg.esm_model_name,
        pairformer_blocks=cfg.pairformer_blocks,
        pair_c=cfg.pair_c,
        allow_inter_chain=cfg.allow_inter_chain,
        attn_from_last_layer_only=cfg.attn_from_last_layer_only,
        torch_dtype=_maybe_cast_dtype(cfg.torch_dtype),
        #precomputed_pkl_gz=cfg.precomputed_pkl_gz,
        #precomputed_shards_dir=cfg.precomputed_shards_dir,  # or None
        #strict_model_match=cfg.strict_model_match,
        #seq_normalize=cfg.seq_normalize,
    )

    # Optional / optimized kwargs: only pass if supported
    sig = inspect.signature(PanimmuneEmbedderPairs.__init__)
    optional_keys = ["pad_to_multiple_of", "enable_cache", "cache_size", "use_amp", "max_length"]
    for k in optional_keys:
        if k in sig.parameters and getattr(cfg, k) is not None:
            kwargs[k] = getattr(cfg, k)

    # Instantiate
    embedder = PanimmuneEmbedderPairs(**kwargs)
    if device is not None:
        embedder = embedder.to(device)

    # Freeze or unfreeze ESM as requested (works for both original and optimized variants)
    try:
        if cfg.freeze_esm:
            embedder.esm.eval().requires_grad_(False)
        else:
            embedder.esm.train().requires_grad_(True)
    except Exception:
        # If your Panimmune implementation manages freezing internally, ignore
        pass

    return embedder


def build_classifier(embedder: PanimmuneEmbedderPairs, cfg: ClassifierConfig, device: Optional[torch.device] = None) -> nn.Module:
    model = PairAwareClassifier(embedder, cfg)
    if device is not None:
        model = model.to(device)
    return model
