# model_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# ========================= pMHC Pair / Embedder Config ========================= #
@dataclass
class PMHCPairConfig:
    """
    Configuration for pMHC pair embedder (one-hot + U-Net + axial bottleneck).

    Assumes fixed-length pMHC inputs:
      - MHC pseudo-sequence length = mhc_len
      - peptide length = pep_len
      - total pMHC length L_pmhc = mhc_len + pep_len
    """

    # ---- single (one-hot -> proj) ----
    proj_dim: int = 16
    aa_vocab: str = "ACDEFGHIKLMNPQRSTVWYX"

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

    # ---- fixed-length pMHC setup ----
    mhc_len: int = 34
    pep_len: int = 15

    @property
    def fixed_len(self) -> int:
        return int(self.mhc_len + self.pep_len)


# ========================= TCR Pair / Embedder Config ========================= #
@dataclass
class TCRPairConfig:
    """
    Configuration for TCR pair embedder (one-hot + U-Net + axial bottleneck).

    Inputs per item:
      - tcra (required)
      - tcrb (optional)
    Internally we concatenate:
      - if tcrb is None: concat = tcra
      - else: concat = tcra + tcrb

    We apply truncation budgets BEFORE concatenation:
      tcra truncated to tcr_a_max_len
      tcrb truncated to tcr_b_max_len
    """

    # ---- single (one-hot -> proj) ----
    proj_dim: int = 16
    aa_vocab: str = "ACDEFGHIKLMNPQRSTVWYX"

    # ---- chain embedding ----
    # 0 = PAD, 1 = alpha, 2 = beta (extra chains clamp to chain_vocab-1)
    chain_vocab: int = 8

    # ---- pair / U-Net / axial bottleneck ----
    pair_dim: int = 32
    mha_heads: int = 4
    dropout: float = 0.1

    unet_depth: int = 3
    unet_base_channels: int = 16

    chunk_rows: int = 0
    n_transformers: int = 4

    # ---- TCR truncation budgets ----
    tcr_a_max_len: int = 70
    tcr_b_max_len: int = 70

    @property
    def max_len(self) -> int:
        """Total TCR diagonal-block budget (alpha + beta)."""
        return int(self.tcr_a_max_len + self.tcr_b_max_len)


# ========================= Full Grid / Refiner / Assembly Config ========================= #
@dataclass
class FullGridPairConfig:
    """
    Configuration for assembling (TCR pair) ⊕ (pMHC pair) into a full grid and refining.

    Output:
      z_out: [B, max_len_total, max_len_total, pair_dim]

    We allocate:
      Lt = max_len_tcr (default: tcr_cfg.max_len)
      Lp = max_len_total - Lt

    The refiner itself is also (U-Net + axial bottleneck), with its own knobs
    so you can tune it independently from the two embedders.
    """

    # ---- output sizing ----
    max_len_total: int = 360
    max_len_tcr: Optional[int] = None  # if None -> use TCRPairConfig.max_len

    def resolved_max_len_tcr(self, tcr_cfg: TCRPairConfig) -> int:
        lt = int(tcr_cfg.max_len if self.max_len_tcr is None else self.max_len_tcr)
        if lt < 1 or lt >= self.max_len_total:
            raise ValueError(
                f"Invalid max_len_tcr={lt}. Must satisfy 1 <= max_len_tcr < max_len_total={self.max_len_total}."
            )
        return lt

    def resolved_max_len_pmhc(self, tcr_cfg: TCRPairConfig) -> int:
        return int(self.max_len_total - self.resolved_max_len_tcr(tcr_cfg))

    # ---- refiner architecture knobs (independent) ----
    pair_dim: int = 32
    mha_heads: int = 4
    dropout: float = 0.1

    unet_depth: int = 3
    unet_base_channels: int = 16

    chunk_rows: int = 0
    n_transformers: int = 4

    # ---- optional off-diagonal seeding (TCR×pMHC blocks) ----
    use_offdiag_seed: bool = True
    offdiag_seed_mode: Literal["outer"] = "outer"
    offdiag_seed_dim: int = 64


# ========================= Head / Task Config ========================= #
@dataclass
class ZClassifierConfig:
    """
    Configuration for a Z-grid conv + flatten head.

    Default:
      - regression
      - sigmoid output in [0,1]
    """
    hidden_dim: int = 512
    num_classes: int = 1
    dropout: float = 0.1

    n_convs: int = 2

    task_type: Literal["regression"] = "regression"
    output_activation: Literal["sigmoid", "none"] = "sigmoid"
    label_range: tuple[float, float] = (0.0, 1.0)


# ========================= Unified Model Config ========================= #
@dataclass
class ModelConfig:
    """
    Unified configuration container for the full TCR+pMHC model.
    """
    pmhc: PMHCPairConfig
    tcr: TCRPairConfig
    full: FullGridPairConfig
    classifier: ZClassifierConfig


def load_default_config() -> ModelConfig:
    """
    Default configuration for TCR+pMHC full-grid tasks.
    """
    pmhc_cfg = PMHCPairConfig()
    tcr_cfg = TCRPairConfig()
    full_cfg = FullGridPairConfig()
    cls_cfg = ZClassifierConfig()
    return ModelConfig(pmhc=pmhc_cfg, tcr=tcr_cfg, full=full_cfg, classifier=cls_cfg)
