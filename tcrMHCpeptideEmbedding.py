# tcrMHCpeptideEmbedding.py
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import PMHCPairConfig, TCRPairConfig, FullGridPairConfig
from MHCpeptideEmbedding import MHCpeptideEmbedderPairs  # existing pMHC embedder (fixed-length internally)


# ============================ Helpers ============================ #

_AA_SET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def _canon_aa_seq(seq: str) -> str:
    if not isinstance(seq, str):
        raise TypeError(f"Sequence must be str, got {type(seq)}.")
    s = seq.upper()
    for ch in (":", "|", " ", "\t", "\n", "\r"):
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch in _AA_SET)
    if not s:
        raise ValueError(f"Empty/invalid sequence after cleaning: {repr(seq)}")
    return s


def _norm_empty_to_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return s


def _pad_list_of_tensors_2d(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(tensors) == 0:
        raise ValueError("Cannot pad empty tensor list.")
    B = len(tensors)
    Lmax = max(t.size(0) for t in tensors)
    D = tensors[0].size(-1)
    dev = tensors[0].device
    out = tensors[0].new_full((B, Lmax, D), pad_value)
    mask = torch.zeros((B, Lmax), dtype=torch.bool, device=dev)
    for i, t in enumerate(tensors):
        l = t.size(0)
        out[i, :l] = t
        mask[i, :l] = True
    return out, mask


def _pad_list_of_1d_long(
    tensors: List[torch.Tensor],
    pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(tensors) == 0:
        raise ValueError("Cannot pad empty tensor list.")
    B = len(tensors)
    Lmax = max(t.numel() for t in tensors)
    dev = tensors[0].device
    out = tensors[0].new_full((B, Lmax), pad_value)
    mask = torch.zeros((B, Lmax), dtype=torch.bool, device=dev)
    for i, t in enumerate(tensors):
        l = t.numel()
        out[i, :l] = t
        mask[i, :l] = True
    return out, mask


def _make_pair_mask(mask_1d: torch.Tensor) -> torch.Tensor:
    return mask_1d.unsqueeze(2) & mask_1d.unsqueeze(1)


def _crop_pad_pair(z: torch.Tensor, L: int) -> torch.Tensor:
    """Crop/pad [B, Lz, Lz, D] -> [B, L, L, D] with zeros."""
    if z.dim() != 4:
        raise ValueError(f"Expected z [B,L,L,D], got {tuple(z.shape)}")
    B, Lz1, Lz2, D = z.shape
    if Lz1 != Lz2:
        raise ValueError("z must be square")
    out = z.new_zeros((B, L, L, D))
    l = min(L, Lz1)
    out[:, :l, :l, :] = z[:, :l, :l, :]
    return out


def _diag_concat_pairs(z_tcr: torch.Tensor, z_pmhc: torch.Tensor) -> torch.Tensor:
    """Block-diagonal concat -> [B, Lt+Lp, Lt+Lp, D]."""
    B1, Lt, _, D1 = z_tcr.shape
    B2, Lp, _, D2 = z_pmhc.shape
    if B1 != B2 or D1 != D2:
        raise ValueError("Batch or D mismatch.")
    L = Lt + Lp
    z_full = z_tcr.new_zeros((B1, L, L, D1))
    z_full[:, :Lt, :Lt, :] = z_tcr
    z_full[:, Lt:, Lt:, :] = z_pmhc
    return z_full


# ===================== One-hot (projected) ====================== #

class OneHotEmbedder(nn.Module):
    def __init__(self, proj_dim: int, vocab: str = "ACDEFGHIKLMNPQRSTVWYX", device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
        if "X" not in self.aa_to_idx:
            raise ValueError("aa_vocab must include 'X'.")
        self.unk = self.aa_to_idx["X"]
        self.projector = nn.Linear(self.vocab_size, proj_dim, bias=True)

    def _encode_one(self, seq: str) -> torch.Tensor:
        idx = torch.full((len(seq),), self.unk, dtype=torch.long, device=self.device)
        for i, ch in enumerate(seq):
            idx[i] = self.aa_to_idx.get(ch, self.unk)
        onehot = F.one_hot(idx, num_classes=self.vocab_size).float()
        h = self.projector(onehot)
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch_concat_seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        reps = [self._encode_one(s) for s in batch_concat_seqs]
        lens = [t.size(0) for t in reps]
        s_pad, smask = _pad_list_of_tensors_2d(reps)
        return s_pad, smask, lens


# ================= Single -> Pair Initialization ================= #

class PairInitOPM(nn.Module):
    def __init__(self, single_dim: int, pair_dim: int, use_bias: bool = True):
        super().__init__()
        self.proj_i = nn.Linear(single_dim, pair_dim, bias=use_bias)
        self.proj_j = nn.Linear(single_dim, pair_dim, bias=use_bias)
        self.mix = nn.Linear(2 * pair_dim, pair_dim, bias=use_bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        si = self.proj_i(s)
        sj = self.proj_j(s)
        mul = si.unsqueeze(2) * sj.unsqueeze(1)
        cat = torch.cat(
            [
                si.unsqueeze(2).expand(-1, -1, si.size(1), -1),
                sj.unsqueeze(1).expand(-1, si.size(1), -1, -1),
            ],
            dim=-1,
        )
        z0 = self.mix(cat) + mul
        return torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0)


# ===================== U-Net + Axial bottleneck ===================== #

class ResBlock2D(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.proj(x)
        return self.act(torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))


class DownStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock2D(c_in, c_out, dropout)
        self.res2 = ResBlock2D(c_out, c_out, dropout)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x)
        x = self.res2(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.res1 = ResBlock2D(c_out + c_skip, c_out, dropout)
        self.res2 = ResBlock2D(c_out, c_out, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
            x = x[..., :skip.size(-2), :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


class AxialSelfAttention2D(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.row_mha = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.col_mha = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.row_ln = nn.LayerNorm(channels)
        self.col_ln = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.chunk_rows = int(chunk_rows) if chunk_rows else 0

    def _run_chunked(self, mha: nn.MultiheadAttention, x: torch.Tensor, kpm: Optional[torch.Tensor]) -> torch.Tensor:
        if self.chunk_rows and self.chunk_rows > 0:
            outs = []
            for i in range(0, x.size(0), self.chunk_rows):
                xi = x[i:i + self.chunk_rows]
                km = None if kpm is None else kpm[i:i + self.chunk_rows]
                oi, _ = mha(xi, xi, xi, key_padding_mask=km, need_weights=False)
                outs.append(oi)
            return torch.cat(outs, dim=0)
        o, _ = mha(x, x, x, key_padding_mask=kpm, need_weights=False)
        return o

    def forward(self, z: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L1, L2 = z.shape
        if L1 != L2:
            raise ValueError("square grid required")
        z = z.permute(0, 2, 3, 1).contiguous()  # [B,L,L,C]
        _, L, _, H = z.shape

        zr = z.reshape(B * L, L, H)
        key_padding = None
        if mask_1d is not None:
            m = (~mask_1d.bool())
            key_padding = m.unsqueeze(1).expand(B, L, L).reshape(B * L, L)
        r_out = self._run_chunked(self.row_mha, zr, key_padding)
        zr = self.row_ln(zr + self.drop(r_out))

        zc = zr.reshape(B, L, L, H).transpose(1, 2).reshape(B * L, L, H)
        key_padding = None
        if mask_1d is not None:
            m = (~mask_1d.bool())
            key_padding = m.unsqueeze(2).expand(B, L, L).reshape(B * L, L)
        c_out = self._run_chunked(self.col_mha, zc, key_padding)
        zc = self.col_ln(zc + self.drop(c_out))

        z = zc.reshape(B, L, L, H).transpose(1, 2)
        return z.permute(0, 3, 1, 2).contiguous()


class FeedForward2D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AxialTransformerStack(nn.Module):
    def __init__(self, n_layers: int, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn": AxialSelfAttention2D(channels, heads, dropout, chunk_rows),
                "ff": FeedForward2D(channels, dropout),
            }))

    def forward(self, x: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer["attn"](x, mask_1d)
            x = layer["ff"](x)
        return x


# ========================= TCR pair embedder ========================= #

class TCREmbedderPairs(nn.Module):
    def __init__(self, pair_cfg: TCRPairConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pair_cfg = pair_cfg
        self.proj_dim = int(pair_cfg.proj_dim)

        aa_vocab = getattr(pair_cfg, "aa_vocab", "ACDEFGHIKLMNPQRSTVWYX")
        self.encoder = OneHotEmbedder(self.proj_dim, vocab=aa_vocab, device=self.device)

        self.chain_vocab = int(getattr(pair_cfg, "chain_vocab", 8))
        self.chain_embed = nn.Embedding(self.chain_vocab, self.proj_dim, padding_idx=0)

        self.alpha_pos = nn.Parameter(torch.tensor(1.0))
        self.alpha_chain = nn.Parameter(torch.tensor(1.0))
        self.single_post = nn.LayerNorm(self.proj_dim)

        self.single_to_pair = PairInitOPM(self.proj_dim, int(pair_cfg.pair_dim), use_bias=True)

        C0 = int(pair_cfg.unet_base_channels)
        self.in_conv = nn.Conv2d(int(pair_cfg.pair_dim), C0, 3, padding=1)

        downs, ch = [], C0
        for _ in range(int(pair_cfg.unet_depth)):
            downs.append(DownStage(ch, ch * 2, float(pair_cfg.dropout)))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        self.bott_stack = AxialTransformerStack(
            n_layers=int(pair_cfg.n_transformers),
            channels=ch,
            heads=int(pair_cfg.mha_heads),
            dropout=float(pair_cfg.dropout),
            chunk_rows=int(getattr(pair_cfg, "chunk_rows", 0)),
        )

        ups = []
        for _ in range(int(pair_cfg.unet_depth)):
            ups.append(UpStage(ch, ch, ch // 2, float(pair_cfg.dropout)))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(C0, int(pair_cfg.pair_dim), 3, padding=1)

        self.tcr_a_max_len = int(getattr(pair_cfg, "tcr_a_max_len", 70))
        self.tcr_b_max_len = int(getattr(pair_cfg, "tcr_b_max_len", 70))

    def _build_sinusoidal_pos(self, L: int, device) -> torch.Tensor:
        d = self.proj_dim
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d // 2, device=device, dtype=torch.float32).unsqueeze(0)
        denom = torch.exp(-math.log(10000.0) * (2 * i) / d)
        pe = torch.cat([torch.sin(pos * denom), torch.cos(pos * denom)], dim=1)
        if d % 2 == 1:
            pe = torch.cat([pe, torch.zeros(L, 1, device=device)], dim=1)
        return pe

    @staticmethod
    def _resize_mask_1d(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.shape[1] == target_len:
            return mask
        m = mask.float().unsqueeze(1).unsqueeze(1)
        m = F.interpolate(m, size=(1, target_len), mode="nearest")
        return (m.squeeze(1).squeeze(1) > 0.5)

    @staticmethod
    def _truncate(s: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        return s[:max_len] if len(s) > max_len else s

    def forward(self, tcra_list: List[str], tcrb_list: List[Optional[str]]) -> torch.Tensor:
        concat_seqs: List[str] = []
        chain_id_vecs: List[torch.Tensor] = []

        for a, b in zip(tcra_list, tcrb_list):
            a = _canon_aa_seq(a)
            a = self._truncate(a, self.tcr_a_max_len)

            b = _norm_empty_to_none(b)
            if b is None:
                concat = a
                ids = [1] * len(a)  # alpha
            else:
                b = _canon_aa_seq(b)
                b = self._truncate(b, self.tcr_b_max_len)
                concat = a + b
                ids = [1] * len(a) + [2] * len(b)  # alpha then beta

            concat_seqs.append(concat)
            ids = [min(i, self.chain_vocab - 1) for i in ids]
            chain_id_vecs.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        s, smask, lens = self.encoder(concat_seqs)
        cid_pad, _ = _pad_list_of_1d_long(chain_id_vecs, pad_value=0)

        pos_list = [self._build_sinusoidal_pos(L, s.device) for L in lens]
        pos_pad, _ = _pad_list_of_tensors_2d(pos_list)

        chain_e = self.chain_embed(cid_pad)
        s = s + self.alpha_pos * pos_pad + self.alpha_chain * chain_e
        s = s * smask.unsqueeze(-1).float()
        s = self.single_post(s)

        z = self.single_to_pair(s).permute(0, 3, 1, 2).contiguous()

        x = self.in_conv(z)
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        _, _, Ld, _ = x.shape
        smask_scaled = self._resize_mask_1d(smask, Ld)
        x = self.bott_stack(x, smask_scaled)

        for up in self.ups:
            x = up(x, skips.pop())

        out = self.out_conv(x).permute(0, 2, 3, 1).contiguous()
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ========================= Full-grid refiner ========================= #

class PairDiagRefinerUNet(nn.Module):
    def __init__(self, cfg: FullGridPairConfig):
        super().__init__()
        self.cfg = cfg
        C0 = int(cfg.unet_base_channels)
        pair_dim = int(cfg.pair_dim)

        self.in_conv = nn.Conv2d(pair_dim, C0, 3, padding=1)

        downs, ch = [], C0
        for _ in range(int(cfg.unet_depth)):
            downs.append(DownStage(ch, ch * 2, float(cfg.dropout)))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        self.bott_stack = AxialTransformerStack(
            n_layers=int(cfg.n_transformers),
            channels=ch,
            heads=int(cfg.mha_heads),
            dropout=float(cfg.dropout),
            chunk_rows=int(getattr(cfg, "chunk_rows", 0)),
        )

        ups = []
        for _ in range(int(cfg.unet_depth)):
            ups.append(UpStage(ch, ch, ch // 2, float(cfg.dropout)))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(C0, pair_dim, 3, padding=1)

    @staticmethod
    def _resize_mask_1d(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.shape[1] == target_len:
            return mask
        m = mask.float().unsqueeze(1).unsqueeze(1)
        m = F.interpolate(m, size=(1, target_len), mode="nearest")
        return (m.squeeze(1).squeeze(1) > 0.5)

    def forward(self, z_full: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_1d is not None:
            z_full = z_full * _make_pair_mask(mask_1d).unsqueeze(-1).float()

        x = z_full.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        m_scaled = None
        if mask_1d is not None:
            _, _, Ld, _ = x.shape
            m_scaled = self._resize_mask_1d(mask_1d, Ld)

        x = self.bott_stack(x, m_scaled)

        for up in self.ups:
            x = up(x, skips.pop())

        out = self.out_conv(x).permute(0, 2, 3, 1).contiguous()
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if mask_1d is not None:
            out = out * _make_pair_mask(mask_1d).unsqueeze(-1).float()
        return out


# ========================= Main wrapper (max_len_total) ========================= #

class TCRpMHCFullPairEmbedderMaxTotal(nn.Module):
    """
    Output: z_out [B, L, L, pair_dim] where L = full_cfg.max_len_total.

    Pipeline:
      1) Build pMHC concat strings "MHC:PEPTIDE" and embed => z_pmhc [B, Lp0, Lp0, D]
      2) Embed TCR (alpha/beta) => z_tcr [B, Lt0, Lt0, D]
      3) Crop/pad each to fixed budgets:
           Lt = full_cfg.resolved_max_len_tcr(tcr_cfg)
           Lp = full_cfg.max_len_total - Lt
      4) Block-diagonal concat => z_full [B, L, L, D]
      5) (optional) seed off-diagonal TCR×pMHC blocks (config-driven)
      6) Refine with U-Net + axial attention => z_out [B, L, L, D]
    """
    def __init__(
        self,
        pmhc_cfg: PMHCPairConfig,
        tcr_cfg: TCRPairConfig,
        full_cfg: FullGridPairConfig,
        *,
        device: Optional[torch.device] = None,
        pmhc_sep: str = ":",
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pmhc_cfg = pmhc_cfg
        self.tcr_cfg = tcr_cfg
        self.full_cfg = full_cfg
        self.pmhc_sep = str(pmhc_sep)

        self.max_len_total = int(full_cfg.max_len_total)
        self.max_len_tcr = int(full_cfg.resolved_max_len_tcr(tcr_cfg))
        self.max_len_pmhc = int(full_cfg.resolved_max_len_pmhc(tcr_cfg))

        if self.max_len_total < 2:
            raise ValueError("full_cfg.max_len_total must be >= 2")

        # Pair dim must match across the system (you can add projections later if you want)
        Dp = int(pmhc_cfg.pair_dim)
        Dt = int(tcr_cfg.pair_dim)
        Dr = int(full_cfg.pair_dim)
        if not (Dp == Dt == Dr):
            raise ValueError(f"pair_dim mismatch: pmhc={Dp} tcr={Dt} full/refiner={Dr}")

        self.pmhc = MHCpeptideEmbedderPairs(pmhc_cfg, device=self.device).to(self.device)
        self.tcr = TCREmbedderPairs(tcr_cfg, device=self.device).to(self.device)
        self.refiner = PairDiagRefinerUNet(full_cfg).to(self.device)

        # Optional off-diagonal seed (uses diag features as cheap single surrogates)
        self.use_offdiag_seed = bool(getattr(full_cfg, "use_offdiag_seed", True))
        self.offdiag_seed_dim = int(getattr(full_cfg, "offdiag_seed_dim", 64))
        self.offdiag_seed_mode = str(getattr(full_cfg, "offdiag_seed_mode", "outer")).lower().strip()
        if self.offdiag_seed_mode not in ("outer",):
            raise ValueError("full_cfg.offdiag_seed_mode must be 'outer' for now.")

        if self.use_offdiag_seed:
            pair_dim = int(full_cfg.pair_dim)
            self.tcr_diag_proj = nn.Linear(pair_dim, self.offdiag_seed_dim, bias=True)
            self.pmhc_diag_proj = nn.Linear(pair_dim, self.offdiag_seed_dim, bias=True)
            self.offdiag_out = nn.Linear(self.offdiag_seed_dim, pair_dim, bias=True)

    def _build_pmhc_batch(self, mhc_list: List[str], peptide_list: List[str]) -> List[str]:
        # NOTE: We leave fixed-length enforcement to your dataset / upstream,
        # but we still canonicalize to be safe.
        batch = []
        for mhc, pep in zip(mhc_list, peptide_list):
            mhc = _canon_aa_seq(mhc)
            pep = _canon_aa_seq(pep)
            batch.append(f"{mhc}{self.pmhc_sep}{pep}")
        return batch

    def _seed_offdiag(self, z_full: torch.Tensor, Lt: int, Lp: int) -> torch.Tensor:
        """
        Seed TCR×pMHC off-diagonal blocks using projected diagonals:
          diag_t: [B,Lt,D], diag_p: [B,Lp,D]
          seed: outer over seed_dim -> map back to pair_dim.
        """
        if not self.use_offdiag_seed:
            return z_full

        B, L, _, D = z_full.shape
        assert L == Lt + Lp

        z_t = z_full[:, :Lt, :Lt, :]
        z_p = z_full[:, Lt:, Lt:, :]

        # diagonal "single surrogates"
        tdiag = z_t[:, torch.arange(Lt, device=z_full.device), torch.arange(Lt, device=z_full.device), :]  # [B,Lt,D]
        pdiag = z_p[:, torch.arange(Lp, device=z_full.device), torch.arange(Lp, device=z_full.device), :]  # [B,Lp,D]

        tfeat = self.tcr_diag_proj(tdiag)   # [B,Lt,S]
        pfeat = self.pmhc_diag_proj(pdiag)  # [B,Lp,S]

        # outer (dot in seed space) => [B,Lt,Lp]
        score = torch.einsum("b i s, b j s -> b i j", tfeat, pfeat) / math.sqrt(max(1, self.offdiag_seed_dim))
        # lift to pair_dim
        seed = self.offdiag_out(score.unsqueeze(-1).expand(-1, -1, -1, self.offdiag_seed_dim))  # [B,Lt,Lp,D]

        # write both off-diagonal blocks symmetrically
        z_full = z_full.clone()
        z_full[:, :Lt, Lt:, :] = z_full[:, :Lt, Lt:, :] + seed
        z_full[:, Lt:, :Lt, :] = z_full[:, Lt:, :Lt, :] + seed.transpose(1, 2)
        return torch.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        peptide_list: List[str],
        mhc_list: List[str],
        tcra_list: List[str],
        tcrb_list: List[Optional[str]],
        *,
        apply_mask: bool = True,
    ) -> torch.Tensor:
        B = len(peptide_list)
        if not (len(mhc_list) == len(tcra_list) == len(tcrb_list) == B):
            raise ValueError("All inputs must have same batch size.")

        # 1) Embed pMHC
        pmhc_batch = self._build_pmhc_batch(mhc_list, peptide_list)
        z_pmhc = self.pmhc(pmhc_batch)  # [B, Lp0, Lp0, D]

        # 2) Embed TCR
        z_tcr = self.tcr(tcra_list, tcrb_list)  # [B, Lt0, Lt0, D]

        # 3) Crop/pad to fixed budgets from full_cfg
        Lt = self.max_len_tcr
        Lp = self.max_len_pmhc
        z_tcr_fix = _crop_pad_pair(z_tcr, Lt)
        z_pmhc_fix = _crop_pad_pair(z_pmhc, Lp)

        # 4) Block-diagonal concat
        z_full = _diag_concat_pairs(z_tcr_fix, z_pmhc_fix)  # [B, L, L, D]

        # 5) Optional off-diagonal seeding (cheap cross-block init)
        z_full = self._seed_offdiag(z_full, Lt=Lt, Lp=Lp)

        # 6) Optional approximate mask
        mask_1d = None
        if apply_mask:
            Lt_dyn = int(z_tcr.shape[1])
            Lp_dyn = int(z_pmhc.shape[1])
            tcr_valid = torch.zeros((B, Lt), dtype=torch.bool, device=z_full.device)
            pmhc_valid = torch.zeros((B, Lp), dtype=torch.bool, device=z_full.device)
            tcr_valid[:, :min(Lt_dyn, Lt)] = True
            pmhc_valid[:, :min(Lp_dyn, Lp)] = True
            mask_1d = torch.cat([tcr_valid, pmhc_valid], dim=1)

        # 7) Refine full grid
        z_out = self.refiner(z_full, mask_1d=mask_1d)
        return z_out


# =============================== Smoke test =============================== #

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pmhc_cfg = PMHCPairConfig()
    tcr_cfg = TCRPairConfig()
    full_cfg = FullGridPairConfig(max_len_total=360, max_len_tcr=None)  # None => tcr_cfg.max_len

    model = TCRpMHCFullPairEmbedderMaxTotal(
        pmhc_cfg=pmhc_cfg,
        tcr_cfg=tcr_cfg,
        full_cfg=full_cfg,
        device=dev,
    ).to(dev)

    peptide = ["NLVPMVATV", "SIINFEKL"]
    mhc     = ["A" * 180, "B" * 180]
    tcra    = ["CASSIRSSYEQYF" * 3, "CASSLSTDTQYF" * 3]
    tcrb    = ["CASSQETQYF" * 3, None]

    z = model(peptide, mhc, tcra, tcrb, apply_mask=True)
    print("z_out shape:", tuple(z.shape))  # [B, full_cfg.max_len_total, full_cfg.max_len_total, pair_dim]
