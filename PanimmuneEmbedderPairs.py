# PanimmuneEmbedderPairs.py
"""
Panimmune Pairwise U-Net with strict multi-chain parsing and additive fusion
- Inputs per item:
    * "A:B:C" or "A|B|C"  (any number of chains)
    * (A, B, C, ...) or [A, B, C, ...]
    * "A" (single chain)
  NOTE: Colons/bars are stripped before ESM.

Pipeline:
  ESM (HF) -> Linear projector (proj_dim)
  s = token_proj + alpha_pos * PosEmbed + alpha_chain * ChainEmbed(chain_id) -> LayerNorm
  Single->Pair (outer-product style) -> U-Net (ResBlocks) + Axial Transformer bottleneck
  Returns: [B, L, L, pair_dim]
"""
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ESMConfig, PairConfig

# ---- ESM (HuggingFace) ----
try:
    from transformers import EsmTokenizer, EsmModel
except Exception:  # pragma: no cover
    EsmTokenizer = None
    EsmModel = None


# ============================ Helpers ============================ #

_AA_SET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")  # ESM AA alphabet (uppercased)

def _canon_aa_seq(seq: str) -> str:
    """Uppercase, drop separators, keep only valid AA; raise on empty after cleaning."""
    if not isinstance(seq, str):
        raise TypeError(f"Sequence must be str, got {type(seq)}.")
    s = seq.upper()
    for ch in (":", "|", " ", "\t"):
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch in _AA_SET)
    if not s:
        raise ValueError(f"Empty/invalid sequence after cleaning: {repr(seq)}")
    return s

def _split_multi(item: Union[str, Tuple[str, ...], List[str]]) -> List[str]:
    """
    Normalize an item to a list of chain sequences: [chain1, chain2, ...].
    STRICT: raises on invalid input; no silent fallback.
    """
    # tuple/list of strings → as-is
    if isinstance(item, (tuple, list)):
        if len(item) < 1:
            raise ValueError("Empty list/tuple for chains.")
        if not all(isinstance(x, str) for x in item):
            raise TypeError(f"All chains must be strings. Got: {[type(x) for x in item]}")
        chains = [_canon_aa_seq(x) for x in item]
        return chains

    # plain string: split on ':' or '|'
    if isinstance(item, str):
        s = item.strip()
        if not s:
            raise ValueError("Empty sequence string.")
        s = s.replace("|", ":")
        parts = s.split(":")
        chains = [_canon_aa_seq(p) for p in parts]
        return chains

    # unsupported input
    raise TypeError(f"Unsupported input type for chain parsing: {type(item)} (value={item})")

def _seq_and_chain_ids_multi(item: Union[str, Tuple[str, ...], List[str]]) -> Tuple[str, List[int]]:
    """
    From item -> (concatenated_seq, chain_ids with labels 1..K).
    Example: ("ACDE","FGHIK","LMN") -> "ACDEFGHIKLMN", [1,1,1,1,2,2,2,2,2,3,3,3]
    """
    chains = _split_multi(item)                 # list[str], len = K >= 1
    concat = "".join(chains)
    ids: List[int] = []
    for k, ch in enumerate(chains, start=1):
        ids.extend([k] * len(ch))
    return concat, ids

def _pad_list_of_tensors_2d(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of [L_i, D] -> [B, Lmax, D], return mask [B, Lmax] (True=valid).
    """
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

def _pad_list_of_1d_long(tensors: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of [L_i] longs -> [B, Lmax], return mask [B, Lmax] (True=valid).
    """
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


# ===================== ESM (auto-projected) ====================== #
class LinearProjectedESM(nn.Module):
    """
    Wrap HF ESM; detect hidden size and project to proj_dim.
    Provides sliding-window stitching for long sequences.
    """
    def __init__(self, cfg: ESMConfig, proj_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        assert EsmTokenizer is not None and EsmModel is not None, "Please install `transformers` for ESM."
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model_name)
        self.model = EsmModel.from_pretrained(cfg.model_name)
        self.device = device if device is not None else torch.device("cpu")

        try:
            self.tokenizer.padding_side = cfg.pad_side
        except Exception:
            pass

        in_dim = int(self.model.config.hidden_size)
        self.projector = nn.Linear(in_dim, proj_dim, bias=True)

        if self.cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.to(self.device)

    def _embed_once(self, seq: str) -> torch.Tensor:
        """Embed one (already sanitized) sequence (no sliding)."""
        tok = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=True)
        hidden = out.last_hidden_state if self.cfg.layer is None else out.hidden_states[self.cfg.layer]
        h = hidden[0, 1:-1, :]                    # strip BOS/EOS -> [L, H]
        h = self.projector(h)                     # [L, proj_dim]
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_concat(self, sequence: str) -> torch.Tensor:
        """Embed a single concatenated sequence (separators removed upstream)."""
        # sequence here already cleaned and joined
        seq = sequence  # assume caller sanitized; we may still ensure non-empty
        if not isinstance(seq, str) or not seq:
            raise ValueError("embed_concat received empty sequence.")
        self.model.train(self.training)
        L_max = self.cfg.max_tokens - 2  # for BOS/EOS
        if len(seq) <= L_max:
            return self._embed_once(seq)

        # sliding-window with overlap, stitched by cosine ramps
        win, stride, L = L_max, self.cfg.stride, len(seq)
        assert 0 < stride <= win, "ESM stride must be in (0, win]"
        pieces, weights, starts = [], [], []
        start = 0
        while start < L:
            end = min(start + win, L)
            rep = self._embed_once(seq[start:end])  # [l, proj_dim]
            pieces.append(rep)
            starts.append(start)

            l = rep.size(0)
            w = rep.new_ones(l)
            if start > 0:
                ramp = torch.linspace(0, math.pi, steps=min(stride, l), device=rep.device, dtype=rep.dtype)
                w[: ramp.numel()] *= 0.5 * (1 - torch.cos(ramp))
            if end < L:
                ramp = torch.linspace(0, math.pi, steps=min(stride, l), device=rep.device, dtype=rep.dtype)
                w[- ramp.numel():] *= 0.5 * (1 - torch.cos(ramp))
            weights.append(w)

            if end == L:
                break
            start = end - stride

        H = pieces[0].size(1)
        out = pieces[0].new_zeros((L, H))
        denom = pieces[0].new_zeros((L,))
        for rep, w, s in zip(pieces, weights, starts):
            l = rep.size(0)
            out[s:s+l] += rep * w[:, None]
            denom[s:s+l] += w
        out = out / denom.clamp_min(1e-6)[:, None]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch_concat_seqs: List[str]) -> Union[List[torch.Tensor], torch.Tensor]:
        reps = [self.embed_concat(seq) for seq in batch_concat_seqs]  # each [L_i, proj_dim]
        Ls = [t.size(0) for t in reps]
        return torch.stack(reps, 0) if len(set(Ls)) == 1 else reps


# ================= Single -> Pair Initialization ================= #
class PairInitOPM(nn.Module):
    """
    Outer-product style init for pair reps.
    Input s: [B, L, P]  ->  z0: [B, L, L, pair_dim]
    """
    def __init__(self, single_dim: int, pair_dim: int, use_bias: bool = True):
        super().__init__()
        self.proj_i = nn.Linear(single_dim, pair_dim, bias=use_bias)
        self.proj_j = nn.Linear(single_dim, pair_dim, bias=use_bias)
        self.mix = nn.Linear(2 * pair_dim, pair_dim, bias=use_bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        si = self.proj_i(s)  # [B, L, Hp]
        sj = self.proj_j(s)  # [B, L, Hp]
        mul = si.unsqueeze(2) * sj.unsqueeze(1)  # [B, L, L, Hp]
        cat = torch.cat(
            [
                si.unsqueeze(2).expand(-1, -1, si.size(1), -1),
                sj.unsqueeze(1).expand(-1, si.size(1), -1, -1),
            ],
            dim=-1,
        )  # [B, L, L, 2Hp]
        z0 = self.mix(cat) + mul
        return torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0)


# ===================== U-Net Building Blocks ==================== #
class ResBlock2D(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out)
        )
        self.act = nn.GELU()

    def forward(self, x):
        y = self.block(x)
        y = y + self.proj(x)
        return self.act(torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))


class DownStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock2D(c_in,  c_out, dropout)
        self.res2 = ResBlock2D(c_out, c_out, dropout)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)  # ↓ by 2

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.res1 = ResBlock2D(c_out + c_skip, c_out, dropout)
        self.res2 = ResBlock2D(c_out, c_out, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)
        # pad/crop to match skip (odd sizes)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
            x = x[..., :skip.size(-2), :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


# =============== Axial Attention Bottleneck =================== #
class AxialSelfAttention2D(nn.Module):
    """
    Row then column attention over [B, C, L, L] (as [B*L, L, C]).
    """
    def __init__(self, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.row_mha = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.col_mha = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.row_ln = nn.LayerNorm(channels)
        self.col_ln = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.chunk_rows = int(chunk_rows) if chunk_rows else 0

    def _run_chunked(self, mha: nn.MultiheadAttention, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.chunk_rows and self.chunk_rows > 0:
            outs = []
            for i in range(0, x.size(0), self.chunk_rows):
                xi = x[i:i+self.chunk_rows]
                kpm = None if key_padding_mask is None else key_padding_mask[i:i+self.chunk_rows]
                oi, _ = mha(xi, xi, xi, key_padding_mask=kpm, need_weights=False)
                outs.append(oi)
            return torch.cat(outs, dim=0)
        else:
            o, _ = mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
            return o

    def forward(self, z: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        # [B, C, L, L] -> [B, L, L, C]
        B, C, L1, L2 = z.shape
        assert L1 == L2, "Pair grid must be square [L, L]"
        z = z.permute(0, 2, 3, 1).contiguous()
        B, L, _, H = z.shape

        # Row attention over B*L sequences of length L
        zr = z.reshape(B * L, L, H)
        key_padding = None
        if mask_1d is not None:
            m = (~mask_1d.bool())
            key_padding = m.unsqueeze(1).expand(B, L, L).reshape(B * L, L)  # True=pad
        r_out = self._run_chunked(self.row_mha, zr, key_padding)
        zr = self.row_ln(zr + self.drop(r_out))

        # Column attention: swap axes
        zc = zr.reshape(B, L, L, H).transpose(1, 2).reshape(B * L, L, H)
        key_padding = None
        if mask_1d is not None:
            m = (~mask_1d.bool())
            key_padding = m.unsqueeze(2).expand(B, L, L).reshape(B * L, L)
        c_out = self._run_chunked(self.col_mha, zc, key_padding)
        zc = self.col_ln(zc + self.drop(c_out))

        # Back to [B, C, L, L]
        z = zc.reshape(B, L, L, H).transpose(1, 2)
        return z.permute(0, 3, 1, 2).contiguous()


class FeedForward2D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class AxialTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.attn = AxialSelfAttention2D(channels, heads, dropout, chunk_rows)
        self.ff   = FeedForward2D(channels, dropout)

    def forward(self, x: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, mask_1d)
        x = self.ff(x)
        return x


class AxialTransformerStack(nn.Module):
    def __init__(self, n_layers: int, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialTransformerBlock(channels, heads, dropout, chunk_rows) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x, mask_1d)
        return x


# ========================= Full Model =========================== #
class PanimmuneEmbedderPairs(nn.Module):
    """
    End-to-end:
      item -> split into chains -> concat seq + chain_ids(1..K)
      ESM -> proj_dim
      s = token + alpha_pos*pos + alpha_chain*chain_embed -> LayerNorm
      Single->Pair -> U-Net (axial bottleneck) -> [B, L, L, pair_dim]
    """
    def __init__(self, esm_cfg: ESMConfig, pair_cfg: PairConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.esm = LinearProjectedESM(esm_cfg, pair_cfg.proj_dim, device=self.device)
        self.pair_cfg = pair_cfg

        # --- Fusion (addition) setup ---
        self.proj_dim = pair_cfg.proj_dim
        # chain_vocab: 0=PAD, 1..(chain_vocab-1)=chain IDs; configurable (default 8)
        self.chain_vocab = int(getattr(pair_cfg, "chain_vocab", 8))
        if self.chain_vocab < 3:
            raise ValueError(f"pair_cfg.chain_vocab must be >= 3, got {self.chain_vocab}.")
        self.chain_embed = nn.Embedding(self.chain_vocab, self.proj_dim, padding_idx=0)

        # Learnable gates to balance contributions
        self.alpha_chain = nn.Parameter(torch.tensor(1.0))
        self.alpha_pos   = nn.Parameter(torch.tensor(1.0))

        # LayerNorm only (no Dropout) after sum
        self.single_post = nn.LayerNorm(self.proj_dim)

        # Single -> Pair
        self.single_to_pair = PairInitOPM(pair_cfg.proj_dim, pair_cfg.pair_dim, use_bias=True)

        # U-Net encoder
        C0 = pair_cfg.unet_base_channels
        self.in_conv = nn.Conv2d(pair_cfg.pair_dim, C0, 3, padding=1)
        downs, ch = [], C0
        for _ in range(pair_cfg.unet_depth):
            downs.append(DownStage(ch, ch * 2, pair_cfg.dropout))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.bott_stack = AxialTransformerStack(
            n_layers=pair_cfg.n_transformers,
            channels=ch,
            heads=pair_cfg.mha_heads,
            dropout=pair_cfg.dropout,
            chunk_rows=pair_cfg.chunk_rows,
        )

        # Decoder
        ups = []
        for _ in range(pair_cfg.unet_depth):
            ups.append(UpStage(c_in=ch, c_skip=ch, c_out=ch // 2, dropout=pair_cfg.dropout))
            ch = ch // 2
        self.ups = nn.ModuleList(ups)
        self.out_conv = nn.Conv2d(C0, pair_cfg.pair_dim, 3, padding=1)

        self.to(self.device)

    # ---------- positional encoding ----------
    def _build_sinusoidal_pos(self, L: int, device) -> torch.Tensor:
        """Return [L, proj_dim] sinusoidal PE (for addition)."""
        d = self.proj_dim
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)      # [L,1]
        i = torch.arange(d // 2, device=device, dtype=torch.float32).unsqueeze(0)   # [1,d/2]
        denom = torch.exp(-math.log(10000.0) * (2 * i) / d)                          # [1,d/2]
        pe = torch.cat([torch.sin(pos * denom), torch.cos(pos * denom)], dim=1)     # [L,d or d-1]
        if d % 2 == 1:
            pe = torch.cat([pe, torch.zeros(L, 1, device=device)], dim=1)
        return pe  # [L, d]

    # ---------- utilities ----------
    @staticmethod
    def _resize_mask_1d(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Exact resize of a boolean mask [B, L] -> [B, target_len] using nearest interpolation.
        """
        B, L = mask.shape
        if L == target_len:
            return mask
        m = mask.float().unsqueeze(1).unsqueeze(1)   # [B,1,1,L]
        m = F.interpolate(m, size=(1, target_len), mode="nearest")
        m = m.squeeze(1).squeeze(1)
        return (m > 0.5)

    # ---------- forward ----------
    def forward(self, batch_items: List[Union[str, Tuple[str, ...], List[str]]], return_intermediates: bool = False):
        # 1) Build concatenated sequences + chain id lists (1..K)
        concat_seqs: List[str] = []
        chain_id_vecs: List[torch.Tensor] = []
        for it in batch_items:
            concat, ids12 = _seq_and_chain_ids_multi(it)         # ids in 1..K
            concat_seqs.append(concat)
            # clamp IDs to chain_vocab-1 if too many chains
            ids = [min(i, self.chain_vocab - 1) for i in ids12]
            chain_id_vecs.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        # 2) ESM on concatenated seqs → ragged reps → pad
        s_list_or_tensor = self.esm(concat_seqs)                 # list([L_i,P]) or [B,L,P]
        if isinstance(s_list_or_tensor, torch.Tensor):
            s = s_list_or_tensor
            smask = torch.ones(s.size(0), s.size(1), dtype=torch.bool, device=s.device)
            lens = [s.size(1)] * s.size(0)
        else:
            s, smask = _pad_list_of_tensors_2d(s_list_or_tensor) # [B, L, P], [B, L]
            lens = [t.size(0) for t in s_list_or_tensor]
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) # [B, L, proj_dim]

        # 3) Pad chain IDs to [B, L] (0 for padding beyond length)
        cid_pad, _ = _pad_list_of_1d_long(chain_id_vecs, pad_value=0)  # [B, L]

        # 4) Positional encodings per item -> pad to [B, L, proj_dim]
        pos_list = [self._build_sinusoidal_pos(L, s.device) for L in lens]
        pos_pad, _ = _pad_list_of_tensors_2d(pos_list)                 # [B, L, proj_dim]

        # 5) ADD fusion at single level: token + alpha_pos*pos + alpha_chain*chain_emb
        chain_e = self.chain_embed(cid_pad)                             # [B, L, proj_dim]
        s = s + self.alpha_pos * pos_pad + self.alpha_chain * chain_e
        # zero padded positions pre-LN (safety)
        s = s * smask.unsqueeze(-1).float()
        s = self.single_post(s)                                         # LayerNorm only

        # 6) Single -> Pair
        z = self.single_to_pair(s)                                      # [B, L, L, pair_dim]
        z = z.permute(0, 3, 1, 2).contiguous()                          # [B, C=pair_dim, L, L]

        # 7) U-Net encoder
        x = self.in_conv(z)
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        # 8) Bottleneck (axial Transformer stack); resize mask to current L
        B, C, Ld, _ = x.shape
        smask_scaled = self._resize_mask_1d(smask, Ld)
        x = self.bott_stack(x, smask_scaled)

        # 9) Decoder with skips
        for up in self.ups:
            x = up(x, skips.pop())

        # 10) Back to pair_dim channels; return [B, L, L, pair_dim]
        x = self.out_conv(x).permute(0, 2, 3, 1).contiguous()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if return_intermediates:
            return x, s, smask, cid_pad
        return x


# ========================= Tiny smoke test ====================== #
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example ESM/Pair configs (fill from your model_config)
    esm_cfg = ESMConfig()   # e.g., model_name="facebook/esm2_t6_8M_UR50D", freeze=True
    pair_cfg = PairConfig() # must define: proj_dim, pair_dim, unet_depth, unet_base_channels,
                            # n_transformers, mha_heads, chunk_rows, dropout, (optional) chain_vocab

    model = PanimmuneEmbedderPairs(esm_cfg, pair_cfg, device=dev)
    model.train()

    batch = [
        ("ACDE", "FGHIK"),           # 2 chains
        "LMNPQRST:GGG",              # colon separated
        ["AAA", "BBBB", "CC"],       # 3 chains from list
        "VVVVVDDDDDDDDD",            # single chain
        "A:B:C:D:E:F:G:H",           # many chains (may clamp ids to chain_vocab-1)
    ]
    z = model(batch)  # [B, L, L, pair_dim]
    print("z shape:", tuple(z.shape))
