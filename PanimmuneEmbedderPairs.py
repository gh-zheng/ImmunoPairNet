# PanimmuneEmbedderPairs.py
"""
Panimmune Pairwise U-Net (Axial Transformer Stack, ESM freeze toggle) — Memory-safe
- ESM -> projector with STREAMING sliding windows (O(1) extra memory)
- Optional ESM-on-CPU; projector/U-Net on GPU (cfg.run_esm_device="cpu")
- Single->Pair: outer-product style init to [B, L, L, pair_dim]
- Early downsampling (pre_pool_factor in {1,2,4}) to save memory
- U-Net over the (downsampled) pair grid; axial Transformer bottleneck
- Exact 1D mask resize to bottleneck size; final output is restored to original L×L
- SDPA/Flash friendly (need_weights=False), optional row chunking
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


# ===================== ESM (auto-projected, streaming) ====================== #
class LinearProjectedESM(nn.Module):
    """
    Wrap ESM; auto-detect hidden size and project to proj_dim.
    Long sequences use streaming sliding windows (no storing all chunks).
    """
    def __init__(self, cfg: ESMConfig, proj_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        assert EsmTokenizer is not None and EsmModel is not None, "Install transformers to use ESM."
        self.cfg = cfg
        self.device = device if device is not None else torch.device("cpu")

        # ESM backbone device (CPU offload optional)
        run_esm_device = getattr(cfg, "run_esm_device", "same")  # "same" | "cpu" | "cuda"
        self.esm_device = self.device if run_esm_device == "same" else torch.device(run_esm_device)

        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model_name)
        self.model = EsmModel.from_pretrained(cfg.model_name).to(self.esm_device)

        try:
            self.tokenizer.padding_side = cfg.pad_side
        except Exception:
            pass

        in_dim = int(self.model.config.hidden_size)
        self.projector = nn.Linear(in_dim, proj_dim, bias=True).to(self.device)

        # freeze or not
        if self.cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _esm_hidden_one(self, seq_piece: str) -> torch.Tensor:
        """Run one chunk through ESM (on esm_device); return [l, H_in] on esm_device."""
        # Normalize chain separators so length accounting stays predictable
        seq_piece = seq_piece.replace(getattr(self.cfg, "sep_token", ":"), "X")

        # Important: restrict hidden_states output unless we actually need a chosen layer
        need_hs = (self.cfg.layer is not None)
        with torch.no_grad() if self.cfg.freeze else torch.enable_grad():
            inputs = self.tokenizer(seq_piece, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.esm_device, non_blocking=True) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=need_hs)
            if need_hs:
                hidden = out.hidden_states[self.cfg.layer][0, 1:-1, :]
            else:
                hidden = out.last_hidden_state[0, 1:-1, :]
            # free HF containers promptly
            del out, inputs
        return hidden  # on esm_device

    def embed(self, sequence: str) -> torch.Tensor:
        """
        Return [L, proj_dim] for a single concatenated sequence.
        Streaming sliding windows: compute→stitch directly into buffers.
        """
        self.model.train(self.training)

        L = len(sequence)
        L_max = int(self.cfg.max_tokens) - 2  # reserve BOS/EOS

        # Fast path (single window)
        if L <= L_max:
            hidden = self._esm_hidden_one(sequence)                             # [L, H_in] on esm_device
            hidden = hidden.to(self.device, non_blocking=True)                  # move once
            rep = self.projector(hidden)                                        # [L, proj_dim]
            del hidden
            return rep

        # ---- Streaming windows (O(1) extra memory) ----
        win, stride = L_max, int(self.cfg.stride)
        assert 0 < stride <= win, "stride must be in (0, win]"

        Hproj = int(self.projector.out_features)
        out = torch.zeros((L, Hproj), device=self.device)
        denom = torch.zeros((L,), device=self.device)

        # reuse cosine fade up to 'stride'
        ramp = torch.linspace(0, math.pi, steps=stride, device=self.device)
        fade = 0.5 * (1 - torch.cos(ramp))

        start = 0
        while start < L:
            end = min(start + win, L)
            subseq = sequence[start:end]

            hidden = self._esm_hidden_one(subseq)                 # [l, H_in] esm_device
            hidden = hidden.to(self.device, non_blocking=True)
            rep = self.projector(hidden)                           # [l, Hproj]
            del hidden

            l = rep.size(0)
            w = torch.ones(l, device=self.device, dtype=rep.dtype)
            if start > 0:
                k = min(stride, l); w[:k] *= fade[:k]
            if end < L:
                k = min(stride, l); w[-k:] *= fade[:k]

            out[start:end] += rep * w[:, None]
            denom[start:end] += w

            del rep, w
            if end == L:
                break
            start = end - stride

        return out / denom.clamp_min(1e-6)[:, None]

    def forward(self, batch: List[str]) -> Union[List[torch.Tensor], torch.Tensor]:
        reps: List[torch.Tensor] = [self.embed(seq) for seq in batch]   # each [L_i, proj_dim]
        Ls = [t.size(0) for t in reps]
        if len(set(Ls)) == 1:
            return torch.stack(reps, dim=0)  # [B, L, proj_dim]
        return reps  # ragged; caller will pad


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
        z0 = self.mix(cat) + mul           # [B, L, L, Hp]
        return z0


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
        return self.act(y + self.proj(x))


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
        x = self.up(x)       # [B, c_in, 2h, 2w]
        x = self.conv(x)     # [B, c_out, 2h, 2w]
        # pad/crop to match skip (odd sizes)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
            x = x[..., :skip.size(-2), :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)  # [B, c_out + c_skip, H, W]
        x = self.res1(x)
        x = self.res2(x)
        return x


# =============== Axial Attention Bottleneck =================== #
class AxialSelfAttention2D(nn.Module):
    """
    Row then column attention over [B, C, L, L] (converted to [B, L, L, C]).
    Chunking over B*L rows/cols bounds memory; need_weights=False enables SDPA/Flash.
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
            # mask_1d: [B, L] (True=valid) -> key_padding (True=pad)
            m = (~mask_1d.bool())
            key_padding = m.unsqueeze(1).expand(B, L, L).reshape(B * L, L)
        r_out = self._run_chunked(self.row_mha, zr, key_padding)
        zr = self.row_ln(zr + self.drop(r_out))

        # Column attention (swap L dims)
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


# =============== Axial Transformer Stack =================== #
class AxialTransformerBlock(nn.Module):
    """One Transformer-style block: axial attention + feed-forward."""
    def __init__(self, channels: int, heads: int, dropout: float, chunk_rows: int = 0):
        super().__init__()
        self.attn = AxialSelfAttention2D(channels, heads, dropout, chunk_rows)
        self.ff   = FeedForward2D(channels, dropout)

    def forward(self, x: torch.Tensor, mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, mask_1d)
        x = self.ff(x)
        return x


class AxialTransformerStack(nn.Module):
    """Stack of axial Transformer blocks."""
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
      batch_seqs (concat) -> ESM -> projector -> Single->Pair -> U-Net
    Forward returns the updated pair tensor: [B, L, L, pair_dim] with L = original token length.
    """
    def __init__(self, esm_cfg: ESMConfig, pair_cfg: PairConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.esm = LinearProjectedESM(esm_cfg, pair_cfg.proj_dim, device=self.device)
        self.pair_cfg = pair_cfg

        # Single -> Pair
        self.single_to_pair = PairInitOPM(pair_cfg.proj_dim, pair_cfg.pair_dim, use_bias=True)

        # U-Net encoder
        C0 = pair_cfg.unet_base_channels
        self.in_conv = nn.Conv2d(pair_cfg.pair_dim, C0, 3, padding=1)

        # Early downsampling to reduce side length; we will restore to original L×L at the end
        self.pre_pool_factor = int(getattr(pair_cfg, "pre_pool_factor", 1))
        assert self.pre_pool_factor in (1, 2, 4), "pre_pool_factor must be 1, 2, or 4"
        self.pre_pool = (
            nn.Identity() if self.pre_pool_factor == 1 else
            nn.AvgPool2d(kernel_size=self.pre_pool_factor, stride=self.pre_pool_factor)
        )

        downs, ch = [], C0
        for _ in range(pair_cfg.unet_depth):
            downs.append(DownStage(ch, ch * 2, pair_cfg.dropout))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        # Bottleneck: axial Transformer stack
        self.bott_stack = AxialTransformerStack(
            n_layers=pair_cfg.n_transformers,
            channels=ch,
            heads=pair_cfg.mha_heads,
            dropout=pair_cfg.dropout,
            chunk_rows=pair_cfg.chunk_rows,
        )

        # U-Net decoder (c_skip equals current 'ch' at each level)
        ups = []
        for _ in range(pair_cfg.unet_depth):
            ups.append(UpStage(c_in=ch, c_skip=ch, c_out=ch // 2, dropout=pair_cfg.dropout))
            ch = ch // 2
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(C0, pair_cfg.pair_dim, 3, padding=1)

        self.to(self.device)

    # ---------- helpers ----------
    def _ensure_batched(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list([L_i, P]) to padded [B, Lmax, P] + mask [B, Lmax].
        """
        if isinstance(x, torch.Tensor):
            B, L, P = x.shape
            mask = torch.ones(B, L, dtype=torch.bool, device=x.device)
            return x, mask
        Lmax = max(t.size(0) for t in x)
        P = x[0].size(-1)
        B = len(x)
        padded = x[0].new_zeros((B, Lmax, P))
        mask = torch.zeros((B, Lmax), dtype=torch.bool, device=x[0].device)
        for i, t in enumerate(x):
            l = t.size(0)
            padded[i, :l] = t
            mask[i, :l] = True
        return padded, mask

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
    def forward(self, batch_seqs: List[str], return_intermediates: bool = False):
        # 1) ESM embeddings (projected), ragged->padded
        s_list_or_tensor = self.esm(batch_seqs)       # list([L_i, proj_dim]) or [B, L, proj_dim]
        s, smask = self._ensure_batched(s_list_or_tensor)  # [B, L, P], [B, L]
        L0 = s.size(1)  # original token length (we will restore to this)

        # 2) Single -> Pair
        z = self.single_to_pair(s)                    # [B, L, L, pair_dim]
        z = z.permute(0, 3, 1, 2).contiguous()        # [B, C=pair_dim, L, L]

        # 2.5) Early downsample to reduce spatial side length
        z = self.pre_pool(z)                          # [B, C, L', L']

        # 3) U-Net encoder
        x = self.in_conv(z)
        skips = []
        for down in self.downs:
            x, skip = down(x)     # skip channels == current 'ch'
            skips.append(skip)

        # 4) Bottleneck (axial Transformer stack), exact mask resize to current side
        B, C, Ld, _ = x.shape
        smask_scaled = self._resize_mask_1d(smask, Ld)  # [B, Ld]
        x = self.bott_stack(x, smask_scaled)

        # 5) U-Net decoder (with skip connections)
        for up in self.ups:
            skip = skips.pop()    # deepest first
            x = up(x, skip)

        # 6) Project back to pair_dim channels at the (downsampled) side
        x = self.out_conv(x)                          # [B, pair_dim, L', L']

        # 7) Recover to original L0 x L0, if early pooled
        if self.pre_pool_factor != 1:
            x = F.interpolate(x, size=(L0, L0), mode='bilinear', align_corners=False)

        # 8) Return [B, L, L, pair_dim]
        x = x.permute(0, 2, 3, 1).contiguous()
        if return_intermediates:
            return x, s, smask
        return x


# ========================= Tiny smoke test ====================== #
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Minimal inline configs for a quick check (real values should come from model_config)
    class _ESM: pass
    class _PAIR: pass
    esm_cfg = _ESM(); pair_cfg = _PAIR()
    esm_cfg.model_name = "facebook/esm2_t12_35M_UR50D"
    esm_cfg.layer = None
    esm_cfg.max_tokens = 1024
    esm_cfg.stride = 256
    esm_cfg.sep_token = ":"
    esm_cfg.pad_side = "right"
    esm_cfg.freeze = True
    esm_cfg.run_esm_device = "same"

    pair_cfg.proj_dim = 128
    pair_cfg.pair_dim = 64
    pair_cfg.mha_heads = 8
    pair_cfg.dropout = 0.1
    pair_cfg.unet_depth = 3
    pair_cfg.unet_base_channels = 64
    pair_cfg.chunk_rows = 512
    pair_cfg.n_transformers = 4
    pair_cfg.pre_pool_factor = 2

    model = PanimmuneEmbedderPairs(esm_cfg, pair_cfg, device=dev).train()
    with torch.cuda.amp.autocast(enabled=(dev.type=="cuda")):
        z = model(["ACDEFGHIKLMNPQRSTVWY" * 40 + ":" + "ACDE" + ":" + "ACDEFGHIKLMNPQRSTVWY" * 40])
    print("z shape:", tuple(z.shape))
