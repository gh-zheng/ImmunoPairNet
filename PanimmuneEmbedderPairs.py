# panimmune_embedder_pairs_chunked_precompute.py
from typing import List, Tuple, Dict, Any, Optional
import os, gzip, pickle
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel


class PanimmuneEmbedderPairs(nn.Module):
    """
    ESM-2 encoder with:
      • Optional precomputation of token embeddings/attentions (per-chain or full concatenated sequence keys)
      • Chunk-and-stitch for >1k residues while preserving attentions
      • Block-diagonal z (intra-chain), optional cross-chain mask
      • Pairformer per-sample (no giant padded [B,Lmax,Lmax,Cz] before compute)

    Precomputation usage order per sample:
      1) If full concatenated key is present in precompute, use it directly.
      2) Else, try to use precomputed per-chain values for all chains; if all present, assemble from them.
      3) Else, encode only the missing chains with ESM; combine with precomputed chains.
      4) If any chain exceeds ESM positional limit, chunk-and-stitch that chain.
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        pairformer_blocks: int = 6,
        pair_c: int = 32,
        allow_inter_chain: bool = False,          # mask allows cross-chain; z off-diagonal remains zero in this module
        attn_from_last_layer_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        freeze_esm: bool = True,
        use_amp: bool = True,
        return_padded: bool = False,              # pad AFTER Pairformer only if required
        # Chunking
        window: int = 1000,
        stride: int = 900,
        proj_chunk_elems: int = 512_000,
        # Precompute stores
        precomputed_pkl_gz: Optional[str] = None,     # e.g., "esm2_precompute.pkl.gz"
        precomputed_shards_dir: Optional[str] = None, # e.g., "esm_shards" with shard_*.pkl.gz
        strict_model_match: bool = True,              # require meta.esm_model == esm_model_name
        seq_normalize: bool = False,                  # normalize seq keys (upper/strip) if sources differ
        pairformer_ctor=None,                         # inject if your Pairformer path differs
    ):
        super().__init__()

        # --- ESM ---
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        self.esm = EsmModel.from_pretrained(
            esm_model_name, output_attentions=True, torch_dtype=torch_dtype
        )
        if freeze_esm:
            self.esm.eval().requires_grad_(False)
        else:
            self.esm.train().requires_grad_(True)

        # --- H->Cz projection for pair map ---
        self.project_z = nn.Linear(self.esm.config.num_attention_heads, pair_c, bias=True)

        # --- Pairformer ---
        if pairformer_ctor is None:
            from src.models.pairformer import PairformerStack
            pairformer_ctor = PairformerStack
        self.pairformer = pairformer_ctor(
            c_s=self.esm.config.hidden_size,
            c_z=pair_c,
            no_blocks=pairformer_blocks,
        )

        self.allow_inter_chain = allow_inter_chain
        self.attn_from_last_layer_only = attn_from_last_layer_only
        self.use_amp = use_amp
        self.return_padded = return_padded
        self.window = int(window)
        self.stride = int(stride)
        self.proj_chunk_elems = int(proj_chunk_elems)

        self.register_buffer("_device_anchor", torch.empty(0))
        self.max_pos = int(getattr(self.esm.config, "max_position_embeddings", 1024))

        # Perf toggles
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
        except Exception:
            pass

        # --- Precompute stores (CPU numpy) ---
        self._pre_tok: Dict[str, np.ndarray] = {}
        self._pre_attn: Optional[Dict[str, np.ndarray]] = None
        self._pre_meta: Dict[str, Any] = {}
        self._strict_model_match = strict_model_match
        self._seq_normalize = seq_normalize

        if precomputed_pkl_gz or precomputed_shards_dir:
            self._load_precomputed(precomputed_pkl_gz, precomputed_shards_dir, esm_model_name)

    # =================== Precompute helpers =================== #
    def _norm_key(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        if self._seq_normalize:
            s = s.upper()
        return s

    def _verify_meta(self, meta: Dict[str, Any], current_model: str) -> bool:
        if not self._strict_model_match:
            return True
        m = meta.get("esm_model", None)
        return str(m) == str(current_model)

    @staticmethod
    def _load_pickle_gz(path: str) -> Any:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def _merge_payload(self, payload: Dict[str, Any]):
        emb = payload.get("embeddings", {}) or {}
        for k, v in emb.items():
            self._pre_tok[self._norm_key(k)] = v
        if "attentions" in payload:
            if self._pre_attn is None:
                self._pre_attn = {}
            for k, v in (payload["attentions"] or {}).items():
                self._pre_attn[self._norm_key(k)] = v
        if payload.get("meta"):
            self._pre_meta = payload["meta"]

    def _load_precomputed(self, pkl_path: Optional[str], shards_dir: Optional[str], current_model: str):
        try:
            if shards_dir:
                shard_files = sorted(glob(os.path.join(shards_dir, "shard_*.pkl.gz")))
                for sp in shard_files:
                    part = self._load_pickle_gz(sp)
                    if self._verify_meta(part.get("meta", {}), current_model):
                        self._merge_payload(part)
            if pkl_path and os.path.isfile(pkl_path):
                part = self._load_pickle_gz(pkl_path)
                if self._verify_meta(part.get("meta", {}), current_model):
                    self._merge_payload(part)
            # Hidden size sanity
            hs_file = self._pre_meta.get("hidden_size", None)
            if hs_file is not None and int(hs_file) != int(self.esm.config.hidden_size):
                # Token embeddings incompatible; keep attentions if any
                self._pre_tok.clear()
        except Exception as e:
            print(f"[precompute] failed to load: {e}. Ignoring precomputed data.")
            self._pre_tok.clear()
            self._pre_attn = None
            self._pre_meta = {}

    # =================== Core utilities =================== #
    def _project_heads_chunked(self, A_llh: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        A_llh: [L, L, H]  =>  z: [L, L, Cz] via chunked linear projection.
        """
        L, _, H = A_llh.shape
        if L == 0:
            return A_llh.new_zeros((0, 0, self.project_z.out_features), dtype=out_dtype or A_llh.dtype)

        out_dtype = out_dtype or (self.esm.dtype if hasattr(self.esm, "dtype") else torch.float32)
        N = L * L
        z = A_llh.new_zeros((N, self.project_z.out_features), dtype=out_dtype)

        proj = self.project_z
        if A_llh.dtype in (torch.float16, torch.bfloat16) and proj.weight.dtype != A_llh.dtype:
            W = proj.weight.to(dtype=A_llh.dtype)
            b = proj.bias.to(dtype=A_llh.dtype) if proj.bias is not None else None
            tmp_wb = (W, b)
        else:
            tmp_wb = None

        start = 0
        flat = A_llh.view(N, H)
        step = self.proj_chunk_elems
        while start < N:
            end = min(N, start + step)
            tile = flat[start:end]
            if tmp_wb is None:
                out = torch.nn.functional.linear(tile, proj.weight, proj.bias)
            else:
                W, b = tmp_wb
                out = torch.nn.functional.linear(tile, W, b)
            if out.dtype != out_dtype:
                out = out.to(out_dtype)
            z[start:end] = out
            del tile, out
            start = end
            if A_llh.is_cuda:
                torch.cuda.empty_cache()
        return z.view(L, L, self.project_z.out_features)

    def _concat_with_seps(self, seqs: List[str]) -> Tuple[str, List[Tuple[int, int]], List[int]]:
        parts, offsets, sep_positions = [], []
        cur = 0
        for i, s in enumerate(seqs):
            if not s:
                continue
            parts.append(s)
            st, en = cur, cur + len(s)
            offsets.append((st, en))
            cur = en
            if i != len(seqs) - 1:
                parts.append("X")
                sep_positions.append(cur)
                cur += 1
        return "".join(parts), offsets, sep_positions

    # =================== ESM encoding (single window) =================== #
    def _esm_encode_once(self, cat: str, device: torch.device):
        tok = self.tokenizer(
            cat, return_tensors="pt", add_special_tokens=True,
            padding=False, truncation=False, max_length=self.max_pos
        )
        tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        amp_ok = (self.use_amp and device.type == "cuda")
        amp_ctx = (torch.autocast(device_type="cuda", dtype=(self.esm.dtype if amp_ok else torch.float32))
                   if amp_ok else torch.no_grad())
        use_inference = not any(p.requires_grad for p in self.esm.parameters())
        ctx = torch.inference_mode if use_inference else torch.no_grad

        with ctx():
            with amp_ctx:
                out = self.esm(**tok)

        hidden = out.last_hidden_state[0]        # [S, C]
        S = int(tok["attention_mask"].sum().item())
        h = hidden[1:S - 1]                      # [L, C]
        if self.attn_from_last_layer_only:
            attn = out.attentions[-1][0]         # [H, S, S]
        else:
            attn = torch.stack(out.attentions, dim=0).mean(dim=0)[0]
        A_last = attn[:, 1:S - 1, 1:S - 1]       # [H, L, L]
        return h, A_last

    # =================== Chunk & stitch (cat coord) =================== #
    def _encode_chunked_cat(
        self, cat: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunk-and-stitch embeddings + attentions in *cat* coordinates.
        Returns:
          h_full: [L_cat, C_s]
          A_full: [H, L_cat, L_cat]  (assembled by window overlap-averaging on blocks)
        """
        C_s = self.esm.config.hidden_size
        L_cat = len(cat)
        H = int(self.esm.config.num_attention_heads)

        # Embedding accumulators
        h_sum = torch.zeros((L_cat, C_s), device=device, dtype=self.esm.dtype if hasattr(self.esm, "dtype") else torch.float32)
        h_cnt = torch.zeros((L_cat,), device=device, dtype=torch.float32)

        # Attention accumulator (diagonal tiles only; we build full A as needed)
        # For simplicity, we accumulate a dense A; windows are bounded so memory is acceptable per step.
        A_sum = torch.zeros((H, L_cat, L_cat), device=device, dtype=torch.float32)
        A_cnt = torch.zeros((L_cat, L_cat), device=device, dtype=torch.float32)

        W, S = self.window, self.stride
        starts = list(range(0, max(1, L_cat - 1), S))
        if not starts or starts[-1] + W < L_cat:
            starts.append(max(0, L_cat - W))

        for g_st in starts:
            g_en = min(L_cat, g_st + W)
            sub = cat[g_st:g_en]
            h_win, A_last = self._esm_encode_once(sub, device=device)  # h_win: [l,C], A_last: [H,l,l]
            l = h_win.size(0)

            # embeddings
            h_sum[g_st:g_st + l] += h_win
            h_cnt[g_st:g_st + l] += 1.0

            # attentions
            A_sum[:, g_st:g_st + l, g_st:g_st + l] += A_last
            A_cnt[g_st:g_st + l, g_st:g_st + l] += 1.0

            del h_win, A_last
            if device.type == "cuda":
                torch.cuda.empty_cache()

        h_cnt = torch.clamp(h_cnt, min=1.0)
        h_full = h_sum / h_cnt.unsqueeze(1)

        A_cnt = torch.clamp(A_cnt, min=1.0)
        A_full = A_sum / A_cnt.unsqueeze(0)

        return h_full, A_full  # [L,C], [H,L,L]

    # =================== Precompute retrieval =================== #
    def _get_pre_tokens(self, seq: str, device: torch.device) -> Optional[torch.Tensor]:
        key = self._norm_key(seq)
        if key and key in self._pre_tok:
            tok_np = self._pre_tok[key]
            if tok_np.ndim == 2 and tok_np.shape[1] == self.esm.config.hidden_size:
                return torch.from_numpy(tok_np).to(device=device, dtype=self.esm.dtype if hasattr(self.esm, "dtype") else torch.float32)
        return None

    def _get_pre_attn(self, seq: str, L_expect: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._pre_attn is None:
            return None
        key = self._norm_key(seq)
        if not key or key not in self._pre_attn:
            return None
        A = self._pre_attn[key]
        # Accept [L,L], [H,L,L], or [L,L,H]; coerce to [H,L,L]
        if A.ndim == 2:
            A = np.repeat(A[..., None], self.esm.config.num_attention_heads, axis=2)  # [L,L,H]
        if A.ndim == 3 and A.shape[0] == self.esm.config.num_attention_heads:
            A_t = torch.from_numpy(A).to(device=device, dtype=torch.float32)          # [H,L,L]
        elif A.ndim == 3 and A.shape[-1] == self.esm.config.num_attention_heads:
            A_t = torch.from_numpy(A).to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        else:
            return None
        if A_t.shape[-1] != L_expect or A_t.shape[-2] != L_expect:
            return None
        return A_t

    # =================== Forward =================== #
    def forward(self, batch_pairs: List[List[str]], return_metadata: bool = True):
        """
        Per sample:
          • Try precompute (full concatenated key, else per-chain).
          • Else encode: single-pass if len<=max_pos; else chunk-and-stitch (cat coord).
          • Build block-diagonal z via chunked projection (per chain).
          • Pairformer per sample; no pre-padding.
        """
        device = self._device_anchor.device
        dt = (self.esm.dtype if hasattr(self.esm, "dtype") else torch.float32)
        z_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                   else torch.float16 if device.type == "cuda" else dt)

        s_out_list, z_out_list = [], []
        sm_list, pm_list = [], []
        meta_offsets, meta_lengths = [], []

        for sample in batch_pairs:
            seqs = [s for s in sample if isinstance(s, str) and len(s) > 0]
            if not seqs:
                continue

            # Build concatenated string and offsets (cat coordinates)
            cat, chain_offsets_cat, _ = self._concat_with_seps(seqs)
            L_cat = len(cat)

            # ---------- 1) Try precompute: full concatenated ----------
            pre_h_cat = self._get_pre_tokens(cat, device=device)
            pre_A_cat = self._get_pre_attn(cat, L_expect=L_cat, device=device) if pre_h_cat is not None else None

            if pre_h_cat is not None and pre_A_cat is not None:
                # Slice per-chain directly
                chain_h = [pre_h_cat[st:en] for (st, en) in chain_offsets_cat]
                z_blocks = []
                A_llh = pre_A_cat.permute(1, 2, 0).contiguous()  # [L,L,H]
                A_llh = A_llh.to(dtype=(torch.bfloat16 if A_llh.is_cuda and torch.cuda.is_bf16_supported()
                                        else torch.float16 if A_llh.is_cuda else torch.float32))
                for (st, en) in chain_offsets_cat:
                    l = en - st
                    z_blk = self._project_heads_chunked(A_llh[st:en, st:en, :], out_dtype=z_dtype) if l > 0 else pre_h_cat.new_zeros((0, 0, self.project_z.out_features), dtype=z_dtype)
                    z_blocks.append(z_blk)
                del pre_A_cat, A_llh
                h_full = pre_h_cat  # keep for consistency (not strictly needed beyond chain slices)

            else:
                # ---------- 2) Try precompute: all chains individually ----------
                pre_ok_all = True
                chain_h, pre_A_chain = [], []
                for s in seqs:
                    h_s = self._get_pre_tokens(s, device=device)
                    if h_s is None:
                        pre_ok_all = False
                        chain_h.append(None)      # placeholder
                        pre_A_chain.append(None)
                        continue
                    A_s = self._get_pre_attn(s, L_expect=h_s.shape[0], device=device)
                    if A_s is None:
                        pre_ok_all = False
                        chain_h.append(None)
                        pre_A_chain.append(None)
                        continue
                    chain_h.append(h_s)
                    pre_A_chain.append(A_s)

                if pre_ok_all:
                    # All chains precomputed; assemble
                    z_blocks = []
                    for h_s, A_s in zip(chain_h, pre_A_chain):
                        if h_s is None or A_s is None:
                            z_blocks.append(torch.zeros((0, 0, self.project_z.out_features), device=device, dtype=z_dtype))
                            continue
                        A_llh = A_s.permute(1, 2, 0).contiguous().to(
                            dtype=(torch.bfloat16 if A_s.is_cuda and torch.cuda.is_bf16_supported()
                                   else torch.float16 if A_s.is_cuda else torch.float32)
                        )  # [L,L,H]
                        z_blk = self._project_heads_chunked(A_llh, out_dtype=z_dtype)
                        z_blocks.append(z_blk)
                    # concat chain_h into cat order (no 'X' indices in this assembled space)
                    h_full = torch.cat(chain_h, dim=0)

                else:
                    # ---------- 3) Encode missing (cat path) ----------
                    if L_cat <= self.max_pos:
                        # single pass
                        h_full, A_full = self._esm_encode_once(cat, device=device)  # [L,C], [H,L,L]
                    else:
                        # chunk-and-stitch for both h and A in cat coords
                        h_full, A_full = self._encode_chunked_cat(cat, device=device)  # [L,C], [H,L,L]

                    # Build per-chain blocks from cat A
                    z_blocks = []
                    A_llh = A_full.permute(1, 2, 0).contiguous().to(
                        dtype=(torch.bfloat16 if A_full.is_cuda and torch.cuda.is_bf16_supported()
                               else torch.float16 if A_full.is_cuda else torch.float32)
                    )  # [L,L,H]
                    for (st, en) in chain_offsets_cat:
                        l = en - st
                        if l <= 0:
                            z_blocks.append(h_full.new_zeros((0, 0, self.project_z.out_features), dtype=z_dtype))
                            continue
                        z_blk = self._project_heads_chunked(A_llh[st:en, st:en, :], out_dtype=z_dtype)
                        z_blocks.append(z_blk)
                    del A_full, A_llh

            # ---------- Assemble s_i, z_i (block-diagonal), masks ----------
            chain_h_slices = [h_full[st:en] for (st, en) in chain_offsets_cat]
            total_L = sum(h.shape[0] for h in chain_h_slices)
            s_i = torch.cat(chain_h_slices, dim=0) if total_L > 0 else h_full.new_zeros((0, self.esm.config.hidden_size))
            z_i = s_i.new_zeros((total_L, total_L, self.project_z.out_features), dtype=z_dtype)

            offsets_assembled = []
            cur = 0
            for (st, en), z_blk in zip(chain_offsets_cat, z_blocks):
                l = en - st
                if l > 0:
                    z_i[cur:cur + l, cur:cur + l, :] = z_blk
                    offsets_assembled.append((cur, cur + l))
                    cur += l
                else:
                    offsets_assembled.append((cur, cur))

            single_mask = torch.ones((1, total_L), dtype=torch.bool, device=device)
            pair_mask = torch.zeros((1, total_L, total_L), dtype=torch.bool, device=device)
            for (st, en) in offsets_assembled:
                if en > st:
                    pair_mask[:, st:en, st:en] = True
            if self.allow_inter_chain:
                pair_mask[:] = True  # off-diagonals in z are zero in this module

            # ---------- Pairformer per sample ----------
            s_b = s_i.unsqueeze(0).to(dtype=dt)
            z_b = z_i.unsqueeze(0)
            s_b_out, z_b_out = self.pairformer(s=s_b, z=z_b, single_mask=single_mask, pair_mask=pair_mask)

            s_out_list.append(s_b_out)
            z_out_list.append(z_b_out)
            sm_list.append(single_mask)
            pm_list.append(pair_mask)
            meta_offsets.append(offsets_assembled)
            meta_lengths.append(total_L)

            # free temps
            del s_i, z_i, chain_h_slices, z_blocks, h_full
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        meta = {
            "offsets_per_sample": meta_offsets,
            "lengths": meta_lengths,
            "Lmax": max(meta_lengths) if meta_lengths else 0,
            "chunking": {"window": self.window, "stride": self.stride},
            "precompute_used": (len(self._pre_tok) > 0) or (self._pre_attn is not None),
            "precompute_model": self._pre_meta.get("esm_model", None),
            "precompute_attn_kind": self._pre_meta.get("attn_kind", None),
        }

        if not self.return_padded:
            return s_out_list, z_out_list, (sm_list, pm_list), meta

        # Optional: pad AFTER Pairformer
        if len(s_out_list) == 0:
            cs = self.esm.config.hidden_size
            cz = self.project_z.out_features
            device = self._device_anchor.device
            empty_s = torch.zeros((0, 0, cs), device=device)
            empty_z = torch.zeros((0, 0, 0, cz), device=device)
            empty_sm = torch.zeros((0, 0), dtype=torch.bool, device=device)
            empty_pm = torch.zeros((0, 0, 0), dtype=torch.bool, device=device)
            return empty_s, empty_z, (empty_sm, empty_pm), meta

        Ls = [x.shape[1] for x in s_out_list]
        Lmax = max(Ls)
        B = len(s_out_list)
        device = s_out_list[0].device
        cs = s_out_list[0].shape[-1]
        cz = z_out_list[0].shape[-1]

        s_pad = torch.zeros((B, Lmax, cs), device=device, dtype=s_out_list[0].dtype)
        z_pad = torch.zeros((B, Lmax, Lmax, cz), device=device, dtype=z_out_list[0].dtype)
        sm_pad = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
        pm_pad = torch.zeros((B, Lmax, Lmax), dtype=torch.bool, device=device)
        for b in range(B):
            L = Ls[b]
            s_pad[b, :L] = s_out_list[b][0]
            z_pad[b, :L, :L] = z_out_list[b][0]
            sm_pad[b, :L] = sm_list[b][0]
            pm_pad[b, :L, :L] = pm_list[b][0]
        return s_pad, z_pad, (sm_pad, pm_pad), meta


# --------------------------- quick test --------------------------- #
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = (torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
             else (torch.float16 if use_cuda else None))

    model = PanimmuneEmbedderPairs(
        allow_inter_chain=False,
        attn_from_last_layer_only=True,
        torch_dtype=dtype,
        freeze_esm=True,
        use_amp=True,
        pairformer_blocks=4,
        pair_c=32,
        window=1000,
        stride=900,
        # point these at your precompute if available:
        precomputed_pkl_gz=None,         # e.g., "esm2_precompute.pkl.gz"
        precomputed_shards_dir=None,     # e.g., "esm_shards"
        strict_model_match=True,
        seq_normalize=False,
    ).to(device)

    long_antigen = "M" * 1800 + "K" * 400  # 2200 aa
    batch_pairs = [
        ["QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSW", "LTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKSGTSPKRWI"],
        ["QVQLVQSGAEVKKPGSSVKVSCKASGYTFTNYW", long_antigen],
    ]

    s_out, z_out, masks, meta = model(batch_pairs, return_metadata=True)
    print(f"#samples={len(s_out)}")
    for i, (s_i, z_i) in enumerate(zip(s_out, z_out)):
        print(f"  sample {i}: s {tuple(s_i.shape)}  z {tuple(z_i.shape)}  L={meta['lengths'][i]}  blocks={len(meta['offsets_per_sample'][i])}")
