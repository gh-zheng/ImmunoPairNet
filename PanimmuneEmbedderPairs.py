# panimmune_embedder_pairs_optimized.py
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict
import os, gzip, pickle
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

# Ensure this import resolves in your project
from src.models.pairformer import PairformerStack


class _LRUSeqCache:
    """
    Very simple LRU cache for sequence -> (s_seq, z_seq) tensors.
    Stores CPU tensors; moves to device on read.
    """
    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.store: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, key: str):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def put(self, key: str, value: Tuple[torch.Tensor, torch.Tensor]):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)


def _load_pickle_gz(path: str) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class PanimmuneEmbedderPairs(nn.Module):
    """
    Optimized version with optional precomputation:
      - Uses precomputed ESM token embeddings & attentions when available.
      - Falls back to live ESM encoding otherwise.
      - Batch ESM encodes by sequence position across the batch.
      - AMP for ESM forward, dtype-safe padding for s/z.
      - Optional caching for repeated sequences (CPU offload).
      - Block-diagonal z per sample, optional inter-chain mixing via pair_mask.
    """
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pairformer_blocks: int = 12,
        pair_c: int = 128,
        allow_inter_chain: bool = False,
        attn_from_last_layer_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: int = 8,
        enable_cache: bool = True,
        cache_size: int = 100_000,
        freeze_esm: bool = True,
        use_amp: bool = True,
        # -------- NEW OPTIONALS (don’t break existing callers) --------
        precomputed_pkl_gz: Optional[str] = None,       # e.g., "esm2_token_embeddings_650M_long.pkl.gz"
        precomputed_shards_dir: Optional[str] = None,   # e.g., "esm_shards" (files: shard_*.pkl.gz)
        strict_model_match: bool = True,                # require meta['esm_model'] == esm_model_name
        seq_normalize: bool = False,                    # normalize seq keys if your sources differ
    ):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        self.esm = EsmModel.from_pretrained(
            esm_model_name, output_attentions=True, torch_dtype=torch_dtype
        )

        # Optional: freeze ESM (no grads) for speed & memory
        if freeze_esm:
            self.esm.eval().requires_grad_(False)
        else:
            self.esm.train().requires_grad_(True)

        self.project_z = nn.Linear(self.esm.config.num_attention_heads, pair_c)

        self.pairformer = PairformerStack(
            c_s=self.esm.config.hidden_size,
            c_z=pair_c,
            no_blocks=pairformer_blocks,
        )

        self.allow_inter_chain = allow_inter_chain
        self.attn_from_last_layer_only = attn_from_last_layer_only
        self.use_amp = use_amp
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length or getattr(self.esm.config, "max_position_embeddings", 1024)

        # Simple sequence cache (CPU)
        self.enable_cache = enable_cache
        self._cache = _LRUSeqCache(max_size=cache_size) if enable_cache else None
        self.register_buffer("_device_anchor", torch.empty(0))

        # Precomputed stores (CPU numpy → converted on demand)
        self._pre_tok: Dict[str, np.ndarray] = {}
        self._pre_attn: Optional[Dict[str, np.ndarray]] = None
        self._pre_meta: Dict[str, Any] = {}
        self._seq_normalize = seq_normalize
        self._strict_model_match = strict_model_match
        if precomputed_pkl_gz or precomputed_shards_dir:
            self._load_precomputed(precomputed_pkl_gz, precomputed_shards_dir, esm_model_name)

        # Fast attention & TF32 friendly settings (safe no-ops on CPU)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
        except Exception:
            pass

    # ---------- precompute helpers ----------
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

    def _merge_payload(self, payload: Dict[str, Any]):
        # Merge embeddings
        emb = payload.get("embeddings", {})
        for k, v in emb.items():
            self._pre_tok[self._norm_key(k)] = v  # numpy arrays kept on CPU
        # Merge attentions (optional)
        if "attentions" in payload:
            if self._pre_attn is None:
                self._pre_attn = {}
            for k, v in payload["attentions"].items():
                self._pre_attn[self._norm_key(k)] = v
        # Keep last meta
        self._pre_meta = payload.get("meta", {}) or self._pre_meta

    def _load_precomputed(self, pkl_path: Optional[str], shards_dir: Optional[str], current_model: str):
        try:
            if shards_dir:
                shard_files = sorted(glob(os.path.join(shards_dir, "shard_*.pkl.gz")))
                for sp in shard_files:
                    part = _load_pickle_gz(sp)
                    # If strict, skip incompatible shards
                    meta = part.get("meta", {})
                    if not self._verify_meta(meta, current_model):
                        continue
                    self._merge_payload(part)
            if pkl_path and os.path.isfile(pkl_path):
                part = _load_pickle_gz(pkl_path)
                meta = part.get("meta", {})
                if self._verify_meta(meta, current_model):
                    self._merge_payload(part)

            # Sanity: verify hidden sizes if present
            hs_file = self._pre_meta.get("hidden_size", None)
            if hs_file is not None and int(hs_file) != int(self.esm.config.hidden_size):
                # If hidden size mismatches, we cannot reuse token embeddings safely
                # Keep attentions if they exist (we can still project those), but drop tokens
                self._pre_tok.clear()

        except Exception as e:
            print(f"[precompute] failed to load: {e}. Ignoring precomputed data.")
            self._pre_tok.clear()
            self._pre_attn = None
            self._pre_meta = {}

    # ---------- encoding helpers ----------
    def _encode_group_batched(
        self,
        seqs_in: List[str],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a list of sequences that belong to the same position/group across the batch.
        Returns:
          tokens_list: list of [Li, C_s] tensors
          atts_list:   list of [Li, Li, C_z] tensors
        Uses precomputed values when available, then cache, then ESM.
        """
        # Filter out empties early
        seqs = [(i, s) for i, s in enumerate(seqs_in) if isinstance(s, str) and len(s) > 0]
        if not seqs:
            return [], []

        # First pass: try precomputed
        tokens_list: List[torch.Tensor] = [None] * len(seqs)  # type: ignore
        atts_list: List[torch.Tensor] = [None] * len(seqs)    # type: ignore
        need_encode_idx: List[int] = []
        need_encode_strs: List[str] = []

        H_current = int(self.esm.config.num_attention_heads)
        C_s_current = int(self.esm.config.hidden_size)
        for pos, s in seqs:
            key = self._norm_key(s)
            took = False
            # 1) exact precomputed tensors (embeddings + attentions)
            if key and key in self._pre_tok:
                tok_np = self._pre_tok[key]  # [L, C_s_pre]
                if tok_np.ndim == 2 and tok_np.shape[1] == C_s_current:
                    # tokens OK
                    t = torch.from_numpy(tok_np).to(device=device, dtype=self.esm.dtype if hasattr(self.esm, "dtype") and self.esm.dtype.is_floating_point else torch.float32)
                    # attentions
                    if self._pre_attn is not None and key in self._pre_attn:
                        A_np = self._pre_attn[key]
                        if A_np.ndim == 2:
                            # [L, L] (last_mean) -> expand to [L, L, H_current]
                            A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, H_current)
                        elif A_np.ndim == 3:
                            # Could be [H, L, L] or [L, L, H]
                            if A_np.shape[0] == H_current:
                                # [H, L, L] -> [L, L, H]
                                A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32).permute(1, 2, 0).contiguous()
                            elif A_np.shape[-1] == H_current:
                                # already [L, L, H]
                                A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
                            else:
                                # head mismatch; fallback to mean across any head axis
                                A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
                                if A.shape[0] == A.shape[1]:
                                    # ambiguous; treat as [L, L] by averaging over dim0 if needed
                                    A = A.mean(dim=0, keepdim=False)
                                    A = A.unsqueeze(-1).repeat(1, 1, H_current)
                                else:
                                    # assume [H?, L, L]
                                    A = A.mean(dim=0).permute(1, 2, 0)
                        else:
                            # Unknown shape: make zeros to keep batch moving
                            L = t.shape[0]
                            A = torch.zeros((L, L, H_current), device=device, dtype=torch.float32)
                    else:
                        # No precomputed attention; fabricate simple diagonal (neutral) to project
                        L = t.shape[0]
                        A = torch.zeros((L, L, H_current), device=device, dtype=torch.float32)

                    z = self.project_z(A)  # [L, L, C_z]

                    tokens_list[pos] = t
                    atts_list[pos] = z
                    took = True
            if not took:
                # 2) cache (CPU) check
                if self.enable_cache:
                    got = self._cache.get(s)
                    if got is not None:
                        h_i, z_i = got
                        tokens_list[pos] = h_i.to(device, non_blocking=True)
                        atts_list[pos] = z_i.to(device, non_blocking=True)
                        continue

                # 3) mark for live ESM encode
                need_encode_idx.append(pos)
                need_encode_strs.append(s)

        # Live ESM for those still missing
        if len(need_encode_strs) > 0:
            raise
            tok = self.tokenizer(
                need_encode_strs,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=False,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

            # AMP around ESM if enabled (CUDA)
            if self.use_amp and device.type == "cuda":
                amp_ctx = torch.autocast(device_type="cuda", dtype=self.esm.dtype if getattr(self.esm, "dtype", torch.float32).is_floating_point else torch.bfloat16)
            else:
                class _NoOp:
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                amp_ctx = _NoOp()

            with amp_ctx:
                out = self.esm(**tok)  # last_hidden_state [B, S, C_s]; attentions: list(L)[B, H, S, S]

            hidden = out.last_hidden_state          # [B, S, C_s]
            attn_mask = tok["attention_mask"]       # [B, S]
            S_vec = attn_mask.sum(dim=1)            # [B] (includes specials)

            # Select attentions (last layer or mean over layers)
            if self.attn_from_last_layer_only:
                attn_all = out.attentions[-1]       # [B, H, S, S]
            else:
                attn_all = torch.stack(out.attentions, dim=0).mean(dim=0)  # [B, H, S, S]

            for j in range(hidden.size(0)):
                S_j = int(S_vec[j].item())
                if S_j <= 2:
                    c_s = hidden.shape[-1]
                    c_z = self.project_z.out_features
                    h_j = hidden.new_zeros((0, c_s))
                    z_j = hidden.new_zeros((0, 0, c_z))
                else:
                    # strip BOS/EOS
                    h_j = hidden[j, 1:S_j-1, :]                         # [L, C_s]
                    a_j = attn_all[j, :, 1:S_j-1, 1:S_j-1]              # [H, L, L]
                    a_j = a_j.permute(1, 2, 0).contiguous()             # [L, L, H]
                    z_j = self.project_z(a_j)                           # [L, L, C_z]

                pos = need_encode_idx[j]
                tokens_list[pos] = h_j
                atts_list[pos] = z_j

                # Offload to CPU cache
                if self.enable_cache:
                    self._cache.put(need_encode_strs[j], (h_j.detach().cpu(), z_j.detach().cpu()))

        # Compact (remove Nones) preserving the original order of non-empty seqs
        final_tokens, final_atts = [], []
        for (pos, _s) in seqs:
            final_tokens.append(tokens_list[pos])
            final_atts.append(atts_list[pos])

        return final_tokens, final_atts

    # ---------- main forward ----------
    def forward(
        self,
        batch_pairs: List[List[str]],               # e.g., [["heavy","light","antigen"], ["heavy","antigen"], ...]
        return_metadata: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Returns:
            s: [B, Lmax, C_s] (padded)
            z: [B, Lmax, Lmax, C_z] (padded)
            (single_mask, pair_mask):
                single_mask: [B, Lmax] (True = valid token)
                pair_mask:   [B, Lmax, Lmax] (True = valid pair)
            meta: { "offsets_per_sample": List[List[(st,en)]], "lengths": List[int], "Lmax": int }
        """
        device = self._device_anchor.device
        B = len(batch_pairs)
        if B == 0:
            c_s = self.esm.config.hidden_size
            c_z = self.project_z.out_features
            empty = (
                torch.zeros((0, 0, c_s), device=device),
                torch.zeros((0, 0, 0, c_z), device=device),
                (torch.zeros((0, 0), dtype=torch.bool, device=device), torch.zeros((0, 0, 0), dtype=torch.bool, device=device)),
                {"offsets_per_sample": [], "lengths": [], "Lmax": 0},
            )
            return empty

        # Determine max K across samples; keep alignment with placeholders (empty string) if needed
        K = max(len(s) for s in batch_pairs)
        grouped: List[List[str]] = [[] for _ in range(K)]
        for sample in batch_pairs:
            for k in range(K):
                grouped[k].append(sample[k] if k < len(sample) else "")

        # Encode each group batched
        group_tokens: List[List[torch.Tensor]] = []
        group_atts: List[List[torch.Tensor]] = []
        for k in range(K):
            non_empty = [s for s in grouped[k] if isinstance(s, str) and len(s) > 0]
            if len(non_empty) == 0:
                group_tokens.append([])
                group_atts.append([])
                continue
            t_k, z_k = self._encode_group_batched(non_empty, device=device)
            group_tokens.append(t_k)  # list of per-seq [L, C_s]
            group_atts.append(z_k)    # list of per-seq [L, L, C_z]

        # Rebuild each sample by concatenating its groups in order
        s_list, z_list, single_masks, pair_masks = [], [], [], []
        offsets_per_sample, lengths = [], []
        g_ptrs = [0] * K  # pointers within each encoded group list

        inferred_dtype: Optional[torch.dtype] = None

        for b in range(B):
            s_blocks, z_blocks, offsets = [], [], []
            total = 0
            for k in range(K):
                if k >= len(batch_pairs[b]) or not isinstance(batch_pairs[b][k], str) or len(batch_pairs[b][k]) == 0:
                    continue
                if g_ptrs[k] >= len(group_tokens[k]):
                    continue
                s_k = group_tokens[k][g_ptrs[k]]   # [Lk, C_s]
                z_k = group_atts[k][g_ptrs[k]]     # [Lk, Lk, C_z]
                g_ptrs[k] += 1

                if inferred_dtype is None:
                    inferred_dtype = s_k.dtype

                Lk = s_k.shape[0]
                if Lk == 0:
                    continue
                s_blocks.append(s_k)
                z_blocks.append(z_k)
                offsets.append((total, total + Lk))
                total += Lk

            if total == 0:
                c_s = self.esm.config.hidden_size
                c_z = self.project_z.out_features
                dt = inferred_dtype or torch.float32
                s_list.append(torch.zeros((0, c_s), device=device, dtype=dt))
                z_list.append(torch.zeros((0, 0, c_z), device=device, dtype=dt))
                single_masks.append(torch.zeros((0,), dtype=torch.bool, device=device))
                pair_masks.append(torch.zeros((0, 0), dtype=torch.bool, device=device))
                offsets_per_sample.append([])
                lengths.append(0)
                continue

            s_i = torch.cat(s_blocks, dim=0)  # [Li, C_s]
            c_z = z_blocks[0].shape[-1]
            z_i = s_i.new_zeros((total, total, c_z))
            pair_mask_i = torch.zeros((total, total), dtype=torch.bool, device=device)

            for z_blk, (st, en) in zip(z_blocks, offsets):
                z_i[st:en, st:en, :] = z_blk
                pair_mask_i[st:en, st:en] = True

            if self.allow_inter_chain:
                pair_mask_i[:, :] = True

            s_list.append(s_i)
            z_list.append(z_i)
            single_masks.append(torch.ones((total,), dtype=torch.bool, device=device))
            pair_masks.append(pair_mask_i)
            offsets_per_sample.append(offsets)
            lengths.append(total)

        # Pad across batch to Lmax
        Lmax = max(lengths) if lengths else 0
        c_s = self.esm.config.hidden_size
        c_z = self.project_z.out_features
        dt = inferred_dtype or (self.esm.dtype if hasattr(self.esm, "dtype") else torch.float32)

        s = torch.zeros((B, Lmax, c_s), device=device, dtype=dt)
        z = torch.zeros((B, Lmax, Lmax, c_z), device=device, dtype=dt)
        single_mask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
        pair_mask = torch.zeros((B, Lmax, Lmax), dtype=torch.bool, device=device)

        for b in range(B):
            L = lengths[b]
            if L == 0:
                continue
            s[b, :L] = s_list[b]
            z[b, :L, :L] = z_list[b]
            single_mask[b, :L] = True
            pair_mask[b, :L, :L] = pair_masks[b]

        # Pairformer pass
        s_out, z_out = self.pairformer(
            s=s,
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
        )

        meta = {
            "offsets_per_sample": offsets_per_sample,
            "lengths": lengths,
            "Lmax": Lmax,
            "precompute_used": (len(self._pre_tok) > 0),
            "precompute_model": self._pre_meta.get("esm_model", None),
            "precompute_attn_kind": self._pre_meta.get("attn_kind", None),
        }
        return s_out, z_out, (single_mask, pair_mask), meta


# ------------------------------- quick test ------------------------------- #
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = (torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
             else (torch.float16 if use_cuda else None))

    model = PanimmuneEmbedderPairs(
        allow_inter_chain=False,
        attn_from_last_layer_only=True,
        torch_dtype=dtype,
        max_length=None,                  # defaults to ESM pos-emb
        pad_to_multiple_of=8,
        enable_cache=True,
        cache_size=200_000,
        freeze_esm=True,                  # no grads for ESM
        use_amp=True,
        # point these at your precompute:
        precomputed_pkl_gz=r"esm2_token_embeddings_650M_long.pkl.gz",
        precomputed_shards_dir=None,  # or None
        strict_model_match=True,
        seq_normalize=False,
    ).to(device)
    print(model._encode_group_batched(["HLMGWDYPK", "MAVMPPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRSAKAHSQTDRVDLGTLRGYYNQSEDGSHTIQIMYGCDVGSDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHAAEQQRAYLEGTCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV"], device=device))

