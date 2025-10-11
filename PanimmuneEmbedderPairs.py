# panimmune_embedder_pairs_optimized.py
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict
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


class PanimmuneEmbedderPairs(nn.Module):
    """
    Optimized version:
      - Batch ESM encodes by sequence position across the batch (A group, B group, ...).
      - Tokenizer padding/truncation with pad_to_multiple_of=8.
      - AMP for ESM forward, dtype-safe padding for s/z.
      - Optional caching for repeated sequences (CPU offload).
      - Block-diagonal z per sample, optional inter-chain mixing via pair_mask.
    """
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pairformer_blocks: int = 4,
        pair_c: int = 128,
        allow_inter_chain: bool = False,        # cross-seq mixing *within* a sample
        attn_from_last_layer_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None,  # e.g., torch.bfloat16/float16 on GPU
        max_length: Optional[int] = None,       # cap for tokenizer; defaults to ESM's pos-emb if None
        pad_to_multiple_of: int = 8,
        enable_cache: bool = True,
        cache_size: int = 100_000,
        freeze_esm: bool = True,                # disable grads for ESM
        use_amp: bool = True,                   # autocast around ESM forward
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

        # Fast attention & TF32 friendly settings (safe no-ops on CPU)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
        except Exception:
            pass

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
        Uses cache for repeated sequences, runs ESM only for uncached ones.
        """
        # Filter out empties early
        seqs = [(i, s) for i, s in enumerate(seqs_in) if isinstance(s, str) and len(s) > 0]
        if not seqs:
            return [], []

        # Split into cached vs to-encode
        cached_out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        to_encode_idx: List[int] = []
        to_encode_strs: List[str] = []
        if self.enable_cache:
            for i, s in seqs:
                got = self._cache.get(s)
                if got is not None:
                    cached_out[i] = got
                else:
                    to_encode_idx.append(i)
                    to_encode_strs.append(s)
        else:
            for i, s in seqs:
                to_encode_idx.append(i)
                to_encode_strs.append(s)

        # Run ESM for the not-yet-cached sequences (batched)
        encoded_new: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if len(to_encode_strs) > 0:
            tok = self.tokenizer(
                to_encode_strs,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=False,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

            # AMP around ESM if enabled
            if self.use_amp and device.type == "cuda":
                amp_ctx = torch.autocast(device_type="cuda", dtype=self.esm.dtype if self.esm.dtype.is_floating_point else torch.bfloat16)
            else:
                # dummy context manager
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
                if S_j <= 2:  # only BOS/EOS or pathological
                    # Make zero shapes that match dtypes
                    c_s = hidden.shape[-1]
                    c_z = self.project_z.out_features
                    h_j = hidden.new_zeros((0, c_s))
                    z_j = hidden.new_zeros((0, 0, c_z))
                else:
                    # strip BOS/EOS
                    h_j = hidden[j, 1:S_j-1, :]                   # [L, C_s]
                    a_j = attn_all[j, :, 1:S_j-1, 1:S_j-1]        # [H, L, L]
                    a_j = a_j.permute(1, 2, 0).contiguous()       # [L, L, H]
                    z_j = self.project_z(a_j)                     # [L, L, C_z]

                # Offload to CPU for caching (saves VRAM)
                if self.enable_cache:
                    self._cache.put(to_encode_strs[j], (h_j.detach().cpu(), z_j.detach().cpu()))

                encoded_new[to_encode_idx[j]] = (h_j, z_j)

        # Stitch outputs in the input order (only for non-empty items)
        tokens_list: List[torch.Tensor] = []
        atts_list: List[torch.Tensor] = []
        for i, _s in seqs:
            pair = cached_out.get(i, None)
            if pair is None:
                pair = encoded_new[i]
            h_i, z_i = pair
            # Move to target device (non_blocking) in case they came from CPU cache
            tokens_list.append(h_i.to(device, non_blocking=True))
            atts_list.append(z_i.to(device, non_blocking=True))

        return tokens_list, atts_list

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
            # Remove empties for compute; we will re-associate by consuming in order later
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

        # dtype inference for final padded tensors
        # We'll sniff dtype from the first produced token tensor
        inferred_dtype: Optional[torch.dtype] = None

        for b in range(B):
            s_blocks, z_blocks, offsets = [], [], []
            total = 0
            for k in range(K):
                # skip empty slot
                if k >= len(batch_pairs[b]) or not isinstance(batch_pairs[b][k], str) or len(batch_pairs[b][k]) == 0:
                    continue
                # fetch next encoded tensor from group k (keeps sample order)
                if g_ptrs[k] >= len(group_tokens[k]):
                    continue  # safety
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
                # empty sample; push zero shapes
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
    ).to(device)

    # Example: two samples, each with variable # of sequences
    batch_pairs = [
        ["MKTAYIAKQRQISFVKSHFSRQDILDL", "GILGFVFTLTVPSER", "MAVMAPRTLVLLLSGALA"],
        ["MKTAYIAKQRQISFVKSHFSRQDILDLI", "LLGATCMFVLMYFGT"],
    ]

    s, z, masks, meta = model(batch_pairs, return_metadata=True)

    single_mask, pair_mask = masks
    print("\n=== Pair-batch result ===")
    print("B:", len(batch_pairs))
    print("L per sample:", meta["lengths"], "Lmax:", meta["Lmax"])
    print("s:", tuple(s.shape), "z:", tuple(z.shape))
    print("single_mask:", tuple(single_mask.shape), "pair_mask:", tuple(pair_mask.shape))
    print("offsets_per_sample:", meta["offsets_per_sample"])
