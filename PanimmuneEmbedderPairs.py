# panimmune_embedder_pair_batch.py
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

# Ensure this import resolves
from src.models.pairformer import PairformerStack


class PanimmuneEmbedderPairs(nn.Module):
    """
    Batch over samples, where each sample contains K sequences (e.g., 2: [a,b]).
    For each sample:
      - Encode each sequence with ESM
      - Concatenate token embeddings along L
      - Build block-diagonal z from per-sequence attentions
      - Optionally allow inter-sequence mixing within a sample (allow_inter_chain)
    Across batch:
      - Pad to max sample length
      - Provide masks for padding and pair blocks
    """
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        pairformer_blocks: int = 4,
        pair_c: int = 128,
        allow_inter_chain: bool = False,        # cross-seq mixing *within* a sample
        attn_from_last_layer_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None,  # e.g., torch.float16 or bfloat16 on GPU
    ):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        self.esm = EsmModel.from_pretrained(
            esm_model_name, output_attentions=True, torch_dtype=torch_dtype
        )
        self.esm.eval().requires_grad_(True)

        self.project_z = nn.Linear(self.esm.config.num_attention_heads, pair_c)

        self.pairformer = PairformerStack(
            c_s=self.esm.config.hidden_size,
            c_z=pair_c,
            no_blocks=pairformer_blocks,
        )

        self.allow_inter_chain = allow_inter_chain
        self.attn_from_last_layer_only = attn_from_last_layer_only

    # ---------- low-level helpers ----------
    def _encode_one(self, seq: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single sequence.
        Returns:
            s_seq: [L, C_s]
            z_seq: [L, L, C_z] from last-layer (or mean) attention heads -> linear proj
        """
        tok = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.esm(**tok)
        hidden = out.last_hidden_state           # [1, S, C_s]
        attn_mask = tok["attention_mask"][0]     # [S]
        S = int(attn_mask.sum().item())          # includes BOS/EOS
        if S < 2:
            # pathological
            L = 0
            c_s = hidden.shape[-1]
            c_z = self.project_z.out_features
            return hidden.new_zeros((L, c_s)), hidden.new_zeros((L, L, c_z))

        # strip BOS/EOS
        s_seq = hidden[0, 1:S-1, :]  # [L, C_s], where L=S-2

        if self.attn_from_last_layer_only:
            attn = out.attentions[-1][0]  # [H, S, S] for batch elem 0
        else:
            all_layers = torch.stack(out.attentions, dim=0)  # [L, B, H, S, S]
            attn = all_layers[:, 0].mean(dim=0)              # [H, S, S]
        attn = attn[:, 1:S-1, 1:S-1]         # [H, L, L]
        attn = attn.permute(1, 2, 0).contiguous()  # [L, L, H]
        z_seq = self.project_z(attn)         # [L, L, C_z]
        return s_seq, z_seq

    def _build_sample(
        self, seqs: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Build one sample from K sequences.
        Returns:
            s_i: [Li, C_s]
            z_i: [Li, Li, C_z] (block-diagonal of K blocks)
            pair_mask_i: [Li, Li] boolean
            offsets: list of (start, end) per sequence inside the sample
        """
        s_blocks, z_blocks, offsets = [], [], []
        total = 0
        for seq in seqs:
            s_seq, z_seq = self._encode_one(seq, device=device)  # [L, Cs], [L,L,Cz]
            L = s_seq.shape[0]
            if L == 0:
                continue
            s_blocks.append(s_seq)
            z_blocks.append(z_seq)
            offsets.append((total, total + L))
            total += L

        if total == 0:
            c_s = self.esm.config.hidden_size
            c_z = self.project_z.out_features
            return (
                torch.zeros((0, c_s), device=device),
                torch.zeros((0, 0, c_z), device=device),
                torch.zeros((0, 0), dtype=torch.bool, device=device),
                [],
            )

        # concat s
        s_i = torch.cat(s_blocks, dim=0)  # [Li, C_s]

        # assemble block-diagonal z and pair_mask
        c_z = z_blocks[0].shape[-1]
        z_i = torch.zeros((total, total, c_z), device=device, dtype=s_i.dtype)
        pair_mask_i = torch.zeros((total, total), dtype=torch.bool, device=device)

        for z_blk, (st, en) in zip(z_blocks, offsets):
            z_i[st:en, st:en, :] = z_blk
            pair_mask_i[st:en, st:en] = True

        if self.allow_inter_chain:
            pair_mask_i[:, :] = True  # allow cross-seq mixing (z off-diagonals are zero unless you add biases)

        return s_i, z_i, pair_mask_i, offsets

    # ---------- main forward ----------
    def forward(
        self,
        batch_pairs: List[List[str]],               # e.g., [["AAAA","BBBB"], ["AAAA","DDDD"], ...]
        return_metadata: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            batch_pairs: list of samples; each sample is a list of sequences (size K can vary).
        Returns:
            s: [B, Lmax, C_s] (padded)
            z: [B, Lmax, Lmax, C_z] (padded)
            masks: dict-like trio combined into a single tuple for convenience:
                single_mask: [B, Lmax] (True = valid token)
                pair_mask:   [B, Lmax, Lmax] (True = valid pair)
            meta: { "offsets_per_sample": List[List[(st,en)]], "lengths": List[int], "Lmax": int }
        """
        device = next(self.parameters()).device

        # Build each sample independently
        s_list, z_list, single_masks, pair_masks = [], [], [], []
        offsets_per_sample, lengths = [], []
        for sample in batch_pairs:
            s_i, z_i, pair_mask_i, offsets = self._build_sample(sample, device=device)
            L = s_i.shape[0]
            lengths.append(L)
            offsets_per_sample.append(offsets)

            s_list.append(s_i)             # [Li, Cs]
            z_list.append(z_i)             # [Li, Li, Cz]
            single_masks.append(torch.ones((L,), dtype=torch.bool, device=device))
            pair_masks.append(pair_mask_i) # [Li, Li]

        B = len(batch_pairs)
        Lmax = max(lengths) if lengths else 0
        c_s = self.esm.config.hidden_size
        c_z = self.project_z.out_features

        # Allocate padded tensors
        s = torch.zeros((B, Lmax, c_s), device=device)
        z = torch.zeros((B, Lmax, Lmax, c_z), device=device)
        single_mask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
        pair_mask = torch.zeros((B, Lmax, Lmax), dtype=torch.bool, device=device)

        for b in range(B):
            L = lengths[b]
            if L == 0:
                continue
            s[b, :L] = s_list[b]
            z[b, :L, :L] = z_list[b]
            single_mask[b, :L] = single_masks[b]
            pair_mask[b, :L, :L] = pair_masks[b]

        # Pairformer pass
        s_out, z_out = self.pairformer(
            s=s,                           # [B, Lmax, C_s]
            z=z,                           # [B, Lmax, Lmax, C_z]
            single_mask=single_mask,       # [B, Lmax]
            pair_mask=pair_mask,           # [B, Lmax, Lmax]
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
    ).to(device)

    # Example matching your [[a,b],[a,d]] idea (two samples, each with 2 sequences)
    batch_pairs = [
        ["MKTAYIAKQRQISFVKSHFSRQDILDLI", "GILGFVFTLTVPSER"],   # sample 0: [a, b]
        ["MKTAYIAKQRQISFVKSHFSRQDILDLI", "LLGATCMFVLMYFGT"],   # sample 1: [a, d]
    ]

    s, z, masks, meta = model(batch_pairs, return_metadata=True)

    single_mask, pair_mask = masks
    print("\n=== Pair-batch result ===")
    print("B:", len(batch_pairs))
    print("L per sample:", meta["lengths"], "Lmax:", meta["Lmax"])
    print("s:", s.shape, "z:", z.shape)
    print("single_mask:", single_mask.shape, "pair_mask:", pair_mask.shape)
    print("offsets_per_sample:", meta["offsets_per_sample"])
