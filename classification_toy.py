# train_bcr_classifier.py
import argparse, csv, random
import pandas as pd
from typing import List, Tuple, Dict, Iterable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from PanimmuneEmbedderPairs import PanimmuneEmbedderPairs


# ============================= Dataset ============================= #
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def _clean_seq(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().upper()
    return "".join(ch for ch in s if ch in VALID_AA)

def _concat_bcr(vh: str, cdr3h: str) -> str:
    return _clean_seq(vh) + _clean_seq(cdr3h)

class BCRAntigenPairDataset(Dataset):
    def __init__(self, csv_path: str, drop_if_empty: bool = True):
        super().__init__()
        self.items: List[Tuple[List[str], int, Dict]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                antigen = _clean_seq(row.get("Antigen", ""))
                bcr_pos = _concat_bcr(row.get("BetterBCR_Vh", ""), row.get("BetterBCR_CDR3h", ""))
                bcr_neg = _concat_bcr(row.get("WorseBCR_Vh", ""), row.get("WorseBCR_CDR3h", ""))
                if drop_if_empty and (len(antigen)==0 or (len(bcr_pos)==0 and len(bcr_neg)==0)):
                    continue
                meta = {"id": row.get("id"), "Project": row.get("Project")}
                if len(bcr_pos) > 0:
                    self.items.append(([bcr_pos, antigen], 1, {**meta, "which":"better"}))
                if len(bcr_neg) > 0:
                    self.items.append(([bcr_neg, antigen], 0, {**meta, "which":"worse"}))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_bcr_antigen(batch):
    batch_samples = [seqs for seqs, _, _ in batch]
    labels = torch.tensor([int(y) for _, y, _ in batch], dtype=torch.long)
    metas = [meta for _, _, meta in batch]
    return batch_samples, labels, metas


# ======================= Pair-aware pooling & head ======================= #
class PairAwarePooling(nn.Module):
    def __init__(self, c_s: int, c_z: int, hidden: int = 256):
        super().__init__()
        self.proj_pair = nn.Linear(c_z, hidden, bias=False)
        self.proj_single = nn.Linear(c_s, hidden, bias=False)
        self.gate = nn.Linear(2*hidden, hidden)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, s, z, single_mask, pair_mask):
        Pz = self.proj_pair(z)
        valid_ij = single_mask.unsqueeze(1) & single_mask.unsqueeze(2) & pair_mask
        valid_ij_f = valid_ij.unsqueeze(-1).to(Pz.dtype)
        denom = valid_ij_f.sum(dim=2).clamp_min(1.0)
        m = (Pz*valid_ij_f).sum(dim=2) / denom
        s_h = self.proj_single(s)
        h = torch.tanh(self.gate(torch.cat([s_h,m], dim=-1)))
        logits = self.attn(h).squeeze(-1)
        logits = logits.masked_fill(~single_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), s).squeeze(1)
        return pooled

class PairAwareClassifier(nn.Module):
    def __init__(self, embedder: PanimmuneEmbedderPairs, num_classes=2):
        super().__init__()
        self.embedder = embedder
        c_s = embedder.esm.config.hidden_size
        c_z = embedder.project_z.out_features
        self.pool = PairAwarePooling(c_s, c_z)
        self.classifier = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch_samples):
        s, z, (single_mask, pair_mask), _ = self.embedder(batch_samples, return_metadata=True)
        s = s.detach().clone()
        z = z.detach().clone()
        pooled = self.pool(s, z, single_mask, pair_mask)
        return self.classifier(pooled)


# =============================== Main =============================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--toy100", action="store_true", help="Use head(100) rows only")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    dataset_path = args.csv
    if args.toy100:
        df = pd.read_csv(args.csv).head(100)
        toy_path = "toy100.csv"
        df.to_csv(toy_path, index=False)
        dataset_path = toy_path
        print(f"Using toy dataset of 100 rows: {toy_path}")

    ds = BCRAntigenPairDataset(dataset_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bcr_antigen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = PanimmuneEmbedderPairs().to(device)
    model = PairAwareClassifier(embedder).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total, correct = 0, 0
        for batch_samples, labels, _ in dl:
            labels = labels.to(device)
            logits = model(batch_samples)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            preds = logits.argmax(dim=1)
            total += labels.numel(); correct += (preds==labels).sum().item()
        print(f"epoch {epoch+1}: acc={correct/total:.3f}")

if __name__ == "__main__":
    main()
