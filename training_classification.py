# train_bcr_classifier.py
import os, random, json
from typing import List, Dict, Any, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---- model & config (your file) ----
from classification import load_default_config, build_embedder, build_classifier, PairAwareClassifier

# ---- datasets (your file) ----
from Panimmnue_dataload import (
    IEDBRetrainMHCDataset, collate_mhc_panimmune,
    IntegratedTCRDataset, collate_tcr_panimmune,
    IntegratedAntibodyDataset, collate_antibody_panimmune,
)

# ============================= Config (edit here) ============================= #
# paths
PATH_MHC = "data\IEDB_retrain_extraction_MHC_final.csv"   # TRAIN ON THIS
PATH_TCR = "integrated_TCR_data.csv"                 # optional smoke test
PATH_AB  = "integrated_antibody_data.csv"            # optional smoke test

# train hyperparams
EPOCHS = 30
BATCH_SIZE = 1
LR = 3e-4                    # smaller LR since we train ESM too
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SEED = 42

# behavior
DO_SMOKE_TEST = True         # one forward pass on each dataset if file exists
LOCAL_TEST_100 = True        # run on first 100 rows for a quick local sanity training

# saving
SAVE_DIR = "model_parameter"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIG_OUT = os.path.join(SAVE_DIR, "config.json")
SAVE_EVERY = 5               # epochs

# ============================= Utils ============================= #
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

@torch.no_grad()
def smoke_test(model: nn.Module, dl: DataLoader, name: str):
    try:
        batch_samples, labels, _ = next(iter(dl))
    except StopIteration:
        print(f"[{name}] dataset empty — skipping.")
        return
    logits = model(batch_samples)
    print(f"[{name}] smoke OK | logits={tuple(logits.shape)} | labels={tuple(labels.shape)}")

def save_embedder_only(embedder: nn.Module, path: str):
    torch.save({"state_dict": embedder.state_dict()}, path)

def save_classification_head_only(model: PairAwareClassifier, path: str):
    """
    Save only the 'classification' model (pool + classifier), excluding embedder params.
    """
    head_state = {k: v for k, v in model.state_dict().items() if not k.startswith("embedder.")}
    torch.save({"state_dict": head_state}, path)

# ============================= Train ============================= #
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- build config, embedder, classifier
    cfg = load_default_config()
    # Train embedder + ESM jointly
    cfg.embedder.freeze_esm = False        # IMPORTANT: unfreeze ESM
    # (Optional precision tweaks)
    # cfg.embedder.torch_dtype = None

    embedder = build_embedder(cfg.embedder, device=device)
    model = build_classifier(embedder, cfg.classifier, device=device)  # PairAwareClassifier

    # Ensure ESM is train mode (since we unfreeze)
    try:
        model.embedder.esm.train().requires_grad_(True)
    except Exception:
        pass

    # ---- optimizer / loss
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()  # labels must be ints in {0,1}

    # ---- dataloaders
    if not os.path.exists(PATH_MHC):
        raise FileNotFoundError(f"Missing file: {PATH_MHC}")

    mhc_ds_full = IEDBRetrainMHCDataset(PATH_MHC, binarize=True, threshold=0.5)
    if LOCAL_TEST_100:
        indices = list(range(min(1, len(mhc_ds_full))))
        mhc_ds = Subset(mhc_ds_full, indices)
        print(f"[LocalTest] Using first {len(indices)} samples for quick training.")
    else:
        mhc_ds = mhc_ds_full


    mhc_dl = DataLoader(
        mhc_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_mhc_panimmune,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )

    # optional smoke tests on other datasets
    if DO_SMOKE_TEST and os.path.exists(PATH_TCR):
        tcr_ds = IntegratedTCRDataset(PATH_TCR, binarize=True, threshold=0.5)
        tcr_dl = DataLoader(
            tcr_ds, batch_size=min(BATCH_SIZE, 2), shuffle=False, collate_fn=collate_tcr_panimmune,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
        )
    else:
        tcr_dl = None
        if DO_SMOKE_TEST: print(f"[TCR] {PATH_TCR} not found — skipping smoke test.")

    if DO_SMOKE_TEST and os.path.exists(PATH_AB):
        ab_ds = IntegratedAntibodyDataset(PATH_AB, require_light=False, binarize=True, threshold=0.5)
        ab_dl = DataLoader(
            ab_ds, batch_size=min(BATCH_SIZE, 2), shuffle=False, collate_fn=collate_antibody_panimmune,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
        )
    else:
        ab_dl = None
        if DO_SMOKE_TEST: print(f"[AB]  {PATH_AB} not found — skipping smoke test.")

    # ---- smoke tests
    model.eval()
    with torch.no_grad():
        smoke_test(model, mhc_dl, "MHC(train)")
        if tcr_dl: smoke_test(model, tcr_dl, "TCR(test)")
        if ab_dl:  smoke_test(model, ab_dl,  "AB(test)")

    # ---- training (embedder + classifier jointly)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(epoch)
        total, correct, total_loss = 0, 0, 0.0

        for batch_samples, labels, _ in mhc_dl:
            labels = labels.to(device, non_blocking=True)

            logits = model(batch_samples)     # [B, 2]
            loss = loss_fn(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total += labels.numel()
                correct += (preds == labels).sum().item()
                total_loss += float(loss.item()) * labels.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"epoch {epoch}: loss={epoch_loss:.4f}  acc={epoch_acc:.3f}")

        # ---- periodic saving every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0:
            emb_path = os.path.join(SAVE_DIR, f"embedder_epoch{epoch}.pt")
            cls_path = os.path.join(SAVE_DIR, f"classification_epoch{epoch}.pt")
            save_embedder_only(model.embedder, emb_path)
            save_classification_head_only(model, cls_path)
            print(f"Saved: {emb_path}  |  {cls_path}")

    # ---- final save
    emb_path = os.path.join(SAVE_DIR, f"embedder_final.pt")
    cls_path = os.path.join(SAVE_DIR, f"classification_final.pt")
    save_embedder_only(model.embedder, emb_path)
    save_classification_head_only(model, cls_path)
    print(f"Saved final: {emb_path}  |  {cls_path}")

    # ---- save config for reproducibility
    with open(CONFIG_OUT, "w") as f:
        json.dump({
            "embedder": cfg.embedder.__dict__,
            "classifier": cfg.classifier.__dict__,
            "train": {
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "seed": SEED,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "local_test_100": LOCAL_TEST_100
            }
        }, f, indent=2)
    print(f"Saved config to {CONFIG_OUT}")

if __name__ == "__main__":
    main()
