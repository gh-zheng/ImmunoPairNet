# train_bcr_classifier.py  (multi-dataset training + simple flags + DP)
import os, random, json, time
from typing import Tuple, Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---- model & config (your file) ----
from classification import load_default_config, build_embedder, build_classifier, PairAwareClassifier

# ---- datasets (your file) ----
from Panimmnue_dataload import (
    IEDBRetrainMHCDataset, collate_mhc_panimmune,
    IntegratedTCRDataset,   collate_tcr_panimmune,
    IntegratedAntibodyDataset, collate_antibody_panimmune,
)

# ============================= Defaults ============================= #
DATA_PATHS = {
    "mhc": "data/IEDB_retrain_extraction_MHC_final.csv",
    "tcr": "data/integrated_TCR_data.csv",
    "ab":  "data/integrated_antibody_data.csv",
}

EPOCHS = 10
BATCH_SIZE = 32                # per-step batch for any dataset
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SEED = 42

DO_SMOKE_TEST = False
LOCAL_TEST_100 = False

SAVE_DIR = "model_parameter"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIG_OUT = os.path.join(SAVE_DIR, "config.json")
SAVE_EVERY = 5

# ============================= Utils ============================= #
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

@torch.no_grad()
def smoke_test(model: nn.Module, dl: DataLoader, name: str):
    if dl is None:
        return
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
    head_state = {k: v for k, v in model.state_dict().items() if not k.startswith("embedder.")}
    torch.save({"state_dict": head_state}, path)

# ============================= Data helpers ============================= #
def make_dataset_and_loader(kind: str, path: str, batch_size: int) -> Tuple[Any, DataLoader]:
    """
    Build a dataset + dataloader for the given kind ('mhc' | 'tcr' | 'ab').
    Returns (dataset, dataloader). Raises FileNotFoundError if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{kind}] Missing file: {path}")

    if kind == "mhc":
        ds = IEDBRetrainMHCDataset(path, binarize=False, threshold=0.5)
        if LOCAL_TEST_100: ds = Subset(ds, list(range(min(1, len(ds)))))
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mhc_panimmune,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, drop_last=True
        )
    elif kind == "tcr":
        ds = IntegratedTCRDataset(path, binarize=False, threshold=0.5)
        if LOCAL_TEST_100: ds = Subset(ds, list(range(min(1, len(ds)))))
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_tcr_panimmune,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, drop_last=True
        )
    elif kind == "ab":
        ds = IntegratedAntibodyDataset(path, require_light=False, binarize=False, threshold=0.5)
        if LOCAL_TEST_100: ds = Subset(ds, list(range(min(1, len(ds)))))
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_antibody_panimmune,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, drop_last=True
        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    return ds, dl

# ============================= Train loop (per-dataloader) ============================= #
def train_one_loader(model: nn.Module, dl: DataLoader, loss_fn, opt, device, grad_clip_norm: float) -> Tuple[float, float]:
    total, correct, total_loss = 0, 0, 0.0
    for batch_samples, labels, _ in dl:
        labels = labels.to(device, non_blocking=True)

        logits = model(batch_samples)     # [B, 2]
        loss = loss_fn(logits, labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            total_loss += float(loss.item()) * labels.size(0)
    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

# ============================= Main (controlled by simple flags below) ============================= #
def run_training(train_sets: list, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 lr: float = LR, weight_decay: float = WEIGHT_DECAY, save_every: int = SAVE_EVERY,
                 save_dir: str = SAVE_DIR, config_out: str = CONFIG_OUT):
    # Seed & device
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[Device] Using {device} | CUDA devices: {torch.cuda.device_count()} | TrainSets={train_sets}")

    # Build config / model
    cfg = load_default_config()
    cfg.embedder.freeze_esm = False  
    embedder = build_embedder(cfg.embedder, device=device)
    model = build_classifier(embedder, cfg.classifier, device=device)

    # Multi-GPU (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"[Parallel] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # Optimizer / loss
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Build loaders for selected datasets
    loaders: Dict[str, DataLoader] = {}
    for kind in train_sets:
        try:
            _, dl = make_dataset_and_loader(kind, DATA_PATHS[kind], batch_size)
            loaders[kind] = dl
        except FileNotFoundError as e:
            print(e)

    if not loaders:
        raise RuntimeError("No datasets available to train on. Please check file paths.")

    # Optional smoke tests
    model.eval()
    with torch.no_grad():
        for kind, dl in loaders.items():
            smoke_test(model, dl, f"{kind.upper()}(train)")

    # Train
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        per_ds_metrics = {}
        start = time.time()

        for kind, dl in loaders.items():
            ds_loss, ds_acc = train_one_loader(model, dl, loss_fn, opt, device, GRAD_CLIP_NORM)
            per_ds_metrics[kind] = (ds_loss, ds_acc)
            print(f"[{kind}] loss={ds_loss:.4f}  acc={ds_acc:.3f}")

        # macro-average across datasets
        avg_loss = sum(l for l, _ in per_ds_metrics.values()) / len(per_ds_metrics)
        avg_acc  = sum(a for _, a in per_ds_metrics.values()) / len(per_ds_metrics)
        dt = time.time() - start
        print(f"[avg]  loss={avg_loss:.4f}  acc={avg_acc:.3f}  time={dt:.1f}s")

        # periodic save (unwrap DP if needed)
        if epoch % save_every == 0:
            msave = model.module if isinstance(model, nn.DataParallel) else model
            emb_path = os.path.join(save_dir, f"embedder_epoch{epoch}.pt")
            cls_path = os.path.join(save_dir, f"classification_epoch{epoch}.pt")
            save_embedder_only(msave.embedder, emb_path)
            save_classification_head_only(msave, cls_path)
            print(f"Saved: {emb_path}  |  {cls_path}")

    # final save
    msave = model.module if isinstance(model, nn.DataParallel) else model
    emb_path = os.path.join(save_dir, f"embedder_final.pt")
    cls_path = os.path.join(save_dir, f"classification_final.pt")
    save_embedder_only(msave.embedder, emb_path)
    save_classification_head_only(msave, cls_path)
    print(f"Saved final: {emb_path}  |  {cls_path}")

    # write config
    with open(config_out, "w") as f:
        json.dump({
            "embedder": cfg.embedder.__dict__,
            "classifier": cfg.classifier.__dict__,
            "train": {
                "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "weight_decay": weight_decay, "seed": SEED,
                "grad_clip_norm": GRAD_CLIP_NORM, "local_test_100": LOCAL_TEST_100,
                "train_sets": train_sets
            }
        }, f, indent=2)
    print(f"Saved config to {config_out}")

# ============================= Simple switches here ============================= #
if __name__ == "__main__":
    # ---- Edit these two lines to control training sets ----
    MHC_ONLY   = True              # if True, forces training on MHC only
    TRAIN_SETS = ["mhc", "tcr", "ab"]    # used only when MHC_ONLY is False; choose any of: "mhc", "tcr", "ab"

    selected_sets = ["mhc"] if MHC_ONLY else TRAIN_SETS
    run_training(train_sets=selected_sets)
