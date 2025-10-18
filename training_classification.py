# train_classifier.py — DDP trainer supporting CUDA & XPU, AMP, grad clipping, rank-0 I/O, and cache freeing
import os, random, json, time
from datetime import timedelta
from typing import Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# ---- model & config (your files) ----

from classification import load_default_config, build_embedder, build_classifier, PairAwareClassifier


# ---- datasets (your files) ----
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
BATCH_SIZE = 1
BASE_LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SEED = 42

DO_SMOKE_TEST = False
LOCAL_TEST_100 = True

SAVE_DIR = "model_parameter"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIG_OUT = os.path.join(SAVE_DIR, "config.json")
SAVE_EVERY = 1

# ============================= Utilities ============================= #
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()

def current_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if is_xpu_available():
        return torch.device("xpu", local_rank)
    return torch.device("cpu")

def free_device_cache(device: torch.device):
    """Release unoccupied cache; call BEFORE training and AFTER each epoch."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "xpu" and is_xpu_available():
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    else:
        import gc
        gc.collect()

# AMP helpers that gracefully degrade if a backend lacks AMP classes
class _NullScaler:
    def __init__(self, enabled: bool = False): self.enabled = False
    def scale(self, x): return x
    def unscale_(self, _): pass
    def step(self, opt): opt.step()
    def update(self): pass

def amp_autocast(device: torch.device):
    """Return the correct autocast context manager for device, or a no-op."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        return torch.xpu.amp.autocast()  # Intel extension
    # no-op context manager
    from contextlib import nullcontext
    return nullcontext()

def amp_scaler(device: torch.device, enabled: bool):
    """Return a GradScaler for device if available, else a no-op scaler."""
    if not enabled:
        return _NullScaler()
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        # Intel extension provides its own GradScaler
        return torch.xpu.amp.GradScaler(enabled=True)
    return _NullScaler()

# ============================= DDP helpers ============================= #
def init_distributed():
    """Initialize torch.distributed with a sensible backend for CUDA/XPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    device = current_device(local_rank)

    backend = None
    if world_size > 1:
        if device.type == "cuda":
            backend = "nccl"
            torch.cuda.set_device(local_rank)
        elif device.type == "xpu":
            # Prefer 'ccl' when oneCCL is installed; allow override via env
            backend = os.environ.get("TORCH_DDP_BACKEND", "ccl")
            try:
                if hasattr(torch, "xpu"):
                    torch.xpu.set_device(local_rank)
            except Exception:
                pass
        else:
            backend = "gloo"

        try:
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=3600))
            is_ddp = True
        except Exception as e:
            # Fallback to single-process if DDP init fails
            if global_rank == 0:
                print(f"[DDP] Failed to init backend='{backend}': {e}. Falling back to single process.")
            is_ddp = False
    else:
        is_ddp = False

    return is_ddp, world_size, global_rank, local_rank, device, backend

def is_main_process(rank: int) -> bool:
    return rank == 0

def barrier_if_distributed(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()

def dist_mean_scalar(x: float, device: torch.device, is_ddp: bool) -> float:
    """Synchronize scalar averages across ranks."""
    if not is_ddp or not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())

# ============================= Data helpers ============================= #
def build_dataset(kind: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{kind}] Missing file: {path}")
    if kind == "mhc":
        ds = IEDBRetrainMHCDataset(path, binarize=False, threshold=0.5)
    elif kind == "tcr":
        ds = IntegratedTCRDataset(path, binarize=False, threshold=0.5)
    elif kind == "ab":
        ds = IntegratedAntibodyDataset(path, require_light=False, binarize=False, threshold=0.5)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    if LOCAL_TEST_100:
        ds = Subset(ds, list(range(min(10, len(ds)))))
    return ds

def make_loader(kind: str, ds, batch_size: int, is_ddp: bool):
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    if kind == "mhc":
        collate_fn = collate_mhc_panimmune
    elif kind == "tcr":
        collate_fn = collate_tcr_panimmune
    else:
        collate_fn = collate_antibody_panimmune
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, drop_last=True
    )
    return dl, sampler

# ============================= Training functions ============================= #
def train_one_loader(model: nn.Module, dl: DataLoader, loss_fn, opt, device, grad_clip_norm: float, use_amp: bool):
    total, correct, total_loss = 0, 0, 0.0
    scaler = amp_scaler(device, enabled=use_amp)

    for batch_samples, labels, _ in dl:
        labels = labels.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(device):
            logits = model(batch_samples)    # model handles encoding & internal device use
            loss = loss_fn(logits, labels)

        if isinstance(scaler, _NullScaler):
            loss.backward()
            if grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(opt)
            scaler.update()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            total_loss += float(loss.item()) * labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def smoke_test(model, dl, name, device, do_print=True):
    try:
        batch_samples, labels, _ = next(iter(dl))
    except StopIteration:
        if do_print:
            print(f"[{name}] dataset empty — skipping.")
        return
    labels = labels.to(device, non_blocking=True)
    logits = model(batch_samples)
    if do_print:
        print(f"[{name}] smoke OK | logits={tuple(logits.shape)} | labels={tuple(labels.shape)}")

# ============================= Main training loop ============================= #
def run_training(train_sets, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 lr=BASE_LR, weight_decay=WEIGHT_DECAY, save_every=SAVE_EVERY,
                 save_dir=SAVE_DIR, config_out=CONFIG_OUT):

    set_seed(SEED)
    is_ddp, world_size, global_rank, local_rank, device, backend = init_distributed()
    is_main = is_main_process(global_rank)

    # Perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type in ("cuda", "xpu")  # enable AMP on CUDA and XPU (if available)

    if is_main:
        print(f"[Init] DDP={is_ddp} backend={backend} world_size={world_size} local_rank={local_rank} device={device} | TrainSets={train_sets}")

    # ==== Build model ====
    cfg = load_default_config()
    cfg.embedder.freeze_esm = True
    embedder = build_embedder(cfg.embedder, device=device)
    model = build_classifier(embedder, cfg.classifier, device=device)

    if is_ddp:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type != "cpu" else {}
        model = DDP(model, find_unused_parameters=True, **ddp_kwargs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # ==== Build loaders ====
    loaders, samplers, sizes = {}, {}, {}
    for kind in train_sets:
        try:
            ds = build_dataset(kind, DATA_PATHS[kind])
            sizes[kind] = len(ds)
            dl, sampler = make_loader(kind, ds, batch_size, is_ddp)
            loaders[kind], samplers[kind] = dl, sampler
        except FileNotFoundError as e:
            if is_main:
                print(e)

    if not loaders:
        raise RuntimeError("No datasets found!")

    if is_main:
        for k, sz in sizes.items():
            print(f"[Data] {k}: {sz} samples (per-rank batch={batch_size})")

    # ==== Optional smoke test (rank-0) ====
    if DO_SMOKE_TEST and is_main:
        model.eval()
        for k, dl in loaders.items():
            smoke_test(model, dl, f"{k.upper()}(train)", device)
        model.train()

    # >>>>>>>>>> EMPTY CACHE BEFORE TRAINING <<<<<<<<<<
    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    # ==== Train loop ====
    for epoch in range(1, epochs + 1):
        # distinct shuffles across epochs (DDP)
        for k, sampler in samplers.items():
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        if is_main:
            print(f"\n=== Epoch {epoch} ===")
        per_ds_metrics = {}
        start = time.time()

        model.train()
        for k, dl in loaders.items():
            loss_local, acc_local = train_one_loader(model, dl, loss_fn, opt, device, GRAD_CLIP_NORM, use_amp)
            # reduce to global means
            loss_avg = dist_mean_scalar(loss_local, device, is_ddp)
            acc_avg  = dist_mean_scalar(acc_local,  device, is_ddp)
            per_ds_metrics[k] = (loss_avg, acc_avg)
            if is_main:
                print(f"[{k}] loss={loss_avg:.4f}  acc={acc_avg:.3f}")

        # average across datasets
        avg_loss = sum(l for l, _ in per_ds_metrics.values()) / len(per_ds_metrics)
        avg_acc  = sum(a for _, a in per_ds_metrics.values()) / len(per_ds_metrics)
        if is_main:
            print(f"[avg] loss={avg_loss:.4f} acc={avg_acc:.3f} time={time.time()-start:.1f}s")

        # save checkpoints (rank 0)
        if (epoch % save_every == 0) and is_main:
            msave = model.module if isinstance(model, DDP) else model
            emb_path = os.path.join(save_dir, f"embedder_epoch{epoch}.pt")
            cls_path = os.path.join(save_dir, f"classification_epoch{epoch}.pt")
            torch.save({"state_dict": msave.embedder.state_dict()}, emb_path)
            head_state = {k: v for k, v in msave.state_dict().items() if not k.startswith("embedder.")}
            torch.save({"state_dict": head_state}, cls_path)
            print(f"Saved: {emb_path} | {cls_path}")

        # sync and free memory caches AFTER each epoch
        barrier_if_distributed(is_ddp)
        free_device_cache(device)

    # ==== Final save ====
    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    if is_main:
        msave = model.module if isinstance(model, DDP) else model
        emb_path = os.path.join(save_dir, "embedder_final.pt")
        cls_path = os.path.join(save_dir, "classification_final.pt")
        torch.save({"state_dict": msave.embedder.state_dict()}, emb_path)
        head_state = {k: v for k, v in msave.state_dict().items() if not k.startswith("embedder.")}
        torch.save({"state_dict": head_state}, cls_path)
        print(f"Saved final: {emb_path} | {cls_path}")

        # config dump
        with open(config_out, "w") as f:
            json.dump({
                "embedder": cfg.embedder.__dict__,
                "classifier": cfg.classifier.__dict__,
                "train": {
                    "epochs": epochs, "batch_size_per_rank": batch_size,
                    "lr": lr, "weight_decay": weight_decay, "seed": SEED,
                    "grad_clip_norm": GRAD_CLIP_NORM, "local_test_100": LOCAL_TEST_100,
                    "train_sets": train_sets, "world_size": world_size, "backend": backend
                }
            }, f, indent=2)
        print(f"Saved config to {config_out}")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

# ============================= Entrypoint ============================= #
if __name__ == "__main__":
    MHC_ONLY = True
    TRAIN_SETS = ["mhc", "tcr", "ab"]
    selected_sets = ["mhc"] if MHC_ONLY else TRAIN_SETS
    run_training(train_sets=selected_sets)
