# train_classifier.py — BCEWithLogitsLoss trainer (float labels in [0,1])
# - Works with your four modules:
#     * model_config.py            -> load_default_config() returning ModelConfig(esm, pair, classifier)
#     * PanImmunologyClassifier.py -> PanImmunologyClassifier.from_config(classifier_cfg, esm_cfg, pair_cfg)
#     * PanimmuneEmbedderPairs.py  -> used internally by the classifier
#     * Panimmune_dataload.py      -> datasets returning (concat_seq_str, label_float_in_[0,1])
#
# - Features:
#     * DDP (CUDA/XPU/CPU), AMP, grad clipping
#     * Checkpoint save/resume (full or split embedder/head)
#     * Cache freeing between epochs
#     * Regression metrics: MAE and RMSE (with BCE loss on logits)

import os, time, json, random, gc, math
from datetime import timedelta
from typing import Tuple, Optional, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# ==== your modules ====
from model_config import ModelConfig, load_default_config
from PanImmunologyClassifier import PanImmunologyClassifier
from Panimmune_dataload import (
    IEDBRetrainMHCDataset,
    IntegratedTCRDataset,
    IntegratedAntibodyDataset,
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

# ============================= Utils ============================= #
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
        try: torch.xpu.synchronize()
        except Exception: pass
        try: torch.xpu.empty_cache()
        except Exception: pass
    gc.collect()

class _NullScaler:
    def __init__(self, enabled: bool = False): self.enabled = False
    def scale(self, x): return x
    def unscale_(self, _): pass
    def step(self, opt): opt.step()
    def update(self): pass

def amp_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        return torch.xpu.amp.autocast()
    from contextlib import nullcontext
    return nullcontext()

def amp_scaler(device: torch.device, enabled: bool):
    if not enabled: return _NullScaler()
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        return torch.xpu.amp.GradScaler(enabled=True)
    return _NullScaler()

# ============================= DDP helpers ============================= #
def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    device = current_device(local_rank)

    backend = None
    is_ddp = False
    if world_size > 1:
        if device.type == "cuda":
            backend = "nccl"
            torch.cuda.set_device(local_rank)
        elif device.type == "xpu":
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
            if global_rank == 0:
                print(f"[DDP] init failed (backend='{backend}'): {e}. Using single process.")
            is_ddp = False

    return is_ddp, world_size, global_rank, local_rank, device, backend

def is_main_process(rank: int) -> bool:
    return rank == 0

def barrier_if_distributed(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()

def dist_mean_scalar(x: float, device: torch.device, is_ddp: bool) -> float:
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
        ds = IEDBRetrainMHCDataset(path)
    elif kind == "tcr":
        ds = IntegratedTCRDataset(path)
    elif kind == "ab":
        ds = IntegratedAntibodyDataset(path, require_light=False)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    if LOCAL_TEST_100:
        ds = Subset(ds, list(range(min(100, len(ds)))))
    return ds

def collate_panimmune(batch: List[Tuple[str, Any]]):
    """
    batch: list of (concat_seq_str, label_float_in_[0,1])
    Returns:
        batch_seqs: List[str]
        labels: FloatTensor [B] in [0,1]
        extras: None
    """
    seqs = [s for (s, _) in batch]
    labels = torch.tensor([float(y) for (_, y) in batch], dtype=torch.float32)
    return seqs, labels, None

def make_loader(kind: str, ds, batch_size: int, is_ddp: bool):
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collate_panimmune,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, drop_last=True
    )
    return dl, sampler

# ============================= Checkpoint helpers ============================= #
def _get_msave(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def save_checkpoint(model, opt, scaler, epoch, cfg: ModelConfig, train_sets, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    msave = _get_msave(model)
    ckpt = {
        "epoch": int(epoch),
        "model": msave.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": (None if isinstance(scaler, _NullScaler) else scaler.state_dict()),
        "cfg_esm": cfg.esm.__dict__,
        "cfg_pair": cfg.pair.__dict__,
        "cfg_classifier": cfg.classifier.__dict__,
        "train_sets": train_sets,
        "rng_state": torch.get_rng_state(),
        "backend_device": device.type,
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    path = os.path.join(save_dir, f"ckpt_epoch{epoch}.pt")
    torch.save(ckpt, path)
    print(f"[Save] Full checkpoint: {path}")

def load_checkpoint_if_any(resume_from: str, model, opt, scaler, device) -> int:
    if not resume_from:
        return 1
    if not os.path.exists(resume_from):
        print(f"[Resume] Not found: {resume_from} — starting fresh.")
        return 1

    map_loc = {"cuda": f"cuda:{device.index}", "xpu": f"xpu:{device.index}"}.get(device.type, "cpu")
    ckpt = torch.load(resume_from, map_location=map_loc)

    msave = _get_msave(model)
    msave.load_state_dict(ckpt["model"], strict=True)

    if ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[Resume] Optimizer state load failed ({e}); continuing with fresh optimizer.")

    if ckpt.get("scaler") and not isinstance(scaler, _NullScaler):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[Resume] AMP scaler state load failed ({e}); continuing with fresh scaler.")

    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

    last_epoch = int(ckpt.get("epoch", 0))
    print(f"[Resume] Loaded checkpoint: {resume_from} (epoch {last_epoch})")
    return last_epoch + 1

def load_split_epoch(model, embedder_path: str, head_path: str, device, strict: bool = True):
    """
    Fallback loader when only split weights exist (no optimizer/scaler/epoch).
    """
    msave = _get_msave(model)
    if not os.path.exists(embedder_path) or not os.path.exists(head_path):
        raise FileNotFoundError("[Resume-split] Missing embedder/head path(s).")

    mp = torch.load(embedder_path, map_location=device)
    hp = torch.load(head_path,    map_location=device)
    msave.embeder.load_state_dict(mp["state_dict"], strict=True)

    head_state = hp["state_dict"]
    missing, unexpected = msave.load_state_dict(head_state, strict=False)
    print(f"[Resume-split] missing={missing} unexpected={unexpected}")

# ============================= Train/Eval ============================= #
def train_one_loader(model: nn.Module, dl: DataLoader, loss_fn, opt, device, grad_clip_norm: float, use_amp: bool):
    """
    Returns: loss_avg, mae_avg, rmse_avg (averaged over the dataset on this rank)
    """
    total = 0
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    scaler = amp_scaler(device, enabled=use_amp)

    for batch_seqs, labels, _ in dl:  # batch_seqs: List[str]; labels: float32 [B] in [0,1]
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with amp_autocast(device):
            logits = model(batch_seqs)            # [B, num_classes or 1]
            # Ensure single logit for BCE; if model outputs >1, take the first channel
            if logits.dim() == 2 and logits.size(1) != 1:
                logits = logits[:, :1]
            logits = logits.squeeze(-1)           # [B]
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
            preds = torch.sigmoid(logits)         # [B] in [0,1]
            bsz = labels.size(0)
            total += bsz
            loss_sum += float(loss.item()) * bsz
            mae_sum  += torch.sum(torch.abs(preds - labels)).item()
            mse_sum  += torch.sum((preds - labels) ** 2).item()

    # per-rank averages
    loss_avg = loss_sum / max(total, 1)
    mae_avg  = mae_sum  / max(total, 1)
    rmse_avg = math.sqrt(mse_sum / max(total, 1)) if total > 0 else 0.0
    return loss_avg, mae_avg, rmse_avg

@torch.no_grad()
def smoke_test(model, dl, name, device, do_print=True):
    try:
        batch_seqs, labels, _ = next(iter(dl))
    except StopIteration:
        if do_print:
            print(f"[{name}] dataset empty — skipping.")
        return
    labels = labels.to(device, non_blocking=True)
    logits = model(batch_seqs)
    if logits.dim() == 2 and logits.size(1) != 1:
        logits = logits[:, :1]
    preds = torch.sigmoid(logits.squeeze(-1))
    if do_print:
        print(f"[{name}] smoke OK | logits={tuple(logits.shape)} | preds≈[{preds.min():.3f},{preds.max():.3f}] | labels={tuple(labels.shape)}")

# ============================= Main loop ============================= #
def run_training(train_sets,
                 epochs=EPOCHS, batch_size=BATCH_SIZE,
                 lr=BASE_LR, weight_decay=WEIGHT_DECAY, save_every=SAVE_EVERY,
                 save_dir=SAVE_DIR, config_out=CONFIG_OUT,
                 resume_from: str = "",
                 resume_split: Optional[Tuple[str, str, int]] = None):

    set_seed(SEED)
    is_ddp, world_size, global_rank, local_rank, device, backend = init_distributed()
    is_main = is_main_process(global_rank)

    # perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type in ("cuda", "xpu")

    if is_main:
        print(f"[Init] DDP={is_ddp} backend={backend} world_size={world_size} local_rank={local_rank} device={device} | TrainSets={train_sets}")

    # ==== Build model from unified config ====
    cfg: ModelConfig = load_default_config()
    # NOTE: BCEWithLogitsLoss expects a single logit; ensure num_classes==1 in cfg.classifier.
    if hasattr(cfg.classifier, "num_classes") and cfg.classifier.num_classes != 1:
        if is_main:
            print(f"[Warn] cfg.classifier.num_classes={cfg.classifier.num_classes} -> expected 1 for BCE. "
                  f"The trainer will slice to the first channel at runtime.")
    model = PanImmunologyClassifier.from_config(cfg.classifier, cfg.esm, cfg.pair)
    model = model.to(device)

    if is_ddp:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type != "cpu" else {}
        model = DDP(model, find_unused_parameters=True, **ddp_kwargs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = amp_scaler(device, enabled=use_amp)

    # ==== Resume ====
    if resume_from:
        start_epoch = load_checkpoint_if_any(resume_from, model, opt, scaler, device)
    elif resume_split:
        emb_path, head_path, start_epoch_hint = resume_split
        load_split_epoch(model, emb_path, head_path, device, strict=True)
        start_epoch = int(start_epoch_hint)
        if is_main:
            print(f"[Resume-split] Starting from epoch {start_epoch} (optimizer/scaler reset).")
    else:
        start_epoch = 1

    # ==== Build loaders ====
    loaders, samplers, sizes = {}, {}, {}
    for kind in train_sets:
        try:
            ds = build_dataset(kind, DATA_PATHS[kind])
            sizes[kind] = len(ds)
            dl, sampler = make_loader(kind, ds, batch_size, is_ddp)
            loaders[kind], samplers[kind] = dl, sampler
        except FileNotFoundError as e:
            if is_main: print(e)

    if not loaders:
        raise RuntimeError("No datasets found!")

    if is_main:
        for k, sz in sizes.items():
            print(f"[Data] {k}: {sz} samples (per-rank batch={batch_size})")

    # Optional smoke
    if DO_SMOKE_TEST and is_main:
        model.eval()
        for k, dl in loaders.items():
            smoke_test(model, dl, f"{k.upper()}(train)", device)
        model.train()

    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    # ==== Train ====
    for epoch in range(start_epoch, epochs + 1):
        # shuffle per-epoch in DDP
        for sampler in samplers.values():
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        if is_main:
            print(f"\n=== Epoch {epoch} ===")
        per_ds_metrics = {}
        t0 = time.time()

        model.train()
        for k, dl in loaders.items():
            loss_local, mae_local, rmse_local = train_one_loader(
                model, dl, loss_fn, opt, device, GRAD_CLIP_NORM, use_amp
            )
            # reduce to global means (per-rank averages -> mean across ranks)
            loss_avg = dist_mean_scalar(loss_local, device, is_ddp)
            mae_avg  = dist_mean_scalar(mae_local,  device, is_ddp)
            rmse_avg = dist_mean_scalar(rmse_local, device, is_ddp)
            per_ds_metrics[k] = (loss_avg, mae_avg, rmse_avg)
            if is_main:
                print(f"[{k}] loss={loss_avg:.4f}  MAE={mae_avg:.4f}  RMSE={rmse_avg:.4f}")

        # average across datasets
        avg_loss = sum(l for l, _, _ in per_ds_metrics.values()) / len(per_ds_metrics)
        avg_mae  = sum(m for _, m, _ in per_ds_metrics.values()) / len(per_ds_metrics)
        avg_rmse = sum(r for _, _, r in per_ds_metrics.values()) / len(per_ds_metrics)
        if is_main:
            print(f"[avg] loss={avg_loss:.4f}  MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}  time={time.time()-t0:.1f}s")

        if (epoch % save_every == 0) and is_main:
            save_checkpoint(model, opt, scaler, epoch, cfg, list(loaders.keys()), SAVE_DIR, device)

        barrier_if_distributed(is_ddp)
        free_device_cache(device)

    # ==== Final save ====
    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    if is_main:
        msave = _get_msave(model)
        emb_path = os.path.join(SAVE_DIR, "embedder_final.pt")
        cls_path = os.path.join(SAVE_DIR, "classification_final.pt")
        torch.save({"state_dict": msave.embeder.state_dict()}, emb_path)
        head_state = {k: v for k, v in msave.state_dict().items() if not k.startswith("embeder.")}
        torch.save({"state_dict": head_state}, cls_path)
        print(f"[Save] Inference weights: {emb_path} | {cls_path}")

        with open(CONFIG_OUT, "w") as f:
            json.dump({
                "esm": cfg.esm.__dict__,
                "pair": cfg.pair.__dict__,
                "classifier": cfg.classifier.__dict__,
                "train": {
                    "epochs": epochs, "batch_size_per_rank": batch_size,
                    "lr": BASE_LR, "weight_decay": WEIGHT_DECAY, "seed": SEED,
                    "grad_clip_norm": GRAD_CLIP_NORM, "local_test_100": LOCAL_TEST_100,
                    "train_sets": train_sets, "world_size": world_size, "backend": backend
                }
            }, f, indent=2)
        print(f"[Save] Config: {CONFIG_OUT}")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

# ============================= Entrypoint ============================= #
if __name__ == "__main__":
    MHC_ONLY = True
    TRAIN_SETS = ["mhc", "tcr", "ab"]
    selected_sets = ["ab"] if MHC_ONLY else TRAIN_SETS

    # Resume options
    RESUME = ""          # e.g., "model_parameter/ckpt_epoch10.pt"
    RESUME_SPLIT = None  # e.g., ("embedder_epoch10.pt", "classification_epoch10.pt", 11)

    run_training(
        train_sets=selected_sets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        save_every=SAVE_EVERY,
        save_dir=SAVE_DIR,
        config_out=CONFIG_OUT,
        resume_from=RESUME,
        resume_split=RESUME_SPLIT
    )
