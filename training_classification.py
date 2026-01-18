# training_classification.py
"""
MHC-I peptide binding REGRESSION trainer (normalized targets in [0,1];
fixed length; 2D conv+flatten head)

- Works with your updated modules:
    * model_config.py               -> load_default_config() returning ModelConfig(pair, classifier)
    * MHCpeptideEmbeddingClassifier.py    -> PanImmunologyRegressor.from_config(pair_cfg, clf_cfg, grid_len)
    * MHCpeptideEmbedding.py     -> one-hot + U-Net + axial bottleneck, returns z [B,L,L,C]
    * MHCpeptide_dataload.py         -> IEDBRetrainMHCDataset + collate_concat_regression
                                      (supports label_fn for raw label processing)
- Key notes:
    * Model output activation is CONFIG-DRIVEN via cfg.classifier.output_activation
      - "sigmoid" => predictions in (0,1) (recommended for normalized labels)
      - "none"    => unbounded predictions
    * Trainer uses MSELoss on the label space (here: normalized [0,1])
    * Fixed peptide length: drop > pep_len, pad < pep_len (pep_len in PairConfig, default 11)
"""

import os, time, json, random, gc, math
from datetime import timedelta
from typing import Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# ==== your modules ====
from model_config import ModelConfig, load_default_config
from MHCpeptideEmbeddingClassifier import MHCpeptideRegressor
from MHCpeptide_dataload import (
    MHCpeptideDataset,
    collate_concat_regression,
)

# ============================= Defaults ============================= #
DATA_PATHS = {
    "mhc": "data/IEDB_retrain_extraction_MHC_final.csv",
}

EPOCHS = 22
BATCH_SIZE = 16
BASE_LR = 1e-2
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SEED = 42

DO_SMOKE_TEST = False
LOCAL_TEST_100 = False  # True for quick debug

SAVE_DIR = "model_parameter"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIG_OUT = os.path.join(SAVE_DIR, "config.json")
SAVE_EVERY = 1

WARMUP_STEPS = 1000  # linear warmup steps for fresh-optimizer resumes

# Optional clamp on predictions (generally keep None if using sigmoid in-model)
# If you want extra safety for fp16/amp, set (0.0, 1.0)
PRED_CLAMP: Optional[Tuple[float, float]] = None  # e.g., (0.0, 1.0)


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
    if not enabled:
        return _NullScaler()
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        return torch.xpu.amp.GradScaler(enabled=True)
    return _NullScaler()

class WarmupLRScheduler:
    """Linear warmup to base_lr for `warmup_steps` steps, then hold base_lr."""
    def __init__(self, optimizer, base_lr: float, warmup_steps: int = 0, start_step: int = 0):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))
        self.global_step = int(max(0, start_step))

    def step(self):
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * float(self.global_step + 1) / float(self.warmup_steps)
        else:
            lr = self.base_lr
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.global_step += 1


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
LabelFn = Callable[[Any], float]

def build_dataset_mhc(path: str, cfg: ModelConfig, label_fn: Optional[LabelFn] = None):
    """
    label_fn: optional raw-label -> float transform applied inside dataset.
              If your CSV Label is already normalized [0,1], keep label_fn=None.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[mhc] Missing file: {path}")

    ds = MHCpeptideDataset(
        csv_path=path,
        pair_cfg=cfg.pair,
        label_fn=label_fn,
        sep=":",
        antigen_col="Antigen",
        mhc_col="MHC_sequence",
        label_col="Label",
    )

    if LOCAL_TEST_100:
        ds = Subset(ds, list(range(min(256, len(ds)))))
    return ds

def make_loader(ds, batch_size: int, is_ddp: bool):
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_concat_regression,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=True,
    )
    return dl, sampler


# ============================= Checkpoint helpers ============================= #
def _get_msave(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def save_checkpoint(model, opt, scaler, epoch, cfg: ModelConfig, save_dir, device, global_step: int = 0):
    os.makedirs(save_dir, exist_ok=True)
    msave = _get_msave(model)
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": msave.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": (None if isinstance(scaler, _NullScaler) else scaler.state_dict()),
        "cfg_pair": cfg.pair.__dict__,
        "cfg_classifier": cfg.classifier.__dict__,
        "rng_state": torch.get_rng_state(),
        "backend_device": device.type,
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    path = os.path.join(save_dir, f"ckpt_epoch{epoch}.pt")
    torch.save(ckpt, path)
    print(f"[Save] Full checkpoint: {path}")

def load_checkpoint_if_any(resume_from: str, model, opt, scaler, device) -> Tuple[int, int]:
    if not resume_from:
        return 1, 0
    if not os.path.exists(resume_from):
        print(f"[Resume] Not found: {resume_from} — starting fresh.")
        return 1, 0

    map_loc = {"cuda": f"cuda:{device.index}", "xpu": f"xpu:{device.index}"}.get(device.type, "cpu")
    ckpt = torch.load(resume_from, map_location=map_loc)

    msave = _get_msave(model)
    msave.load_state_dict(ckpt["model"], strict=True)

    loaded_opt = False
    if ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
            loaded_opt = True
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
    global_step = int(ckpt.get("global_step", 0))

    print(f"[Resume] Loaded checkpoint: {resume_from} (epoch {last_epoch})"
          f"{' with optimizer/scaler' if loaded_opt else ' (fresh optimizer)'}")
    return last_epoch + 1, global_step if loaded_opt else 0


# ============================= Train/Eval ============================= #
def _maybe_clamp(pred: torch.Tensor) -> torch.Tensor:
    if PRED_CLAMP is None:
        return pred
    lo, hi = PRED_CLAMP
    return torch.clamp(pred, float(lo), float(hi))

def train_one_loader(
    model: nn.Module,
    dl: DataLoader,
    loss_fn,
    opt,
    device,
    grad_clip_norm: float,
    use_amp: bool,
    scheduler: WarmupLRScheduler,
):
    total = 0
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    scaler = amp_scaler(device, enabled=use_amp)

    for batch_seqs, y in dl:
        y = y.to(device, non_blocking=True).view(-1)  # [B]

        scheduler.step()
        opt.zero_grad(set_to_none=True)

        with amp_autocast(device):
            pred = model(batch_seqs).view(-1)  # [B]
            pred = _maybe_clamp(pred)
            loss = loss_fn(pred, y)

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
            bsz = y.size(0)
            total += bsz
            loss_sum += float(loss.item()) * bsz
            mae_sum += torch.sum(torch.abs(pred - y)).item()
            mse_sum += torch.sum((pred - y) ** 2).item()

    loss_avg = loss_sum / max(total, 1)
    mae_avg  = mae_sum  / max(total, 1)
    rmse_avg = math.sqrt(mse_sum / max(total, 1)) if total > 0 else 0.0
    return loss_avg, mae_avg, rmse_avg

@torch.no_grad()
def smoke_test(model, dl, name, device, do_print=True):
    try:
        batch_seqs, y = next(iter(dl))
    except StopIteration:
        if do_print:
            print(f"[{name}] dataset empty — skipping.")
        return
    y = y.to(device, non_blocking=True).view(-1)
    pred = model(batch_seqs).view(-1)
    pred = _maybe_clamp(pred)
    if do_print:
        print(
            f"[{name}] smoke OK | pred={tuple(pred.shape)} "
            f"| pred≈[{pred.min():.3f},{pred.max():.3f}] "
            f"| y≈[{y.min():.3f},{y.max():.3f}]"
        )


# ============================= Main loop ============================= #
def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
    save_every=SAVE_EVERY,
    save_dir=SAVE_DIR,
    config_out=CONFIG_OUT,
    resume_from: str = "",
    label_fn: Optional[LabelFn] = None,
):
    """
    label_fn: optional raw-label -> float transform used by the dataset.
              If your CSV Label is already normalized [0,1], set label_fn=None.
    """
    set_seed(SEED)
    is_ddp, world_size, global_rank, local_rank, device, backend = init_distributed()
    is_main = is_main_process(global_rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type in ("cuda", "xpu")

    if is_main:
        print(f"[Init] DDP={is_ddp} backend={backend} world_size={world_size} local_rank={local_rank} device={device}")

    cfg: ModelConfig = load_default_config()

    # Grid length must match fixed_len
    grid_len = int(getattr(cfg.pair, "fixed_len", cfg.pair.mhc_len + cfg.pair.pep_len))

    # Model output behavior is config-driven (cfg.classifier.output_activation)
    model = MHCpeptideRegressor.from_config(
        pair_cfg=cfg.pair,
        clf_cfg=cfg.classifier,
        grid_len=grid_len,
        device=str(device),
    ).to(device)

    if is_ddp:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type != "cpu" else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scaler = amp_scaler(device, enabled=use_amp)

    start_epoch, loaded_global_step = (
        load_checkpoint_if_any(resume_from, model, opt, scaler, device) if resume_from else (1, 0)
    )
    scheduler = WarmupLRScheduler(opt, base_lr=lr, warmup_steps=WARMUP_STEPS, start_step=loaded_global_step)

    ds = build_dataset_mhc(DATA_PATHS["mhc"], cfg, label_fn=label_fn)
    dl, sampler = make_loader(ds, batch_size, is_ddp)

    if is_main:
        act = getattr(cfg.classifier, "output_activation", "sigmoid")
        lrng = getattr(cfg.classifier, "label_range", None)
        print(f"[Data] mhc: {len(ds)} samples (per-rank batch={batch_size})")
        print(f"[Fixed] mhc_len={cfg.pair.mhc_len} pep_len={cfg.pair.pep_len} => grid_len={grid_len}")
        print(f"[Model] output_activation={act}  label_range={lrng}")
        if PRED_CLAMP is not None:
            print(f"[Warn] PRED_CLAMP={PRED_CLAMP} (usually keep None when using sigmoid)")

    if DO_SMOKE_TEST and is_main:
        model.eval()
        smoke_test(model, dl, "MHC(train)", device)
        model.train()

    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    for epoch in range(start_epoch, epochs + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        if is_main:
            print(f"\n=== Epoch {epoch} ===")
        t0 = time.time()

        model.train()
        loss_local, mae_local, rmse_local = train_one_loader(
            model, dl, loss_fn, opt, device, GRAD_CLIP_NORM, use_amp, scheduler
        )

        loss_avg = dist_mean_scalar(loss_local, device, is_ddp)
        mae_avg  = dist_mean_scalar(mae_local,  device, is_ddp)
        rmse_avg = dist_mean_scalar(rmse_local, device, is_ddp)

        if is_main:
            print(f"[mhc] loss={loss_avg:.4f}  MAE={mae_avg:.4f}  RMSE={rmse_avg:.4f}  time={time.time()-t0:.1f}s")

        if (epoch % save_every == 0) and is_main:
            save_checkpoint(model, opt, scaler, epoch, cfg, save_dir, device, global_step=scheduler.global_step)

        barrier_if_distributed(is_ddp)
        free_device_cache(device)

    barrier_if_distributed(is_ddp)
    free_device_cache(device)

    if is_main:
        msave = _get_msave(model)
        torch.save({"state_dict": msave.state_dict()}, os.path.join(save_dir, "model_final.pt"))

        with open(config_out, "w") as f:
            json.dump({
                "pair": cfg.pair.__dict__,
                "classifier": cfg.classifier.__dict__,
                "train": {
                    "epochs": epochs,
                    "batch_size_per_rank": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "seed": SEED,
                    "grad_clip_norm": GRAD_CLIP_NORM,
                    "local_test_100": LOCAL_TEST_100,
                    "world_size": world_size,
                    "backend": backend,
                    "warmup_steps": WARMUP_STEPS,
                    "pred_clamp": PRED_CLAMP,
                    "grid_len": grid_len,
                }
            }, f, indent=2)

        print(f"[Save] Final model + config in: {save_dir}")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


# ============================= Entrypoint ============================= #
if __name__ == "__main__":
    RESUME = ""  # e.g., "model_parameter/ckpt_epoch10.pt"

    # Your labels are normalized to [0,1] => keep label_fn=None
    label_fn = None

    # Optional: extra safety clamp if you want (normally not needed with sigmoid output)
    # PRED_CLAMP = (0.0, 1.0)

    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        save_every=SAVE_EVERY,
        save_dir=SAVE_DIR,
        config_out=CONFIG_OUT,
        resume_from=RESUME,
        label_fn=label_fn,
    )
