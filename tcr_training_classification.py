# training_tcr_classification.py
"""
TCR+pMHC BINARY CLASSIFICATION trainer (labels 0/1; BCEWithLogitsLoss)

Enhanced with:
- Different learning rates for pMHC (pretrained), TCR, and Classifier
- Pretrained pMHC embedder with optional freezing
- Comprehensive verification checks
"""

import os, time, json, random, gc
from datetime import timedelta
from typing import Tuple, Optional, Any, Callable, Dict
import hashlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# ==== your modules ====
from src.model_config import ModelConfig, load_default_config

# IMPORTANT: your file defines class "TCRpMHCClassifier"
from TCRmhcEmbeddingClassifier import TCRpMHCClassifier

from PanTCR_dataload import (
    IntegratedTCRDataset,
    collate_tcr_pmhc,
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TCR-pMHC Training')
    parser.add_argument(
        '--data_path',
        type=str,
        default="/scratch/10119/ghzheng/ImmunoPairNet/data/PISTE_unipep.csv",
        help='Path to TCR dataset CSV file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='model_parameter_vdjdb\ckpt_epoch36.pt',
        help='provide checkpoint dir'
    )
    return parser.parse_args()

# Parse arguments
args = parse_args()

# ============================= Editable config at top ============================= #
# Data
DATA_PATHS = {
    "tcr": args.data_path,
}

# Data
DATA_PATHS = {
    "tcr": r"data\PMTnet.csv",
}

# Train hyperparams
EPOCHS = 120
BATCH_SIZE = 16

# Learning rates for different components
PMHC_LR = 1e-5          # For pMHC embedder (pretrained, needs smaller updates)
BASE_LR = 4e-4          # For TCR embedder (randomly initialized)
CLASSIFIER_LR = 4e-4    # For classifier head (can learn faster)

WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SEED = 42

# Debug
DO_SMOKE_TEST = False
LOCAL_TEST_100 = True  # quick debug subset (takes first up to 64 samples)

# Saving
# Extract filename without extension from the data path
data_file = DATA_PATHS["tcr"]
filename = os.path.basename(data_file)  # "vdjdb_data.csv"
filename_no_ext = os.path.splitext(filename)[0]  # "vdjdb_data"

# Create save directory name
SAVE_DIR = f"model_parameter_{filename_no_ext}"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIG_OUT = os.path.join(SAVE_DIR, "config.json")
SAVE_EVERY = 1

# LR warmup
WARMUP_STEPS = 500  # linear warmup steps

# Optional probability clamp (numerical safety for BCELoss)
PRED_CLAMP: Optional[Tuple[float, float]] = (1e-6, 1.0 - 1e-6)

# ===== Pretrained pMHC settings =====
# Point this to your pretrained pMHC state_dict / checkpoint.
PMHC_PRETRAIN_CKPT = "./model_parameter/pmhc_embedder_only_epoch160.pt"
FREEZE_PMHC = True        # True => keep pMHC fixed during training


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

def load_pretrained_pmhc_into_model_with_checks(
    model: nn.Module,
    ckpt_path: str,
    *,
    freeze: bool = True,
    device: Optional[torch.device] = None,
    strict: bool = False,
    assert_no_missing: bool = False,
    assert_no_unexpected: bool = True,
    do_sha1_check: bool = True,
    do_grad_check: bool = True,
    do_param_step_check: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Loads pretrained weights into: model.embedder.pmhc
    Optionally freezes pmhc.
    Performs 3 checks (combined):
      (1) load_state_dict missing/unexpected reporting (+ optional asserts)
      (2) SHA1 hash before/after load to prove weights changed
      (3) freeze/grad/one-step param-delta checks (if requested)

    Returns a dict with diagnostics and a closure you can call inside training
    to do grad + one-step param-delta checks after backward/step.

    Usage (recommended):
      info = load_pretrained_pmhc_into_model_with_checks(model, PMHC_PRETRAIN_CKPT, freeze=True, device=device)

      # inside train loop:
      # after backward (grads exist):
      info["after_backward_check"](model)

      # after optimizer step:
      info["after_step_check"](model)

    Notes:
      - Call this AFTER model is created, BEFORE wrapping with DDP.
      - For DDP-wrapped model, the returned closures also work (they unwrap model.module).
    """

    def _unwrap(m: nn.Module) -> nn.Module:
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        # Handle pMHC-specific checkpoint format: {"embedder_state_dict": ..., "pair_cfg": ..., "grid_len": ...}
        if isinstance(ckpt_obj, dict) and "embedder_state_dict" in ckpt_obj:
            return ckpt_obj["embedder_state_dict"]
        if isinstance(ckpt_obj, dict):
            return ckpt_obj
        raise ValueError("Unrecognized checkpoint format; expected dict-like state_dict.")

    def _sd_sha1(sd: Dict[str, torch.Tensor]) -> str:
        h = hashlib.sha1()
        for k in sorted(sd.keys()):
            v = sd[k]
            h.update(k.encode("utf-8"))
            # deterministic bytes
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    info: Dict[str, Any] = {
        "loaded": False,
        "ckpt_path": ckpt_path,
        "freeze": bool(freeze),
        "missing": [],
        "unexpected": [],
        "sha1_before": None,
        "sha1_after": None,
        "sha1_changed": None,
        "pmhc_num_params": None,
        "pmhc_trainable_tensors": None,
        "grad_any_nonzero": None,
        "param_delta_max_after_step": None,
    }

    # --- basic guards ---
    if not ckpt_path:
        if verbose:
            print("[pMHC] No pretrained ckpt set; training from scratch.")
        return info
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[pMHC] Pretrained checkpoint not found: {ckpt_path}")

    m = _unwrap(model)
    if not hasattr(m, "embedder") or not hasattr(m.embedder, "pmhc"):
        raise AttributeError("Model must have model.embedder.pmhc to load pMHC pretrained weights.")

    pmhc_mod = m.embedder.pmhc

    # --- SHA1 before ---
    if do_sha1_check:
        info["sha1_before"] = _sd_sha1(pmhc_mod.state_dict())

    # --- load ---
    map_location = "cpu" if device is None else device
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = _extract_state_dict(ckpt)

    missing, unexpected = pmhc_mod.load_state_dict(sd, strict=strict)
    info["missing"] = list(missing)
    info["unexpected"] = list(unexpected)
    info["loaded"] = True

    # --- SHA1 after ---
    if do_sha1_check:
        info["sha1_after"] = _sd_sha1(pmhc_mod.state_dict())
        info["sha1_changed"] = (info["sha1_before"] != info["sha1_after"])

    # --- stats ---
    info["pmhc_num_params"] = int(sum(p.numel() for p in pmhc_mod.parameters()))
    info["pmhc_trainable_tensors"] = int(sum(1 for p in pmhc_mod.parameters() if p.requires_grad))

    # --- optional asserts on keys ---
    if assert_no_unexpected and len(unexpected) > 0:
        raise RuntimeError(f"[pMHC] Unexpected keys when loading: {unexpected[:30]}")
    if assert_no_missing and len(missing) > 0:
        raise RuntimeError(f"[pMHC] Missing keys when loading: {missing[:30]}")

    # --- freeze ---
    if freeze:
        for p in pmhc_mod.parameters():
            p.requires_grad = False
        pmhc_mod.eval()
        info["pmhc_trainable_tensors"] = int(sum(1 for p in pmhc_mod.parameters() if p.requires_grad))

    # --- print summary ---
    if verbose:
        print(f"[pMHC] Loaded pretrained weights from: {ckpt_path}")
        if do_sha1_check:
            print(f"[pMHC] SHA1 before={info['sha1_before']}")
            print(f"[pMHC] SHA1 after ={info['sha1_after']}")
            print(f"[pMHC] SHA1 changed? {info['sha1_changed']}")
        if info["missing"]:
            print(f"[pMHC] Missing keys (up to 10): {info['missing'][:10]}{' ...' if len(info['missing'])>10 else ''}")
        if info["unexpected"]:
            print(f"[pMHC] Unexpected keys (up to 10): {info['unexpected'][:10]}{' ...' if len(info['unexpected'])>10 else ''}")
        print(f"[pMHC] Num params: {info['pmhc_num_params']:,}")
        print(f"[pMHC] Frozen={freeze} | trainable tensors in pmhc_mod: {info['pmhc_trainable_tensors']}")

    # --- prepare step-check state (captures first param tensor snapshot) ---
    step_state: Dict[str, Any] = {"w0": None, "did_snapshot": False, "did_delta": False, "did_grad": False}

    def after_backward_check(model_in: nn.Module) -> Optional[bool]:
        """
        Call this AFTER backward (grads populated), BEFORE optimizer.step().
        Returns whether pMHC has any nonzero grads (True/False), or None if disabled.
        """
        if not do_grad_check:
            return None
        if step_state["did_grad"]:
            return info.get("grad_any_nonzero", None)

        mm = _unwrap(model_in)
        pm = mm.embedder.pmhc

        any_grad = False
        for p in pm.parameters():
            if p.grad is not None and p.grad.detach().abs().sum().item() > 0:
                any_grad = True
                break

        info["grad_any_nonzero"] = bool(any_grad)
        step_state["did_grad"] = True
        if verbose:
            print(f"[pMHC][Check] Any nonzero grads after backward? {any_grad} (expect False if frozen)")
        return bool(any_grad)

    def before_step_snapshot(model_in: nn.Module) -> Optional[torch.Tensor]:
        """
        Call this right BEFORE optimizer.step() (or at start of first iteration).
        Takes a snapshot of one representative parameter tensor to compare later.
        """
        if not do_param_step_check:
            return None
        if step_state["did_snapshot"]:
            return step_state["w0"]

        mm = _unwrap(model_in)
        pm = mm.embedder.pmhc
        w0 = next(iter(pm.parameters())).detach().clone()
        step_state["w0"] = w0
        step_state["did_snapshot"] = True
        return w0

    def after_step_check(model_in: nn.Module) -> Optional[float]:
        """
        Call this AFTER optimizer.step().
        Returns max absolute delta for one representative pMHC parameter tensor, or None if disabled.
        """
        if not do_param_step_check:
            return None
        if step_state["did_delta"]:
            return info.get("param_delta_max_after_step", None)

        if not step_state["did_snapshot"]:
            # automatically snapshot if user forgot
            before_step_snapshot(model_in)

        mm = _unwrap(model_in)
        pm = mm.embedder.pmhc

        w1 = next(iter(pm.parameters())).detach()
        delta = (w1 - step_state["w0"]).abs().max().item()
        info["param_delta_max_after_step"] = float(delta)
        step_state["did_delta"] = True

        if verbose:
            print(f"[pMHC][Check] max|Δ| after 1 optimizer step: {delta:.3e} (expect 0 if frozen)")
        return float(delta)

    # return closures so you can run the checks exactly once in training
    info["after_backward_check"] = after_backward_check
    info["before_step_snapshot"] = before_step_snapshot
    info["after_step_check"] = after_step_check
    return info


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
    """Linear warmup for ALL param groups, each to their own base_lr."""
    def __init__(self, optimizer, warmup_steps: int = 0, start_step: int = 0):
        self.opt = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self.global_step = int(max(0, start_step))
        
        # Store each group's base LR
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            scale = float(self.global_step + 1) / float(self.warmup_steps)
            for g, base_lr in zip(self.opt.param_groups, self.base_lrs):
                g["lr"] = base_lr * scale
        else:
            for g, base_lr in zip(self.opt.param_groups, self.base_lrs):
                g["lr"] = base_lr
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


# ============================= Optimizer with param groups ============================= #

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def build_optimizer_with_param_groups(model: nn.Module, freeze_pmhc: bool = True, verbose: bool = True):
    """
    Build optimizer with different learning rates for different components.
    
    - pMHC embedder: PMHC_LR (or excluded if frozen)
    - TCR embedder: BASE_LR
    - Classifier head: CLASSIFIER_LR
    - Other parameters: BASE_LR
    """
    m = _unwrap_model(model)
    
    param_groups = []
    
    # 1. pMHC parameters (only if not frozen)
    if not freeze_pmhc:
        pmhc_params = [p for p in m.embedder.pmhc.parameters() if p.requires_grad]
        if pmhc_params:
            param_groups.append({
                'params': pmhc_params,
                'lr': PMHC_LR,
                'name': 'pmhc_embedder'
            })
            if verbose:
                print(f"[Optimizer] pMHC embedder: {len(pmhc_params)} param tensors, LR={PMHC_LR}")
    
    # 2. TCR parameters
    tcr_params = [p for p in m.embedder.tcr.parameters() if p.requires_grad]
    if tcr_params:
        param_groups.append({
            'params': tcr_params,
            'lr': BASE_LR,
            'name': 'tcr_embedder'
        })
        if verbose:
            print(f"[Optimizer] TCR embedder: {len(tcr_params)} param tensors, LR={BASE_LR}")
    
    # 3. Classifier head parameters (it's called 'head' in TCRpMHCClassifier)
    head_params = [p for p in m.head.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': CLASSIFIER_LR,
            'name': 'classifier_head'
        })
        if verbose:
            print(f"[Optimizer] Classifier head: {len(head_params)} param tensors, LR={CLASSIFIER_LR}")
    
    # 4. Any remaining parameters (e.g., from full_model or other components)
    covered_params = set()
    if not freeze_pmhc:
        covered_params.update(id(p) for p in m.embedder.pmhc.parameters())
    covered_params.update(id(p) for p in m.embedder.tcr.parameters())
    covered_params.update(id(p) for p in m.head.parameters())
    
    other_params = [p for p in m.parameters() if id(p) not in covered_params and p.requires_grad]
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': BASE_LR,
            'name': 'other'
        })
        if verbose:
            print(f"[Optimizer] Other parameters: {len(other_params)} param tensors, LR={BASE_LR}")
    
    if not param_groups:
        raise ValueError("No trainable parameters found! Check if model is properly initialized.")
    
    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


# ============================= Data helpers ============================= #

LabelFn = Callable[[Any], float]

def _default_label_fn(x: Any) -> float:
    return float(x)

def build_dataset_tcr(path: str, cfg: ModelConfig, label_fn: Optional[LabelFn] = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[tcr] Missing file: {path}")

    ds = IntegratedTCRDataset(
        csv_path=path,
        pmhc_cfg=cfg.pmhc,
        tcr_cfg=cfg.tcr,
        label_fn=label_fn if label_fn is not None else _default_label_fn,
        antigen_col="Antigen",
        mhc_col="MHC_sequence",
        tcra_col="TCR_alpha",
        tcrb_col="TCR_beta",
        label_col="Label",
    )

    if LOCAL_TEST_100:
        ds = Subset(ds, list(range(min(64, len(ds)))))
    return ds

def make_loader(ds, batch_size: int, is_ddp: bool):
    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_tcr_pmhc,
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
        "cfg_pmhc": cfg.pmhc.__dict__,
        "cfg_tcr": cfg.tcr.__dict__,
        "cfg_full": cfg.full.__dict__ if getattr(cfg, "full", None) is not None else None,
        "cfg_classifier": cfg.tcr_classifier.__dict__,
        "rng_state": torch.get_rng_state(),
        "backend_device": device.type,
        "pmhc_pretrain_ckpt": PMHC_PRETRAIN_CKPT,
        "freeze_pmhc": bool(FREEZE_PMHC),
        "pmhc_lr": PMHC_LR,
        "base_lr": BASE_LR,
        "classifier_lr": CLASSIFIER_LR,
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


# ============================= Metrics ============================= #

@torch.no_grad()
def binary_metrics_from_probs(probs: torch.Tensor, y: torch.Tensor, thresh: float = 0.5):
    """
    probs: [B] in (0,1)
    y:     [B] in {0,1}
    """
    probs = torch.clamp(probs, 0.0, 1.0)
    pred = (probs >= thresh).float()
    acc = (pred == y).float().mean().item()

    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    prec = tp / max(tp + fp, 1.0)
    rec  = tp / max(tp + fn, 1.0)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, prec, rec, f1


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
    acc_sum = 0.0
    f1_sum = 0.0

    scaler = amp_scaler(device, enabled=use_amp)

    for peps, mhcs, tcras, tcrbs, y in dl:
        y = y.to(device, non_blocking=True).view(-1).float()  # [B] 0/1

        scheduler.step()
        opt.zero_grad(set_to_none=True)

        with amp_autocast(device):
            logits = model(peps, mhcs, tcras, tcrbs).view(-1)  # raw logits (any real value)
            loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits)  # only for metrics
            acc, _, _, f1 = binary_metrics_from_probs(probs.detach(), y.detach())

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

        bsz = y.size(0)
        total += bsz
        loss_sum += float(loss.item()) * bsz
        acc_sum  += float(acc) * bsz
        f1_sum   += float(f1) * bsz

    loss_avg = loss_sum / max(total, 1)
    acc_avg  = acc_sum  / max(total, 1)
    f1_avg   = f1_sum   / max(total, 1)
    return loss_avg, acc_avg, f1_avg

@torch.no_grad()
def smoke_test(model, dl, name, device, do_print=True):
    try:
        peps, mhcs, tcras, tcrbs, y = next(iter(dl))
    except StopIteration:
        if do_print:
            print(f"[{name}] dataset empty — skipping.")
        return
    y = y.to(device, non_blocking=True).view(-1).float()
    logits = model(peps, mhcs, tcras, tcrbs).view(-1)
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 0.0, 1.0)
    if do_print:
        print(f"[{name}] smoke OK | logits={tuple(logits.shape)} logits≈[{logits.min():.3f},{logits.max():.3f}] probs≈[{probs.min():.3f},{probs.max():.3f}] y≈[{y.min():.1f},{y.max():.1f}]")


# ============================= Main loop ============================= #

def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_every=SAVE_EVERY,
    save_dir=SAVE_DIR,
    config_out=CONFIG_OUT,
    resume_from: str = "",
    label_fn: Optional[LabelFn] = None,
):
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

    # Build model (NOT DDP yet)
    model = TCRpMHCClassifier.from_config(
        pmhc_cfg=cfg.pmhc,
        tcr_cfg=cfg.tcr,
        full_cfg=cfg.full,
        clf_cfg=cfg.tcr_classifier,
        device=str(device),
        clamp_to_label_range=True,
        apply_mask_in_embedder=True,
    ).to(device)

    # Load pretrained pMHC weights + optionally freeze (do this BEFORE DDP wrap)
    if is_main and PMHC_PRETRAIN_CKPT:
        print(f"[pMHC] Will load pretrained: {PMHC_PRETRAIN_CKPT} | freeze={FREEZE_PMHC}")
    load_pretrained_pmhc_into_model_with_checks(model, PMHC_PRETRAIN_CKPT, freeze=FREEZE_PMHC, device=device)

    if is_ddp:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type != "cpu" else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)

    # Optimizer with different learning rates for different components
    if is_main:
        print(f"\n[Optimizer] Building optimizer with differentiated learning rates:")
        print(f"  pMHC LR: {PMHC_LR} (frozen={FREEZE_PMHC})")
        print(f"  TCR LR: {BASE_LR}")
        print(f"  Classifier LR: {CLASSIFIER_LR}")
    
    opt = build_optimizer_with_param_groups(model, freeze_pmhc=FREEZE_PMHC, verbose=is_main)

    # BCEWithLogitsLoss (expects raw logits, not probabilities)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = amp_scaler(device, enabled=use_amp)

    start_epoch, loaded_global_step = (
        load_checkpoint_if_any(resume_from, model, opt, scaler, device) if resume_from else (1, 0)
    )
    scheduler = WarmupLRScheduler(opt, warmup_steps=WARMUP_STEPS, start_step=loaded_global_step)

    ds = build_dataset_tcr(DATA_PATHS["tcr"], cfg, label_fn=label_fn)
    dl, sampler = make_loader(ds, batch_size, is_ddp)

    if is_main:
        print(f"\n[Data] tcr: {len(ds)} samples (per-rank batch={batch_size})")
        print(f"[Fixed pMHC] mhc_len={cfg.pmhc.mhc_len} pep_len={cfg.pmhc.pep_len} fixed_len={cfg.pmhc.fixed_len}")
        print(f"[TCR max] a_max={cfg.tcr.tcr_a_max_len} b_max={cfg.tcr.tcr_b_max_len} (b optional)")
        print(f"[Full grid] max_len_total={cfg.full.max_len_total} pair_dim={cfg.full.pair_dim}")
        print(f"[Pretrain] pmhc_ckpt='{PMHC_PRETRAIN_CKPT}' freeze_pmhc={FREEZE_PMHC}")

    if DO_SMOKE_TEST and is_main:
        model.eval()
        smoke_test(model, dl, "TCR(train)", device)
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
        loss_local, acc_local, f1_local = train_one_loader(
            model, dl, loss_fn, opt, device, GRAD_CLIP_NORM, use_amp, scheduler
        )

        loss_avg = dist_mean_scalar(loss_local, device, is_ddp)
        acc_avg  = dist_mean_scalar(acc_local,  device, is_ddp)
        f1_avg   = dist_mean_scalar(f1_local,   device, is_ddp)

        if is_main:
            current_lrs = {g.get('name', f'group_{i}'): g['lr'] for i, g in enumerate(opt.param_groups)}
            lr_str = ", ".join([f"{name}={lr:.2e}" for name, lr in current_lrs.items()])
            print(f"[tcr] loss={loss_avg:.4f}  ACC={acc_avg:.4f}  F1={f1_avg:.4f}  LRs=[{lr_str}]  time={time.time()-t0:.1f}s")

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
                "pmhc": cfg.pmhc.__dict__,
                "tcr": cfg.tcr.__dict__,
                "full": cfg.full.__dict__,
                "classifier": cfg.tcr_classifier.__dict__,
                "train": {
                    "epochs": epochs,
                    "batch_size_per_rank": batch_size,
                    "pmhc_lr": PMHC_LR,
                    "base_lr": BASE_LR,
                    "classifier_lr": CLASSIFIER_LR,
                    "weight_decay": weight_decay,
                    "seed": SEED,
                    "grad_clip_norm": GRAD_CLIP_NORM,
                    "local_test_100": LOCAL_TEST_100,
                    "world_size": world_size,
                    "backend": backend,
                    "warmup_steps": WARMUP_STEPS,
                    "pred_clamp": PRED_CLAMP,
                    "expect_logits": True,
                    "loss_function": "BCEWithLogitsLoss",
                    "pmhc_pretrain_ckpt": PMHC_PRETRAIN_CKPT,
                    "freeze_pmhc": bool(FREEZE_PMHC),
                }
            }, f, indent=2)

        print(f"\n[Save] Final model + config in: {save_dir}")

    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


# ============================= Entrypoint ============================= #

if __name__ == "__main__":
    RESUME = args.checkpoint  # e.g., "model_parameter_vdjdb_data/ckpt_epoch10.pt"

    # Labels are already 0/1 => label_fn=None
    label_fn = None

    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_every=SAVE_EVERY,
        save_dir=SAVE_DIR,
        config_out=CONFIG_OUT,
        resume_from=RESUME,
        label_fn=label_fn,
    )