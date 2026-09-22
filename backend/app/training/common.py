"""Shared training loop, progress reporting and helpers used by every model plugin."""
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from ..core.gpu import detect_gpu
from .config import TrainConfig
from .methods.params import param_groups as build_param_groups


class Progress:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "progress.jsonl"

    def emit(self, type_: str, **kw) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": type_, "t": time.time(), **kw}) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu_used_gb": round((total - free) / 1024**3, 2),
        "gpu_total_gb": round(total / 1024**3, 2),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu_util_pct": detect_gpu().utilization_pct,
    }


def amp_dtype(cfg: TrainConfig):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.precision)


def to_device(batch: dict, dev: str) -> dict:
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_optimizer(groups: list[dict], cfg: TrainConfig):
    """`groups`: [{"params", "lr", "weight_decay"}]."""
    if cfg.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=cfg.learning_rate)
    return torch.optim.AdamW(groups, lr=cfg.learning_rate)


def optimizer_groups(groups: list[dict], cfg: TrainConfig, lr: float | None = None) -> list[dict]:
    base = cfg.learning_rate if lr is None else lr
    return [{"params": g["params"], "lr": base * g["lr_scale"], "weight_decay": g["weight_decay"]} for g in groups]


def plan_steps(n_items: int, cfg: TrainConfig) -> tuple[int, int]:
    """(optimizer steps per epoch, total optimizer steps)."""
    spe = max(1, math.ceil(math.ceil(n_items / cfg.batch_size) / cfg.grad_accum))
    return spe, (cfg.max_steps or spe * cfg.epochs)


def snapshot(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def restore(model, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.device, p.dtype))


def fit(model, train_items, val_items, collate, forward, cfg: TrainConfig, run_dir: Path, progress: Progress, save_best, *,
        groups: list[dict] | None = None, ema=None, post_step=None, extra_eval=None, val_forward=None) -> dict:
    """Generic loop: gradient accumulation, AMP, LR schedule with warmup, per-group LRs / weight decay, optional EMA,
    periodic evaluation (val loss, plus e.g. val WER via `extra_eval`), selection of the best checkpoint by the chosen
    metric, early stopping, cooperative stop (a STOP file in the run dir), and -- importantly -- the best weights are
    loaded back into `model` before returning, so anything evaluated afterwards is exactly what was checkpointed."""
    dev = device()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    groups = groups or build_param_groups(model, cfg)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    opt = make_optimizer(optimizer_groups(groups, cfg), cfg)
    loader = DataLoader(train_items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_items, batch_size=cfg.batch_size, collate_fn=collate) if val_items else None
    steps_per_epoch, total_steps = plan_steps(len(train_items), cfg)
    sched = get_scheduler(cfg.lr_scheduler, opt, num_warmup_steps=min(cfg.warmup_steps, total_steps // 2),
                          num_training_steps=total_steps)
    adt = amp_dtype(cfg)
    use_amp = adt is not None and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16" and dev == "cuda")
    eval_every = cfg.eval_steps or steps_per_epoch
    log_every = max(1, total_steps // 200)

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                with torch.autocast(dev, dtype=adt, enabled=use_amp):
                    size = len(next(iter(b.values())))
                    tot += (val_forward or forward)(model, to_device(b, dev)).item() * size
                n += size
        return tot / max(n, 1)

    def evaluate() -> dict:
        with (ema.swapped() if ema else nullcontext()):
            m = {"val_loss": val_loss()}
            if extra_eval:
                m.update(extra_eval(model))
        model.train()
        return m

    def checkpoint() -> dict:
        with (ema.swapped() if ema else nullcontext()):
            save_best()
            return snapshot(model)

    progress.emit("start", total_steps=total_steps, steps_per_epoch=steps_per_epoch, trainable_params=n_train,
                  train_samples=len(train_items), val_samples=len(val_items or []), learning_rate=cfg.learning_rate,
                  groups=len(groups), ema=bool(ema))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step, bad = 0, 0
    best_score, best_metrics, best_state = (math.inf, math.inf), None, None
    monitor = cfg.select_by
    status, early = "completed", False
    model.train()
    stop_file = Path(run_dir) / "STOP"
    for _epoch in range(cfg.epochs):
        micro, acc = 0, 0.0
        for i, batch in enumerate(loader):
            with torch.autocast(dev, dtype=adt, enabled=use_amp):
                loss = forward(model, to_device(batch, dev))
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became NaN/inf; lower the learning rate or use bf16/fp32.")
            scaler.scale(loss / cfg.grad_accum).backward()
            acc += loss.item()
            micro += 1
            if micro < cfg.grad_accum and i < len(loader) - 1:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if post_step:
                post_step(step)
            if ema:
                ema.update()
            opt.zero_grad(set_to_none=True)
            el = time.time() - t0
            if step % log_every == 0 or step == total_steps:
                progress.emit("train", step=step, total_steps=total_steps, epoch=round(step / steps_per_epoch, 2),
                              loss=round(acc / micro, 5), lr=sched.get_last_lr()[0], elapsed=round(el, 1),
                              eta=round(el / step * (total_steps - step), 1), **gpu_stats())
            micro, acc = 0, 0.0
            if val_loader and (step % eval_every == 0 or step == total_steps):
                m = evaluate()
                key = monitor if monitor in m else "val_loss"
                score = (m[key], m["val_loss"])  # ties (common with WER on small sets) fall back to loss
                improved = score < best_score
                if improved:
                    best_score, best_metrics, bad = score, m, 0
                    best_state = checkpoint()
                else:
                    bad += 1
                progress.emit("eval", step=step, improved=improved, monitor=key, best=round(best_metrics["val_loss"], 5),
                              **{k: round(v, 5) for k, v in m.items()})
                if cfg.early_stopping and bad >= cfg.patience:
                    early = True
                    break
            if stop_file.exists():
                status = "stopped"
                break
            if step >= total_steps:
                break
        if early or status == "stopped" or step >= total_steps:
            break
    if best_state is None:  # no validation data (or stopped before the first eval): keep the final weights
        best_state = checkpoint()
    else:
        restore(model, best_state)
    return {"steps": step, "total_steps": total_steps,
            "best_val_loss": None if best_metrics is None else round(best_metrics["val_loss"], 5),
            "best_metrics": None if best_metrics is None else {k: round(v, 5) for k, v in best_metrics.items()},
            "monitor": monitor, "train_seconds": round(time.time() - t0, 1), "early_stopped": early, "status": status,
            "trainable_params": n_train, "learning_rate": cfg.learning_rate, "scheduler": cfg.lr_scheduler,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if dev == "cuda" else None}
