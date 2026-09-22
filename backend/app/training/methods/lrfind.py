"""LR range test: sweep the learning rate exponentially over a few real batches, read the loss curve, restore everything."""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import amp_dtype, device, make_optimizer, optimizer_groups, to_device


def find_lr(model, train_items, collate, forward, cfg, groups, *, steps: int = 40, lo: float = 1e-7, hi: float = 1.0) -> dict:
    dev = device()
    params = [p for g in groups for p in g["params"]]
    saved = [p.detach().to("cpu", copy=True) for p in params]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if dev == "cuda" else None
    opt = make_optimizer(optimizer_groups(groups, cfg, lo), cfg)
    loader = iter(DataLoader(train_items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate))
    adt = amp_dtype(cfg)
    use_amp = adt is not None and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16" and dev == "cuda")
    lrs, raw, smooth = [], [], []
    avg, best = 0.0, math.inf
    model.train()
    try:
        for i in range(steps):
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(DataLoader(train_items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate))
                batch = next(loader)
            lr = lo * (hi / lo) ** (i / (steps - 1))
            for g, base in zip(opt.param_groups, groups):
                g["lr"] = lr * base["lr_scale"]
            with torch.autocast(dev, dtype=adt, enabled=use_amp):
                loss = forward(model, to_device(batch, dev))
            if not torch.isfinite(loss):
                break
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            v = loss.item()
            avg = 0.9 * avg + 0.1 * v
            s = avg / (1 - 0.9 ** (i + 1))  # bias-corrected smoothing
            lrs.append(lr), raw.append(v), smooth.append(s)
            best = min(best, s)
            if i > 8 and s > 4 * best:  # diverged
                break
    finally:  # always put the model back exactly as it was
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s.to(p.device, p.dtype))
        opt.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        del opt
    return {"suggested": suggest(lrs, smooth), "steps": len(lrs),
            "curve": [[float(a), float(b)] for a, b in zip(lrs[:: max(1, len(lrs) // 60)], smooth[:: max(1, len(lrs) // 60)])]}


def suggest(lrs: list[float], smooth: list[float]) -> float | None:
    """LR one order of magnitude below the loss minimum, unless the curve shows no real descent (then None)."""
    if len(lrs) < 12:
        return None
    s = np.array(smooth)
    i_min = int(np.argmin(s))
    if s[i_min] > 0.98 * s[: max(3, len(s) // 6)].mean():  # essentially flat: no signal
        return None
    return float(lrs[i_min] / 10)
