"""Optimizer parameter groups: no weight decay on biases/norms, layer-wise LR decay, LoRA+ (higher LR for B matrices)."""
import re

from ..config import TrainConfig

_LAYER = re.compile(r"^(.*?layers)\.(\d+)\.")
_LOW_LEVEL = ("embed", "conv1", "conv2", "prenet", "positions")


def _stack_depths(names: list[str]) -> dict[str, int]:
    depth: dict[str, int] = {}
    for n in names:
        m = _LAYER.match(n)
        if m:
            depth[m.group(1)] = max(depth.get(m.group(1), 0), int(m.group(2)) + 1)
    return depth


def layer_scale(name: str, depths: dict[str, int], llrd: float) -> float:
    """1.0 for the top layer of each stack, llrd^k for k layers below it; input-side modules get the smallest LR."""
    if llrd >= 1.0:
        return 1.0
    m = _LAYER.match(name)
    if m:
        return llrd ** (depths[m.group(1)] - 1 - int(m.group(2)))
    if any(k in name for k in _LOW_LEVEL):
        return llrd ** max(depths.values(), default=1)
    return 1.0


def param_groups(model, cfg: TrainConfig) -> list[dict]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    depths = _stack_depths([n for n, _ in named])
    groups: dict[tuple[float, float], list] = {}
    for n, p in named:
        scale = layer_scale(n, depths, cfg.llrd)
        if cfg.loraplus_ratio > 0 and "lora_B" in n:
            scale *= cfg.loraplus_ratio
        decay = 0.0 if (p.ndim <= 1 or "norm" in n.lower() or n.endswith(".bias")) else cfg.weight_decay
        groups.setdefault((round(scale, 8), decay), []).append(p)
    return [{"params": ps, "lr_scale": s, "weight_decay": d} for (s, d), ps in sorted(groups.items())]
