"""Token-level loss with optional label smoothing and per-token weights (used to emphasise domain vocabulary)."""
import torch
import torch.nn.functional as F


def find_subseq(row: list[int], pat: list[int]) -> list[int]:
    n = len(pat)
    return [i for i in range(len(row) - n + 1) if row[i:i + n] == pat] if n else []


def domain_weights(labels: list[list[int]], patterns: list[list[int]], weight: float, width: int) -> torch.Tensor:
    """(B, width) tensor: `weight` on tokens that belong to a domain-term occurrence, 1 elsewhere."""
    w = torch.ones(len(labels), width)
    for b, row in enumerate(labels):
        for pat in patterns:
            for i in find_subseq(row, pat):
                w[b, i:i + len(pat)] = weight
    return w


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None,
                label_smoothing: float = 0.0) -> torch.Tensor:
    """Mean cross-entropy over non-ignored tokens (labels == -100 are ignored), normalised by the weight mass."""
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.float().reshape(-1, v), labels.reshape(-1), ignore_index=-100,
                         label_smoothing=label_smoothing, reduction="none").view_as(labels)
    mask = (labels != -100).float()
    w = mask if weights is None else weights.to(mask.device) * mask
    return (ce * w).sum() / w.sum().clamp(min=1e-8)
