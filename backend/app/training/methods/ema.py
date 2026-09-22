"""Exponential moving average of the trainable weights. Evaluation and checkpoints use the smoothed copy."""
from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.step = 0
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {n: p.detach().clone().float() for n, p in self.params.items()}

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))  # warm-up so early steps aren't dominated by the init
        for n, p in self.params.items():
            self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @contextmanager
    def swapped(self):
        """Temporarily load the averaged weights into the model."""
        backup = {n: p.detach().clone() for n, p in self.params.items()}
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(self.shadow[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(backup[n])
