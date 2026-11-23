"""Shared training utilities: EMA weights and a diffusion training loop on pre-built tensors."""

import copy
import math
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from pusht_wm_dit.data.latent_arrays import condition
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner


class EMA:
    """Exponential moving average of model weights (used for inference, as in Diffusion Policy)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # warm-up
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)


def train_diffusion(
    model: DiTTrajectoryPlanner,
    T: Dict[str, torch.Tensor],
    target_key: str,
    obs_history: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
    val_fn: Callable[[nn.Module], float],
    eval_every: int = 10,
    on_best: Optional[Callable[[nn.Module, nn.Module, int, float], None]] = None,
    log: Callable[[str], None] = print,
) -> float:
    """Trains `model` to denoise T[target_key] conditioned on the latents; keeps the EMA copy with the
    best validation score (lower is better). Returns the best score."""
    device = T["latent"].device
    n = len(T["latent"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05)
    ema = EMA(model, ema_decay)
    best, t0 = float("inf"), time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            out = model(path_future=T[target_key][idx], s_t=condition(T, idx, obs_history))
            opt.zero_grad(set_to_none=True)
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            tot += out["total_loss"].item()
        if ep % eval_every == 0 or ep == epochs:
            score = val_fn(ema.model)
            is_best = score < best
            if is_best:
                best = score
                if on_best is not None:
                    on_best(ema.model, model, ep, score)
            log(f"  epoch {ep:03d} | train loss {tot / steps_per_epoch:.4f} | val {score:.4f}{' *' if is_best else ''} | {time.perf_counter() - t0:.0f}s")
    return best
