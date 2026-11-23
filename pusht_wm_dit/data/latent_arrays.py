"""Vectorised training/validation tensors for the DiT planner and the inverse dynamics.

Builds everything once from the zarr arrays and the cached World Model latents, instead of
decoding images per sample every epoch (which made train_dit_planner.py slow). The ordering and the
anti-diagonal reflection match PushTDataset exactly, so cached latent i belongs to sample i.
"""

from typing import Dict

import numpy as np
import torch

from pusht_wm_dit.data.pusht_dataset import PushTDataset
from pusht_wm_dit.utils.normalizer import normalize_image


def _norm(p):
    return (p - 256.0) / 256.0


def _pose4(state_rows):
    return np.stack([_norm(state_rows[..., 2]), _norm(state_rows[..., 3]),
                     np.sin(state_rows[..., 4]), np.cos(state_rows[..., 4])], axis=-1)


def _reflect_xy(v):
    return -v[..., [1, 0]]


def _reflect_pose4(p):
    return np.stack([-p[..., 1], -p[..., 0], p[..., 3], p[..., 2]], axis=-1)


def base_arrays(ds: PushTDataset, pred_horizon: int, action_horizon: int) -> Dict[str, np.ndarray]:
    """Arrays for ds.indices order (no augmentation): path (N,H,4), actions (N,Ta,2), agent (N,2), prev (N,)."""
    t = np.array([i[0] for i in ds.indices])
    start = np.array([i[1] for i in ds.indices])
    end = np.array([i[2] for i in ds.indices])
    ph = np.minimum(t[:, None] + np.arange(pred_horizon)[None], end[:, None] - 1)
    ah = np.minimum(t[:, None] + np.arange(action_horizon)[None], end[:, None] - 1)
    prev = np.where(t > start, np.arange(len(t)) - 1, np.arange(len(t)))  # index of the t-1 sample
    return {
        "t": t,
        "path": _pose4(ds.state_arr[ph]).astype(np.float32),
        "actions": _norm(ds.action_arr[ah]).astype(np.float32),
        "agent": _norm(ds.state_arr[t, :2]).astype(np.float32),
        "prev": prev,
    }


def train_tensors(ds: PushTDataset, latents: torch.Tensor, pred_horizon: int, action_horizon: int,
                  device: str) -> Dict[str, torch.Tensor]:
    """Training tensors aligned with the latent cache (base samples, then reflected copies if augmented)."""
    a = base_arrays(ds, pred_horizon, action_horizon)
    n = len(a["t"])
    path, acts, agent, prev = a["path"], a["actions"], a["agent"], a["prev"]
    if ds.augment_symmetry:
        path = np.concatenate([path, _reflect_pose4(path)])
        acts = np.concatenate([acts, _reflect_xy(acts)])
        agent = np.concatenate([agent, _reflect_xy(agent)])
        prev = np.concatenate([prev, prev + n])
    assert len(latents) == len(path), f"latent cache ({len(latents)}) does not match dataset ({len(path)})"
    return {
        "latent": latents.float().to(device),
        "prev": torch.from_numpy(prev).long().to(device),
        "path": torch.from_numpy(path).to(device),
        "actions": torch.from_numpy(acts).to(device),
        "agent": torch.from_numpy(agent).to(device),
    }


@torch.no_grad()
def val_tensors(ds: PushTDataset, wm, pred_horizon: int, action_horizon: int, device: str) -> Dict[str, torch.Tensor]:
    """Validation tensors; latents are computed with the frozen World Model encoder."""
    a = base_arrays(ds, pred_horizon, action_horizon)
    lat = []
    for i in range(0, len(a["t"]), 256):
        tt = a["t"][i:i + 256]
        fr = torch.from_numpy(normalize_image(ds.img_arr[tt])).float().to(device)
        ag = torch.from_numpy(a["agent"][i:i + 256]).to(device)
        lat.append(wm.encode(fr, ag))
    return {
        "latent": torch.cat(lat),
        "prev": torch.from_numpy(a["prev"]).long().to(device),
        "path": torch.from_numpy(a["path"]).to(device),
        "actions": torch.from_numpy(a["actions"]).to(device),
        "agent": torch.from_numpy(a["agent"]).to(device),
    }


def condition(T: Dict[str, torch.Tensor], idx: torch.Tensor, obs_history: int) -> torch.Tensor:
    """DiT/ID conditioning: current latent, optionally concatenated with the previous step's latent."""
    s = T["latent"][idx]
    if obs_history == 1:
        return s
    return torch.cat([s, T["latent"][T["prev"][idx]]], dim=-1)
