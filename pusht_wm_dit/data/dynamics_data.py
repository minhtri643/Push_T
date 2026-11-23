"""Windows of (state, K actions, K future states) for training/evaluating the dynamics model.

Uses the same episode split as PushTDataset (same seed and val_ratio), so validation episodes
never leak into training. Works on raw states, no images, so it is fast.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

HALF = 256.0


def episode_split(n_episodes: int, val_ratio: float, seed: int):
    """Identical to PushTDataset: returns (train_ids, val_ids)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_episodes)
    n_val = int(n_episodes * val_ratio)
    return sorted(set(perm[n_val:].tolist())), sorted(set(perm[:n_val].tolist()))


def load_demo_sequences(zarr_path: str, split: str, val_ratio: float = 0.1, seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """Episodes as dicts of pixel-space arrays: agent (T,2), block (T,3), action (T,2)."""
    root = zarr.open(zarr_path, mode="r")
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    action = np.asarray(root["data/action"][:], dtype=np.float64)
    ends = root["meta/episode_ends"][:]
    train_ids, val_ids = episode_split(len(ends), val_ratio, seed)
    keep = set(train_ids if split == "train" else val_ids)
    seqs, start = [], 0
    for ep, end in enumerate(ends):
        if ep in keep and end > start:
            seqs.append({"agent": state[start:end, :2], "block": state[start:end, 2:5], "action": action[start:end], "episode": ep})
        start = end
    return seqs


def load_sim_sequences(npz_path: str) -> List[Dict[str, np.ndarray]]:
    data = np.load(npz_path)
    lens, seqs, start = data["lengths"], [], 0
    for n in lens:
        seqs.append({"agent": data["agent"][start:start + n], "block": data["block"][start:start + n],
                     "action": data["action"][start:start + n], "episode": -1})
        start += n
    return seqs


def _reflect(seq: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Anti-diagonal reflection through the goal centre (same symmetry as the image augmentation)."""
    def refl(p):
        return np.stack([2 * HALF - p[:, 1], 2 * HALF - p[:, 0]], axis=1)
    block = np.concatenate([refl(seq["block"][:, :2]), (np.pi / 2 - seq["block"][:, 2:3])], axis=1)
    return {"agent": refl(seq["agent"]), "block": block, "action": refl(seq["action"]), "episode": seq["episode"]}


def make_windows(seqs: List[Dict[str, np.ndarray]], horizon: int = 16, augment: bool = False) -> Dict[str, torch.Tensor]:
    """One window per time step t: inputs at t, actions t..t+K-1, targets t+1..t+K (masked past the end)."""
    if augment:
        seqs = list(seqs) + [_reflect(s) for s in seqs]
    out = {k: [] for k in ("block0", "agent0", "vel0", "actions", "tgt_block", "tgt_agent", "mask", "moved")}
    K = horizon
    for s in seqs:
        T = len(s["agent"])
        if T < 2:
            continue
        ag = (s["agent"] - HALF) / HALF
        bl = np.concatenate([(s["block"][:, :2] - HALF) / HALF, np.unwrap(s["block"][:, 2:3], axis=0)], axis=1)
        ac = (s["action"] - HALF) / HALF
        vel = np.concatenate([np.zeros((1, 2)), np.diff(ag, axis=0)], axis=0)
        t_idx = np.arange(T)
        fut = np.minimum(t_idx[:, None] + 1 + np.arange(K)[None], T - 1)          # (T, K) target indices
        act_idx = np.minimum(t_idx[:, None] + np.arange(K)[None], T - 1)          # (T, K) action indices
        mask = (t_idx[:, None] + 1 + np.arange(K)[None]) <= (T - 1)
        out["block0"].append(bl)
        out["agent0"].append(ag)
        out["vel0"].append(vel)
        out["actions"].append(ac[act_idx])
        out["tgt_block"].append(bl[fut])
        out["tgt_agent"].append(ag[fut])
        out["mask"].append(mask)
        move = np.linalg.norm(bl[fut[:, -1], :2] - bl[:, :2], axis=1) * HALF
        rot = np.abs(np.rad2deg(bl[fut[:, -1], 2] - bl[:, 2]))
        out["moved"].append((move > 2.0) | (rot > 1.0))
    return {k: torch.from_numpy(np.concatenate(v).astype(np.float32 if k not in ("mask", "moved") else bool)) for k, v in out.items()}
