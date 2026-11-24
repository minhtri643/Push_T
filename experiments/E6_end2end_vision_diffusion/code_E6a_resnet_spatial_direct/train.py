"""Training script for Stage E6a:
End-to-End ResNet-18 + Spatial-Softmax + DiT Action Diffusion Policy.

Learns direct continuous actions from video frames without Inverse Dynamics.
Uses T_obs = 2 consecutive frames and anti-diagonal symmetry augmentation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import zarr

from experiments.E6_end2end_vision_diffusion.code_E6a_resnet_spatial_direct.model import ResNetSpatialDirectPolicy


def norm_xy(p: np.ndarray) -> np.ndarray:
    """[0, 512] -> [-1, 1]."""
    return (p - 256.0) / 256.0


def reflect_xy(p: np.ndarray) -> np.ndarray:
    """Anti-diagonal reflection of coordinates (..., 2): (x, y) -> (-y, -x)."""
    return -p[..., [1, 0]]


def reflect_img_np(img: np.ndarray) -> np.ndarray:
    """Anti-diagonal reflection for (H, W, C) or (..., H, W, C):
    pixel (x=col, y=row) -> (W-1-row, H-1-col).
    In numpy: transpose H and W, then flip along both H and W axes.
    """
    return np.flip(np.swapaxes(img, -3, -2), axis=(-3, -2))


def episode_split(n_episodes: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_episodes).tolist()
    n_val = max(1, int(n_episodes * val_ratio))
    val_ids = sorted(perm[:n_val])
    train_ids = sorted(perm[n_val:])
    return train_ids, val_ids


class End2EndPushTDataset(Dataset):
    """Dataset serving 2-frame RGB clips, agent pos, and future 16 actions."""

    def __init__(
        self,
        imgs: np.ndarray,
        agents: np.ndarray,
        actions: np.ndarray,
        windows: List[Tuple[int, int, int, bool]],
        mean: np.ndarray,
        std: np.ndarray,
    ):
        """windows: List of tuples (t_prev, t_curr, ep_end, is_reflected)"""
        self.imgs = imgs
        self.agents = agents
        self.actions = actions
        self.windows = windows
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_prev, t_curr, ep_end, is_refl = self.windows[idx]

        # 1. Image frames (2, 96, 96, 3) uint8 -> (2, 3, 96, 96) float32 normalized
        f_prev = self.imgs[t_prev]
        f_curr = self.imgs[t_curr]
        if is_refl:
            f_prev = reflect_img_np(f_prev)
            f_curr = reflect_img_np(f_curr)

        clip = np.stack([f_prev, f_curr], axis=0).astype(np.float32) / 255.0  # (2, 96, 96, 3)
        clip = (clip - self.mean) / self.std
        clip = np.transpose(clip, (0, 3, 1, 2))  # (2, 3, 96, 96)

        # 2. Agent pos (2, 2) in [-1, 1]
        p_prev = self.agents[t_prev]
        p_curr = self.agents[t_curr]
        if is_refl:
            p_prev = reflect_xy(p_prev)
            p_curr = reflect_xy(p_curr)
        agent_pos = np.stack([p_prev, p_curr], axis=0).astype(np.float32)

        # 3. Future actions (16, 2) in [-1, 1]
        H = 16
        act_slice = self.actions[t_curr:min(t_curr + H, ep_end)]
        if len(act_slice) < H:
            pad = np.repeat(act_slice[-1:], H - len(act_slice), axis=0)
            act_slice = np.concatenate([act_slice, pad], axis=0)
        if is_refl:
            act_slice = reflect_xy(act_slice)
        actions_future = act_slice.astype(np.float32)

        return (
            torch.from_numpy(clip),
            torch.from_numpy(agent_pos),
            torch.from_numpy(actions_future),
        )


def build_windows(ends: np.ndarray, ep_ids: List[int], include_reflections: bool = True) -> List[Tuple[int, int, int, bool]]:
    wins = []
    ep_set = set(ep_ids)
    start = 0
    for ep, end in enumerate(ends):
        end_int = int(end)
        if ep in ep_set:
            for t in range(start, end_int):
                t_prev = max(start, t - 1)
                wins.append((t_prev, t, end_int, False))
                if include_reflections:
                    wins.append((t_prev, t, end_int, True))
        start = end_int
    return wins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=80, help="number of training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="batch size")
    p.add_argument("--lr-backbone", type=float, default=1e-4, help="learning rate for ResNet-18")
    p.add_argument("--lr-dit", type=float, default=2e-4, help="learning rate for DiT planner")
    p.add_argument("--num-kp", type=int, default=16, help="number of spatial keypoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="run a quick 1-epoch test")
    args = p.parse_args()

    out_dir = "results/vision_world_models/E6a_resnet_spatial_direct"
    ckpt_dir = "checkpoints/E6_vision"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "resnet_spatial_direct.pt")
    log_file = os.path.join(out_dir, "training_log.txt")

    print(f"=== Stage E6a: ResNet-18 + Spatial-Softmax + DiT Action Diffusion ===")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size}")

    # 1. Load data into RAM
    t_load0 = time.perf_counter()
    zarr_path = "data/pusht_hf.zarr"
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Push-T dataset not found at {zarr_path}.")

    root = zarr.open(zarr_path, mode="r")
    imgs = np.asarray(root["data/img"][:])  # (25650, 96, 96, 3) uint8 (~700 MB)
    st = np.asarray(root["data/state"][:], dtype=np.float32)
    ac = np.asarray(root["data/action"][:], dtype=np.float32)
    ends = root["meta/episode_ends"][:]
    print(f"Loaded {len(imgs):,} frames into RAM in {time.perf_counter() - t_load0:.2f}s.")

    agents_norm = norm_xy(st[:, :2])
    actions_norm = norm_xy(ac)

    train_ids, val_ids = episode_split(len(ends), val_ratio=0.1, seed=42)
    train_wins = build_windows(ends, train_ids, include_reflections=True)
    val_wins = build_windows(ends, val_ids, include_reflections=False)

    if args.smoke:
        train_wins = train_wins[:500]
        val_wins = val_wins[:200]
        args.epochs = 1

    print(f"Train windows: {len(train_wins):,} | Val windows: {len(val_wins):,}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    train_ds = End2EndPushTDataset(imgs, agents_norm, actions_norm, train_wins, mean, std)
    val_ds = End2EndPushTDataset(imgs, agents_norm, actions_norm, val_wins, mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=min(256, len(val_ds)), shuffle=False, num_workers=0)

    # 2. Build model
    model = ResNetSpatialDirectPolicy(
        num_keypoints=args.num_kp,
        horizon=16,
        model_dim=256,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4.0,
        diffusion_steps=100,
        beta_schedule="squaredcos_cap_v2",
    ).to(args.device)

    # Differential learning rates for vision backbone and DiT
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.spatial.parameters(), "lr": args.lr_backbone},
        {"params": model.dit.parameters(), "lr": args.lr_dit},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    best_val_rmse = float("inf")
    history = []
    t_start = time.perf_counter()

    with open(log_file, "w", encoding="utf-8") as log_f:
        header = f"Stage E6a Training Log | ResNet-18 + Spatial-Softmax (K={args.num_kp}) | Direct Actions\n"
        print(header, end="")
        log_f.write(header)

        for ep in range(1, args.epochs + 1):
            ep_t0 = time.perf_counter()
            model.train()
            train_loss = 0.0

            for clips, agents, actions in train_loader:
                clips = clips.to(args.device)
                agents = agents.to(args.device)
                actions = actions.to(args.device)

                opt.zero_grad(set_to_none=True)
                out = model(clips, agents, actions)
                loss = out["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))

            # Validation evaluation: Sample actions and compute Action RMSE in pixels
            model.eval()
            val_mse = 0.0
            val_batches = 0
            with torch.no_grad():
                for clips, agents, actions in val_loader:
                    clips = clips.to(args.device)
                    agents = agents.to(args.device)
                    actions = actions.to(args.device)

                    pred_actions = model.sample_actions(clips, agents, num_inference_steps=10)
                    val_mse += F.mse_loss(pred_actions[:, :8], actions[:, :8]).item()
                    val_batches += 1
                    if (args.smoke and val_batches >= 2) or (val_batches >= 2 and ep < args.epochs):
                        break

            val_rmse_px = math.sqrt(val_mse / max(1, val_batches)) * 256.0
            is_best = val_rmse_px < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse_px
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_rmse_px": val_rmse_px,
                    "num_kp": args.num_kp,
                    "horizon": 16,
                }, ckpt_path)

            ep_elapsed = time.perf_counter() - ep_t0
            star = " *" if is_best else "  "
            log_line = f"Epoch {ep:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Action RMSE: {val_rmse_px:5.2f} px{star} | Elapsed: {ep_elapsed:2.0f}s\n"
            print(log_line, end="")
            log_f.write(log_line)
            log_f.flush()

            history.append({
                "epoch": ep,
                "train_loss": train_loss,
                "val_rmse_px": val_rmse_px,
                "best_val_rmse_px": best_val_rmse,
            })

    total_time = time.perf_counter() - t_start
    print(f"\n[DONE] Training complete in {total_time:.1f}s. Best Val Action RMSE: {best_val_rmse:.2f} px.")
    print(f"Checkpoint saved to {ckpt_path}.")

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_rmse": best_val_rmse, "total_seconds": total_time}, f, indent=2)


if __name__ == "__main__":
    main()
