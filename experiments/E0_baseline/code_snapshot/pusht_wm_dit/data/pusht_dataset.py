"""Push-T Dataset Implementation with Sliding Window Trajectory Extraction.

Extracts:
1. Current observation frame (3x96x96) and normalized agent position (x_agent, y_agent).
2. Continuous 4D pose representation: [x_T, y_T, sin(theta_T), cos(theta_T)] in [-1, 1]^4.
3. Ground-truth future trajectory over horizon H=16 (padded to avoid boundary truncation).
4. Future low-level action sequence over Ta=8.
5. Transition pair (t, t+1) for World Model dynamics modeling.
"""

from typing import Dict, Tuple, Optional
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pusht_wm_dit.utils.normalizer import (
    normalize_pos,
    pose_to_4d,
    normalize_image,
)


class PushTDataset(Dataset):
    """PyTorch Dataset for Push-T Zarr replay buffers."""

    def __init__(
        self,
        zarr_path: str,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        augment_symmetry: bool = False,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.split = split
        self.augment_symmetry = augment_symmetry and (split == "train")

        # Open Zarr group and preload into RAM for 600x high-throughput indexing
        self.root = zarr.open(zarr_path, mode="r")
        raw_imgs = self.root["data/img"][:]
        if raw_imgs.dtype != np.uint8:
            self.img_arr = np.clip(raw_imgs, 0, 255).astype(np.uint8)
        else:
            self.img_arr = raw_imgs
        self.state_arr = np.asarray(self.root["data/state"][:], dtype=np.float32)
        self.action_arr = np.asarray(self.root["data/action"][:], dtype=np.float32)
        self.episode_ends = self.root["meta/episode_ends"][:]

        num_episodes = len(self.episode_ends)
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(num_episodes)
        num_val = int(num_episodes * val_ratio)

        if split == "val":
            selected_episodes = set(shuffled_indices[:num_val])
        else:
            selected_episodes = set(shuffled_indices[num_val:])

        # Build episode intervals and valid timesteps
        self.indices = []
        self.episode_ranges = []

        start_idx = 0
        for ep_idx, end_idx in enumerate(self.episode_ends):
            ep_len = end_idx - start_idx
            if ep_idx in selected_episodes and ep_len > 0:
                self.episode_ranges.append((start_idx, end_idx))
                for t in range(start_idx, end_idx):
                    self.indices.append((t, start_idx, end_idx))
            start_idx = end_idx

        print(
            f"PushTDataset [{split}]: {len(self.indices)} timesteps from "
            f"{len(self.episode_ranges)} episodes (pred_horizon={pred_horizon}, "
            f"action_horizon={action_horizon})"
        )

    def __len__(self) -> int:
        return len(self.indices) * (2 if self.augment_symmetry else 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        is_augmented = False
        if self.augment_symmetry and idx >= len(self.indices):
            idx = idx - len(self.indices)
            is_augmented = True

        t, ep_start, ep_end = self.indices[idx]

        # 1. Current frame observation: (96, 96, 3) -> (3, 96, 96) in [0, 1]
        raw_img = self.img_arr[t]
        frame = normalize_image(raw_img)
        frame_tensor = torch.from_numpy(frame).float()

        # 2. Current state: state[0:2]=agent_pos, state[2:4]=T_pos, state[4]=theta
        raw_state = self.state_arr[t]
        agent_pos_px = raw_state[0:2]
        agent_pos_norm = normalize_pos(agent_pos_px)
        agent_pos_tensor = torch.from_numpy(agent_pos_norm).float()

        bx, by, btheta = raw_state[2], raw_state[3], raw_state[4]
        block_pose_4d = pose_to_4d(bx, by, btheta)
        block_pose_4d_tensor = torch.from_numpy(block_pose_4d).float()

        # 3. Future T-block trajectory over horizon H=16: tau_{t:t+H} in [-1, 1]^4
        path_future_list = []
        for h in range(self.pred_horizon):
            step_idx = min(t + h, ep_end - 1)
            st = self.state_arr[step_idx]
            p4 = pose_to_4d(st[2], st[3], st[4])
            path_future_list.append(p4)
        path_future = np.stack(path_future_list, axis=0)  # (H, 4)
        path_future_tensor = torch.from_numpy(path_future).float()

        # 4. Action sequence over action_horizon Ta=8
        action_future_list = []
        for h in range(self.action_horizon):
            step_idx = min(t + h, ep_end - 1)
            act = normalize_pos(self.action_arr[step_idx])
            action_future_list.append(act)
        action_future = np.stack(action_future_list, axis=0)  # (Ta, 2)
        action_future_tensor = torch.from_numpy(action_future).float()

        # Immediate next action for step transition
        action_next = normalize_pos(self.action_arr[t])
        action_next_tensor = torch.from_numpy(action_next).float()

        # 5. Transition target at t+1 for World Model dynamics
        next_t = min(t + 1, ep_end - 1)
        raw_next_img = self.img_arr[next_t]
        next_frame = normalize_image(raw_next_img)
        next_frame_tensor = torch.from_numpy(next_frame).float()

        raw_next_state = self.state_arr[next_t]
        next_agent_pos_norm = normalize_pos(raw_next_state[0:2])
        next_agent_pos_tensor = torch.from_numpy(next_agent_pos_norm).float()

        next_block_pose_4d = pose_to_4d(raw_next_state[2], raw_next_state[3], raw_next_state[4])
        next_block_pose_4d_tensor = torch.from_numpy(next_block_pose_4d).float()

        if is_augmented:
            # Planar diagonal reflection (x <-> y, theta <-> pi/2 - theta)
            frame_tensor = frame_tensor.transpose(1, 2)
            agent_pos_tensor = agent_pos_tensor[[1, 0]]
            block_pose_4d_tensor = block_pose_4d_tensor[[1, 0, 3, 2]]
            path_future_tensor = path_future_tensor[:, [1, 0, 3, 2]]
            action_future_tensor = action_future_tensor[:, [1, 0]]
            action_next_tensor = action_next_tensor[[1, 0]]
            next_frame_tensor = next_frame_tensor.transpose(1, 2)
            next_agent_pos_tensor = next_agent_pos_tensor[[1, 0]]
            next_block_pose_4d_tensor = next_block_pose_4d_tensor[[1, 0, 3, 2]]

        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "frame": frame_tensor,                          # (3, 96, 96) in [0, 1]
            "agent_pos": agent_pos_tensor,                  # (2,) in [-1, 1]
            "block_pose_4d": block_pose_4d_tensor,          # (4,) in [-1, 1]
            "path_future": path_future_tensor,              # (H, 4) in [-1, 1]
            "action_future": action_future_tensor,          # (Ta, 2) in [-1, 1]
            "action": action_next_tensor,                   # (2,) in [-1, 1]
            "next_frame": next_frame_tensor,                # (3, 96, 96) in [0, 1]
            "next_agent_pos": next_agent_pos_tensor,        # (2,) in [-1, 1]
            "next_block_pose_4d": next_block_pose_4d_tensor, # (4,) in [-1, 1]
        }


def create_dataloaders(
    zarr_path: str,
    pred_horizon: int = 16,
    action_horizon: int = 8,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation DataLoaders."""
    train_dataset = PushTDataset(
        zarr_path=zarr_path,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    val_dataset = PushTDataset(
        zarr_path=zarr_path,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
