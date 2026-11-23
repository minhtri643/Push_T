"""Latent State Precomputation and Caching for Push-T World Model.

Caches frozen WM latent states (s_t) to disk to:
1. Drastically reduce VRAM requirements during Diffusion Transformer (DiT) training.
2. Eliminate repetitive ResNet-18 forward passes for each diffusion training step.
3. Stabilize training representations with guaranteed zero drift.
"""

from typing import Optional, Dict
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pusht_wm_dit.data.pusht_dataset import PushTDataset


class CachedLatentDataset(Dataset):
    """Wraps PushTDataset to serve precomputed latent state vectors s_t."""

    def __init__(
        self,
        base_dataset: PushTDataset,
        latents_cache_path: str,
        device: str = "cpu",
    ):
        self.base_dataset = base_dataset
        self.latents_cache_path = latents_cache_path
        self.device = device

        if not os.path.exists(latents_cache_path):
            raise FileNotFoundError(
                f"Latent cache not found at {latents_cache_path}. "
                f"Please run scripts/03_train_wm.py or data/precompute_latents.py first."
            )

        print(f"Loading precomputed latents from {latents_cache_path}...")
        cache_data = torch.load(latents_cache_path, map_location=device, weights_only=False)
        self.latents = cache_data["latents"]  # Tensor of shape (N, latent_dim)

        assert len(self.latents) == len(self.base_dataset), (
            f"Latents count ({len(self.latents)}) does not match dataset length ({len(self.base_dataset)})"
        )
        print(f"Successfully loaded {len(self.latents)} latent vectors (dim={self.latents.shape[-1]})")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        latent_st = self.latents[idx]

        # Add latent state vector s_t
        sample["latent_st"] = latent_st
        return sample


def precompute_and_cache_latents(
    world_model: torch.nn.Module,
    dataset: PushTDataset,
    cache_path: str,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
) -> str:
    """Passes all dataset frames through frozen World Model encoder and caches latents to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    world_model.eval()
    world_model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    all_latents = []
    print(f"Precomputing latents for {len(dataset)} samples using frozen World Model...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding frames to latents"):
            frames = batch["frame"].to(device)
            agent_pos = batch["agent_pos"].to(device)

            # Encode frame + agent position to compact latent state s_t
            s_t = world_model.encode(frames, agent_pos)
            all_latents.append(s_t.cpu())

    concatenated_latents = torch.cat(all_latents, dim=0)
    assert len(concatenated_latents) == len(dataset)

    torch.save(
        {
            "latents": concatenated_latents,
            "latent_dim": concatenated_latents.shape[-1],
            "dataset_len": len(dataset),
        },
        cache_path,
    )
    print(f"Cached {len(concatenated_latents)} latents to {cache_path} (shape: {concatenated_latents.shape})")
    return cache_path
