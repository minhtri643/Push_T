"""Latent State Precomputation and Caching for Push-T World Model.

Caches frozen WM latent states (s_t) to disk to:
1. Drastically reduce VRAM requirements during Diffusion Transformer (DiT) training.
2. Eliminate repetitive ResNet-18 forward passes for each diffusion training step.
3. Stabilize training representations with guaranteed zero drift.
"""

from typing import Optional, Dict, Tuple
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pusht_wm_dit.data.pusht_dataset import PushTDataset

# Bump whenever the meaning of cached latents changes (e.g. augmentation transform).
# v2: anti-diagonal reflection augmentation (v1 caches used a wrong main-diagonal flip).
LATENT_CACHE_VERSION = 2


def check_latent_cache(cache_path: str, dataset: PushTDataset) -> Tuple[bool, str]:
    """Checks whether a latent cache file matches the given dataset.

    Returns:
        (is_valid, reason) where reason explains why the cache is stale.
    """
    if not os.path.exists(cache_path):
        return False, f"cache not found: {cache_path}"
    meta = torch.load(cache_path, map_location="cpu", weights_only=False)
    if meta.get("cache_version") != LATENT_CACHE_VERSION:
        return False, f"cache_version {meta.get('cache_version')} != {LATENT_CACHE_VERSION} (stale augmentation)"
    if meta.get("augment_symmetry") != dataset.augment_symmetry:
        return False, f"augment_symmetry {meta.get('augment_symmetry')} != dataset {dataset.augment_symmetry}"
    if meta.get("dataset_len") != len(dataset) or len(meta["latents"]) != len(dataset):
        return False, f"length {len(meta['latents'])} != dataset length {len(dataset)}"
    src = os.path.normpath(getattr(dataset, "zarr_path", ""))
    if os.path.normpath(meta.get("zarr_path", "")) != src:
        return False, f"cache built from {meta.get('zarr_path')!r}, dataset is {src!r}"
    return True, "ok"


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

        is_valid, reason = check_latent_cache(latents_cache_path, base_dataset)
        if not is_valid:
            raise RuntimeError(
                f"Latent cache at {latents_cache_path} does not match the dataset ({reason}). "
                f"Re-run scripts/3_world_model/train_encoder.py, or delete the file and let scripts/4_diffusion/train_dit_planner.py regenerate it."
            )

        print(f"Loading precomputed latents from {latents_cache_path}...")
        cache_data = torch.load(latents_cache_path, map_location=device, weights_only=False)
        self.latents = cache_data["latents"]  # Tensor of shape (N, latent_dim)
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
            "augment_symmetry": bool(getattr(dataset, "augment_symmetry", False)),
            "cache_version": LATENT_CACHE_VERSION,
            "zarr_path": os.path.normpath(getattr(dataset, "zarr_path", "")),
        },
        cache_path,
    )
    print(f"Cached {len(concatenated_latents)} latents to {cache_path} (shape: {concatenated_latents.shape})")
    return cache_path
