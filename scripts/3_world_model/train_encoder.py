"""Pipeline 3 (World Model, part 1 = image encoder): Train World Model (ResNet-18 GroupNorm + Latent Dynamics) & Freeze Weights.

Workflow:
1. Loads PushTDataset training and validation splits.
2. Instantiates ResNet-18 (GroupNorm) World Model with Latent Transition Predictor.
3. Optimizes pose reconstruction, agent state grounding, and latent forward transition consistency.
4. Tracks validation loss and saves best checkpoint to 'checkpoints/final/world_model_best.pt'.
5. Freezes World Model weights.
6. Calls precompute_latents to cache latent states s_t to disk for high-speed DiT training.
"""

import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

from typing import Optional, Dict
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pusht_wm_dit.data.pusht_dataset import PushTDataset, create_dataloaders
from pusht_wm_dit.data.precompute_latents import precompute_and_cache_latents
from pusht_wm_dit.models.world_model import WorldModel


def train_one_epoch(
    model: WorldModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    model.train()
    total_loss_sum = 0.0
    loss_pose_sum = 0.0
    loss_agent_sum = 0.0
    loss_trans_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Training WM", leave=False):
        frames = batch["frame"].to(device, non_blocking=True)
        agent_pos = batch["agent_pos"].to(device, non_blocking=True)
        block_pose_4d = batch["block_pose_4d"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        next_frames = batch["next_frame"].to(device, non_blocking=True)
        next_agent_pos = batch["next_agent_pos"].to(device, non_blocking=True)
        next_block_pose_4d = batch["next_block_pose_4d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device == "cuda":
            with torch.amp.autocast("cuda"):
                loss_dict = model.compute_loss(
                    frame=frames,
                    agent_pos=agent_pos,
                    block_pose_4d=block_pose_4d,
                    action=actions,
                    next_frame=next_frames,
                    next_agent_pos=next_agent_pos,
                    next_block_pose_4d=next_block_pose_4d,
                )
                loss = loss_dict["total_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model.compute_loss(
                frame=frames,
                agent_pos=agent_pos,
                block_pose_4d=block_pose_4d,
                action=actions,
                next_frame=next_frames,
                next_agent_pos=next_agent_pos,
                next_block_pose_4d=next_block_pose_4d,
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

        total_loss_sum += loss.item()
        loss_pose_sum += loss_dict["loss_pose"].item()
        loss_agent_sum += loss_dict["loss_agent"].item()
        loss_trans_sum += loss_dict["loss_transition"].item()
        n_batches += 1

    return {
        "total_loss": total_loss_sum / n_batches,
        "loss_pose": loss_pose_sum / n_batches,
        "loss_agent": loss_agent_sum / n_batches,
        "loss_trans": loss_trans_sum / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: WorldModel,
    loader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    total_loss_sum = 0.0
    loss_pose_sum = 0.0
    loss_agent_sum = 0.0
    loss_trans_sum = 0.0
    n_batches = 0

    for batch in loader:
        frames = batch["frame"].to(device, non_blocking=True)
        agent_pos = batch["agent_pos"].to(device, non_blocking=True)
        block_pose_4d = batch["block_pose_4d"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        next_frames = batch["next_frame"].to(device, non_blocking=True)
        next_agent_pos = batch["next_agent_pos"].to(device, non_blocking=True)
        next_block_pose_4d = batch["next_block_pose_4d"].to(device, non_blocking=True)

        loss_dict = model.compute_loss(
            frame=frames,
            agent_pos=agent_pos,
            block_pose_4d=block_pose_4d,
            action=actions,
            next_frame=next_frames,
            next_agent_pos=next_agent_pos,
            next_block_pose_4d=next_block_pose_4d,
        )
        total_loss_sum += loss_dict["total_loss"].item()
        loss_pose_sum += loss_dict["loss_pose"].item()
        loss_agent_sum += loss_dict["loss_agent"].item()
        loss_trans_sum += loss_dict["loss_transition"].item()
        n_batches += 1

    return {
        "val_total_loss": total_loss_sum / n_batches,
        "val_loss_pose": loss_pose_sum / n_batches,
        "val_loss_agent": loss_agent_sum / n_batches,
        "val_loss_trans": loss_trans_sum / n_batches,
    }


def main():
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    wm_config = config["world_model"]
    data_config = config["data"]

    augment = bool(data_config.get("augment_symmetry", False))

    print("=" * 70)
    print(f"STEP 3: TRAINING WORLD MODEL (DEVICE: {device.upper()}, augment_symmetry={augment})")
    print("=" * 70)

    # 1. Dataloaders (augmentation on the train split only)
    train_loader, val_loader = create_dataloaders(
        zarr_path=data_config["zarr_path"],
        pred_horizon=data_config["pred_horizon"],
        action_horizon=data_config["action_horizon"],
        batch_size=wm_config["batch_size"],
        val_ratio=data_config["val_ratio"],
        num_workers=data_config["num_workers"],
        seed=config["seed"],
        augment_symmetry=augment,
    )

    # 2. Instantiate Model
    model = WorldModel(
        latent_dim=wm_config["latent_dim"],
        agent_pos_dim=wm_config["agent_pos_dim"],
        pose_dim=wm_config["pose_dim"],
        action_dim=wm_config["action_dim"],
        num_groups=wm_config["num_groups"],
        transition_hidden_dim=wm_config["transition_hidden_dim"],
        lambda_recon=wm_config["lambda_recon"],
        lambda_agent=wm_config["lambda_agent"],
        lambda_transition=wm_config["lambda_transition"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=wm_config["lr"],
        weight_decay=wm_config["weight_decay"],
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=wm_config["epochs"]
    )
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    save_path = wm_config["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    epochs = wm_config["epochs"]

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, device)
        lr_scheduler.step()

        val_loss = val_metrics["val_total_loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": wm_config,
                },
                save_path,
            )

        star = " ★ (Best Saved)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} (Pose: {train_metrics['loss_pose']:.4f}, "
            f"Ag: {train_metrics['loss_agent']:.4f}, Trans: {train_metrics['loss_trans']:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Pose: {val_metrics['val_loss_pose']:.4f}){star}"
        )

    print("\n" + "=" * 70)
    print("FREEZING WORLD MODEL & PRECOMPUTING LATENTS")
    print("=" * 70)

    # Load best checkpoint
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze()

    # Precompute latents for the training dataset to save VRAM and accelerate Step 4.
    # Must use the same augment flag as train_dit_planner.py (checked via cache metadata).
    full_dataset = PushTDataset(
        zarr_path=data_config["zarr_path"],
        pred_horizon=data_config["pred_horizon"],
        action_horizon=data_config["action_horizon"],
        split="train",
        val_ratio=data_config["val_ratio"],
        seed=config["seed"],
        augment_symmetry=augment,
    )
    latents_cache_path = data_config["latents_cache_path"]
    precompute_and_cache_latents(
        world_model=model,
        dataset=full_dataset,
        cache_path=latents_cache_path,
        device=device,
        batch_size=128,
    )

    print(f"World Model training and latent caching completed!")
    print(f"  Best Model Checkpoint: {save_path}")
    print(f"  Latents Cache Path:    {latents_cache_path}")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Optional
    main()
