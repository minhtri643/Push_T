"""Script 04: Self-Supervised Training of DiT Trajectory Planner + Inverse Dynamics.

Workflow:
1. Loads cached World Model latents (s_t) using CachedLatentDataset (saving VRAM and computation).
2. Optimizes 1D Diffusion Transformer with AdaLN-Zero conditioning:
   L_total = ||path_future - hat_path||_2^2 + lambda_fde * FDE^2
   where FDE = sqrt(w_xy * (dx_H^2 + dy_H^2) + w_theta * dtheta_H^2).
3. Optimizes Inverse Dynamics Controller: maps (s_t, hat_path) -> action commands [x_cmd, y_cmd].
4. Tracks loss metrics and saves best checkpoints to 'checkpoints/'.
"""

import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from typing import Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pusht_wm_dit.data.pusht_dataset import PushTDataset
from pusht_wm_dit.data.precompute_latents import CachedLatentDataset
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.models.world_model import WorldModel


def train_epoch(
    dit_model: DiTTrajectoryPlanner,
    id_model: InverseDynamicsController,
    loader: DataLoader,
    optimizer_dit: optim.Optimizer,
    optimizer_id: optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    dit_model.train()
    id_model.train()

    loss_total_sum = 0.0
    loss_mse_sum = 0.0
    loss_fde_sum = 0.0
    fde_eval_sum = 0.0
    loss_id_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Training DiT + ID", leave=False):
        # Latent state vector s_t from frozen WM (cached)
        s_t = batch["latent_st"].to(device, non_blocking=True)
        path_future = batch["path_future"].to(device, non_blocking=True)
        action_future = batch["action_future"].to(device, non_blocking=True)
        agent_pos = batch["agent_pos"].to(device, non_blocking=True)

        # Update DiT Planner
        optimizer_dit.zero_grad(set_to_none=True)
        dit_out = dit_model(path_future=path_future, s_t=s_t)
        loss_dit = dit_out["total_loss"]
        loss_dit.backward()
        torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
        optimizer_dit.step()

        # Update Inverse Dynamics Controller (with explicit agent_pos feedback and trajectory perturbation)
        optimizer_id.zero_grad(set_to_none=True)
        # Robustness augmentation: inject small realistic variation into input trajectory
        traj_perturbed = torch.clamp(path_future + torch.randn_like(path_future) * 0.02, -1.0, 1.0)
        id_out = id_model.compute_loss(
            s_t=s_t,
            predicted_path=traj_perturbed,
            target_actions=action_future,
            agent_pos=agent_pos,
        )
        loss_id = id_out["loss"]
        loss_id.backward()
        torch.nn.utils.clip_grad_norm_(id_model.parameters(), max_norm=1.0)
        optimizer_id.step()

        loss_total_sum += loss_dit.item()
        loss_mse_sum += dit_out["loss_mse"].item()
        loss_fde_sum += dit_out["loss_fde"].item()
        fde_eval_sum += dit_out["fde_eval"].item()
        loss_id_sum += loss_id.item()
        n_batches += 1

    return {
        "dit_total_loss": loss_total_sum / n_batches,
        "loss_mse": loss_mse_sum / n_batches,
        "loss_fde": loss_fde_sum / n_batches,
        "fde_eval": fde_eval_sum / n_batches,
        "loss_id": loss_id_sum / n_batches,
    }


def main():
    config_path = "pusht_wm_dit/configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    data_config = config["data"]
    dit_config = config["dit_planner"]
    id_config = config["inverse_dynamics"]
    wm_config = config["world_model"]

    print("=" * 70)
    print(f"STEP 4: TRAINING DiT TRAJECTORY PLANNER & INVERSE DYNAMICS (DEVICE: {device.upper()})")
    print("=" * 70)

    # 1. Prepare Base Dataset with SE(2) Diagonal Symmetry Augmentation
    base_train_ds = PushTDataset(
        zarr_path=data_config["zarr_path"],
        pred_horizon=data_config["pred_horizon"],
        action_horizon=data_config["action_horizon"],
        split="train",
        val_ratio=data_config["val_ratio"],
        seed=config["seed"],
        augment_symmetry=True,
    )

    # 2. Load Cached Latents or generate if missing
    latents_cache_path = data_config["latents_cache_path"]
    if not os.path.exists(latents_cache_path):
        print(f"Latents cache not found. Generating now with World Model...")
        wm = WorldModel(
            latent_dim=wm_config["latent_dim"],
            agent_pos_dim=wm_config["agent_pos_dim"],
            pose_dim=wm_config["pose_dim"],
            action_dim=wm_config["action_dim"],
            num_groups=wm_config["num_groups"],
        ).to(device)
        if os.path.exists(wm_config["save_path"]):
            ckpt = torch.load(wm_config["save_path"], map_location=device, weights_only=False)
            wm.load_state_dict(ckpt["model_state_dict"])
        wm.freeze()
        from pusht_wm_dit.data.precompute_latents import precompute_and_cache_latents
        precompute_and_cache_latents(wm, base_train_ds, latents_cache_path, device=device)

    cached_train_ds = CachedLatentDataset(
        base_dataset=base_train_ds,
        latents_cache_path=latents_cache_path,
        device="cpu",
    )
    loader = DataLoader(
        cached_train_ds,
        batch_size=dit_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=(device == "cuda"),
    )

    # 3. Instantiate DiT Planner & Inverse Dynamics Controller
    dit_model = DiTTrajectoryPlanner(
        horizon=dit_config["horizon"],
        traj_dim=dit_config["traj_dim"],
        latent_dim=wm_config["latent_dim"],
        model_dim=dit_config["model_dim"],
        num_heads=dit_config["num_heads"],
        num_layers=dit_config["num_layers"],
        mlp_ratio=dit_config["mlp_ratio"],
        diffusion_steps=dit_config["diffusion_steps"],
        beta_schedule=dit_config["beta_schedule"],
        w_xy=dit_config["w_xy"],
        w_theta=dit_config["w_theta"],
        lambda_fde=dit_config["lambda_fde"],
    ).to(device)

    id_model = InverseDynamicsController(
        latent_dim=wm_config["latent_dim"],
        horizon=dit_config["horizon"],
        traj_dim=dit_config["traj_dim"],
        action_dim=id_config["action_dim"],
        exec_steps=id_config["exec_steps"],
        hidden_dims=tuple(id_config["hidden_dims"]),
    ).to(device)

    optimizer_dit = optim.AdamW(
        dit_model.parameters(),
        lr=dit_config["lr"],
        weight_decay=dit_config["weight_decay"],
    )
    optimizer_id = optim.AdamW(
        id_model.parameters(),
        lr=id_config["lr"],
        weight_decay=id_config["weight_decay"],
    )

    epochs = dit_config["epochs"]
    lr_scheduler_dit = optim.lr_scheduler.CosineAnnealingLR(optimizer_dit, T_max=epochs)
    lr_scheduler_id = optim.lr_scheduler.CosineAnnealingLR(optimizer_id, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_fde = float("inf")

    dit_save_path = dit_config["save_path"]
    id_save_path = id_config["save_path"]
    os.makedirs(os.path.dirname(dit_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(id_save_path), exist_ok=True)

    print(f"Starting DiT + Inverse Dynamics training for {epochs} epochs...")
    print(f"  FDE Objective: minimize sqrt(w_xy * (dx^2 + dy^2) + w_theta * dtheta^2) [w_xy={dit_config['w_xy']}, w_th={dit_config['w_theta']}]")
    print(f"  Auxiliary FDE Weight (lambda_fde): {dit_config['lambda_fde']}")

    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            dit_model,
            id_model,
            loader,
            optimizer_dit,
            optimizer_id,
            device,
            scaler,
        )
        lr_scheduler_dit.step()
        lr_scheduler_id.step()

        fde_curr = metrics["fde_eval"]
        is_best = fde_curr < best_fde
        if is_best:
            best_fde = fde_curr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": dit_model.state_dict(),
                    "optimizer_state_dict": optimizer_dit.state_dict(),
                    "best_fde": best_fde,
                    "config": dit_config,
                },
                dit_save_path,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": id_model.state_dict(),
                    "optimizer_state_dict": optimizer_id.state_dict(),
                    "config": id_config,
                },
                id_save_path,
            )

        star = " ★ (Best Saved)" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"DiT Loss: {metrics['dit_total_loss']:.4f} | "
            f"MSE: {metrics['loss_mse']:.4f} | "
            f"FDE Penalty: {metrics['loss_fde']:.4f} | "
            f"FDE: {fde_curr:.4f} | "
            f"ID Action Loss: {metrics['loss_id']:.4f}{star}"
        )

    print("\n" + "=" * 70)
    print("DiT PLANNER & INVERSE DYNAMICS TRAINING COMPLETED!")
    print(f"  Best FDE achieved:        {best_fde:.4f}")
    print(f"  DiT Checkpoint:           {dit_save_path}")
    print(f"  Inverse Dynamics Checkpoint: {id_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
