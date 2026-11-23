"""Pipeline 4 (baseline): action-diffusion baseline in the style of Diffusion Policy (course tutorial day-17).

Same frozen World Model encoder and the same DiT architecture, but the diffusion model generates
the robot ACTIONS directly (16 predicted, 8 executed) instead of the object path. Comparing it with
"DiT path -> inverse dynamics" shows what the intermediate object-path representation buys.
Note: the official Diffusion Policy uses a 1D U-Net trained for 200k steps; this is a small,
same-backbone baseline, not a reproduction of the paper.

    python scripts/4_diffusion/train_action_diffusion.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import math

import torch
import torch.nn.functional as F
import yaml

from pusht_wm_dit.data.latent_arrays import condition, train_tensors, val_tensors
from pusht_wm_dit.data.pusht_dataset import PushTDataset
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.training import train_diffusion

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

load_world_model = import_module("train_dit_planner").load_world_model


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    acfg, dcfg, wcfg, data = cfg["action_diffusion"], cfg["dit_planner"], cfg["world_model"], cfg["data"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=acfg["epochs"])
    p.add_argument("--obs-history", type=int, default=acfg.get("obs_history", 1), choices=(1, 2))
    p.add_argument("--out", default=acfg["save_path"])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    H, hist = acfg["horizon"], args.obs_history

    train_ds = PushTDataset(data["zarr_path"], dcfg["horizon"], H, split="train", val_ratio=data["val_ratio"],
                            seed=cfg["seed"], augment_symmetry=bool(data.get("augment_symmetry", False)))
    wm = load_world_model(cfg, device)
    latents = torch.load(data["latents_cache_path"], map_location="cpu", weights_only=False)["latents"]
    T = train_tensors(train_ds, latents, dcfg["horizon"], H, device)
    val_ds = PushTDataset(data["zarr_path"], dcfg["horizon"], H, split="val", val_ratio=data["val_ratio"], seed=cfg["seed"])
    V = val_tensors(val_ds, wm, dcfg["horizon"], H, device)
    v_idx = torch.arange(len(V["latent"]), device=device)
    exec_steps = cfg["evaluation"]["exec_horizon"]

    model = DiTTrajectoryPlanner(horizon=H, traj_dim=2, latent_dim=wcfg["latent_dim"] * hist, model_dim=dcfg["model_dim"],
                                 num_heads=dcfg["num_heads"], num_layers=dcfg["num_layers"], mlp_ratio=dcfg["mlp_ratio"],
                                 diffusion_steps=dcfg["diffusion_steps"], beta_schedule=dcfg["beta_schedule"],
                                 lambda_fde=0.0).to(device)

    @torch.no_grad()
    def val_rmse(m):
        pred = m.sample(condition(V, v_idx, hist), num_inference_steps=dcfg["sample_steps"], best_of_n=1)
        return math.sqrt(F.mse_loss(pred[:, :exec_steps], V["actions"][:, :exec_steps]).item()) * 256  # px

    def save(ema_model, raw_model, ep, score):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model_state_dict": ema_model.state_dict(), "epoch": ep, "val_action_rmse_px": score,
                    "obs_history": hist, "ema": True, "config": acfg}, args.out)

    print(f"Training action-diffusion baseline ({len(T['latent']):,} samples, val = RMSE of executed actions in px)")
    best = train_diffusion(model, T, "actions", hist, args.epochs, dcfg["batch_size"], dcfg["lr"], dcfg["weight_decay"],
                           dcfg.get("ema_decay", 0.999), val_rmse, eval_every=dcfg.get("eval_every", 10), on_best=save)
    print(f"Best action diffusion: val RMSE {best:.1f}px -> {args.out}")


if __name__ == "__main__":
    main()
