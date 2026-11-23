"""Pipeline 4 (diffusion): train the DiT trajectory planner and the inverse dynamics controller.

Self-supervised planner training as in the course sketch: s_t + noisy future path -> DiT -> path,
loss = MSE + lambda_fde * FDE^2. Compared with the first version:
- trains on tensors built once from the latent cache (data/latent_arrays.py), so the planner can
  train for many more epochs in the same time;
- keeps an EMA copy of the planner (used for inference, as in Diffusion Policy);
- optional 2-frame observation history (dit_planner.obs_history or --obs-history);
- picks the planner checkpoint by validation FDE of sampled plans on the 20 held-out episodes;
- trains the inverse dynamics on true paths (+ small noise) and picks its snapshot by closed-loop
  simulation on validation seeds, because its offline action error did not predict closed-loop
  success (training it on DiT-sampled paths kept the offline error but cut success to 10%).

    python scripts/4_diffusion/train_dit_planner.py
    python scripts/4_diffusion/train_dit_planner.py --skip-dit          # only retrain the inverse dynamics
    python scripts/4_diffusion/train_dit_planner.py --obs-history 2 --dit-out ... --id-out ...
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import math
import time

import torch
import torch.nn.functional as F
import yaml

from pusht_wm_dit.data.latent_arrays import condition, train_tensors, val_tensors
from pusht_wm_dit.data.precompute_latents import check_latent_cache, precompute_and_cache_latents
from pusht_wm_dit.data.pusht_dataset import PushTDataset
from pusht_wm_dit.models.dit_planner import DiTTrajectoryPlanner
from pusht_wm_dit.models.inverse_dynamics import InverseDynamicsController
from pusht_wm_dit.models.world_model import WorldModel
from pusht_wm_dit.training import EMA, train_diffusion
from pusht_wm_dit.utils.metrics import compute_ade_torch, compute_fde_torch


def load_world_model(cfg, device):
    w = cfg["world_model"]
    wm = WorldModel(latent_dim=w["latent_dim"], agent_pos_dim=w["agent_pos_dim"], pose_dim=w["pose_dim"],
                    action_dim=w["action_dim"], num_groups=w["num_groups"],
                    transition_hidden_dim=w["transition_hidden_dim"]).to(device)
    if not os.path.exists(w["save_path"]):
        raise FileNotFoundError(f"World Model checkpoint not found at {w['save_path']}. Run scripts/3_world_model/train_encoder.py first.")
    wm.load_state_dict(torch.load(w["save_path"], map_location=device, weights_only=False)["model_state_dict"])
    wm.freeze()
    return wm


@torch.no_grad()
def sample_paths(dit, T, idx_all, obs_history, steps, bs=2048):
    out = []
    for i in range(0, len(idx_all), bs):
        idx = idx_all[i:i + bs]
        out.append(dit.sample(condition(T, idx, obs_history), num_inference_steps=steps,
                              current_pose_4d=T["path"][idx, 0], best_of_n=1))
    return torch.cat(out)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    dcfg, icfg, wcfg, data = cfg["dit_planner"], cfg["inverse_dynamics"], cfg["world_model"], cfg["data"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs-history", type=int, default=dcfg.get("obs_history", 1), choices=(1, 2))
    p.add_argument("--epochs", type=int, default=dcfg["epochs"])
    p.add_argument("--id-epochs", type=int, default=icfg["epochs"])
    p.add_argument("--dit-out", default=dcfg["save_path"])
    p.add_argument("--id-out", default=icfg["save_path"])
    p.add_argument("--skip-dit", action="store_true", help="keep the planner at --dit-out and only (re)train the inverse dynamics")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    H, Ta, hist = dcfg["horizon"], icfg["exec_steps"], args.obs_history
    print("=" * 78)
    print(f"STEP 4: DiT PLANNER + INVERSE DYNAMICS | device {device} | obs_history {hist} | epochs {args.epochs}")
    print("=" * 78)

    # 1. Data: cached latents of the frozen World Model + vectorised targets
    train_ds = PushTDataset(data["zarr_path"], H, Ta, split="train", val_ratio=data["val_ratio"], seed=cfg["seed"],
                            augment_symmetry=bool(data.get("augment_symmetry", False)))
    wm = load_world_model(cfg, device)
    ok, reason = check_latent_cache(data["latents_cache_path"], train_ds)
    if not ok:
        print(f"Latent cache unusable ({reason}); regenerating...")
        precompute_and_cache_latents(wm, train_ds, data["latents_cache_path"], device=device)
    latents = torch.load(data["latents_cache_path"], map_location="cpu", weights_only=False)["latents"]
    T = train_tensors(train_ds, latents, H, Ta, device)
    val_ds = PushTDataset(data["zarr_path"], H, Ta, split="val", val_ratio=data["val_ratio"], seed=cfg["seed"])
    V = val_tensors(val_ds, wm, H, Ta, device)
    v_idx = torch.arange(len(V["latent"]), device=device)
    print(f"Train samples {len(T['latent']):,} | val samples {len(V['latent']):,}")

    # 2. DiT planner with EMA, checkpoint chosen by validation FDE of sampled plans
    def new_dit():
        return DiTTrajectoryPlanner(horizon=H, traj_dim=dcfg["traj_dim"], latent_dim=wcfg["latent_dim"] * hist,
                                    model_dim=dcfg["model_dim"], num_heads=dcfg["num_heads"], num_layers=dcfg["num_layers"],
                                    mlp_ratio=dcfg["mlp_ratio"], diffusion_steps=dcfg["diffusion_steps"],
                                    beta_schedule=dcfg["beta_schedule"], w_xy=dcfg["w_xy"], w_theta=dcfg["w_theta"],
                                    lambda_fde=dcfg["lambda_fde"]).to(device)

    def val_fde(m):
        pred = sample_paths(m, V, v_idx, hist, dcfg["sample_steps"])
        return float(compute_fde_torch(V["path"], pred).item())

    def save_dit(ema_model, raw_model, ep, score):
        os.makedirs(os.path.dirname(args.dit_out), exist_ok=True)
        torch.save({"model_state_dict": ema_model.state_dict(), "raw_state_dict": raw_model.state_dict(), "epoch": ep,
                    "val_fde": score, "obs_history": hist, "ema": True, "config": dcfg}, args.dit_out)

    if not args.skip_dit:
        dit = new_dit()
        print("Training DiT planner (val = FDE of sampled plans, normalized units)...")
        train_diffusion(dit, T, "path", hist, args.epochs, dcfg["batch_size"], dcfg["lr"], dcfg["weight_decay"],
                        dcfg.get("ema_decay", 0.999), val_fde, eval_every=dcfg.get("eval_every", 10), on_best=save_dit)
    best = new_dit()
    best.load_state_dict(torch.load(args.dit_out, map_location=device, weights_only=False)["model_state_dict"])
    best.eval()
    with torch.no_grad():
        vp = sample_paths(best, V, v_idx, hist, dcfg["sample_steps"])
    print(f"DiT {args.dit_out}: val FDE {compute_fde_torch(V['path'], vp).item():.4f} | val ADE {compute_ade_torch(V['path'], vp).item():.4f}")

    # 3. Inverse dynamics. Offline action error does NOT predict closed-loop quality (measured: two
    #    controllers with equal val RMSE gave 46% vs 10% success), so snapshots are compared by
    #    closed-loop simulation on validation seeds that the benchmark never uses.
    ratio = float(icfg.get("dit_path_ratio", 0.0))
    tp = None
    if ratio > 0:
        print(f"Sampling DiT plans for every training state (dit_path_ratio={ratio})...")
        tp = sample_paths(best, T, torch.arange(len(T["latent"]), device=device), hist, dcfg["sample_steps"])
    idm = InverseDynamicsController(latent_dim=wcfg["latent_dim"] * hist, horizon=H, traj_dim=dcfg["traj_dim"],
                                    action_dim=icfg["action_dim"], exec_steps=Ta, hidden_dims=tuple(icfg["hidden_dims"])).to(device)
    opt = torch.optim.AdamW(idm.parameters(), lr=icfg["lr"], weight_decay=icfg["weight_decay"])
    n, bs = len(T["latent"]), icfg["batch_size"]
    spe = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.id_epochs * spe)
    snapshots = {}
    every = int(icfg.get("snapshot_every", 5))
    for ep in range(1, args.id_epochs + 1):
        idm.train()
        perm = torch.randperm(n, device=device)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            path = torch.clamp(T["path"][idx] + torch.randn_like(T["path"][idx]) * 0.02, -1, 1)
            if tp is not None:
                path = torch.where((torch.rand(len(idx), device=device) < ratio)[:, None, None], tp[idx], path)
            loss = F.mse_loss(idm(condition(T, idx, hist), path, agent_pos=T["agent"][idx]), T["actions"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(idm.parameters(), 1.0)
            opt.step()
            sched.step()
        if ep % every == 0 or ep == args.id_epochs:
            with torch.no_grad():
                rmse = math.sqrt(F.mse_loss(idm(condition(V, v_idx, hist), vp, agent_pos=V["agent"]), V["actions"]).item()) * 256
            snapshots[ep] = {k: v.detach().clone() for k, v in idm.state_dict().items()}
            print(f"  ID epoch {ep:03d} | val action RMSE on DiT plans {rmse:.1f}px (snapshot kept)")

    n_eps = int(icfg.get("select_closed_loop_episodes", 20))
    print(f"Choosing the inverse-dynamics snapshot by closed-loop simulation ({n_eps} validation seeds)...")
    from pusht_wm_dit.envs.pusht_sim import PushTSim
    from pusht_wm_dit.policy import WMDiTPolicy, run_episode
    sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=tuple(cfg["env"]["render_size"]), max_steps=cfg["env"]["max_steps"])
    best.obs_history = hist
    scores = {}
    for ep, sd in snapshots.items():
        idm.load_state_dict(sd)
        idm.eval()
        pol = WMDiTPolicy(wm, best, idm, device, sample_steps=dcfg["sample_steps"])
        res = []
        for j in range(n_eps):
            torch.manual_seed(j)
            res.append(run_episode(sim, pol, seed=90_000 + j, max_steps=cfg["env"]["max_steps"], exec_horizon=Ta))
        cov = float(sum(r["max_coverage"] for r in res) / n_eps)
        succ = float(sum(r["success"] for r in res) / n_eps)
        scores[ep] = (succ, cov)
        print(f"  snapshot epoch {ep:03d}: closed-loop success {succ * 100:.0f}% | mean max coverage {cov * 100:.1f}%")
    best_ep = max(scores, key=lambda e: scores[e])
    torch.save({"model_state_dict": snapshots[best_ep], "epoch": best_ep, "closed_loop_val": scores[best_ep],
                "obs_history": hist, "trained_on": f"true paths + noise (dit_path_ratio={ratio})", "config": icfg}, args.id_out)
    print(f"Best inverse dynamics: epoch {best_ep} (success {scores[best_ep][0] * 100:.0f}%, coverage {scores[best_ep][1] * 100:.1f}%) -> {args.id_out}")


if __name__ == "__main__":
    main()
