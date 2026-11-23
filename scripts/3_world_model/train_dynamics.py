"""Pipeline 3 (World Model, part 2 = dynamics): train the action-conditioned dynamics model.

The model imagines K=16 steps open-loop from the current state and the robot's actions; the loss
is the course's displacement error f(dx, dy, dtheta) accumulated over all K steps (in px / deg,
Huber), with moving windows weighted x3 so the static majority does not dominate.
Checkpoints are selected on the held-out validation episodes.

    python scripts/3_world_model/train_dynamics.py                      # demonstrations only
    python scripts/3_world_model/train_dynamics.py --sim --out checkpoints/E3_dynamics_variants/dynamics_demo_sim.pt
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json
import math
import time

import numpy as np
import torch
import yaml

from pusht_wm_dit.data.dynamics_data import load_demo_sequences, load_sim_sequences, make_windows
from pusht_wm_dit.models.dynamics import PushDynamics, pose_errors


def huber(x: torch.Tensor, delta: float = 2.0) -> torch.Tensor:
    return torch.where(x < delta, 0.5 * x ** 2 / delta, x - 0.5 * delta)


def rollout_errors(model, W, idx):
    blocks, agents = model.rollout(W["block0"][idx], W["agent0"][idx], W["vel0"][idx], W["actions"][idx])
    e = pose_errors(blocks, W["tgt_block"][idx])
    e["agent_px"] = torch.linalg.norm(agents - W["tgt_agent"][idx], dim=-1) * 256.0
    return e


def loss_fn(model, W, idx):
    e = rollout_errors(model, W, idx)
    w = (1.0 + 2.0 * W["moved"][idx].float())[:, None] * W["mask"][idx].float()
    per = huber(e["pos_px"]) + huber(e["ang_deg"]) + 0.3 * huber(e["agent_px"])
    return (per * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, W, bs: int = 8192) -> dict:
    """Masked mean errors per horizon step, for all windows and for windows where the block moves."""
    n = len(W["block0"])
    pos, ang = [], []
    for i in range(0, n, bs):
        idx = torch.arange(i, min(n, i + bs), device=W["block0"].device)
        e = rollout_errors(model, W, idx)
        pos.append(e["pos_px"])
        ang.append(e["ang_deg"])
    pos, ang = torch.cat(pos), torch.cat(ang)
    m = W["mask"].float()
    mv = m * W["moved"].float()[:, None]
    res = {
        "pos_px_all": ((pos * m).sum(0) / m.sum(0)).tolist(),
        "ang_deg_all": ((ang * m).sum(0) / m.sum(0)).tolist(),
        "pos_px_moving": ((pos * mv).sum(0) / mv.sum(0)).tolist(),
        "ang_deg_moving": ((ang * mv).sum(0) / mv.sum(0)).tolist(),
    }
    res["score"] = float(np.mean(res["pos_px_all"]) + np.mean(res["ang_deg_all"]))
    return res


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["dynamics"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="add simulator interaction data (experiments/E3_dynamics_world_model/code/collect_sim_interactions.py)")
    p.add_argument("--out", default=dcfg["save_path"])
    p.add_argument("--epochs", type=int, default=dcfg["epochs"])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    K = dcfg["horizon"]

    train_seqs = load_demo_sequences(cfg["data"]["zarr_path"], "train", cfg["data"]["val_ratio"], cfg["seed"])
    val_seqs = load_demo_sequences(cfg["data"]["zarr_path"], "val", cfg["data"]["val_ratio"], cfg["seed"])
    if args.sim:
        sim = load_sim_sequences(dcfg["sim_data_path"])
        print(f"+ {len(sim)} simulated rollouts ({sum(len(s['agent']) for s in sim):,} steps)")
        train_seqs = train_seqs + sim
    W = {k: v.to(device) for k, v in make_windows(train_seqs, K, augment=bool(cfg["data"].get("augment_symmetry"))).items()}
    V = {k: v.to(device) for k, v in make_windows(val_seqs, K, augment=False).items()}
    n = len(W["block0"])
    print(f"Train windows: {n:,} | val windows: {len(V['block0']):,} (moving: {int(V['moved'].sum()):,}) | horizon K={K}")

    model = PushDynamics(hidden=dcfg["hidden"], layers=dcfg["layers"]).to(device)
    base = evaluate(model, V)  # zero-initialised model == "nothing moves" baseline
    print(f"Static baseline on val: pos k=1/8/16 = {base['pos_px_all'][0]:.1f}/{base['pos_px_all'][7]:.1f}/{base['pos_px_all'][15]:.1f}px")

    opt = torch.optim.AdamW(model.parameters(), lr=dcfg["lr"], weight_decay=dcfg["weight_decay"])
    steps_per_epoch = math.ceil(n / dcfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=dcfg["lr"], total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    best, t0 = None, time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * dcfg["batch_size"]:(i + 1) * dcfg["batch_size"]]
            loss = loss_fn(model, W, idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        val = evaluate(model, V)
        is_best = best is None or val["score"] < best["score"]
        if is_best:
            best = val
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "hparams": model.hparams, "epoch": ep,
                        "val": val, "static_baseline": base, "data": "demo+sim" if args.sim else "demo",
                        "horizon": K}, args.out)
        if ep % 5 == 0 or ep == 1 or is_best and ep > args.epochs - 5:
            print(f"Epoch {ep:03d} | train loss {tot / steps_per_epoch:.3f} | val pos k=1/8/16 "
                  f"{val['pos_px_all'][0]:.2f}/{val['pos_px_all'][7]:.2f}/{val['pos_px_all'][15]:.2f}px | "
                  f"moving k=16 {val['pos_px_moving'][15]:.2f}px {val['ang_deg_moving'][15]:.2f}deg"
                  f"{' *' if is_best else ''} | {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"Best val score {best['score']:.3f} -> {args.out}")
    print(json.dumps({k: [round(x, 2) for x in v[:16:3]] if isinstance(v, list) else v for k, v in best.items()}))


if __name__ == "__main__":
    main()
