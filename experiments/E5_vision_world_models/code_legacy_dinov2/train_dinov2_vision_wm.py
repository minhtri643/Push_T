"""Stage E5a (legacy): train and evaluate the vision World Model (frozen DINOv2 features + action-conditioned predictor).

    python experiments/E5_vision_world_models/code_legacy_dinov2/extract_dino_features.py   # once
    python experiments/E5_vision_world_models/code_legacy_dinov2/train_dinov2_vision_wm.py [--no-sim] [--epochs N]

The predictor is trained on image features only (no block pose). A pose probe is trained
separately to read the block pose from features, which is how the imagination is measured.
Evaluation on the 20 held-out episodes, horizons 4/8/12/16 control steps:
  - "Đoán đứng yên"                       block never moves (true current pose)
  - "Đầu dò trên ảnh thật"                 probe on the TRUE future frame = floor for any vision model
  - "WM ảnh"                               probe on IMAGINED features (the vision World Model)
  - "WM ảnh, hành động xáo trộn"           same, actions from another window
  - "Động lực học trạng thái thật"         models/dynamics.py from the true state (upper bound)
plus the simulator counterfactual test (demonstrated / stand still / shifted 25 px actions).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import json
import math
import time

import gymnasium as gym
import gym_pusht  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr

from pusht_wm_dit.data.dynamics_data import episode_split
from pusht_wm_dit.models.dynamics import load_dynamics, pose_errors
from pusht_wm_dit.models.latent_world_model import FeatureNorm, LatentPredictor, PoseProbe, pose4_to_xyth

DEMO_FEATS = "data/cache/dino_feats_demo.npy"
SIM_FEATS = "data/cache/dino_feats_sim.npy"
SIM_STATES = "data/legacy/sim_interactions_img.npz"
OUT = "results/vision_world_models/E5a_dinov2_zarr_sim"


def norm_xy(p):
    return (p - 256.0) / 256.0


def load_sources(cfg, use_sim):
    """Concatenated arrays (demo first, then sim) and per-sequence (offset, length, split)."""
    root = zarr.open(cfg["data"]["zarr_path"], mode="r")
    st = np.asarray(root["data/state"][:], np.float64)
    ac = np.asarray(root["data/action"][:], np.float64)
    ends = root["meta/episode_ends"][:]
    train_ids, val_ids = episode_split(len(ends), cfg["data"]["val_ratio"], cfg["seed"])
    feats = [np.load(DEMO_FEATS)]
    agents, blocks, actions = [st[:, :2]], [st[:, 2:5]], [ac]
    seqs, start = [], 0
    for ep, end in enumerate(ends):
        seqs.append((start, int(end - start), "val" if ep in set(val_ids) else "train"))
        start = int(end)
    if use_sim:
        sim = np.load(SIM_STATES)
        off = len(st)
        feats.append(np.load(SIM_FEATS))
        agents.append(sim["agent"]); blocks.append(sim["block"]); actions.append(sim["action"])
        for n in sim["lengths"]:
            seqs.append((off, int(n), "train"))
            off += int(n)
    agent, block, action = np.concatenate(agents), np.concatenate(blocks), np.concatenate(actions)
    return {
        "feats": np.concatenate(feats) if len(feats) > 1 else feats[0],
        "agent": norm_xy(agent).astype(np.float32),
        "block": np.concatenate([norm_xy(block[:, :2]), block[:, 2:3]], 1).astype(np.float32),
        "action": norm_xy(action).astype(np.float32),
        "seqs": seqs,
    }


def windows(seqs, split, S, K):
    """Rows of absolute indices: [hist, t, fut_1..fut_K] and the first action index."""
    rows = []
    for off, n, sp in seqs:
        if sp != split:
            continue
        for t in range(0, n - K * S):
            rows.append([off + max(t - S, 0), off + t] + [off + t + (k + 1) * S for k in range(K)])
    return np.array(rows, dtype=np.int64)


class Batcher:
    def __init__(self, D, rows, S, K, device):
        self.D, self.rows, self.S, self.K, self.device = D, rows, S, K, device

    def get(self, sel):
        r = self.rows[sel]
        z = torch.from_numpy(self.D["feats"][r.reshape(-1)]).view(len(r), r.shape[1], 64, -1)
        act_idx = r[:, 1:2] + np.arange(self.K * self.S)[None]
        g = lambda a, idx: torch.from_numpy(self.D[a][idx]).to(self.device)  # noqa: E731
        return {
            "z": z.to(self.device, non_blocking=True),
            "agent": g("agent", r),            # (B, 2+K, 2)
            "block": g("block", r),            # (B, 2+K, 3)
            "actions": g("action", act_idx),   # (B, K*S, 2)
            "agent_prev_step": g("agent", np.maximum(r[:, 1] - 1, 0)),
        }


def wm_loss(pred, norm, b):
    zn = norm(b["z"])
    zs, ags = pred.rollout(zn[:, 0], zn[:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"])
    lf = F.mse_loss(zs, zn[:, 2:])
    la = F.mse_loss(ags, b["agent"][:, 2:]) * 100.0
    return lf + la, lf.item(), la.item()


@torch.no_grad()
def predict_poses(pred, norm, probe, b, actions=None):
    zn = norm(b["z"])
    zs, _ = pred.rollout(zn[:, 0], zn[:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"] if actions is None else actions)
    B, K = zs.shape[:2]
    return pose4_to_xyth(probe(zs.reshape(B * K, 64, -1))).view(B, K, 3)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    vcfg = cfg["vision_wm"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-sim", action="store_true", help="train on demonstrations only")
    p.add_argument("--epochs", type=int, default=vcfg["epochs"])
    p.add_argument("--out", default=vcfg["save_path"])
    p.add_argument("--results-dir", default=OUT)
    p.add_argument("--dim", type=int, default=vcfg["dim"])
    p.add_argument("--depth", type=int, default=vcfg["depth"])
    p.add_argument("--heads", type=int, default=vcfg["heads"])
    p.add_argument("--probe-epochs", type=int, default=vcfg["probe_epochs"])
    args = p.parse_args()
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    S, K = vcfg["skip"], vcfg["rollout_steps"]
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    D = load_sources(cfg, use_sim=not args.no_sim)
    tr, va = windows(D["seqs"], "train", S, K), windows(D["seqs"], "val", S, K)
    print(f"Loaded {len(D['feats']):,} frames in {time.perf_counter() - t0:.0f}s | train windows {len(tr):,} | val windows {len(va):,}")

    sample = torch.from_numpy(D["feats"][np.random.choice(tr[:, 1], 4000, replace=False)]).float().view(-1, D["feats"].shape[-1])
    norm = FeatureNorm(sample.mean(0), sample.std(0) + 1e-3).to(dev)
    TB, VB = Batcher(D, tr, S, K, dev), Batcher(D, va, S, K, dev)

    # ---- 1. pose probe on TRUE features (measurement tool only) ----
    probe = PoseProbe().to(dev)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    frames_tr = np.unique(tr[:, 1])
    psched = torch.optim.lr_scheduler.CosineAnnealingLR(popt, args.probe_epochs * math.ceil(len(frames_tr) / 256))
    for ep in range(args.probe_epochs):
        perm = np.random.permutation(frames_tr)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            z = norm(torch.from_numpy(D["feats"][idx]).to(dev))
            bl = torch.from_numpy(D["block"][idx]).to(dev)
            tgt = torch.stack([bl[:, 0], bl[:, 1], torch.sin(bl[:, 2]), torch.cos(bl[:, 2])], -1)
            loss = F.mse_loss(probe(z), tgt)
            popt.zero_grad(set_to_none=True)
            loss.backward()
            popt.step()
            psched.step()
    probe.eval()
    for p_ in probe.parameters():
        p_.requires_grad_(False)

    # ---- 2. latent predictor ----
    pred = LatentPredictor(skip=S, dim=args.dim, depth=args.depth, heads=args.heads).to(dev)
    opt = torch.optim.AdamW(pred.parameters(), lr=vcfg["lr"], weight_decay=1e-4)
    bs = vcfg["batch_size"]
    spe = math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=vcfg["lr"], total_steps=args.epochs * spe, pct_start=0.05)
    best, t0 = float("inf"), time.perf_counter()
    for ep in range(1, args.epochs + 1):
        pred.train()
        perm = np.random.permutation(len(tr))
        tf = ta = 0.0
        for i in range(spe):
            b = TB.get(perm[i * bs:(i + 1) * bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, lf, la = wm_loss(pred, norm, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            opt.step()
            sched.step()
            tf, ta = tf + lf, ta + la
        pred.eval()
        vl, vpos = 0.0, []
        with torch.no_grad():
            for i in range(0, len(va), 256):
                b = VB.get(np.arange(i, min(len(va), i + 256)))
                vl += wm_loss(pred, norm, b)[1] * len(b["z"])
                pp = predict_poses(pred, norm, probe, b)
                vpos.append(pose_errors(pp, b["block"][:, 2:])["pos_px"])
        vl /= len(va)
        vpos = torch.cat(vpos).mean(0).tolist()
        is_best = vl < best
        if is_best:
            best = vl
            torch.save({"predictor": pred.state_dict(), "hparams": pred.hparams, "norm": norm.state_dict(),
                        "probe": probe.state_dict(), "epoch": ep, "val_feature_mse": vl,
                        "data": "demo" if args.no_sim else "demo+sim"}, args.out)
        print(f"Epoch {ep:02d} | train feat {tf / spe:.4f} agent {ta / spe:.4f} | val feat {vl:.4f} | "
              f"val block pos err (probe) k=4/8/12/16: {' / '.join(f'{x:.1f}' for x in vpos)} px{' *' if is_best else ''} | {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- 3. evaluation ----
    ck = torch.load(args.out, map_location=dev, weights_only=False)
    pred.load_state_dict(ck["predictor"])
    pred.eval()
    dyn = load_dynamics(cfg["dynamics"]["save_path"], dev)
    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(va))
    acc = {k: [] for k in ("static", "probe_true", "wm", "wm_shuffled", "state_dyn", "moving")}
    with torch.no_grad():
        for i in range(0, len(va), 256):
            sel = np.arange(i, min(len(va), i + 256))
            b = VB.get(sel)
            truth = b["block"][:, 2:]
            zn = norm(b["z"])
            B = len(sel)
            acc["static"].append(pose_errors(b["block"][:, 1:2].expand(-1, K, -1), truth))
            pt = pose4_to_xyth(probe(zn[:, 2:].reshape(B * K, 64, -1))).view(B, K, 3)
            acc["probe_true"].append(pose_errors(pt, truth))
            acc["wm"].append(pose_errors(predict_poses(pred, norm, probe, b), truth))
            acc["wm_shuffled"].append(pose_errors(predict_poses(pred, norm, probe, b, VB.get(shuf[sel])["actions"]), truth))
            vel = b["agent"][:, 1] - b["agent_prev_step"]
            dyn_b, _ = dyn.rollout(b["block"][:, 1], b["agent"][:, 1], vel, b["actions"])
            acc["state_dyn"].append(pose_errors(dyn_b[:, S - 1::S], truth))
            acc["moving"].append(torch.linalg.norm(truth[:, -1, :2] - b["block"][:, 1, :2], dim=-1) * 256 > 5)
    moving = torch.cat(acc.pop("moving"))
    names = {"static": "Đoán đứng yên", "probe_true": "Đầu dò trên ảnh thật (sàn)", "wm": "WM ảnh (tưởng tượng)",
             "wm_shuffled": "WM ảnh, hành động xáo trộn", "state_dyn": "Động lực học trạng thái thật (cận trên)"}
    report = {"num_windows": len(va), "num_moving": int(moving.sum()), "horizons_steps": [S * (k + 1) for k in range(K)],
              "data": ck["data"], "epoch": ck["epoch"], "models": {}}
    for k, lst in acc.items():
        pos = torch.cat([e["pos_px"] for e in lst])
        ang = torch.cat([e["ang_deg"] for e in lst])
        report["models"][names[k]] = {"pos_px": pos.mean(0).tolist(), "ang_deg": ang.mean(0).tolist(),
                                      "pos_px_moving": pos[moving].mean(0).tolist(), "ang_deg_moving": ang[moving].mean(0).tolist()}

    # ---- counterfactual test in the simulator ----
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    u = env.unwrapped
    mv_idx = np.nonzero(moving.cpu().numpy())[0]
    pick = mv_idx[np.linspace(0, len(mv_idx) - 1, min(150, len(mv_idx))).astype(int)]
    real, pr_wm, pr_dyn = [], [], []
    for i in pick:
        b = VB.get(np.array([i]))
        ag = b["agent"][0, 1].cpu().numpy() * 256 + 256
        bl = b["block"][0, 1].cpu().numpy()
        blk = [bl[0] * 256 + 256, bl[1] * 256 + 256, bl[2]]
        demo = b["actions"][0].cpu().numpy() * 256 + 256
        ang = rng.uniform(-np.pi, np.pi)
        seqs = [demo, np.repeat(ag[None], K * S, 0), demo + 25.0 * np.array([np.cos(ang), np.sin(ang)])]
        outs = []
        for a_seq in seqs:
            env.reset(seed=0)
            u._set_state(list(ag) + blk)
            u._set_state(list(ag) + blk)
            for a in a_seq:
                u.step(np.clip(a, 0, 512))
            outs.append(np.array(u._get_info()["block_pose"])[:2])
        real.append(outs)
        acts = torch.from_numpy((np.stack(seqs) - 256.0) / 256.0).float().to(dev)
        bb = {k: (v.expand(3, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            pr_wm.append(predict_poses(pred, norm, probe, bb, acts)[:, -1, :2].cpu().numpy() * 256 + 256)
            d_b, _ = dyn.rollout(bb["block"][:, 1], bb["agent"][:, 1], torch.zeros(3, 2, device=dev), acts)
            pr_dyn.append(d_b[:, -1, :2].cpu().numpy() * 256 + 256)
    real = np.array(real)
    cf = {}
    for name, pr in (("WM ảnh (tưởng tượng)", np.array(pr_wm)), ("Động lực học trạng thái thật (cận trên)", np.array(pr_dyn))):
        ratios, errs = [], []
        for j in (1, 2):
            dr, dp = real[:, 0] - real[:, j], pr[:, 0] - pr[:, j]
            ok = np.linalg.norm(dr, axis=1) > 5
            ratios += list(np.linalg.norm(dp[ok], axis=1) / np.linalg.norm(dr[ok], axis=1))
            errs += list(np.linalg.norm(pr[:, j] - real[:, j], axis=1))
        cf[name] = {"sensitivity_median": float(np.median(ratios)), "final_pos_err_px_counterfactual_median": float(np.median(errs))}
    report["counterfactual"] = cf
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    hs = report["horizons_steps"]
    print("\n" + "=" * 100)
    print(f"VISION WORLD MODEL – {len(va):,} cửa sổ kiểm tra ({report['num_moving']:,} có khối di chuyển > 5px), dữ liệu: {ck['data']}")
    print("=" * 100)
    print(f"{'Mô hình':<44}{'vị trí (px) tại bước ' + '/'.join(map(str, hs)):<36}{'góc (°) bước ' + str(hs[-1]):>16}")
    for n, r in report["models"].items():
        print(f"{n:<44}{' / '.join(f'{x:.1f}' for x in r['pos_px']):<36}{r['ang_deg'][-1]:>16.2f}")
    print("\nPhép thử phản thực tế (độ nhạy lý tưởng 100%):")
    for n, r in cf.items():
        print(f"  {n:<44} độ nhạy {r['sensitivity_median'] * 100:5.0f}% | sai vị trí cuối với hành động lạ {r['final_pos_err_px_counterfactual_median']:5.1f}px")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    style = {"Đoán đứng yên": ("#9aa5b1", "--"), "Đầu dò trên ảnh thật (sàn)": ("#23804a", ":"), "WM ảnh (tưởng tượng)": ("#3457d5", "-"),
             "WM ảnh, hành động xáo trộn": ("#d9534f", "-."), "Động lực học trạng thái thật (cận trên)": ("#1f2933", "-")}
    for n, r in report["models"].items():
        c, ls = style[n]
        ax[0].plot(hs, r["pos_px_moving"], ls, color=c, marker="o", lw=2, label=n)
        ax[1].plot(hs, r["ang_deg_moving"], ls, color=c, marker="o", lw=2, label=n)
    for a, t, y in ((ax[0], "Sai số vị trí khối T (khi khối di chuyển)", "pixel"), (ax[1], "Sai số góc khối T (khi khối di chuyển)", "độ")):
        a.set_title(t, fontweight="bold")
        a.set_xlabel("bước tưởng tượng (0,1 s/bước)")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves.png"))
    print(f"\nĐã lưu {args.out}, {out_dir}/metrics.json, {out_dir}/curves.png")


if __name__ == "__main__":
    main()
