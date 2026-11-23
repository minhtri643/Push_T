"""Stage E5b: vision World Model with a SELF-TRAINED encoder (JEPA style), lerobot/pusht data only.

    python scripts/1_data/fit_block_pose.py hf        # once: builds data/pusht_hf.zarr
    python scripts/6_vision_world_model/train_jepa_wm.py [--epochs N]

What is learned (no pretrained weights, no block pose, no simulator data):
    encoder E        96x96 frame -> 12x12 tokens (models/jepa_encoder.py)
    predictor P      (E(o_{t-S}), E(o_t), pusher positions, S actions) -> E(o_{t+S}), rolled out K times
    loss             || P-rollout - sg(E_ema(o_{t+kS})) ||^2           JEPA prediction in latent space
                   + VICReg variance/covariance of E's tokens across the batch   (anti-collapse)
                   + inverse dynamics: actions between o_{t-S} and o_t from E(o_{t-S}), E(o_t)
                   + pusher position predicted by P (proprioception)
Measurement only (never used to train E or P):
    a pose probe trained afterwards on frozen features -> block pose, compared with the poses recovered
    from the images by fit_block_pose.py, and on simulator frames with exactly known pose
    (results/pose_recovery/synthetic_frames.npz, clean and AV1-compressed);
    the simulator counterfactual test (same start, different actions).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import copy
import json
import math
import time

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr

from pusht_wm_dit.data.dynamics_data import episode_split
from pusht_wm_dit.models.dynamics import load_dynamics, pose_errors
from pusht_wm_dit.models.jepa_encoder import ConvEncoder, encode_frames, token_norm
from pusht_wm_dit.models.latent_world_model import FeatureNorm, LatentPredictor, PoseProbe, pose4_to_xyth

OUT = "results/vision_world_models/E5b_jepa"
SYNTH = "results/pose_recovery/synthetic_frames.npz"


def norm_xy(p):
    return (p - 256.0) / 256.0


def reflect_xy(p):
    """Anti-diagonal reflection through the goal centre, in normalised coordinates: (x, y) -> (-y, -x)."""
    return torch.stack([-p[..., 1], -p[..., 0]], -1)


def reflect_img(img):
    """uint8 (..., 96, 96, 3): transpose + flip both axes (matches reflect_xy)."""
    return img.transpose(-3, -2).flip(-3, -2)


class IDMHead(nn.Module):
    """Actions a_{t-S}..a_{t-1} (normalised) from the tokens of o_{t-S} and o_t."""

    def __init__(self, feat_dim, skip, hidden=256):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2 * feat_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * skip))
        self.skip = skip

    def forward(self, z0, z1):
        h = self.tok(torch.cat([z0, z1], -1))
        return self.out(torch.cat([h.mean(1), h.amax(1)], -1)).view(-1, self.skip, 2)


def vicreg(z, gamma):
    """z (B, N, C): variance of every token channel across the batch >= gamma, decorrelated channels."""
    zc = z - z.mean(0, keepdim=True)
    std = torch.sqrt(zc.pow(2).mean(0) + 1e-4)                         # (N, C)
    var_loss = F.relu(gamma - std).mean()
    x = zc.reshape(-1, z.shape[-1])
    cov = (x.T @ x) / (len(x) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, off.pow(2).sum() / z.shape[-1], std.mean()


def windows(seqs, split, S, K):
    rows = []
    for off, n, sp in seqs:
        if sp != split:
            continue
        for t in range(0, n - K * S):
            rows.append([off + max(t - S, 0), off + t] + [off + t + (k + 1) * S for k in range(K)])
    return np.array(rows, dtype=np.int64)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    jc = cfg["jepa_wm"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=jc["epochs"])
    p.add_argument("--out", default=jc["save_path"])
    p.add_argument("--results-dir", default=OUT)
    p.add_argument("--var-gamma", type=float, default=jc["var_gamma"])
    p.add_argument("--smoke", action="store_true", help="tiny run to check the code path (few windows, 1 probe epoch)")
    args = p.parse_args()
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    S, K = jc["skip"], jc["rollout_steps"]
    os.makedirs(args.results_dir, exist_ok=True)

    # ---------------------------------------------------------------- data (lerobot/pusht only)
    root = zarr.open(cfg["data"]["zarr_path"], mode="r")
    print("data:", root.attrs.get("source", cfg["data"]["zarr_path"]))
    imgs = np.asarray(root["data/img"][:])
    st = np.asarray(root["data/state"][:], np.float64)
    ac = np.asarray(root["data/action"][:], np.float64)
    ends = root["meta/episode_ends"][:]
    train_ids, val_ids = episode_split(len(ends), cfg["data"]["val_ratio"], cfg["seed"])
    seqs, start = [], 0
    for ep, end in enumerate(ends):
        seqs.append((start, int(end - start), "val" if ep in set(val_ids) else "train"))
        start = int(end)
    agent = torch.from_numpy(norm_xy(st[:, :2]).astype(np.float32)).to(dev)
    block = torch.from_numpy(np.concatenate([norm_xy(st[:, 2:4]), st[:, 4:5]], 1).astype(np.float32)).to(dev)
    action = torch.from_numpy(norm_xy(ac).astype(np.float32)).to(dev)
    frames = torch.from_numpy(imgs)                                            # uint8 on CPU; batches are copied (4 GB GPU)
    tr, va = windows(seqs, "train", S, K), windows(seqs, "val", S, K)
    if args.smoke:
        tr, va = tr[::40], va[::20]
        jc["probe_epochs"] = 1
    print(f"{len(imgs):,} frames | train windows {len(tr):,} | val windows {len(va):,}")

    def batch(rows, augment=False):
        r = torch.from_numpy(rows).to(dev)
        act_idx = r[:, 1:2] + torch.arange(K * S, device=dev)[None]
        hist_idx = r[:, 0:1] + torch.arange(S, device=dev)[None]
        b = {"img": frames[torch.from_numpy(rows)].to(dev, non_blocking=True), "agent": agent[r], "block": block[r], "actions": action[act_idx],
             "hist_actions": action[hist_idx], "hist_ok": (r[:, 1] - r[:, 0]) == S,
             "agent_prev_step": agent[(r[:, 1] - 1).clamp(min=0)]}
        if augment:
            m = torch.rand(len(r), device=dev) < 0.5
            b["img"] = torch.where(m[:, None, None, None, None], reflect_img(b["img"]), b["img"])
            for k in ("agent", "actions", "hist_actions"):
                b[k] = torch.where(m.view(-1, *[1] * (b[k].dim() - 1)), reflect_xy(b[k]), b[k])
        return b

    # ---------------------------------------------------------------- models
    enc = ConvEncoder(width=jc["width"], feat_dim=jc["feat_dim"]).to(dev)
    enc_ema = copy.deepcopy(enc).requires_grad_(False)
    C, N = enc.feat_dim, enc.n_tok
    pred = LatentPredictor(skip=S, dim=jc["dim"], depth=jc["depth"], heads=jc["heads"], n_tok=N, feat_dim=C).to(dev)
    idm = IDMHead(C, S).to(dev)
    params = list(enc.parameters()) + list(pred.parameters()) + list(idm.parameters())
    opt = torch.optim.AdamW(params, lr=jc["lr"], weight_decay=1e-4)
    bs = jc["batch_size"]
    spe = len(tr) // bs
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=jc["lr"], total_steps=total, pct_start=0.05)
    print(f"encoder {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M params, predictor {sum(p.numel() for p in pred.parameters()) / 1e6:.2f}M")

    def losses(b, train=True):
        B = len(b["img"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = enc(b["img"][:, :2].reshape(-1, 96, 96, 3)).view(B, 2, N, C)
            with torch.no_grad():
                tgt = token_norm(enc_ema(b["img"][:, 2:].reshape(-1, 96, 96, 3)).view(B, K, N, C))
            zc = token_norm(ctx)
            zs, ags = pred.rollout(zc[:, 0], zc[:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"])
            a_hat = idm(zc[:, 0], zc[:, 1])
        l_pred = F.mse_loss(zs.float(), tgt)
        l_agent = F.mse_loss(ags.float(), b["agent"][:, 2:]) * 100.0
        ok = b["hist_ok"]
        l_idm = F.mse_loss(a_hat.float()[ok], b["hist_actions"][ok]) * 10.0 if ok.any() else zs.new_zeros(())
        l_var, l_cov, std = vicreg(zc[:, 1].float(), args.var_gamma)
        loss = l_pred + l_agent + l_idm + jc["var_weight"] * l_var + jc["cov_weight"] * l_cov
        return loss, {"pred": l_pred.item(), "agent": l_agent.item(), "idm": l_idm.item(), "var": l_var.item(),
                      "cov": l_cov.item(), "std": std.item()}

    # ---------------------------------------------------------------- 1. JEPA training
    step, t0, best = 0, time.perf_counter(), float("inf")
    for ep in range(1, args.epochs + 1):
        enc.train(); pred.train(); idm.train()
        perm = np.random.permutation(len(tr))
        acc = {}
        for i in range(spe):
            b = batch(tr[perm[i * bs:(i + 1) * bs]], augment=True)
            loss, logs = losses(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            m = jc["ema"][1] - (jc["ema"][1] - jc["ema"][0]) * 0.5 * (1 + math.cos(math.pi * step / total))
            with torch.no_grad():
                for pe, pt in zip(enc.parameters(), enc_ema.parameters()):
                    pt.lerp_(pe, 1 - m)
            step += 1
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v / spe
        enc.eval(); pred.eval(); idm.eval()
        vacc, nv = {}, 0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                b = batch(va[i:i + 256])
                _, logs = losses(b)
                for k, v in logs.items():
                    vacc[k] = vacc.get(k, 0.0) + v * len(b["img"])
                nv += len(b["img"])
        vacc = {k: v / nv for k, v in vacc.items()}
        print(f"Epoch {ep:02d} | train pred {acc['pred']:.4f} agent {acc['agent']:.4f} idm {acc['idm']:.4f} var {acc['var']:.3f} "
              f"cov {acc['cov']:.3f} std {acc['std']:.3f} | val pred {vacc['pred']:.4f} idm {vacc['idm']:.4f} std {vacc['std']:.3f} | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    # the last epoch is kept: the validation prediction loss is not comparable across epochs (the targets move)

    # ---------------------------------------------------------------- 2. frozen features + pose probe (measurement)
    enc.eval()
    feats = encode_frames(enc, imgs, dev)                                       # (N, 144, C) float16, CPU
    probe = PoseProbe(n_tok=N, feat_dim=C).to(dev)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    frames_tr = np.unique(tr[:, 1])
    pe = jc["probe_epochs"]
    psched = torch.optim.lr_scheduler.CosineAnnealingLR(popt, pe * math.ceil(len(frames_tr) / 256))
    blk_np = np.concatenate([norm_xy(st[:, 2:4]), st[:, 4:5]], 1).astype(np.float32)
    for e in range(pe):
        perm = np.random.permutation(frames_tr)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            z = feats[idx].to(dev).float()
            bl = torch.from_numpy(blk_np[idx]).to(dev)
            tgt = torch.stack([bl[:, 0], bl[:, 1], torch.sin(bl[:, 2]), torch.cos(bl[:, 2])], -1)
            loss = F.mse_loss(probe(z), tgt)
            popt.zero_grad(set_to_none=True)
            loss.backward()
            popt.step()
            psched.step()
    probe.eval()

    def probe_err(z, bl):
        with torch.no_grad():
            pp = torch.cat([pose4_to_xyth(probe(z[i:i + 512].to(dev).float())) for i in range(0, len(z), 512)])
        e = pose_errors(pp[:, None], torch.as_tensor(bl, device=dev)[:, None])
        return float(e["pos_px"].mean()), float(e["ang_deg"].mean())

    frames_va = np.unique(va[:, 1])
    probe_report = {"val_frames_vs_fitted_pose": probe_err(feats[frames_va], blk_np[frames_va])}
    if os.path.exists(SYNTH):
        sy = np.load(SYNTH)
        bl_true = np.concatenate([norm_xy(sy["block"][:, :2]), sy["block"][:, 2:3]], 1).astype(np.float32)
        for k in ("clean", "av1"):
            probe_report[f"sim_frames_{k}_vs_true_pose"] = probe_err(encode_frames(enc, sy[k], dev), bl_true)
    print("\nĐầu dò vị trí khối T trên đặc trưng tự học (vị trí px, góc °):")
    for k, (a, g) in probe_report.items():
        print(f"  {k:<36} {a:6.2f} px | {g:5.2f}°")

    ckpt = {"encoder": enc.state_dict(), "encoder_hparams": enc.hparams, "predictor": pred.state_dict(), "hparams": pred.hparams,
            "norm": FeatureNorm(feat_dim=C).state_dict(), "probe": probe.state_dict(), "epoch": args.epochs,
            "data": "lerobot/pusht (Hugging Face) only", "var_gamma": args.var_gamma}
    torch.save(ckpt, args.out)

    # ---------------------------------------------------------------- 3. imagination accuracy on held-out episodes
    def get(rows):
        b = batch(rows)
        b["z"] = feats[torch.from_numpy(rows.reshape(-1))].view(len(rows), rows.shape[1], N, C).to(dev).float()
        return b

    def predict_poses(b, actions=None):
        with torch.no_grad():
            zs, _ = pred.rollout(b["z"][:, 0], b["z"][:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"] if actions is None else actions)
            B = zs.shape[0]
            return pose4_to_xyth(probe(zs.reshape(B * K, N, C))).view(B, K, 3)

    dyn_path = cfg["dynamics"]["save_path"]
    dyn = load_dynamics(dyn_path, dev) if os.path.exists(dyn_path) else None
    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(va))
    acc = {k: [] for k in ("static", "probe_true", "wm", "wm_shuffled", "state_dyn", "moving")}
    with torch.no_grad():
        for i in range(0, len(va), 256):
            sel = np.arange(i, min(len(va), i + 256))
            b = get(va[sel])
            truth = b["block"][:, 2:]
            B = len(sel)
            acc["static"].append(pose_errors(b["block"][:, 1:2].expand(-1, K, -1), truth))
            pt = pose4_to_xyth(probe(b["z"][:, 2:].reshape(B * K, N, C))).view(B, K, 3)
            acc["probe_true"].append(pose_errors(pt, truth))
            acc["wm"].append(pose_errors(predict_poses(b), truth))
            acc["wm_shuffled"].append(pose_errors(predict_poses(b, batch(va[shuf[sel]])["actions"]), truth))
            if dyn is not None:
                vel = b["agent"][:, 1] - b["agent_prev_step"]
                dyn_b, _ = dyn.rollout(b["block"][:, 1], b["agent"][:, 1], vel, b["actions"])
                acc["state_dyn"].append(pose_errors(dyn_b[:, S - 1::S], truth))
            acc["moving"].append(torch.linalg.norm(truth[:, -1, :2] - b["block"][:, 1, :2], dim=-1) * 256 > 5)
    moving = torch.cat(acc.pop("moving"))
    names = {"static": "Đoán đứng yên", "probe_true": "Đầu dò trên ảnh thật (sàn)", "wm": "WM ảnh tự học (tưởng tượng)",
             "wm_shuffled": "WM ảnh, hành động xáo trộn", "state_dyn": "Động lực học trạng thái (cận trên)"}
    report = {"data": ckpt["data"], "num_windows": len(va), "num_moving": int(moving.sum()),
              "horizons_steps": [S * (k + 1) for k in range(K)], "probe": probe_report, "models": {}}
    for k, lst in acc.items():
        if not lst:
            continue
        pos = torch.cat([e["pos_px"] for e in lst])
        ang = torch.cat([e["ang_deg"] for e in lst])
        report["models"][names[k]] = {"pos_px": pos.mean(0).tolist(), "ang_deg": ang.mean(0).tolist(),
                                      "pos_px_moving": pos[moving].mean(0).tolist(), "ang_deg_moving": ang[moving].mean(0).tolist()}

    # ---------------------------------------------------------------- 4. counterfactual test in the simulator
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    u = env.unwrapped
    mv_idx = np.nonzero(moving.cpu().numpy())[0]
    pick = mv_idx[np.linspace(0, len(mv_idx) - 1, min(10 if args.smoke else 150, len(mv_idx))).astype(int)]
    real, pr_wm = [], []
    for i in pick:
        b = get(va[[i]])
        ag = b["agent"][0, 1].cpu().numpy() * 256 + 256
        bl = b["block"][0, 1].cpu().numpy()
        blk = [bl[0] * 256 + 256, bl[1] * 256 + 256, bl[2]]
        demo = b["actions"][0].cpu().numpy() * 256 + 256
        ang = rng.uniform(-np.pi, np.pi)
        seqs_cf = [demo, np.repeat(ag[None], K * S, 0), demo + 25.0 * np.array([np.cos(ang), np.sin(ang)])]
        outs = []
        for a_seq in seqs_cf:
            env.reset(seed=0)
            u._set_state(list(ag) + blk)
            u._set_state(list(ag) + blk)
            for a in a_seq:
                u.step(np.clip(a, 0, 512))
            outs.append(np.array(u._get_info()["block_pose"])[:2])
        real.append(outs)
        acts = torch.from_numpy((np.stack(seqs_cf) - 256.0) / 256.0).float().to(dev)
        bb = {k: (v.expand(3, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in b.items()}
        pr_wm.append(predict_poses(bb, acts)[:, -1, :2].cpu().numpy() * 256 + 256)
    real, pr = np.array(real), np.array(pr_wm)
    ratios, errs = [], []
    for j in (1, 2):
        dr, dp = real[:, 0] - real[:, j], pr[:, 0] - pr[:, j]
        ok = np.linalg.norm(dr, axis=1) > 5
        ratios += list(np.linalg.norm(dp[ok], axis=1) / np.linalg.norm(dr[ok], axis=1))
        errs += list(np.linalg.norm(pr[:, j] - real[:, j], axis=1))
    report["counterfactual"] = {"sensitivity_median": float(np.median(ratios)), "final_pos_err_px_counterfactual_median": float(np.median(errs))}
    with open(os.path.join(args.results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    hs = report["horizons_steps"]
    print("\n" + "=" * 100)
    print(f"JEPA WORLD MODEL (bộ mã hoá tự học) – {len(va):,} cửa sổ kiểm tra ({report['num_moving']:,} có khối di chuyển > 5px)")
    print("=" * 100)
    print(f"{'Mô hình':<44}{'vị trí (px) tại bước ' + '/'.join(map(str, hs)):<36}{'góc (°) bước ' + str(hs[-1]):>16}")
    for n, r in report["models"].items():
        print(f"{n:<44}{' / '.join(f'{x:.1f}' for x in r['pos_px']):<36}{r['ang_deg'][-1]:>16.2f}")
    cf = report["counterfactual"]
    print(f"\nPhép thử phản thực tế: độ nhạy {cf['sensitivity_median'] * 100:.0f}% (lý tưởng 100%) | "
          f"sai vị trí cuối với hành động lạ {cf['final_pos_err_px_counterfactual_median']:.1f}px")
    print(f"Đã lưu {args.out} và {args.results_dir}/metrics.json")


if __name__ == "__main__":
    main()
