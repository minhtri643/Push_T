"""Stage E5c/E5d: vision World Model on a FROZEN pretrained encoder (V-JEPA 2 or DINOv2), lerobot/pusht only.

"Standing on the shoulders of giants" (transfer learning with a frozen backbone): the encoder is frozen and
never fine-tuned; only the action-conditioned latent predictor and the pose probe are trained on the
206 demonstrations. Same recipe as V-JEPA 2-AC and DINO-WM.

    python scripts/6_vision_world_model/train_frozen_encoder_wm.py --encoder vjepa2      # -> checkpoints/E5_vision/vjepa2_wm.pt, results/vision_world_models/E5d_vjepa2_frozen/
    python scripts/6_vision_world_model/train_frozen_encoder_wm.py --encoder dinov2      # same recipe with DINOv2 -> results/vision_world_models/E5c_dinov2_frozen/
    python scripts/6_vision_world_model/train_frozen_encoder_wm.py --encoder jepa --eval-only   # same measurements for the self-trained JEPA

Steps
 1. features of all 25,650 frames and of their anti-diagonal reflections (augmentation, as everywhere else
    in the pipeline), cached in data/cache/<encoder>_feats_hf.npy (float16, 8x8 tokens)
 2. pose probe on the frozen features: reads the block pose (x, y, theta), so that the World Model's
    imagination can be expressed as the change of the object position (dx, dy, dtheta) and FDE
 3. latent predictor: (z_{t-S}, z_t, pusher positions, S actions) -> z_{t+S}, rolled out K times,
    MSE in (normalised) feature space; checkpoint chosen by validation feature MSE (targets are fixed)
 4. evaluation on the 20 held-out episodes (same windows as train_jepa_wm.py): pose error and error on the CHANGE
    of pose at 4/8/12/16 steps, FDE, shuffled actions, simulator counterfactual test, and probe
    generalisation on simulator frames with known pose (expert episodes + the 95 benchmark starts).
    The simulator is a measuring instrument only.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import gc
import importlib.util
import json
import math
import time

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr

from pusht_wm_dit.data.dynamics_data import episode_split
from pusht_wm_dit.envs.pusht_sim import PushTSim
from pusht_wm_dit.models.dynamics import load_dynamics, pose_errors
from pusht_wm_dit.models.latent_world_model import FeatureNorm, LatentPredictor, PoseProbe, pose4_to_xyth

SYNTH = "results/pose_recovery/synthetic_frames.npz"
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "5_evaluation", "benchmark.py")


def norm_xy(p):
    return (p - 256.0) / 256.0


def reflect_xy(p):
    """Anti-diagonal reflection through the goal centre, normalised coordinates: (x, y) -> (-y, -x)."""
    return np.stack([-p[..., 1], -p[..., 0]], -1)


def reflect_block(b):
    """(x, y, theta) normalised -> reflected pose, theta -> pi/2 - theta."""
    return np.concatenate([reflect_xy(b[..., :2]), np.pi / 2 - b[..., 2:3]], -1)


def reflect_img(img):
    """uint8 (N, 96, 96, 3): transpose + flip both axes (matches reflect_xy, verified on renders)."""
    return np.ascontiguousarray(np.flip(np.swapaxes(img, -3, -2), axis=(-3, -2)))


def windows(seqs, split, S, K):
    rows = []
    for off, n, sp in seqs:
        if sp != split:
            continue
        for t in range(0, n - K * S):
            rows.append([off + max(t - S, 0), off + t] + [off + t + (k + 1) * S for k in range(K)])
    return np.array(rows, dtype=np.int64)


def pose4(bl):
    return torch.stack([bl[..., 0], bl[..., 1], torch.sin(bl[..., 2]), torch.cos(bl[..., 2])], -1)


def change_errors(pred, start, truth, cur):
    """Error on the CHANGE of pose: |(pred_k - start) - (truth_k - cur)| in px and deg. (B, K, 3) / (B, 3)."""
    dp = (pred[..., :2] - start[:, None, :2]) - (truth[..., :2] - cur[:, None, :2])
    da = (pred[..., 2] - start[:, None, 2]) - (truth[..., 2] - cur[:, None, 2])
    return torch.linalg.norm(dp, dim=-1) * 256.0, torch.rad2deg(torch.abs(torch.atan2(torch.sin(da), torch.cos(da))))


def benchmark_start_frames():
    """Frames and true block poses of the 95 benchmark starts (scripts/5_evaluation/benchmark.py)."""
    spec = importlib.util.spec_from_file_location("bench07", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sim = PushTSim(max_steps=300)
    frames, poses, suites = [], [], []
    for suite, _, init in mod.starts():
        obs, _ = sim.reset(seed=init["seed"])
        if "init_state" in init:
            s = init["init_state"]
            obs = sim.set_state(s[:2], s[2:5])
        frames.append(obs["pixels"])
        poses.append(obs["block_pose"])
        suites.append(suite)
    sim.close()
    return np.stack(frames), np.stack(poses).astype(np.float64), np.array(suites)


@torch.no_grad()
def probe_errors(probe, norm, z, block_px, dev):
    """z float16 (N, T, C) CPU, block_px (N, 3) world px / rad -> position (px) and angle (deg) errors per frame."""
    pp = torch.cat([pose4_to_xyth(probe(norm(z[i:i + 512].to(dev)))) for i in range(0, len(z), 512)]).cpu().numpy()
    pos = np.linalg.norm(pp[:, :2] * 256 + 256 - block_px[:, :2], axis=1)
    ang = np.rad2deg(np.abs((pp[:, 2] - block_px[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    return pos, ang


def summary(pos, ang):
    return {"pos_mean_px": float(pos.mean()), "pos_median_px": float(np.median(pos)), "pos_p90_px": float(np.percentile(pos, 90)),
            "ang_mean_deg": float(ang.mean()), "ang_median_deg": float(np.median(ang)), "n": int(len(pos))}


def probe_generalisation(encode, probe, norm, dev, val_feats=None, val_block_px=None):
    """Pose read-out on HF validation frames and on simulator frames whose pose is exactly known."""
    rep = {}
    if val_feats is not None:
        rep["hf_val_frames_vs_fitted_pose"] = summary(*probe_errors(probe, norm, val_feats, val_block_px, dev))
    if os.path.exists(SYNTH):
        sy = np.load(SYNTH)
        for k in ("clean", "av1"):
            rep[f"sim_expert_frames_{k}"] = summary(*probe_errors(probe, norm, encode(sy[k]), sy["block"].astype(np.float64), dev))
    fr, po, su = benchmark_start_frames()
    pos, ang = probe_errors(probe, norm, encode(fr), po, dev)
    rep["benchmark_starts_all95"] = summary(pos, ang)
    for s in ("seeds50", "hard20", "random25"):
        rep[f"benchmark_starts_{s}"] = summary(pos[su == s], ang[su == s])
    return rep


def print_generalisation(rep, title):
    print(f"\n{title}")
    for k, r in rep.items():
        print(f"  {k:<34} vị trí TB {r['pos_mean_px']:6.1f} px (trung vị {r['pos_median_px']:5.1f}, p90 {r['pos_p90_px']:5.1f}) | "
              f"góc TB {r['ang_mean_deg']:5.1f}° | n={r['n']}")


def eval_jepa_only(cfg, dev):
    """Same measurements for the self-trained JEPA (train_jepa_wm.py), for a uniform table: probe generalisation, and
    the imagination error on the CHANGE of pose and FDE (train_jepa_wm.py reports only the absolute pose error).
    Same val windows, same probe frames and same shuffled-action permutation as main()."""
    from pusht_wm_dit.models.jepa_encoder import encode_frames
    from pusht_wm_dit.models.latent_world_model import load_vision_wm
    bundle = load_vision_wm(cfg["jepa_wm"]["save_path"], dev)
    enc, pred, probe, norm = bundle["encoder"], bundle["predictor"], bundle["probe"], bundle["norm"]
    root = zarr.open(cfg["data"]["zarr_path"], mode="r")
    st = np.asarray(root["data/state"][:], np.float64)
    ac = np.asarray(root["data/action"][:], np.float64)
    ends = root["meta/episode_ends"][:]
    _, val_ids = episode_split(len(ends), cfg["data"]["val_ratio"], cfg["seed"])
    encode = lambda fr: encode_frames(enc, fr, dev)  # noqa: E731
    feats = encode(np.asarray(root["data/img"][:]))                              # (N, 144, 64) float16, CPU
    T, C = feats.shape[1:]
    S, K = pred.skip, cfg["jepa_wm"]["rollout_steps"]
    seqs, s0 = [], 0
    for ep, end in enumerate(ends):
        seqs.append((s0, int(end - s0), "val" if ep in set(val_ids) else "train"))
        s0 = int(end)
    va = windows(seqs, "val", S, K)
    frames_va = np.unique(va[:, 1])
    report = {"data": "lerobot/pusht only; self-trained JEPA encoder (train_jepa_wm.py)", "encoder": "jepa", "num_windows": len(va),
              "horizons_steps": [S * (k + 1) for k in range(K)],
              "probe_generalisation": probe_generalisation(encode, probe, norm, dev, feats[frames_va], st[frames_va, 2:5]),
              "models": {}}

    agent = torch.from_numpy(norm_xy(st[:, :2]).astype(np.float32)).to(dev)
    block = torch.from_numpy(np.concatenate([norm_xy(st[:, 2:4]), st[:, 4:5]], 1).astype(np.float32)).to(dev)
    action = torch.from_numpy(norm_xy(ac).astype(np.float32)).to(dev)

    def get(rows):
        r = torch.from_numpy(rows).to(dev)
        return {"z": norm(feats[torch.from_numpy(rows.reshape(-1))].view(len(rows), rows.shape[1], T, C).to(dev)),
                "agent": agent[r], "block": block[r], "actions": action[r[:, 1:2] + torch.arange(K * S, device=dev)[None]]}

    def predict(b, actions=None):
        zs, _ = pred.rollout(b["z"][:, 0], b["z"][:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"] if actions is None else actions)
        return pose4_to_xyth(probe(zs.reshape(-1, T, C))).view(len(zs), K, 3)

    names = {"probe_true": "Đầu dò trên ảnh thật (sàn)", "wm": "WM ảnh jepa (tưởng tượng)", "wm_shuffled": "WM ảnh jepa, hành động xáo trộn"}
    acc = {k: {"pos": [], "ang": [], "dpos": [], "dang": []} for k in names}
    moving_l = []
    shuf = np.random.default_rng(0).permutation(len(va))
    with torch.no_grad():
        for i in range(0, len(va), 256):
            sel = np.arange(i, min(len(va), i + 256))
            b = get(va[sel])
            truth, cur = b["block"][:, 2:], b["block"][:, 1]
            own = pose4_to_xyth(probe(b["z"][:, 1]))
            preds = {"probe_true": (pose4_to_xyth(probe(b["z"][:, 2:].reshape(-1, T, C))).view(len(sel), K, 3), own),
                     "wm": (predict(b), own), "wm_shuffled": (predict(b, get(va[shuf[sel]])["actions"]), own)}
            for k, (pr, s_own) in preds.items():
                e = pose_errors(pr, truth)
                dp, da = change_errors(pr, s_own, truth, cur)
                acc[k]["pos"].append(e["pos_px"]); acc[k]["ang"].append(e["ang_deg"])
                acc[k]["dpos"].append(dp); acc[k]["dang"].append(da)
            moving_l.append(torch.linalg.norm(truth[:, -1, :2] - cur[:, :2], dim=-1) * 256 > 5)
    moving = torch.cat(moving_l)
    report["num_moving"] = int(moving.sum())
    for k, d in acc.items():
        pos, ang, dpos, dang = (torch.cat(d[x]) for x in ("pos", "ang", "dpos", "dang"))
        fde = torch.sqrt((pos[:, -1] / 256.0) ** 2 + torch.deg2rad(ang[:, -1]) ** 2)
        report["models"][names[k]] = {"pos_px": pos.mean(0).tolist(), "ang_deg": ang.mean(0).tolist(),
                                      "change_pos_px": dpos.mean(0).tolist(), "change_ang_deg": dang.mean(0).tolist(),
                                      "pos_px_moving": pos[moving].mean(0).tolist(), "ang_deg_moving": ang[moving].mean(0).tolist(),
                                      "fde_last_step": float(fde.mean())}
    os.makedirs("results/vision_world_models/E5b_jepa", exist_ok=True)
    with open("results/vision_world_models/E5b_jepa/metrics_uniform.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJEPA tự học (train_jepa_wm.py) – {len(va):,} cửa sổ kiểm tra ({report['num_moving']:,} có khối di chuyển > 5px)")
    for n, r in report["models"].items():
        print(f"  {n:<40} vị trí {' / '.join(f'{x:.1f}' for x in r['pos_px'])} px | độ thay đổi "
              f"{' / '.join(f'{x:.1f}' for x in r['change_pos_px'])} px | góc bước 16 {r['ang_deg'][-1]:.2f}° | FDE {r['fde_last_step']:.3f}")
    print_generalisation(report["probe_generalisation"], "Đầu dò tư thế trên đặc trưng JEPA tự học (train_jepa_wm.py):")
    print("\nĐã lưu results/vision_world_models/E5b_jepa/metrics_uniform.json")


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    fc = cfg["frozen_wm"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", choices=("vjepa2", "dinov2", "jepa"), default="vjepa2")
    p.add_argument("--epochs", type=int, default=fc["epochs"])
    p.add_argument("--probe-epochs", type=int, default=fc["probe_epochs"])
    p.add_argument("--eval-only", action="store_true", help="only the probe-generalisation measurement (required for --encoder jepa)")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    args = p.parse_args()
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if args.encoder == "jepa":
        eval_jepa_only(cfg, dev)
        return

    from pusht_wm_dit.models.frozen_encoders import get_frozen_encoder
    key = {"vjepa2": "vjepa2_wm", "dinov2": "dinov2_hf_wm"}[args.encoder]
    save_path = cfg[key]["save_path"]
    out_dir = {"vjepa2_wm": "results/vision_world_models/E5d_vjepa2_frozen",
               "dinov2_hf_wm": "results/vision_world_models/E5c_dinov2_frozen"}[key]
    feats_path = f"data/cache/{args.encoder}_feats_hf.npy"
    os.makedirs(out_dir, exist_ok=True)
    S, K = fc["skip"], fc["rollout_steps"]

    # ---------------------------------------------------------------- data (lerobot/pusht only) + reflected copies
    root = zarr.open(cfg["data"]["zarr_path"], mode="r")
    print("data:", root.attrs.get("source", cfg["data"]["zarr_path"]), "| encoder (frozen):", args.encoder)
    imgs = np.asarray(root["data/img"][:])
    st = np.asarray(root["data/state"][:], np.float64)
    ac = np.asarray(root["data/action"][:], np.float64)
    ends = root["meta/episode_ends"][:]
    N = len(imgs)
    train_ids, val_ids = episode_split(len(ends), cfg["data"]["val_ratio"], cfg["seed"])
    seqs, start = [], 0
    for ep, end in enumerate(ends):
        sp = "val" if ep in set(val_ids) else "train"
        seqs.append((start, int(end - start), sp))
        if sp == "train":
            seqs.append((N + start, int(end - start), "train"))      # reflected copy (training only)
        start = int(end)
    agent0 = norm_xy(st[:, :2])
    block0 = np.concatenate([norm_xy(st[:, 2:4]), st[:, 4:5]], 1)
    act0 = norm_xy(ac)
    D = {"agent": np.concatenate([agent0, reflect_xy(agent0)]).astype(np.float32),
         "block": np.concatenate([block0, reflect_block(block0)]).astype(np.float32),
         "action": np.concatenate([act0, reflect_xy(act0)]).astype(np.float32)}

    enc = get_frozen_encoder(args.encoder, dev)
    C, T = enc.feat_dim, enc.n_tok
    meta = {"encoder": args.encoder, "frames": int(N), "reflected_copy": True, "zarr_path": cfg["data"]["zarr_path"], "n_tok": T, "feat_dim": C}
    ok = os.path.exists(feats_path) and os.path.exists(feats_path + ".json") and json.load(open(feats_path + ".json")) == meta
    if not ok:
        tmp = feats_path[:-4] + "_tmp.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(2 * N, T, C))
        t0 = time.perf_counter()
        for i in range(0, N, 512):
            chunk = imgs[i:i + 512]
            mm[i:i + len(chunk)] = enc.encode(chunk).numpy()
            mm[N + i:N + i + len(chunk)] = enc.encode(reflect_img(chunk)).numpy()
            if (i // 512) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  features {i + len(chunk):,}/{N:,} frames (+ reflections) | {el:.0f}s, ~{el / (i + len(chunk)) * (N - i - len(chunk)):.0f}s left", flush=True)
        mm.flush()
        del mm
        gc.collect()
        os.replace(tmp, feats_path)
        with open(feats_path + ".json", "w") as f:
            json.dump(meta, f)
        print(f"features cached: {feats_path} ({2 * N:,} x {T} x {C}, {time.perf_counter() - t0:.0f}s)")
    t0 = time.perf_counter()
    D["feats"] = np.load(feats_path)
    print(f"features loaded in {time.perf_counter() - t0:.0f}s: {D['feats'].shape}, {D['feats'].nbytes / 1e9:.1f} GB")

    tr, va = windows(seqs, "train", S, K), windows(seqs, "val", S, K)
    if args.smoke:
        tr, va = tr[::40], va[::20]
        args.epochs, args.probe_epochs = 1, 1
    print(f"train windows {len(tr):,} (incl. reflections) | val windows {len(va):,}")

    cand = np.unique(tr[:, 1])
    sample = torch.from_numpy(D["feats"][np.sort(np.random.choice(cand, min(4000, len(cand)), replace=False))]).float().view(-1, C)
    norm = FeatureNorm(sample.mean(0), sample.std(0) + 1e-3, feat_dim=C).to(dev)

    def get(rows):
        z = torch.from_numpy(D["feats"][rows.reshape(-1)]).view(len(rows), rows.shape[1], T, C)
        act_idx = rows[:, 1:2] + np.arange(K * S)[None]
        g = lambda a, idx: torch.from_numpy(D[a][idx]).to(dev)  # noqa: E731
        return {"z": z.to(dev, non_blocking=True), "agent": g("agent", rows), "block": g("block", rows),
                "actions": g("action", act_idx), "agent_prev_step": g("agent", np.maximum(rows[:, 1] - 1, 0))}

    # ---------------------------------------------------------------- 1. pose probe (read-out of the block pose)
    probe = PoseProbe(n_tok=T, feat_dim=C).to(dev)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    frames_tr = np.unique(tr[:, 1])
    psched = torch.optim.lr_scheduler.CosineAnnealingLR(popt, args.probe_epochs * math.ceil(len(frames_tr) / 256))
    t0 = time.perf_counter()
    for ep in range(args.probe_epochs):
        perm = np.random.permutation(frames_tr)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            z = norm(torch.from_numpy(D["feats"][idx]).to(dev))
            loss = F.mse_loss(probe(z), pose4(torch.from_numpy(D["block"][idx]).to(dev)))
            popt.zero_grad(set_to_none=True)
            loss.backward()
            popt.step()
            psched.step()
    probe.eval()
    for q in probe.parameters():
        q.requires_grad_(False)
    frames_va = np.unique(va[:, 1])
    val_block_px = np.concatenate([st[frames_va, 2:4], st[frames_va, 4:5]], 1)
    pv, av = probe_errors(probe, norm, torch.from_numpy(D["feats"][frames_va]), val_block_px, dev)
    print(f"pose probe ({args.probe_epochs} epochs, {time.perf_counter() - t0:.0f}s): HF val frames {pv.mean():.2f} px (median {np.median(pv):.2f}) | {av.mean():.2f} deg")

    # ---------------------------------------------------------------- 2. action-conditioned latent predictor
    pred = LatentPredictor(skip=S, dim=fc["dim"], depth=fc["depth"], heads=fc["heads"], n_tok=T, feat_dim=C).to(dev)
    opt = torch.optim.AdamW(pred.parameters(), lr=fc["lr"], weight_decay=1e-4)
    bs = fc["batch_size"]
    spe = math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=fc["lr"], total_steps=args.epochs * spe, pct_start=0.05)
    print(f"predictor {sum(q.numel() for q in pred.parameters()) / 1e6:.2f}M params | probe {sum(q.numel() for q in probe.parameters()) / 1e6:.2f}M")

    def wm_loss(b):
        zn = norm(b["z"])
        zs, ags = pred.rollout(zn[:, 0], zn[:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"])
        lf = F.mse_loss(zs.float(), zn[:, 2:])
        la = F.mse_loss(ags.float(), b["agent"][:, 2:]) * 100.0
        return lf + la, lf.item(), la.item()

    @torch.no_grad()
    def predict_poses(b, actions=None):
        zn = norm(b["z"])
        zs, _ = pred.rollout(zn[:, 0], zn[:, 1], b["agent"][:, 0], b["agent"][:, 1], b["actions"] if actions is None else actions)
        B = zs.shape[0]
        return pose4_to_xyth(probe(zs.reshape(B * K, T, C).float())).view(B, K, 3)

    best, t0 = float("inf"), time.perf_counter()
    for ep in range(1, args.epochs + 1):
        pred.train()
        perm = np.random.permutation(len(tr))
        tf = ta = 0.0
        for i in range(spe):
            b = get(tr[perm[i * bs:(i + 1) * bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, lf, la = wm_loss(b)
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
                b = get(va[i:i + 256])
                vl += wm_loss(b)[1] * len(b["z"])
                vpos.append(pose_errors(predict_poses(b), b["block"][:, 2:])["pos_px"])
        vl /= len(va)
        vpos = torch.cat(vpos).mean(0).tolist()
        is_best = vl < best
        if is_best:
            best = vl
            torch.save({"encoder_type": args.encoder, "predictor": pred.state_dict(), "hparams": pred.hparams, "norm": norm.state_dict(),
                        "probe": probe.state_dict(), "epoch": ep, "val_feature_mse": vl, "reflect_augmentation": True,
                        "data": f"lerobot/pusht only; encoder {args.encoder} frozen (pretrained, never fine-tuned)"}, save_path)
        print(f"Epoch {ep:02d} | train feat {tf / spe:.4f} agent {ta / spe:.4f} | val feat {vl:.4f} | "
              f"val block pos err (probe) k=4/8/12/16: {' / '.join(f'{x:.1f}' for x in vpos)} px{' *' if is_best else ''} | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- 3. evaluation on the held-out episodes
    ck = torch.load(save_path, map_location=dev, weights_only=False)
    pred.load_state_dict(ck["predictor"])
    pred.eval()
    dyn_path = cfg["dynamics"]["save_path"]
    dyn = load_dynamics(dyn_path, dev) if os.path.exists(dyn_path) else None
    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(va))
    names = {"static": "Đoán đứng yên", "probe_true": "Đầu dò trên ảnh thật (sàn)", "wm": f"WM ảnh {args.encoder} (tưởng tượng)",
             "wm_shuffled": f"WM ảnh {args.encoder}, hành động xáo trộn", "state_dyn": "Động lực học trạng thái (cận trên)"}
    acc = {k: {"pos": [], "ang": [], "dpos": [], "dang": []} for k in names}
    moving_l = []
    with torch.no_grad():
        for i in range(0, len(va), 256):
            sel = np.arange(i, min(len(va), i + 256))
            b = get(va[sel])
            truth, cur = b["block"][:, 2:], b["block"][:, 1]
            B = len(sel)
            zn = norm(b["z"])
            own = pose4_to_xyth(probe(zn[:, 1].float()))                        # the vision model's own reading of "now"
            preds = {"static": (cur[:, None].expand(-1, K, -1), cur),
                     "probe_true": (pose4_to_xyth(probe(zn[:, 2:].reshape(B * K, T, C).float())).view(B, K, 3), own),
                     "wm": (predict_poses(b), own),
                     "wm_shuffled": (predict_poses(b, get(va[shuf[sel]])["actions"]), own)}
            if dyn is not None:
                vel = b["agent"][:, 1] - b["agent_prev_step"]
                dyn_b, _ = dyn.rollout(cur, b["agent"][:, 1], vel, b["actions"])
                preds["state_dyn"] = (dyn_b[:, S - 1::S], cur)
            for k, (pr, s0) in preds.items():
                e = pose_errors(pr, truth)
                dp, da = change_errors(pr, s0, truth, cur)
                acc[k]["pos"].append(e["pos_px"]); acc[k]["ang"].append(e["ang_deg"])
                acc[k]["dpos"].append(dp); acc[k]["dang"].append(da)
            moving_l.append(torch.linalg.norm(truth[:, -1, :2] - cur[:, :2], dim=-1) * 256 > 5)
    moving = torch.cat(moving_l)
    report = {"data": ck["data"], "encoder": args.encoder, "epoch": ck["epoch"], "num_windows": len(va), "num_moving": int(moving.sum()),
              "horizons_steps": [S * (k + 1) for k in range(K)], "models": {}}
    for k, d in acc.items():
        if not d["pos"]:
            continue
        pos, ang, dpos, dang = (torch.cat(d[x]) for x in ("pos", "ang", "dpos", "dang"))
        # course FDE at the last step, normalised units: sqrt(dx^2 + dy^2 + dtheta^2)
        fde = torch.sqrt((pos[:, -1] / 256.0) ** 2 + torch.deg2rad(ang[:, -1]) ** 2)
        report["models"][names[k]] = {"pos_px": pos.mean(0).tolist(), "ang_deg": ang.mean(0).tolist(),
                                      "change_pos_px": dpos.mean(0).tolist(), "change_ang_deg": dang.mean(0).tolist(),
                                      "pos_px_moving": pos[moving].mean(0).tolist(), "ang_deg_moving": ang[moving].mean(0).tolist(),
                                      "fde_last_step": float(fde.mean())}

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
    env.close()
    real, pr = np.array(real), np.array(pr_wm)
    ratios, errs = [], []
    for j in (1, 2):
        dr, dp = real[:, 0] - real[:, j], pr[:, 0] - pr[:, j]
        okm = np.linalg.norm(dr, axis=1) > 5
        ratios += list(np.linalg.norm(dp[okm], axis=1) / np.linalg.norm(dr[okm], axis=1))
        errs += list(np.linalg.norm(pr[:, j] - real[:, j], axis=1))
    report["counterfactual"] = {"sensitivity_median": float(np.median(ratios)), "final_pos_err_px_counterfactual_median": float(np.median(errs)),
                                "states": int(len(pick))}

    # ---------------------------------------------------------------- 5. probe generalisation (unusual poses)
    report["probe_generalisation"] = probe_generalisation(lambda fr: enc.encode(fr), probe, norm, dev,
                                                          torch.from_numpy(D["feats"][frames_va]), val_block_px)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    hs = report["horizons_steps"]
    print("\n" + "=" * 112)
    print(f"WORLD MODEL ẢNH, BỘ MÃ HOÁ {args.encoder.upper()} ĐÓNG BĂNG – {len(va):,} cửa sổ kiểm tra ({report['num_moving']:,} có khối di chuyển > 5px)")
    print("=" * 112)
    print(f"{'Mô hình':<44}{'vị trí (px) bước ' + '/'.join(map(str, hs)):<32}{'độ thay đổi (px)':<28}{'góc bước 16':>12}{'FDE':>8}")
    for n, r in report["models"].items():
        print(f"{n:<44}{' / '.join(f'{x:.1f}' for x in r['pos_px']):<32}{' / '.join(f'{x:.1f}' for x in r['change_pos_px']):<28}"
              f"{r['ang_deg'][-1]:>11.2f}°{r['fde_last_step']:>8.3f}")
    cf = report["counterfactual"]
    print(f"\nPhép thử phản thực tế: độ nhạy {cf['sensitivity_median'] * 100:.0f}% (lý tưởng 100%) | "
          f"sai vị trí cuối với hành động lạ {cf['final_pos_err_px_counterfactual_median']:.1f}px")
    print_generalisation(report["probe_generalisation"], f"Đầu dò tư thế trên đặc trưng {args.encoder} (đóng băng):")
    print(f"\nĐã lưu {save_path} và {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
