"""Pipeline 3 (World Model evaluation): does the World Model predict how the object moves under the robot's actions?

Project brief: "Từ trạng thái hiện tại và hành động của robot, mô hình dự đoán sự thay đổi vị trí vật thể."

All numbers are on the 20 held-out validation episodes (2,235 windows, horizon K=16):
1. Open-loop multi-step error (block position px, angle deg) at every horizon step, for
   - "Đoán đứng yên" baseline (block never moves),
   - the original latent transition of the World Model (frame -> s_t -> T(s_t, a_t) -> decoder),
   - every trained dynamics model (checkpoints/dynamics_*.pt), started from the true state and
     from the state read off the image by the World Model encoder ("từ ảnh"),
   - the same models fed actions from a different window ("hành động xáo trộn"): if the error
     does not grow, the model ignores the action.
2. Displacement error: |(p_hat_k - p_hat_0) - (p_k - p_0)|, i.e. the error on the CHANGE of pose.
3. Counterfactual test in the simulator: from real validation states, run three action sequences
   (the demonstrated one, "stand still", and the demonstrated one shifted by 25 px) and compare
   how differently each model predicts the block to end up with how differently it really does.

Outputs: results/world_model_eval/E4_final/metrics.json, curves.png, trajectories.png
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
import glob
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import yaml

from pusht_wm_dit.data.dynamics_data import load_demo_sequences, make_windows
from pusht_wm_dit.data.pusht_dataset import PushTDataset
from pusht_wm_dit.models.dynamics import load_dynamics, pose_errors
from pusht_wm_dit.policy import load_models
from pusht_wm_dit.utils.metrics import create_tee_polygon
from pusht_wm_dit.utils.normalizer import normalize_image

KS = (1, 2, 4, 8, 12, 16)
OUT = "results/world_model_eval/E4_final"


def masked_mean(x, m):
    return ((x * m).sum(0) / m.sum(0).clamp_min(1)).cpu().numpy()


@torch.no_grad()
def latent_rollout(wm, s0, actions):
    s, out = s0, []
    for k in range(actions.shape[1]):
        s = wm.predict_next_state(s, actions[:, k])
        p4 = wm.decode(s)[0]
        out.append(torch.stack([p4[:, 0], p4[:, 1], torch.atan2(p4[:, 2], p4[:, 3])], -1))
    return torch.stack(out, 1)


def summarize(pred, start, V):
    """pred (N,K,3), start (N,3) = the model's own idea of the current pose."""
    e = pose_errors(pred, V["tgt_block"])
    true_disp = V["tgt_block"][..., :2] - V["block0"][:, None, :2]
    pred_disp = pred[..., :2] - start[:, None, :2]
    disp = torch.linalg.norm(pred_disp - true_disp, dim=-1) * 256.0
    m = V["mask"].float()
    mv = m * V["moved"].float()[:, None]
    return {
        "pos_px": masked_mean(e["pos_px"], m).tolist(),
        "ang_deg": masked_mean(e["ang_deg"], m).tolist(),
        "disp_px": masked_mean(disp, m).tolist(),
        "pos_px_moving": masked_mean(e["pos_px"], mv).tolist(),
        "ang_deg_moving": masked_mean(e["ang_deg"], mv).tolist(),
    }


def simulate(env, agent_px, block_px, actions_px):
    u = env.unwrapped
    env.reset(seed=0)
    st = list(agent_px) + list(block_px)
    u._set_state(st)
    u._set_state(st)
    for a in actions_px:
        u.step(np.clip(a, 0, 512))
    return np.array(u._get_info()["block_pose"], dtype=np.float64)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cf-states", type=int, default=150, help="validation states for the counterfactual test")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = cfg["dynamics"]["horizon"]
    os.makedirs(OUT, exist_ok=True)

    # ---- validation windows and images (same order) ----
    V = {k: v.to(device) for k, v in make_windows(load_demo_sequences(cfg["data"]["zarr_path"], "val",
                                                                       cfg["data"]["val_ratio"], cfg["seed"]), K).items()}
    ds = PushTDataset(cfg["data"]["zarr_path"], pred_horizon=K, action_horizon=K, split="val",
                      val_ratio=cfg["data"]["val_ratio"], seed=cfg["seed"])
    ts = np.array([t for t, _, _ in ds.indices])
    assert len(ts) == len(V["block0"]), "window / image order mismatch"
    N = len(ts)

    wm, _, _ = load_models(cfg, device)
    s0, start_img = [], []
    with torch.no_grad():
        for i in range(0, N, 256):
            fr = torch.from_numpy(normalize_image(ds.img_arr[ts[i:i + 256]])).float().to(device)
            ag = torch.from_numpy((ds.state_arr[ts[i:i + 256], :2] - 256.0) / 256.0).float().to(device)
            s = wm.encode(fr, ag)
            p4 = wm.decode(s)[0]
            s0.append(s)
            start_img.append(torch.stack([p4[:, 0], p4[:, 1], torch.atan2(p4[:, 2], p4[:, 3])], -1))
    s0, start_img = torch.cat(s0), torch.cat(start_img)
    # keep the decoded angle on the same branch as the (unwrapped) true angle
    d = start_img[:, 2] - V["block0"][:, 2]
    start_img[:, 2] = V["block0"][:, 2] + torch.atan2(torch.sin(d), torch.cos(d))

    g = torch.Generator(device="cpu").manual_seed(0)
    shuffled = V["actions"][torch.randperm(N, generator=g).to(device)]

    results, preds = {}, {}
    static = V["block0"][:, None, :].expand(-1, K, -1)
    results["Đoán đứng yên"] = summarize(static, V["block0"], V)
    with torch.no_grad():
        preds["WM cũ (latent)"] = latent_rollout(wm, s0, V["actions"])
        results["WM cũ (latent)"] = summarize(preds["WM cũ (latent)"], start_img, V)
        results["WM cũ, hành động xáo trộn"] = summarize(latent_rollout(wm, s0, shuffled), start_img, V)

    dyn_paths = sorted(set(glob.glob("checkpoints/final/dynamics_*.pt")))
    dyn_models = {}
    for path in dyn_paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        name = f"Động lực học ({ck.get('data', os.path.basename(path))})"
        if name in dyn_models:  # e.g. dynamics_best.pt is a copy of one of the variants
            continue
        model = load_dynamics(path, device)
        dyn_models[name] = model
        with torch.no_grad():
            pr = model.rollout(V["block0"], V["agent0"], V["vel0"], V["actions"])[0]
            preds[name] = pr
            results[name] = summarize(pr, V["block0"], V)
            results[name + ", từ ảnh"] = summarize(model.rollout(start_img, V["agent0"], V["vel0"], V["actions"])[0], start_img, V)
            results[name + ", hành động xáo trộn"] = summarize(model.rollout(V["block0"], V["agent0"], V["vel0"], shuffled)[0], V["block0"], V)

    # ---- counterfactual test in the simulator ----
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    moving = torch.nonzero(V["moved"] & V["mask"][:, -1]).squeeze(1).cpu().numpy()
    pick = moving[np.linspace(0, len(moving) - 1, min(args.cf_states, len(moving))).astype(int)]
    rng = np.random.default_rng(0)
    cf_real, cf_pred = [], {n: [] for n in ["WM cũ (latent)"] + list(dyn_models)}
    for i in pick:
        agent_px = V["agent0"][i].cpu().numpy() * 256 + 256
        block = V["block0"][i].cpu().numpy()
        block_px = np.array([block[0] * 256 + 256, block[1] * 256 + 256, block[2]])
        demo = V["actions"][i].cpu().numpy() * 256 + 256
        still = np.repeat(agent_px[None], K, 0)
        ang = rng.uniform(-np.pi, np.pi)
        shifted = demo + 25.0 * np.array([np.cos(ang), np.sin(ang)])
        seqs = [demo, still, shifted]
        cf_real.append([simulate(env, agent_px, block_px, a) for a in seqs])
        acts = torch.from_numpy((np.stack(seqs) - 256.0) / 256.0).float().to(device)
        with torch.no_grad():
            lat = latent_rollout(wm, s0[i:i + 1].expand(3, -1), acts)[:, -1].cpu().numpy()
            cf_pred["WM cũ (latent)"].append(lat)
            for n, m in dyn_models.items():
                b0 = V["block0"][i:i + 1].expand(3, -1)
                a0 = V["agent0"][i:i + 1].expand(3, -1)
                cf_pred[n].append(m.rollout(b0, a0, torch.zeros_like(a0), acts)[0][:, -1].cpu().numpy())
    cf_real = np.array(cf_real)  # (M, 3, 3) pixels / radians
    counterfactual = {}
    for n, pr in cf_pred.items():
        pr = np.array(pr)
        pr_px = np.concatenate([pr[..., :2] * 256 + 256, pr[..., 2:]], -1)
        ratios, errs = [], []
        for j in (1, 2):  # demo vs still, demo vs shifted
            d_real = cf_real[:, 0, :2] - cf_real[:, j, :2]
            d_pred = pr_px[:, 0, :2] - pr_px[:, j, :2]
            ok = np.linalg.norm(d_real, axis=1) > 5
            ratios += list(np.linalg.norm(d_pred[ok], axis=1) / np.linalg.norm(d_real[ok], axis=1))
            errs += list(np.linalg.norm(pr_px[:, j, :2] - cf_real[:, j, :2], axis=1))
        counterfactual[n] = {"sensitivity_median": float(np.median(ratios)),
                             "final_pos_err_px_counterfactual_actions": float(np.median(errs))}

    report = {"num_windows": N, "num_moving": int(V["moved"].sum()), "horizon": K,
              "models": results, "counterfactual": counterfactual, "counterfactual_states": int(len(pick))}
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 96)
    print(f"WORLD MODEL – {N} cửa sổ kiểm tra ({report['num_moving']} có khối di chuyển), sai số trung bình tại k = {KS}")
    print("=" * 96)
    print(f"{'Mô hình':<46}{'vị trí (px)':<30}{'góc (°) k=16':>12}{'độ dịch k=16':>14}")
    for n, r in results.items():
        pos = " / ".join(f"{r['pos_px'][k - 1]:.1f}" for k in KS)
        print(f"{n:<46}{pos:<30}{r['ang_deg'][K - 1]:>12.2f}{r['disp_px'][K - 1]:>14.2f}")
    print(f"\nPhép thử phản thực tế ({len(pick)} trạng thái): độ nhạy = |dự đoán khác nhau| / |thực tế khác nhau| (lý tưởng 100%)")
    for n, r in counterfactual.items():
        print(f"  {n:<44} độ nhạy {r['sensitivity_median'] * 100:5.0f}% | sai vị trí cuối với hành động lạ {r['final_pos_err_px_counterfactual_actions']:5.1f}px")

    # ---- figures ----
    ks = np.arange(1, K + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), dpi=160)
    style = {"Đoán đứng yên": ("#9aa5b1", "--"), "WM cũ (latent)": ("#d9534f", "-")}
    palette = ["#3457d5", "#23804a", "#8e44ad", "#e67e22"]
    for i, n in enumerate(preds):
        if n not in style:
            style[n] = (palette[i % len(palette)], "-")
    for n in ["Đoán đứng yên"] + list(preds):
        c, ls = style[n]
        ax[0].plot(ks, results[n]["pos_px"], ls, color=c, lw=2, marker="o", ms=3, label=n)
        ax[1].plot(ks, results[n]["ang_deg"], ls, color=c, lw=2, marker="o", ms=3, label=n)
    for a, t, y in ((ax[0], "Sai số vị trí khối T", "pixel"), (ax[1], "Sai số góc khối T", "độ")):
        a.set_title(t, fontweight="bold")
        a.set_xlabel("bước tưởng tượng k (0,1 s/bước)")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "curves.png"))
    plt.close()

    best = list(dyn_models)[-1] if dyn_models else None
    sample = moving[np.linspace(0, len(moving) - 1, 4).astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=140)
    for a, i in zip(axes, sample):
        def px(b):
            return b[..., 0] * 256 + 256, b[..., 1] * 256 + 256
        tb = V["tgt_block"][i].cpu().numpy()
        # truth drawn last (dashed) so it stays visible when the dynamics model lies right on top of it
        for name, col, arr, ls in (("WM cũ", "#d9534f", preds["WM cũ (latent)"][i].cpu().numpy(), "-o"),
                                   (best or "", "#3457d5", preds[best][i].cpu().numpy() if best else None, "-o"),
                                   ("Thực tế", "#1f2933", tb, "--")):
            if arr is None:
                continue
            x, y = px(arr)
            a.plot(x, y, ls, color=col, ms=3, lw=1.8 if ls == "-o" else 1.4, label=name.split(" (")[0])
            poly = create_tee_polygon(float(x[-1]), float(y[-1]), float(arr[-1, 2]))
            gx, gy = poly.exterior.xy
            a.fill(gx, gy, color=col, alpha=0.12)
            a.plot(gx, gy, color=col, lw=1)
        b0 = V["block0"][i].cpu().numpy()
        gx, gy = create_tee_polygon(b0[0] * 256 + 256, b0[1] * 256 + 256, float(b0[2])).exterior.xy
        a.plot(gx, gy, ":", color="#9aa5b1")
        a.set_aspect("equal")
        a.invert_yaxis()
        a.grid(alpha=0.25)
        a.set_title(f"cửa sổ #{i}: 16 bước tưởng tượng", fontsize=10)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "trajectories.png"))
    plt.close()
    print(f"\nĐã lưu: {OUT}/metrics.json, curves.png, trajectories.png")


if __name__ == "__main__":
    main()
