"""Step 1 of the Hugging-Face-only pipeline: recover the T-block pose from the lerobot/pusht frames.

lerobot/pusht has no block pose, so it is estimated from each 96x96 frame by render-and-compare
(pusht_wm_dit/data/block_pose_fit.py). Two modes:

  python scripts/1_data/fit_block_pose.py synthetic
      Accuracy of the fitter on frames whose pose is known: the simulator renders contact-rich pushing
      episodes, the frames are AV1-compressed like the dataset video, then fitted. The simulator is only a
      measuring instrument here; nothing from it is used for training.

  python scripts/1_data/fit_block_pose.py hf
      Fits all 25,650 dataset frames and checks the result against signals inside the dataset itself:
        * next.reward is the goal coverage computed by LeRobot from the true pose -> compare with the
          coverage of the fitted pose,
        * when the pusher is far from the block the block cannot move -> frame-to-frame jitter,
        * pusher position from the image vs observation.state is not needed (it is given).
      Writes data/pusht_hf.zarr (same layout as the old zarr: data/img, data/state, data/action,
      meta/episode_ends) so the training scripts can use it unchanged.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

from pusht_wm_dit.data.lerobot_pusht import decode_video, load_lerobot_pusht, _ffmpeg_exe  # noqa: E402
from pusht_wm_dit.data import block_pose_fit as tf  # noqa: E402

HF_DIR = os.path.join(ROOT, "data", "lerobot_pusht")
OUT_ZARR = os.path.join(ROOT, "data", "pusht_hf.zarr")
RES = os.path.join(ROOT, "results", "pose_recovery")


def pose_errors(fit, true):
    pos = np.linalg.norm(fit[:, :2] - true[:, :2], axis=1)
    ang = np.rad2deg(np.abs(tf.wrap_angle(fit[:, 2] - true[:, 2])))
    kp = np.array([np.linalg.norm(tf.tee_polygon(a) - tf.tee_polygon(b), axis=1).mean() for a, b in zip(fit, true)])
    return pos, ang, kp


def summary(name, x, unit):
    return {f"{name}_mean": float(x.mean()), f"{name}_median": float(np.median(x)), f"{name}_p90": float(np.percentile(x, 90)),
            f"{name}_p99": float(np.percentile(x, 99)), f"{name}_max": float(x.max()), "unit_" + name: unit}


def av1_roundtrip(frames: np.ndarray, path: str, chunk: int = 100) -> np.ndarray:
    """Encode like LeRobot (AV1, yuv420p, GOP 2, crf 30) and decode again. LeRobot uses libsvtav1, which the
    bundled ffmpeg lacks; its libaom crashes (0xC0000005) on these frames unless look-ahead is disabled
    (-lag-in-frames 0), so the compression is similar but not bit-identical to the dataset's."""
    out = []
    raw = path + ".raw"
    for i in range(0, len(frames), chunk):
        part = np.ascontiguousarray(frames[i:i + chunk])
        part.tofile(raw)
        cmd = [_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x96", "-r", "10",
               "-i", raw, "-c:v", "libaom-av1", "-crf", "30", "-b:v", "0", "-g", "2", "-lag-in-frames", "0",
               "-row-mt", "0", "-threads", "1",
               "-pix_fmt", "yuv420p", path]
        subprocess.run(cmd, check=True)
        out.append(decode_video(path, len(part)))
    os.remove(raw)
    return np.concatenate(out)


def synthetic(args):
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    from pusht_wm_dit.envs.pusht_sim import PushTExpertPolicy

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    expert, rng = PushTExpertPolicy(), np.random.RandomState(0)
    frames, agent, block = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=70000 + ep)
        for t in range(args.steps):
            frames.append(obs["pixels"])
            agent.append(info["pos_agent"])
            block.append(info["block_pose"])
            act = expert.get_action(info["block_pose"], info["pos_agent"]) + rng.normal(0, 8, 2)
            obs, _, _, _, info = env.step(np.clip(act, 10, 502))
    frames, agent, block = np.stack(frames), np.stack(agent).astype(np.float32), np.stack(block)
    os.makedirs(RES, exist_ok=True)
    comp = av1_roundtrip(frames, os.path.join(RES, "synthetic_av1.mp4"))
    print(f"{len(frames)} simulator frames, AV1 round trip done; mean abs pixel change {np.abs(comp.astype(int) - frames).mean():.2f}")
    np.savez_compressed(os.path.join(RES, "synthetic_frames.npz"), clean=frames, av1=comp, agent=agent, block=block)

    ren = tf.TeeRenderer("cuda")
    report = {"frames": int(len(frames)), "episodes": args.episodes}
    for label, imgs in (("clean", frames), ("av1", comp)):
        t0 = time.perf_counter()
        pal, s = tf.calibrate(ren, imgs, agent)
        res = tf.fit_all(ren, imgs, agent, pal, s, log=lambda *_: None)
        pos, ang, kp = pose_errors(res["pose"], block)
        dist = tf.distance_agent_to_tee(block, agent)
        contact = dist < 25
        r = {**summary("pos", pos, "world px"), **summary("ang", ang, "deg"), **summary("keypoint", kp, "world px"),
             "residual_loss_median": float(np.median(res["loss"])),
             "contact_frames": int(contact.sum()), "keypoint_mean_contact": float(kp[contact].mean()),
             "keypoint_mean_free": float(kp[~contact].mean()), "frac_keypoint_gt_5px": float((kp > 5).mean()),
             "blur_s": s, "seconds": time.perf_counter() - t0}
        report[label] = r
        print(f"[{label}] pos {r['pos_mean']:.2f} px (median {r['pos_median']:.2f}, p99 {r['pos_p99']:.2f}, max {r['pos_max']:.1f}) | "
              f"angle {r['ang_mean']:.2f} deg (p99 {r['ang_p99']:.2f}) | keypoints {r['keypoint_mean']:.2f} px "
              f"(contact {r['keypoint_mean_contact']:.2f}, free {r['keypoint_mean_free']:.2f}) | >5px {100 * r['frac_keypoint_gt_5px']:.2f}% | "
              f"residual median {r['residual_loss_median']:.4f} | {r['seconds']:.0f}s")
    json.dump(report, open(os.path.join(RES, "synthetic_report.json"), "w"), indent=2)


def hf(args):
    d = load_lerobot_pusht(HF_DIR)
    frames, agent = d["frames"], d["agent"]
    os.makedirs(RES, exist_ok=True)
    raw_path = os.path.join(RES, "hf_tpose_raw.npz")
    n = len(frames) if args.limit is None else args.limit
    if args.reuse_fit and os.path.exists(raw_path):
        r = np.load(raw_path)
        pose, loss, margin, pal, s = r["pose"][:n], r["loss"][:n], r["margin"][:n], r["palette"], float(r["blur"])
        print(f"reusing the image fit from {raw_path}")
    else:
        ren = tf.TeeRenderer("cuda")
        t0 = time.perf_counter()
        print("calibrating colours / blur on 256 dataset frames ...")
        pal_t, s = tf.calibrate(ren, frames, agent)
        pal = pal_t.cpu().numpy()
        res = tf.fit_all(ren, frames[:n], agent[:n], pal_t, s)
        print(f"fitted {n} frames in {time.perf_counter() - t0:.0f}s")
        pose, loss, margin = res["pose"], res["loss"], res["margin"]
        np.savez(raw_path, pose=pose, loss=loss, margin=margin, palette=pal, blur=s)

    # ---- image -> world offset, estimated from the dataset's own reward (see block_pose_fit.offset_from_reward)
    ep, rew = d["episode_index"][:n], d["reward"][:n]
    offset, mae_before, mae_after = tf.offset_from_reward(pose, rew, ep)
    pose = pose.copy()
    pose[:, :2] += offset
    np.savez(os.path.join(RES, "hf_tpose.npz"), pose=pose, loss=loss, margin=margin, palette=pal, blur=s, offset=offset)

    # ---- check 1: coverage of the fitted pose vs next.reward (computed by LeRobot from the true pose)
    cov = tf.coverage(pose)
    same_ep_next = np.append(ep[1:] == ep[:-1], False)
    cov_next = np.where(same_ep_next, np.append(cov[1:], 0), np.nan)
    hyps = {"reward = coverage(t)": cov, "reward = coverage(t)/0.95": np.clip(cov / 0.95, 0, 1),
            "reward = coverage(t+1)": cov_next, "reward = coverage(t+1)/0.95": np.clip(cov_next / 0.95, 0, 1)}
    check1 = {}
    for name, h in hyps.items():
        ok = ~np.isnan(h)
        err = np.abs(h[ok] - rew[ok])
        check1[name] = {"mae": float(err.mean()), "median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
                        "frac_within_0.02": float((err < 0.02).mean())}
        print(f"  {name:<30} |reward - predicted| mean {err.mean():.4f}, median {np.median(err):.4f}, p95 {np.percentile(err, 95):.4f}, "
              f"within 0.02: {100 * (err < 0.02).mean():.1f}%")

    # ---- check 2: static block when the pusher is far away (pose must not change between t and t+1)
    dist = tf.distance_agent_to_tee(pose, agent[:n])
    far = (dist > 30) & same_ep_next
    far &= np.append(dist[1:] > 30, False)
    i = np.flatnonzero(far)
    dpos = np.linalg.norm(pose[i + 1, :2] - pose[i, :2], axis=1)
    dang = np.rad2deg(np.abs(tf.wrap_angle(pose[i + 1, 2] - pose[i, 2])))
    check2 = {"pairs": int(len(i)), "dpos_mean": float(dpos.mean()), "dpos_p95": float(np.percentile(dpos, 95)),
              "dang_mean": float(dang.mean()), "dang_p95": float(np.percentile(dang, 95)),
              "implied_per_frame_pos_noise": float(np.sqrt((dpos ** 2).mean() / 2)),
              "implied_per_frame_ang_noise": float(np.sqrt((dang ** 2).mean() / 2))}
    print(f"  static pairs (pusher > 30 px away): {len(i)} | jump pos mean {dpos.mean():.2f} px (p95 {np.percentile(dpos, 95):.2f}) | "
          f"angle mean {dang.mean():.2f} deg (p95 {np.percentile(dang, 95):.2f})")

    report = {"frames": int(n), "blur_s": s, "palette": np.round(np.asarray(pal) * 255).astype(int).tolist(),
              "world_offset_px_from_reward": offset.tolist(),
              "reward_mae_reward_gt_0.2_before_after_offset": [mae_before, mae_after],
              "loss_mean": float(loss.mean()), "loss_p99": float(np.percentile(loss, 99)),
              "ambiguous_frames(margin<0.002)": int((margin < 0.002).sum()), "reward_check": check1, "static_check": check2}
    json.dump(report, open(os.path.join(RES, "hf_report.json"), "w"), indent=2)
    if args.limit is None:
        write_zarr(d, pose, loss, report)


def write_zarr(d, pose, loss, report):
    import shutil
    import zarr
    if os.path.exists(OUT_ZARR):
        shutil.rmtree(OUT_ZARR)
    root = zarr.open_group(OUT_ZARR, mode="w")
    data, meta = root.require_group("data"), root.require_group("meta")
    state = np.concatenate([d["agent"], pose[:, :2], np.mod(pose[:, 2:3], 2 * np.pi)], axis=1).astype(np.float32)
    data.create_array("img", data=np.asarray(d["frames"]), chunks=(512, 96, 96, 3))
    data.create_array("state", data=state)
    data.create_array("action", data=d["action"].astype(np.float32))
    data.create_array("reward", data=d["reward"].astype(np.float32))
    data.create_array("tpose_fit_loss", data=loss.astype(np.float32))
    meta.create_array("episode_ends", data=d["episode_ends"].astype(np.int64))
    root.attrs.update({
        "source": "https://huggingface.co/datasets/lerobot/pusht (only data source)",
        "state_columns": "agent_x, agent_y (observation.state) | block_x, block_y, block_theta (ESTIMATED from the image by scripts/1_data/fit_block_pose.py)",
        "fit_report": report,
    })
    print(f"wrote {OUT_ZARR}: img {np.asarray(d['frames']).shape}, state {state.shape}, episodes {len(d['episode_ends'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "hf"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="hf: fit only the first N frames (no zarr written)")
    ap.add_argument("--reuse-fit", action="store_true", help="hf: reuse results/pose_recovery/hf_tpose_raw.npz (skip the 15-min image fit)")
    args = ap.parse_args()
    synthetic(args) if args.mode == "synthetic" else hf(args)
