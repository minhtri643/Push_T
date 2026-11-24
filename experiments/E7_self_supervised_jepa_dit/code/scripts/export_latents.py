"""Freeze the trained world model and export S_t for every frame (5 versions: 1 clean + 4 augmented).

Run: python -m scripts.export_latents --run_name wm_main
Outputs: data/cache/latents_<run>.npy, latents_random_init.npy (+ .json with checksums)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.dataset import episode_bounds, load_cache
from src.utils import get_device, set_seed
from src.world_model import JEPAWorldModel, preprocess, random_shift


def param_checksum(model: torch.nn.Module) -> float:
    return float(sum(p.detach().double().abs().sum() for p in model.parameters()))


@torch.no_grad()
def export(model, images, prev, device, n_versions: int, pad: int, seed: int, bs: int = 256) -> np.ndarray:
    N = len(images)
    out = np.zeros((n_versions, N, model.s_dim), dtype=np.float32)
    for v in range(n_versions):
        torch.manual_seed(seed + v)
        for i in range(0, N, bs):
            idx = np.arange(i, min(i + bs, N))
            pair = np.stack([images[prev[idx]], images[idx]], axis=1)          # [B, 2, 96, 96, 3] uint8
            x = preprocess(torch.from_numpy(pair).to(device))
            if v > 0:
                x = random_shift(x, pad)                                        # same offset for both frames
            out[v, idx] = model.aggregator(model.encode_frames(x)).cpu().numpy()
        print(f"  version {v} ({'clean' if v == 0 else 'shift aug'}) done")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="wm_main")
    ap.add_argument("--out_root", default="outputs/runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = get_device()
    ckpt_path = Path(args.out_root) / args.run_name / "ckpt" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    wm = cfg["wm"]
    n_versions = int(cfg["policy"]["n_latent_aug"]) + 1
    pad = int(wm["shift_aug"])

    cache = load_cache(cfg["data"]["cache_dir"])
    images = cache["images"]
    bounds = episode_bounds(cache["episode_ends"])
    N = len(images)
    prev = np.arange(N) - 1
    for s, _ in bounds:
        prev[s] = s

    model = JEPAWorldModel(wm).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval().requires_grad_(False)
    chk_before = param_checksum(model)
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, val_l_pred {ckpt['val_l_pred']:.4f})")
    print(f"param_checksum = {chk_before:.6f}")

    cache_dir = Path(cfg["data"]["cache_dir"])
    print("\nexporting JEPA latents...")
    lat = export(model, images, prev, device, n_versions, pad, args.seed)
    assert np.isfinite(lat).all(), "NaN/inf in exported latents"
    assert abs(param_checksum(model) - chk_before) < 1e-6, "model changed during export"

    # sanity: recompute 64 random frames with encode_state and compare with stored version 0
    rng = np.random.RandomState(0)
    pick = np.sort(rng.choice(N, size=64, replace=False))
    with torch.no_grad():
        pair = np.stack([images[prev[pick]], images[pick]], axis=1)
        s_re = model.encode_state(torch.from_numpy(pair).to(device)).cpu().numpy()
    diff = float(np.abs(s_re - lat[0, pick]).max())
    print(f"\nrecompute check: max |diff| = {diff:.2e} (expected < 1e-3)")
    assert diff < 1e-3

    path = cache_dir / f"latents_{args.run_name}.npy"
    np.save(path, lat)
    meta = {
        "checkpoint": str(ckpt_path), "epoch": int(ckpt["epoch"]), "val_l_pred": float(ckpt["val_l_pred"]),
        "date": datetime.now().isoformat(timespec="seconds"), "param_checksum": chk_before,
        "shape": list(lat.shape), "shift_aug_pad": pad, "seed": args.seed,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n--- JEPA latents ---")
    print(f"shape {lat.shape} dtype {lat.dtype} | size {lat.nbytes / 1e6:.0f} MB")
    print(f"mean {lat.mean():.4f}  std {lat.std():.4f}  per-dim std (clean) mean {lat[0].std(0).mean():.3f}")
    print("S_0 (frame 0, clean), first 8 values:", np.round(lat[0, 0, :8], 4).tolist())
    cos = (lat[0] * lat[1]).sum(1) / (np.linalg.norm(lat[0], axis=1) * np.linalg.norm(lat[1], axis=1) + 1e-8)
    print(f"cosine(clean, shifted v1) mean = {cos.mean():.3f}  min = {cos.min():.3f}")
    print("saved:", path)

    # random-init encoder (ablation A1 in Step 5): same architecture, untrained, same pipeline
    set_seed(123)
    rmodel = JEPAWorldModel(wm).to(device)
    rmodel.eval().requires_grad_(False)
    print("\nexporting random-init latents...")
    lat_r = export(rmodel, images, prev, device, n_versions, pad, args.seed)
    rpath = cache_dir / "latents_random_init.npy"
    np.save(rpath, lat_r)
    rpath.with_suffix(".json").write_text(json.dumps({
        "seed": 123, "param_checksum": param_checksum(rmodel), "shape": list(lat_r.shape),
        "date": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"random-init latents: shape {lat_r.shape} | per-dim std mean {lat_r[0].std(0).mean():.3f} | saved {rpath}")
    print("\nexport_latents done")


if __name__ == "__main__":
    main()
