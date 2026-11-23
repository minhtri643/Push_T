"""Stage E5a (legacy): frozen DINOv2 patch features for every frame (vision World Model input).

    python experiments/E5_vision_world_models/code_legacy_dinov2/extract_dino_features.py

Writes float16 arrays of shape (N, 64, 384):
  data/cache/dino_feats_demo.npy  - the 25,650 demonstration frames (zarr order)
  data/cache/dino_feats_sim.npy   - the simulator frames from collect_sim_interactions.py --images
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import time

import numpy as np
import yaml
import zarr

from pusht_wm_dit.models.dinov2_encoder import FEAT_DIM, GRID, get_encoder

OUT_DEMO = "data/cache/dino_feats_demo.npy"
OUT_SIM = "data/cache/dino_feats_sim.npy"
SIM_FRAMES = "data/legacy/sim_interactions_img.npz.frames.npy"


def extract(frames, out_path, enc, chunk=4096):
    n = frames.shape[0]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, GRID * GRID, FEAT_DIM))
    t0 = time.perf_counter()
    for i in range(0, n, chunk):
        f = np.asarray(frames[i:i + chunk])
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        out[i:i + len(f)] = enc.encode(f).cpu().numpy()
    out.flush()
    print(f"  {n:,} frames -> {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB, {time.perf_counter() - t0:.0f}s)")


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    enc = get_encoder("cuda")
    print("Demonstration frames:")
    extract(zarr.open(cfg["data"]["zarr_path"], mode="r")["data/img"], OUT_DEMO, enc)
    if os.path.exists(SIM_FRAMES):
        print("Simulator frames:")
        extract(np.load(SIM_FRAMES, mmap_mode="r"), OUT_SIM, enc)
    else:
        print(f"(no {SIM_FRAMES}; run collect_sim_interactions.py --images --out data/legacy/sim_interactions_img.npz)")


if __name__ == "__main__":
    main()
