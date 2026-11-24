"""Inspect lerobot/pusht and the Diffusion Policy zarr; cross-check the two sources.

Run: python -m scripts.inspect_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_data import find_zarr
from src.utils import load_config

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def stack_col(series: pd.Series) -> np.ndarray:
    """A parquet column of per-row arrays -> [N, D] float64."""
    return np.stack([np.asarray(v, dtype=np.float64) for v in series.to_numpy()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    root = Path(cfg["data"]["lerobot_dir"])
    checks: dict[str, bool] = {}

    # 1. info.json
    section("1. meta/info.json")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    for key in ("codebase_version", "total_episodes", "total_frames", "fps"):
        print(f"{key}: {info.get(key)}")
    print("features:")
    for name, spec in info["features"].items():
        print(f"  {name}: dtype={spec.get('dtype')} shape={spec.get('shape')}")

    # 2. parquet / mp4 files
    section("2. Data and video files")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    mp4_files = sorted((root / "videos").rglob("*.mp4"))
    print(f"parquet files: {len(parquet_files)}")
    for p in parquet_files[:5]:
        print("  ", p.relative_to(root))
    print(f"mp4 files: {len(mp4_files)}")
    for p in mp4_files[:5]:
        print("  ", p.relative_to(root))
    layout = "v3-style (merged files)" if len(mp4_files) <= 2 else "v2-style (one file per episode)"
    print("layout guess:", layout)

    # 3. dataframe
    section("3. DataFrame")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print(df.dtypes)
    print(df.head().to_string())

    # 4. episodes
    section("4. Episode lengths")
    lengths_s = df.groupby("episode_index").size().sort_index()
    lengths = lengths_s.to_numpy()
    print(f"episodes: {len(lengths)}  min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.2f}")
    contiguous = all(
        np.array_equal(np.sort(g["frame_index"].to_numpy()), np.arange(len(g)))
        for _, g in df.groupby("episode_index")
    )
    print("frame_index contiguous 0..L-1 in every episode:", contiguous)
    checks["206 episodes"] = len(lengths) == 206
    checks["25650 frames"] = len(df) == 25650
    checks["fps == 10"] = info.get("fps") == 10
    checks["frame_index contiguous"] = contiguous

    # 5. ranges
    section("5. Ranges")
    state = stack_col(df["observation.state"])
    action = stack_col(df["action"])
    print("observation.state shape", state.shape, "min", state.min(0), "max", state.max(0))
    print("action            shape", action.shape, "min", action.min(0), "max", action.max(0))
    checks["state/action in [0, 512]"] = bool(
        state.min() >= 0 and state.max() <= 512 and action.min() >= 0 and action.max() <= 512
    )

    # 6. meta/episodes
    section("6. meta/episodes")
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet")) if (root / "meta" / "episodes").exists() else []
    if ep_files:
        ep_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
        print("columns:", list(ep_df.columns))
        print(ep_df.head(3).T.to_string())
    elif (root / "meta" / "episodes.jsonl").exists():
        lines = (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        print("episodes.jsonl, first 3 lines:")
        for ln in lines[:3]:
            print("  ", ln)
    else:
        print("No episode metadata file found.")

    # 7. zarr
    section("7. Zarr (Diffusion Policy original)")
    import zarr

    zpath = cfg["data"].get("zarr_path") or find_zarr(Path(cfg["data"]["dp_dir"]))
    print("zarr path:", zpath)
    zroot = zarr.open(str(zpath), mode="r")
    try:
        print(zroot.tree())
    except Exception as e:  # tree() can need optional deps
        print("tree() unavailable:", e)
    z_img, z_state, z_action = zroot["data/img"], zroot["data/state"], zroot["data/action"]
    z_ends = zroot["meta/episode_ends"][:]
    print("img   ", z_img.shape, z_img.dtype)
    print("state ", z_state.shape, z_state.dtype)
    print("action", z_action.shape, z_action.dtype)
    print("episode_ends", z_ends.shape, "first 3:", z_ends[:3], "last:", z_ends[-1])

    # 8. cross-check
    section("8. Cross-check lerobot vs zarr")
    z_state_np = np.asarray(z_state[:])
    z_action_np = np.asarray(z_action[:])
    same_len = len(df) == len(z_state_np)
    print("same number of rows:", same_len)
    if same_len:
        d_state = np.abs(state - z_state_np[:, :2]).max()
        d_action = np.abs(action - z_action_np).max()
        print(f"max |state  diff| = {d_state:.6f}")
        print(f"max |action diff| = {d_action:.6f}")
        checks["state matches zarr (<1e-3)"] = bool(d_state < 1e-3)
        checks["action matches zarr (<1e-3)"] = bool(d_action < 1e-3)
    z_lengths = np.diff(np.concatenate([[0], z_ends]))
    len_match = len(z_lengths) == len(lengths) and np.array_equal(z_lengths, lengths)
    print("episode lengths identical:", len_match)
    checks["episode lengths identical"] = bool(len_match)

    section("SUMMARY")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")


if __name__ == "__main__":
    main()
