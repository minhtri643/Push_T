"""Reader for the Hugging Face dataset lerobot/pusht (LeRobot format v3.0) - the project's only data source.

Download once (7.7 MB):
    python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/pusht', repo_type='dataset', local_dir='data/lerobot_pusht')"

What the dataset contains (206 human demonstrations, 25,650 frames at 10 Hz):
    observation.image  96x96 RGB frames, all episodes in one AV1 video
    observation.state  (x, y) of the pusher (agent), workspace pixels [0, 512]
    action             (x, y) target of the pusher
    next.reward        T-block coverage of the goal (see fit_tpose.py for how it is used)
It does NOT contain the T-block pose: that is recovered from the images by `scripts/1_data/fit_block_pose.py`.
"""

import json
import os
import subprocess
from typing import Dict

import numpy as np

H = W = 96


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_video(path: str, n_expected: int = None) -> np.ndarray:
    """AV1 mp4 -> uint8 (N, 96, 96, 3). Uses the libaom software decoder bundled with imageio-ffmpeg."""
    cmd = [_ffmpeg_exe(), "-v", "error", "-c:v", "libaom-av1", "-i", path,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if n_expected is not None and len(frames) != n_expected:
        raise ValueError(f"decoded {len(frames)} frames, expected {n_expected}")
    return frames


def load_lerobot_pusht(root: str, cache: bool = True) -> Dict[str, np.ndarray]:
    """Returns numpy arrays for the whole dataset (frames are cached next to the download as frames.npy)."""
    import pyarrow.parquet as pq

    info = json.load(open(os.path.join(root, "meta", "info.json")))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"expected LeRobot v3 layout, got {info.get('codebase_version')}")
    tab = pq.read_table(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    col = lambda name: np.asarray(tab.column(name).to_pylist())
    out = {
        "agent": col("observation.state").astype(np.float32),        # (N, 2)
        "action": col("action").astype(np.float32),                  # (N, 2)
        "reward": col("next.reward").astype(np.float32),             # (N,)
        "episode_index": col("episode_index").astype(np.int64),
        "frame_index": col("frame_index").astype(np.int64),
        "index": col("index").astype(np.int64),
    }
    n = len(out["index"])
    if not np.array_equal(out["index"], np.arange(n)):
        raise ValueError("rows are not in global frame order")
    ep = out["episode_index"]
    out["episode_ends"] = np.flatnonzero(np.diff(np.append(ep, ep[-1] + 1))) + 1   # exclusive ends, like zarr meta

    cache_path = os.path.join(root, "frames.npy")
    if cache and os.path.exists(cache_path):
        frames = np.load(cache_path, mmap_mode="r")
    else:
        frames = decode_video(os.path.join(root, "videos", "observation.image", "chunk-000", "file-000.mp4"), n)
        if cache:
            np.save(cache_path, frames)
    if len(frames) != n:
        raise ValueError(f"{len(frames)} frames vs {n} rows")
    out["frames"] = frames
    return out
