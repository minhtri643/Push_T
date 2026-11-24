"""Dataset utilities. Part A: build and load the .npy cache (read-once, mmap afterwards).

Part B (WMClipDataset, PolicyDataset) is added in Steps 3 and 4.
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.download_data import find_zarr
from src.utils import normalize_pos


# --------------------------------------------------------------------------- helpers
def decode_video(path) -> np.ndarray:
    """Decode an mp4 file to uint8 RGB frames. Returns [T, H, W, 3]."""
    frames = []
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def make_episode_split(n_episodes: int, n_val: int, seed: int) -> dict:
    """Random split BY EPISODE (never by frame, to avoid leakage)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_episodes)
    return {
        "train": sorted(int(i) for i in perm[n_val:]),
        "val": sorted(int(i) for i in perm[:n_val]),
    }


def episode_bounds(episode_ends: np.ndarray) -> np.ndarray:
    """episode_ends [E] (exclusive) -> [E, 2] array of (start, end_exclusive)."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return np.stack([starts, ends], axis=1)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Zarr images are float32. Convert to uint8 in 0..255 whatever the original scale."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    scale = 255.0 if float(arr.max()) <= 1.0 + 1e-6 else 1.0
    return np.clip(np.rint(arr * scale), 0, 255).astype(np.uint8)


def _stack_col(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in series.to_numpy()])


# --------------------------------------------------------------------------- lerobot readers
def read_lerobot_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "index" in df.columns:
        df = df.sort_values("index")
    return df.reset_index(drop=True)


def read_lerobot_images(root: Path, df: pd.DataFrame, fps: int) -> np.ndarray:
    """Return uint8 [N, 96, 96, 3] aligned with the rows of df."""
    n = len(df)
    mp4s = sorted((root / "videos").rglob("*.mp4"))

    # Fast path (v3, single file): frame i <-> row with index == i
    if len(mp4s) == 1:
        frames = decode_video(mp4s[0])
        print(f"decoded {len(frames)} frames from {mp4s[0].name}")
        if len(frames) == n:
            return frames
        print(f"[warn] frame count {len(frames)} != rows {n}; using episode timestamps")

    # General path (v3 with several files): cut by from_timestamp
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet")) if (root / "meta" / "episodes").exists() else []
    if ep_files:
        ep_df = pd.concat([pd.read_parquet(p) for p in ep_files]).sort_values("episode_index")
        ep_arr = df["episode_index"].to_numpy()
        fi = df["frame_index"].to_numpy()
        images = None
        cur_key, cur_frames = None, None
        for _, row in tqdm(ep_df.iterrows(), total=len(ep_df), desc="slice episodes"):
            key = (
                int(row["videos/observation.image/chunk_index"]),
                int(row["videos/observation.image/file_index"]),
            )
            if key != cur_key:
                path = root / "videos" / "observation.image" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.mp4"
                cur_frames, cur_key = decode_video(path), key
            if images is None:
                images = np.zeros((n, *cur_frames.shape[1:]), dtype=np.uint8)
            rows = np.where(ep_arr == int(row["episode_index"]))[0]
            t0 = float(row["videos/observation.image/from_timestamp"])
            idx = np.rint((t0 + fi[rows] / fps) * fps).astype(np.int64)
            images[rows] = cur_frames[idx]
        return images

    # v2.x fallback: one mp4 per episode, ordered by file name
    per_ep = [decode_video(p) for p in tqdm(mp4s, desc="decode episodes")]
    images = np.concatenate(per_ep)
    assert len(images) == n, f"v2 fallback: {len(images)} frames vs {n} rows"
    return images


# --------------------------------------------------------------------------- build / load
def build_cache(cfg: dict) -> None:
    import zarr

    root = Path(cfg["data"]["lerobot_dir"])
    cache = Path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    fps = int(cfg["data"]["fps"])

    df = read_lerobot_table(root)
    ep_idx = df["episode_index"].to_numpy().astype(np.int64)
    frame_idx = df["frame_index"].to_numpy().astype(np.int64)
    assert np.all(np.diff(ep_idx) >= 0), "rows are not grouped by episode"
    agent_pos = _stack_col(df["observation.state"])
    action = _stack_col(df["action"])

    lengths = np.bincount(ep_idx)
    episode_ends = np.cumsum(lengths).astype(np.int64)

    zpath = cfg["data"].get("zarr_path") or find_zarr(Path(cfg["data"]["dp_dir"]))
    zroot = zarr.open(str(zpath), mode="r")
    state5 = np.asarray(zroot["data/state"][:], dtype=np.float32)
    assert np.array_equal(zroot["meta/episode_ends"][:], episode_ends), "episode_ends mismatch"
    zimg = zroot["data/img"]
    print(f"zarr img dtype={zimg.dtype} shape={zimg.shape}")

    source = cfg["data"]["image_source"]
    if source == "lerobot":
        images = read_lerobot_images(root, df, fps)
    elif source == "zarr":
        images = _to_uint8(zimg[:])
    else:
        raise ValueError(f"Unknown image_source: {source}")
    assert images.shape == (len(df), 96, 96, 3) and images.dtype == np.uint8, images.shape

    # Optional sanity check: AV1 compression error vs the raw zarr images
    rng = np.random.RandomState(0)
    pick = np.sort(rng.choice(len(df), size=500, replace=False))
    z_pick = _to_uint8(np.stack([zimg[int(i)] for i in pick]))
    print(f"zarr img value range on sample: {float(np.asarray(zimg[0]).min()):.2f} .. {float(np.asarray(zimg[0]).max()):.2f}")
    diff = np.abs(images[pick].astype(np.float32) - z_pick.astype(np.float32))
    print(f"mean|img_{source} - img_zarr| over 500 frames = {diff.mean():.3f} (0..255 scale)")

    np.save(cache / "images.npy", images)
    np.save(cache / "agent_pos.npy", agent_pos)
    np.save(cache / "action.npy", action)
    np.save(cache / "state5.npy", state5)
    np.save(cache / "episode_index.npy", ep_idx)
    np.save(cache / "frame_index.npy", frame_idx)
    np.save(cache / "episode_ends.npy", episode_ends)

    split = make_episode_split(len(lengths), int(cfg["data"]["n_val_episodes"]), int(cfg["project"]["seed"]))
    split_path = Path(cfg["data"]["split_file"])
    split_path.write_text(json.dumps(split), encoding="utf-8")

    print("\n--- cache summary ---")
    for name in ["images", "agent_pos", "action", "state5", "episode_index", "frame_index", "episode_ends"]:
        a = np.load(cache / f"{name}.npy", mmap_mode="r")
        print(f"{name:14s} {str(a.shape):20s} {a.dtype}  {a.nbytes / 1e6:8.1f} MB")
    print(f"episodes={len(lengths)}  mean length={lengths.mean():.2f}  total={lengths.sum()}")
    print(f"split: train={len(split['train'])} val={len(split['val'])} overlap={len(set(split['train']) & set(split['val']))}")


def load_cache(cache_dir: str | Path, mmap: bool = True) -> dict:
    """Load every .npy in cache_dir (memory-mapped by default) plus the split."""
    cache_dir = Path(cache_dir)
    out: dict = {}
    for p in sorted(cache_dir.glob("*.npy")):
        out[p.stem] = np.load(p, mmap_mode="r" if mmap else None)
    split_path = cache_dir / "split.json"
    if split_path.exists():
        out["split"] = json.loads(split_path.read_text(encoding="utf-8"))
    return out


# --------------------------------------------------------------------------- Part B
class WMClipDataset(Dataset):
    """Overlapping clips for world-model training (one sample per frame t, t in [s, E-2]).

    obs         [t_ctx, 96, 96, 3] uint8   frames t-t_ctx+1 .. t        (clamped to the episode)
    future_obs  [K, 96, 96, 3]     uint8   frames t+1 .. t+K            (clamped)
    actions     [K, 2]             float32 actions t .. t+K-1, scaled to [-1, 1]
    future_mask [K]                bool    True where frame t+k really exists in the episode
    index       []                 int64   global frame index t
    state5      [5]                float32 only if return_state=True (probe / eval, never in the loss)
    """

    def __init__(self, cache: dict, episode_ids, t_ctx: int = 2, k_future: int = 8, stride: int = 1,
                 return_state: bool = False, in_memory: bool = True, ws: float = 512.0):
        self.images = np.array(cache["images"]) if in_memory else cache["images"]
        self.actions = normalize_pos(np.asarray(cache["action"], dtype=np.float32), ws).astype(np.float32)
        self.state5 = np.asarray(cache["state5"], dtype=np.float32)
        self.return_state = return_state
        self.k_future = k_future

        bounds = episode_bounds(cache["episode_ends"])
        ts, starts, ends = [], [], []
        for ep in episode_ids:
            s, e = int(bounds[ep][0]), int(bounds[ep][1])
            t = np.arange(s, e - 1, stride, dtype=np.int64)
            ts.append(t)
            starts.append(np.full_like(t, s))
            ends.append(np.full_like(t, e))
        self.t = np.concatenate(ts)
        self.s = np.concatenate(starts)
        self.e = np.concatenate(ends)

        self._ctx_off = np.arange(-(t_ctx - 1), 1)
        self._fut_off = np.arange(1, k_future + 1)
        self._act_off = np.arange(0, k_future)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int) -> dict:
        t, s, e = int(self.t[i]), int(self.s[i]), int(self.e[i])
        ctx = np.clip(t + self._ctx_off, s, e - 1)
        fut = np.clip(t + self._fut_off, s, e - 1)
        act = np.clip(t + self._act_off, s, e - 1)
        item = {
            "obs": torch.from_numpy(self.images[ctx]),
            "future_obs": torch.from_numpy(self.images[fut]),
            "actions": torch.from_numpy(self.actions[act]),
            "future_mask": torch.from_numpy(t + self._fut_off <= e - 1),
            "index": torch.tensor(t),
        }
        if self.return_state:
            item["state5"] = torch.from_numpy(self.state5[t])
        return item


# --------------------------------------------------------------------------- Part C
class LatentPathDataset:
    """One sample per frame t of the chosen episodes (everything lives on `device`).

    s_t        [256]    latent S_t (random augmentation version when random_aug=True, else version 0)
    agent_pos  [2]      px
    path       [H, 2]   action[t .. t+H-1] px (clamped to the last action of the episode = stand still)
    full_h     bool     True if t+H-1 <= E-1 (only these samples are used for FDE)
    """

    def __init__(self, cache: dict, latents: np.ndarray, episode_ids, horizon: int = 20,
                 n_versions: int = 1, device="cpu"):
        bounds = episode_bounds(cache["episode_ends"])
        ts, es = [], []
        for ep in episode_ids:
            s, e = int(bounds[ep][0]), int(bounds[ep][1])
            t = np.arange(s, e, dtype=np.int64)
            ts.append(t)
            es.append(np.full_like(t, e))
        t, e = np.concatenate(ts), np.concatenate(es)
        path_idx = np.minimum(t[:, None] + np.arange(horizon)[None, :], (e - 1)[:, None])
        self.t = torch.from_numpy(t).to(device)
        self.path_idx = torch.from_numpy(path_idx).to(device)
        self.full_h = torch.from_numpy(t + horizon - 1 <= e - 1).to(device)
        self.latents = torch.from_numpy(np.ascontiguousarray(latents[:n_versions])).float().to(device)
        self.agent_pos = torch.from_numpy(np.array(cache["agent_pos"], dtype=np.float32, copy=True)).to(device)
        self.action = torch.from_numpy(np.array(cache["action"], dtype=np.float32, copy=True)).to(device)
        self.n_versions = n_versions
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.t)

    def get(self, idx: torch.Tensor, random_aug: bool = False):
        t = self.t[idx]
        if random_aug and self.n_versions > 1:
            v = torch.randint(0, self.n_versions, (len(idx),), device=idx.device)
        else:
            v = torch.zeros(len(idx), dtype=torch.long, device=idx.device)
        return self.latents[v, t], self.agent_pos[t], self.action[self.path_idx[idx]], self.full_h[idx]


if __name__ == "__main__":
    import time

    from torch.utils.data import DataLoader

    from src.utils import load_config

    cfg = load_config("configs/default.yaml")
    wm = cfg["wm"]
    K = wm["k_future"]
    cache = load_cache(cfg["data"]["cache_dir"])
    split = cache["split"]

    def make(ids):
        return WMClipDataset(cache, ids, wm["t_ctx"], K, wm["clip_stride"], return_state=True)

    train, val = make(split["train"]), make(split["val"])
    print(f"len(train) = {len(train)} | len(val) = {len(val)} | sum = {len(train) + len(val)} "
          f"(expected {25650 - 206})")
    assert len(train) + len(val) == 25650 - 206

    def show(name, item):
        print(f"\n--- {name} (global t = {int(item['index'])}) ---")
        for k, v in item.items():
            print(f"{k:12s} {str(tuple(v.shape)):18s} {v.dtype}")

    first = train[0]
    show("first sample of first train episode", first)
    assert first["future_mask"].all()
    assert torch.equal(first["obs"][0], first["obs"][1]), "t = start of episode: ctx[0] must be clamped to ctx[1]"
    t0 = int(first["index"])
    assert np.array_equal(first["future_obs"].numpy(), np.asarray(cache["images"][t0 + 1: t0 + 1 + K]))

    j = int(np.where(train.t == train.e - 2)[0][0])  # last valid t of the first train episode
    last = train[j]
    show("last valid t of first train episode", last)
    print("future_mask:", last["future_mask"].tolist())
    assert int((~last["future_mask"]).sum()) == K - 1, "only k=1 should be valid at t = E-2"

    for it in (first, last, train[len(train) // 2]):
        assert it["actions"].min() >= -1.0 and it["actions"].max() <= 1.0
    print("\nactions in [-1, 1]: OK")

    loader = DataLoader(train, batch_size=wm["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)
    next(it)
    t_start = time.perf_counter()
    for _ in range(20):
        batch = next(it)
    ms = (time.perf_counter() - t_start) / 20 * 1000
    print(f"DataLoader: {ms:.0f} ms / batch of {wm['batch_size']} (num_workers=0)")
    print({k: tuple(v.shape) for k, v in batch.items()})
    print("WMClipDataset dummy test passed")
