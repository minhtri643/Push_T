"""Download lerobot/pusht (Hugging Face) and the original Diffusion Policy zarr.

The zarr is used ONLY for evaluation (ground-truth T-block pose), never as model input.
Run: python -m scripts.download_data
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

from src.utils import load_config

DP_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


class _Progress(tqdm):
    """tqdm wrapper usable as urlretrieve reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_lerobot(local_dir: Path) -> None:
    print(f"[1/2] Downloading lerobot/pusht -> {local_dir}")
    snapshot_download(repo_id="lerobot/pusht", repo_type="dataset", local_dir=str(local_dir))


def find_zarr(dp_dir: Path) -> Path | None:
    found = sorted(dp_dir.rglob("*.zarr"), key=lambda p: len(p.parts))
    return found[0] if found else None


def download_dp_zarr(dp_dir: Path) -> Path:
    dp_dir.mkdir(parents=True, exist_ok=True)
    existing = find_zarr(dp_dir)
    if existing is not None:
        print(f"[2/2] Zarr already present: {existing}")
        return existing

    zip_path = dp_dir / "pusht.zip"
    if not zip_path.exists():
        print(f"[2/2] Downloading {DP_URL}")
        part = zip_path.with_suffix(".part")
        with _Progress(unit="B", unit_scale=True, unit_divisor=1024, desc="pusht.zip") as bar:
            urllib.request.urlretrieve(DP_URL, part, reporthook=bar.update_to)
        part.replace(zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dp_dir)

    zarr_path = find_zarr(dp_dir)
    if zarr_path is None:
        raise FileNotFoundError(f"No *.zarr directory found under {dp_dir} after extraction")
    return zarr_path


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def print_tree(root: Path, max_depth: int = 2, _depth: int = 0) -> None:
    """Print a directory tree (2 levels) with sizes."""
    if _depth == 0:
        print(f"{root}/  [{_fmt(_dir_size(root))}]")
    for p in sorted(root.iterdir()):
        indent = "  " * (_depth + 1)
        if p.is_dir():
            print(f"{indent}{p.name}/  [{_fmt(_dir_size(p))}]")
            if _depth + 1 < max_depth:
                print_tree(p, max_depth, _depth + 1)
        else:
            print(f"{indent}{p.name}  [{_fmt(p.stat().st_size)}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    lerobot_dir = Path(cfg["data"]["lerobot_dir"])
    dp_dir = Path(cfg["data"]["dp_dir"])

    download_lerobot(lerobot_dir)
    zarr_path = download_dp_zarr(dp_dir)

    print("\n=== lerobot dataset ===")
    print_tree(lerobot_dir)
    print("\n=== Diffusion Policy data ===")
    print_tree(dp_dir)
    print(f"\nZarr path: {zarr_path}")


if __name__ == "__main__":
    main()
