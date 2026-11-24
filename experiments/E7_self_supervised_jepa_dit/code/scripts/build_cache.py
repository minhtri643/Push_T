"""Build the .npy cache. Run: python -m scripts.build_cache --config configs/default.yaml"""
import argparse

from src.dataset import build_cache
from src.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="e.g. data.image_source=zarr")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg["project"]["seed"])
    build_cache(cfg)


if __name__ == "__main__":
    main()
