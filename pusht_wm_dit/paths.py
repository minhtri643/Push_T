"""Repository layout.

Every relative path in configs/default.yaml and in the scripts is relative to the repository root.
Entry-point scripts call `chdir_root()` first, so they can be started from any working directory
(command-line paths such as --ckpt-dir are then also read relative to the repository root).

    <repo>/configs/default.yaml   configuration of the whole system
    <repo>/checkpoints/           trained models (GitHub Releases, scripts/download_assets.py)
    <repo>/data/                  lerobot/pusht + processed data (download_assets.py) and caches
    <repo>/results/               every measured number (JSON, logs, figures)
"""

import os
from typing import Any, Dict, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")


def chdir_root() -> str:
    """Makes the repository root the working directory and returns it."""
    os.chdir(ROOT)
    return ROOT


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configs/default.yaml (or `path`)."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
