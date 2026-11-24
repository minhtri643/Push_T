"""Shared helpers: config, seeding, device, coordinate normalization, logging, run dirs."""
from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- config
def _parse_value(raw: str):
    """Parse an override string with YAML rules; fall back to float for forms like '1e-4'."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply dotted overrides such as 'wm.lr=1e-4'."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b=value, got: {item}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"Unknown config section: {p} (from override {item})")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = _parse_value(raw)
    return cfg


# --------------------------------------------------------------------------- runtime
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- geometry
def normalize_pos(x, ws: float = 512.0):
    """Map pixel-space positions in [0, ws] to [-1, 1]. Works for numpy and torch."""
    return x / ws * 2.0 - 1.0


def unnormalize_pos(x, ws: float = 512.0):
    """Inverse of normalize_pos. Works for numpy and torch."""
    return (x + 1.0) / 2.0 * ws


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi]. Works for numpy and torch."""
    if isinstance(a, torch.Tensor):
        return torch.atan2(torch.sin(a), torch.cos(a))
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------- logging
class CSVLogger:
    """Append one dict per row to a CSV file. Header is taken from the first row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict) -> None:
        new_file = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore", restval=""
            )
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def make_run_dir(cfg: dict, name: str) -> Path:
    """Create outputs/runs/<name>_<YYYYmmdd-HHMM>/ (+ ckpt/) and save the config there."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path(cfg["project"]["out_dir"]) / f"{name}_{stamp}"
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return run_dir


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import tempfile

    cfg = load_config("configs/default.yaml", ["wm.lr=1e-4", "policy.use_proprio=false"])
    assert cfg["wm"]["lr"] == 1e-4 and cfg["policy"]["use_proprio"] is False
    assert cfg["wm"]["enc_channels"] == [32, 64, 128, 256]

    x_np = np.array([0.0, 256.0, 512.0])
    assert np.allclose(normalize_pos(x_np), [-1.0, 0.0, 1.0])
    assert np.allclose(unnormalize_pos(normalize_pos(x_np)), x_np)
    x_t = torch.tensor([0.0, 256.0, 512.0])
    assert torch.allclose(unnormalize_pos(normalize_pos(x_t)), x_t)

    a = np.array([0.0, 3.5 * np.pi, -3.5 * np.pi])
    w = wrap_angle(a)
    assert np.all(np.abs(w) <= np.pi + 1e-9)
    assert torch.all(torch.abs(wrap_angle(torch.tensor(a))) <= np.pi + 1e-6)

    set_seed(0)
    r1 = torch.rand(1).item()
    set_seed(0)
    assert r1 == torch.rand(1).item()

    with tempfile.TemporaryDirectory() as tmp:
        lg = CSVLogger(Path(tmp) / "log.csv")
        lg.log({"epoch": 0, "loss": 1.5})
        lg.log({"epoch": 1, "loss": 1.2})
        lines = (Path(tmp) / "log.csv").read_text().strip().splitlines()
        assert lines[0] == "epoch,loss" and len(lines) == 3
        cfg2 = dict(cfg)
        cfg2["project"] = {"seed": 42, "out_dir": tmp}
        rd = make_run_dir(cfg2, "unit_test")
        assert (rd / "config.yaml").exists() and (rd / "ckpt").is_dir()

    print("device:", get_device())
    print("All utils tests passed")
