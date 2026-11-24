"""Train the DiT diffusion policy on frozen JEPA latents.

Step 1 (sanity)  : python -m scripts.train_policy --overfit_one_batch
Step 2 (full run): python -m scripts.train_policy --run_name dit_main
Ablation (A1)    : python -m scripts.train_policy --run_name dit_random_init --latents data/cache/latents_random_init.npy
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.export_latents import param_checksum
from src.dataset import LatentPathDataset, load_cache
from src.dit_policy import DiffusionPolicy
from src.utils import CSVLogger, get_device, load_config, normalize_pos, set_seed
from src.world_model import JEPAWorldModel


def check_frozen(latents_path: Path, device) -> None:
    meta = json.loads(latents_path.with_suffix(".json").read_text(encoding="utf-8"))
    if "checkpoint" not in meta:
        print("latents without a checkpoint (random-init): checksum check skipped")
        return
    ckpt = torch.load(meta["checkpoint"], map_location=device, weights_only=False)
    model = JEPAWorldModel(ckpt["cfg"]["wm"]).to(device)
    model.load_state_dict(ckpt["model"])
    chk = param_checksum(model)
    assert abs(chk - meta["param_checksum"]) < 1e-3, f"WM changed! {chk} vs {meta['param_checksum']}"
    print(f"World model frozen: OK (param_checksum {chk:.6f}, epoch {ckpt['epoch']})")


@torch.no_grad()
def evaluate(ema, val_ds, sub_idx, device, n_infer: int = 16) -> dict:
    ema.eval()
    fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(1234)
        tot, n = 0.0, 0
        for i in range(0, len(val_ds), 1024):
            idx = torch.arange(i, min(i + 1024, len(val_ds)), device=device)
            s, p, path, _ = val_ds.get(idx)
            tot += float(ema.compute_loss(s, p, path)) * len(idx)
            n += len(idx)
        s, p, path, _ = val_ds.get(sub_idx)
        pred = ema.sample(s, p, n_infer)
    d = (pred - path).norm(dim=-1)  # [B, H] px
    still = (p[:, None, :] - path).norm(dim=-1)
    return {"val_loss": tot / n, "val_ade": float(d.mean()), "val_fde": float(d[:, -1].mean()),
            "still_ade": float(still.mean()), "still_fde": float(still[:, -1].mean())}


def overfit_one_batch(pol_cfg, train_ds, s_dim, ws, device) -> None:
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_ds), generator=g)[:256].to(device)
    s, p, path, _ = train_ds.get(idx, random_aug=False)
    pol = DiffusionPolicy({**pol_cfg, "dropout": 0.0}, s_dim, ws).to(device)
    opt = torch.optim.AdamW(pol.parameters(), lr=5e-4, weight_decay=0.0)
    losses = []
    for it in range(1, 1001):
        lr = 5e-4 * min(1.0, it / 50)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad()
        loss = pol.compute_loss(s, p, path)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if it % 100 == 0:
            print(f"iter {it:4d} | loss (mean of last 100) = {np.mean(losses[-100:]):.4f}")
    final = float(np.mean(losses[-100:]))

    # loss at fixed noise levels: small k needs very precise x0, so it stays higher (expected)
    pol.eval()
    print("\nloss per timestep k (avg of 20 noise draws):")
    with torch.no_grad():
        x0 = normalize_pos(path, ws)
        for k in (0, 10, 25, 50, 75, 99):
            ls = []
            for _ in range(20):
                noise = torch.randn_like(x0)
                kk = torch.full((len(x0),), k, device=device, dtype=torch.long)
                eps = pol.dit(pol.scheduler.add_noise(x0, noise, kk), kk, pol._cond(s, p))
                ls.append(float(((eps - noise) ** 2).mean()))
            print(f"  k={k:3d}: {np.mean(ls):.4f}")
    print(f"\nfinal overfit loss = {final:.4f} -> {'PASS' if final < 0.02 else 'FAIL'} (threshold 0.02)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run_name", default="dit_main")
    ap.add_argument("--latents", default="data/cache/latents_wm_main.npy")
    ap.add_argument("--overfit_one_batch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    P = cfg["policy"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    ws = float(cfg["data"]["workspace_size"])

    latents_path = Path(args.latents)
    check_frozen(latents_path, device)
    latents = np.load(latents_path)
    cache = load_cache(cfg["data"]["cache_dir"])
    n_versions = int(P["n_latent_aug"]) + 1
    H = int(P["horizon"])
    train_ds = LatentPathDataset(cache, latents, cache["split"]["train"], H, n_versions, device)
    val_ds = LatentPathDataset(cache, latents, cache["split"]["val"], H, 1, device)
    s_dim = latents.shape[-1]
    print(f"latents {latents.shape} | train samples {len(train_ds)} | val samples {len(val_ds)} "
          f"| aug versions used: {n_versions} | use_proprio={P['use_proprio']}")

    if args.overfit_one_batch:
        overfit_one_batch(P, train_ds, s_dim, ws, device)
        return

    run_dir = Path(cfg["project"]["out_dir"]) / args.run_name
    if (run_dir / "ckpt" / "best.pt").exists() and not (args.overwrite or args.run_name.endswith("debug")):
        raise SystemExit(f"{run_dir} already has a trained policy. Use another --run_name or --overwrite.")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    csv_path = run_dir / "log.csv"
    if csv_path.exists():
        csv_path.unlink()
    logger = CSVLogger(csv_path)

    pol = DiffusionPolicy(P, s_dim, ws).to(device)
    ema = copy.deepcopy(pol).eval().requires_grad_(False)
    opt = torch.optim.AdamW(pol.parameters(), lr=float(P["lr"]), weight_decay=float(P["weight_decay"]))
    bs, epochs = int(P["batch_size"]), int(P["epochs"])
    spe = len(train_ds) // bs
    total, warm = epochs * spe, int(P["warmup_steps"])

    def lr_at(it: int) -> float:
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * (it - warm) / max(1, total - warm)))

    full = torch.nonzero(val_ds.full_h).squeeze(1).cpu()
    g = torch.Generator().manual_seed(0)
    sub_idx = full[torch.randperm(len(full), generator=g)[:1000]].to(device)
    base = evaluate(ema, val_ds, sub_idx, device)
    print(f"params {sum(p.numel() for p in pol.parameters()):,} | {spe} steps/epoch x {epochs} epochs | "
          f"stand-still baseline on {len(sub_idx)} val samples: ADE {base['still_ade']:.1f} px, "
          f"FDE {base['still_fde']:.1f} px")

    decay, it, best, t0 = float(P["ema_decay"]), 0, float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        pol.train()
        perm = torch.randperm(len(train_ds), device=device)
        tl = 0.0
        for b in range(spe):
            idx = perm[b * bs:(b + 1) * bs]
            s, p, path, _ = train_ds.get(idx, random_aug=True)
            for gp in opt.param_groups:
                gp["lr"] = float(P["lr"]) * lr_at(it)
            opt.zero_grad(set_to_none=True)
            loss = pol.compute_loss(s, p, path)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {it}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pp in zip(ema.parameters(), pol.parameters()):
                    pe.mul_(decay).add_(pp.detach(), alpha=1.0 - decay)
            tl += float(loss)
            it += 1
        row = {"epoch": epoch, "train_loss": tl / spe, "lr": opt.param_groups[0]["lr"],
               "val_loss": None, "val_ade": None, "val_fde": None}
        msg = f"ep {epoch:3d} | train {row['train_loss']:.4f}"
        if epoch % 10 == 0 or epoch == epochs:
            ev = evaluate(ema, val_ds, sub_idx, device)
            row.update({k: ev[k] for k in ("val_loss", "val_ade", "val_fde")})
            state = {"model": ema.state_dict(), "cfg": cfg, "epoch": epoch, **ev}
            torch.save(state, run_dir / "ckpt" / "last.pt")
            if ev["val_fde"] < best:
                best = ev["val_fde"]
                torch.save(state, run_dir / "ckpt" / "best.pt")
            msg += (f" | val_loss {ev['val_loss']:.4f} | ADE {ev['val_ade']:.1f} px | FDE {ev['val_fde']:.1f} px "
                    f"(still {ev['still_fde']:.1f}) | best FDE {best:.1f} | {time.time() - t0:.0f}s")
        logger.log(row)
        print(msg)

    df = pd.read_csv(csv_path)
    ev_df = df.dropna(subset=["val_fde"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], "-", c="gray", label="train loss")
    axes[0].plot(ev_df["epoch"], ev_df["val_loss"], "-o", c="black", ms=3, label="val loss (EMA)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE on epsilon")
    axes[0].legend()
    axes[1].plot(ev_df["epoch"], ev_df["val_fde"], "-o", c="black", ms=3, label="FDE (pusher, @2 s)")
    axes[1].plot(ev_df["epoch"], ev_df["val_ade"], "--s", c="black", ms=3, label="ADE")
    axes[1].axhline(base["still_fde"], ls=":", c="gray", label="stand-still FDE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("px (0..512)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "curves.png", dpi=130)
    print(f"\nDone. best val FDE = {best:.1f} px (stand-still {base['still_fde']:.1f}). Files in {run_dir}")


if __name__ == "__main__":
    main()
