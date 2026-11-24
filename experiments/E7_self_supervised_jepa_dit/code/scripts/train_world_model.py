"""Train the JEPA latent world model.

Debug (1 epoch): python -m scripts.train_world_model --override wm.epochs=1 --run_name wm_debug
Full run       : python -m scripts.train_world_model --config configs/default.yaml --run_name wm_main
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import WMClipDataset, load_cache
from src.utils import CSVLogger, get_device, load_config, set_seed
from src.world_model import JEPAWorldModel, count_params, effective_rank, preprocess, random_shift


def prep_batch(batch: dict, device, pad: int) -> dict:
    """uint8 clips -> normalised float frames with a shared random shift per sample."""
    obs = batch["obs"].to(device, non_blocking=True)
    fut = batch["future_obs"].to(device, non_blocking=True)
    t_ctx = obs.shape[1]
    frames = random_shift(preprocess(torch.cat([obs, fut], dim=1)), pad)
    return {
        "obs": frames[:, :t_ctx],
        "future_obs": frames[:, t_ctx:],
        "actions": batch["actions"].to(device, non_blocking=True),
        "future_mask": batch["future_mask"].to(device, non_blocking=True),
    }


def copy_baseline(model: JEPAWorldModel, b: dict) -> torch.Tensor:
    """L1 between LN(f_xi(o_t)) and the future targets: 'nothing moves' predictor."""
    z_now = F.layer_norm(model.encode_frames(b["obs"][:, -1:], target=True), (model.z_dim,))
    z_bar = F.layer_norm(model.encode_frames(b["future_obs"], target=True), (model.z_dim,))
    err = (z_now - z_bar).abs().mean(dim=-1)
    m = b["future_mask"].float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model: JEPAWorldModel, loader, device, max_stat_batches: int = 32) -> dict:
    """Optimized evaluation: single-pass forward, avoiding redundant target/obs re-encodings."""
    model.eval()
    s_pred = s_copy = 0.0
    n = 0
    S_list, z_list = [], []
    use_amp = False  # wm_main was trained in fp32 (GTX 1650 has no tensor cores)
    for i, batch in enumerate(loader):
        b = prep_batch(batch, device, pad=0)
        bs = len(b["obs"])
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            z_ctx = model.encode_frames(b["obs"])
            S = model.aggregator(z_ctx)
            z_target = model.encode_frames(b["future_obs"], target=True)
            z_bar = F.layer_norm(z_target, (model.z_dim,))
            z_hat = model.rollout(S, b["actions"])

            m = b["future_mask"].float()
            err_pred = (z_hat - z_bar).abs().mean(dim=-1)
            l_pred = (err_pred * m).sum() / m.sum().clamp_min(1.0)

            z_now = F.layer_norm(model.encode_frames(b["obs"][:, -1:], target=True), (model.z_dim,))
            err_copy = (z_now - z_bar).abs().mean(dim=-1)
            l_copy = (err_copy * m).sum() / m.sum().clamp_min(1.0)

        s_pred += float(l_pred) * bs
        s_copy += float(l_copy) * bs
        n += bs
        if i < max_stat_batches:
            S_list.append(S.float().cpu())
            z_list.append(z_ctx.reshape(-1, model.z_dim).float().cpu())
    S, z = torch.cat(S_list), torch.cat(z_list)
    model.train()
    return {
        "val_l_pred": s_pred / n, "val_copy_baseline": s_copy / n,
        "std_s": float(S.std(dim=0).mean()), "std_z": float(z.std(dim=0).mean()),
        "erank_s": effective_rank(S),
    }


def plot_curves(csv_path: Path, out_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_l_pred"], "-", c="gray", label="train l_pred")
    axes[0].plot(df["epoch"], df["val_l_pred"], "-o", c="black", ms=3, label="val l_pred")
    axes[0].plot(df["epoch"], df["val_copy_baseline"], "--s", c="black", ms=3, label="val copy baseline")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("L1 in latent space")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Prediction loss")
    axes[1].plot(df["epoch"], df["erank_s"], "-o", c="black", ms=3, label="erank(S_t) (left)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("effective rank")
    ax2 = axes[1].twinx()
    ax2.plot(df["epoch"], df["std_s"], "--s", c="gray", ms=3, label="std(S_t) (right)")
    ax2.set_ylabel("mean per-dim std")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8)
    axes[1].set_title("Collapse diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run_name", default="wm_main")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    wm = cfg["wm"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True

    run_dir = Path(cfg["project"]["out_dir"]) / args.run_name
    if (run_dir / "ckpt" / "best.pt").exists() and not (args.overwrite or args.run_name.endswith("debug")):
        raise SystemExit(f"{run_dir} already has a trained model. Use another --run_name or --overwrite.")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    csv_path = run_dir / "log.csv"
    if csv_path.exists():
        csv_path.unlink()
    logger = CSVLogger(csv_path)

    cache = load_cache(cfg["data"]["cache_dir"])
    K = int(wm["k_future"])
    train_ds = WMClipDataset(cache, cache["split"]["train"], wm["t_ctx"], K, wm["clip_stride"])
    val_ds = WMClipDataset(cache, cache["split"]["val"], wm["t_ctx"], K, wm["clip_stride"])
    nw = int(wm["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=wm["batch_size"], shuffle=True, drop_last=True,
                              num_workers=nw, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=nw)

    model = JEPAWorldModel(wm).to(device)
    params = model.trainable_parameters()
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(wm["weight_decay"])}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(wm["lr"]),
    )
    use_amp = False  # wm_main was trained in fp32 (GTX 1650 has no tensor cores)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(wm["epochs"])
    ipe = len(train_loader)
    total = epochs * ipe
    warm = max(1, min(int(wm["warmup_epochs"] * ipe), total // 5))

    def lr_lambda(it: int) -> float:
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * (it - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    tau0, tau1 = float(wm["ema_start"]), float(wm["ema_end"])

    print(f"device={device} (AMP={'on' if use_amp else 'off'}) | params={count_params(model, True):,} | "
          f"train={len(train_ds)} val={len(val_ds)} | {ipe} iters/epoch x {epochs} epochs | run dir: {run_dir}")

    best = float("inf")
    it = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        sum_loss = sum_pred = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            b = prep_batch(batch, device, int(wm["shift_aug"]))
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                out = model.compute_loss(b)
                loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, iter {it}: {float(loss)}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, float(wm["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            tau = tau1 - (tau1 - tau0) * 0.5 * (1.0 + math.cos(math.pi * it / max(1, total)))
            model.update_ema(tau)
            it += 1
            sum_loss += float(loss)
            sum_pred += float(out["l_pred"])
            if it % 20 == 0:
                pbar.set_postfix(loss=f"{float(loss):.3f}", l_pred=f"{float(out['l_pred']):.3f}",
                                 std_s=f"{float(out['std_s']):.2f}")
        ev = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch, "train_loss": sum_loss / ipe, "train_l_pred": sum_pred / ipe,
            "val_l_pred": ev["val_l_pred"], "val_copy_baseline": ev["val_copy_baseline"],
            "std_s": ev["std_s"], "std_z": ev["std_z"], "erank_s": ev["erank_s"],
            "lr": opt.param_groups[0]["lr"], "tau": tau, "epoch_time_s": time.perf_counter() - t0,
        }
        logger.log(row)
        state = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_l_pred": ev["val_l_pred"]}
        torch.save(state, run_dir / "ckpt" / "last.pt")
        if ev["val_l_pred"] < best:
            best = ev["val_l_pred"]
            torch.save(state, run_dir / "ckpt" / "best.pt")
        plot_curves(csv_path, run_dir / "figures" / "curves.png")
        print(f"ep {epoch:3d} | loss {row['train_loss']:.3f} | l_pred tr {row['train_l_pred']:.3f} "
              f"val {row['val_l_pred']:.3f} (copy {row['val_copy_baseline']:.3f}) | std_s {row['std_s']:.3f} "
              f"erank {row['erank_s']:.1f} | lr {row['lr']:.2e} tau {tau:.4f} | {row['epoch_time_s']:.0f}s")

    print(f"\nDone. best val_l_pred = {best:.4f}. Files in {run_dir}")


if __name__ == "__main__":
    main()
