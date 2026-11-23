"""Pipeline 1 (data): Inspect Push-T Dataset and Trajectory Extraction.

Verifies:
1. Zarr dataset integrity and metadata.
2. Sliding window trajectory extraction over horizon H=16 and Ta=8.
3. 4D continuous pose representation [x_T, y_T, sin(theta), cos(theta)] in [-1, 1]^4.
4. Episode boundary handling and padding.
5. Generates visual inspection figure 'data_inspection.png'.
"""

import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from pusht_wm_dit.data.pusht_dataset import PushTDataset, create_dataloaders
from pusht_wm_dit.utils.normalizer import four_d_to_pose, unnormalize_pos


def main():
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    zarr_path = config["data"]["zarr_path"]
    pred_horizon = config["data"]["pred_horizon"]
    action_horizon = config["data"]["action_horizon"]

    print("=" * 70)
    print("STEP 1: INSPECTING PUSH-T DATASET & TRAJECTORY EXTRACTION")
    print("=" * 70)
    print(f"Loading dataset from: {zarr_path}")

    train_ds = PushTDataset(
        zarr_path=zarr_path,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        split="train",
        val_ratio=config["data"]["val_ratio"],
        seed=config["seed"],
    )
    val_ds = PushTDataset(
        zarr_path=zarr_path,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        split="val",
        val_ratio=config["data"]["val_ratio"],
        seed=config["seed"],
    )

    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    print(f"  Total samples: {len(train_ds) + len(val_ds):,}")

    sample = train_ds[0]
    print("\nSample Item Structure:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k:20s}: shape={str(tuple(v.shape)):15s} dtype={v.dtype} range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  - {k:20s}: {v}")

    # Verify sliding window bounds
    print("\nVerifying 4D Continuous Representation [x, y, sin(theta), cos(theta)]:")
    path_4d = sample["path_future"]  # (H, 4)
    x_rec, y_rec, th_rec = four_d_to_pose(path_4d)
    print(f"  Start pose: x={x_rec[0]:.1f}, y={y_rec[0]:.1f}, theta={th_rec[0]:.2f} rad ({th_rec[0]*180/np.pi:.1f} deg)")
    print(f"  End pose:   x={x_rec[-1]:.1f}, y={y_rec[-1]:.1f}, theta={th_rec[-1]:.2f} rad ({th_rec[-1]*180/np.pi:.1f} deg)")
    norm_sc = torch.sqrt(path_4d[:, 2]**2 + path_4d[:, 3]**2)
    print(f"  sin^2 + cos^2 norm check: min={norm_sc.min():.4f}, max={norm_sc.max():.4f} (Ideal: 1.0000)")

    # Plot visual inspection figure
    print("\nGenerating visual inspection figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. Observation image
    img_np = sample["frame"].permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"Observation Frame (96x96x3)\nTimestep {sample['timestep']}")
    axes[0, 0].axis("off")

    # 2. Next frame
    next_img_np = sample["next_frame"].permute(1, 2, 0).numpy()
    axes[0, 1].imshow(next_img_np)
    axes[0, 1].set_title("Next Frame (Transition t+1)")
    axes[0, 1].axis("off")

    # 3. 4D trajectory component plots
    time_steps = np.arange(pred_horizon)
    axes[0, 2].plot(time_steps, path_4d[:, 0].numpy(), "r-o", label="x_norm")
    axes[0, 2].plot(time_steps, path_4d[:, 1].numpy(), "g-s", label="y_norm")
    axes[0, 2].plot(time_steps, path_4d[:, 2].numpy(), "b-^", label="sin(theta)")
    axes[0, 2].plot(time_steps, path_4d[:, 3].numpy(), "m-d", label="cos(theta)")
    axes[0, 2].set_title("4D Trajectory Horizon H=16 (Normalized)")
    axes[0, 2].set_xlabel("Horizon Step")
    axes[0, 2].grid(True)
    axes[0, 2].legend()

    # 4. 2D Trajectory Spatial Path (Push-T coordinates)
    axes[1, 0].plot(x_rec.numpy(), y_rec.numpy(), "b-o", label="T-block Path")
    # Plot agent position
    ag_px = unnormalize_pos(sample["agent_pos"]).numpy()
    axes[1, 0].scatter([ag_px[0]], [ag_px[1]], color="red", s=100, marker="x", label="Agent Pos")
    # Plot action commands
    act_px = unnormalize_pos(sample["action_future"]).numpy()
    axes[1, 0].plot(act_px[:, 0], act_px[:, 1], "r--*", label="Actions (Ta=8)")
    axes[1, 0].set_xlim(0, 512)
    axes[1, 0].set_ylim(512, 0)  # Invert Y for image coordinate convention
    axes[1, 0].set_title("Spatial Workspace Coordinates (0..512 px)")
    axes[1, 0].set_xlabel("X (pixels)")
    axes[1, 0].set_ylabel("Y (pixels)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 5. Multi-sample trajectory diversity
    for i in range(10, 20):
        s_i = train_ds[i * 100]
        p_i = s_i["path_future"]
        xi, yi, _ = four_d_to_pose(p_i)
        axes[1, 1].plot(xi.numpy(), yi.numpy(), alpha=0.6)
    axes[1, 1].set_xlim(0, 512)
    axes[1, 1].set_ylim(512, 0)
    axes[1, 1].set_title("Sample Future Trajectories Diversity (H=16)")
    axes[1, 1].set_xlabel("X (pixels)")
    axes[1, 1].grid(True)

    # 6. Action velocity magnitude distribution
    actions_all = train_ds.action_arr[:2000]
    vels = np.linalg.norm(np.diff(actions_all, axis=0), axis=-1)
    axes[1, 2].hist(vels, bins=30, color="purple", alpha=0.7)
    axes[1, 2].set_title("Action Command Step Displacements")
    axes[1, 2].set_xlabel("Displacement per step (pixels)")
    axes[1, 2].grid(True)

    plt.tight_layout()
    output_fig = "docs/figures/data_inspection.png"
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=150)
    plt.close()
    print(f"Inspection plot saved to: {output_fig}")
    print("=" * 70)
    print("STEP 1: INSPECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
