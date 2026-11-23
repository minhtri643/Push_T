"""Evaluation and Loss Metrics for Push-T Trajectory Planning.

Implements:
1. Final Displacement Error (FDE): f(dx, dy, dtheta) = sqrt(w_xy * (dx_H^2 + dy_H^2) + w_theta * dtheta_H^2)
2. Average Displacement Error (ADE)
3. Target Coverage IoU using Shapely polygon intersection
"""

from typing import Tuple, Union, Optional
import numpy as np
import torch
import shapely.geometry as sg
from shapely.affinity import rotate, translate


def compute_fde_torch(
    target_path: torch.Tensor,
    pred_path: torch.Tensor,
    w_xy: float = 1.0,
    w_theta: float = 1.0,
    return_squared: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Computes differentiable Final Displacement Error (FDE) in PyTorch.

    FDE = sqrt(w_xy * (dx_H^2 + dy_H^2) + w_theta * dtheta_H^2)
    where dtheta_H = atan2(sin(theta_H - hat_theta_H), cos(theta_H - hat_theta_H)).

    Directly operates on 4D continuous vectors [x, y, sin(theta), cos(theta)].

    Args:
        target_path: Ground truth trajectory of shape (B, H, 4) in [-1, 1]^4
        pred_path: Predicted trajectory of shape (B, H, 4) in [-1, 1]^4
        w_xy: Weight for position displacement (dx^2 + dy^2)
        w_theta: Weight for angular displacement (dtheta^2)
        return_squared: If True, returns FDE^2 (directly used for loss penalty)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        FDE or FDE^2 tensor
    """
    # Extract final timestep H (index -1)
    target_H = target_path[:, -1, :]  # (B, 4)
    pred_H = pred_path[:, -1, :]      # (B, 4)

    dx = target_H[:, 0] - pred_H[:, 0]
    dy = target_H[:, 1] - pred_H[:, 1]

    # Trigonometric identity for angle difference without explicit atan2 inversion:
    # sin(theta - hat_theta) = sin(theta) * cos(hat_theta) - cos(theta) * sin(hat_theta)
    # cos(theta - hat_theta) = cos(theta) * cos(hat_theta) + sin(theta) * sin(hat_theta)
    s_tgt, c_tgt = target_H[:, 2], target_H[:, 3]
    s_prd, c_prd = pred_H[:, 2], pred_H[:, 3]

    # Normalize pred vector to avoid scale artifacts
    norm_prd = torch.sqrt(s_prd**2 + c_prd**2 + 1e-6)
    s_p = s_prd / norm_prd
    c_p = c_prd / norm_prd

    sin_diff = s_tgt * c_p - c_tgt * s_p
    cos_diff = c_tgt * c_p + s_tgt * s_p
    eps = 1e-6 * torch.where(cos_diff >= 0, torch.ones_like(cos_diff), -torch.ones_like(cos_diff))
    dtheta = torch.atan2(sin_diff, cos_diff + eps)

    fde_sq = w_xy * (dx**2 + dy**2) + w_theta * (dtheta**2)

    if return_squared:
        res = fde_sq
    else:
        res = torch.sqrt(fde_sq + 1e-8)

    if reduction == "mean":
        return res.mean()
    elif reduction == "sum":
        return res.sum()
    return res


def compute_ade_torch(
    target_path: torch.Tensor,
    pred_path: torch.Tensor,
    w_xy: float = 1.0,
    w_theta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Computes Average Displacement Error (ADE) across all timesteps 1..H in PyTorch.

    Args:
        target_path: Shape (B, H, 4)
        pred_path: Shape (B, H, 4)
        w_xy: Weight for position displacement
        w_theta: Weight for angular displacement
        reduction: 'mean', 'sum', or 'none'

    Returns:
        ADE tensor
    """
    dx = target_path[..., 0] - pred_path[..., 0]
    dy = target_path[..., 1] - pred_path[..., 1]

    s_tgt, c_tgt = target_path[..., 2], target_path[..., 3]
    s_prd, c_prd = pred_path[..., 2], pred_path[..., 3]
    norm_prd = torch.sqrt(s_prd**2 + c_prd**2 + 1e-6)
    s_p = s_prd / norm_prd
    c_p = c_prd / norm_prd

    sin_diff = s_tgt * c_p - c_tgt * s_p
    cos_diff = c_tgt * c_p + s_tgt * s_p
    eps = 1e-6 * torch.where(cos_diff >= 0, torch.ones_like(cos_diff), -torch.ones_like(cos_diff))
    dtheta = torch.atan2(sin_diff, cos_diff + eps)

    step_de = torch.sqrt(w_xy * (dx**2 + dy**2) + w_theta * (dtheta**2) + 1e-8)
    ade = step_de.mean(dim=-1)  # Average over horizon H

    if reduction == "mean":
        return ade.mean()
    elif reduction == "sum":
        return ade.sum()
    return ade


def compute_fde_numpy(
    target_path: np.ndarray,
    pred_path: np.ndarray,
    w_xy: float = 1.0,
    w_theta: float = 1.0,
) -> float:
    """Computes FDE for NumPy arrays (evaluation script).

    Supports:
        (4,) vs (4,) -> single step vector
        (H, 4) vs (H, 4) -> trajectory
        (B, H, 4) vs (B, H, 4) -> batch trajectory
    """
    target = np.asarray(target_path, dtype=np.float32)
    pred = np.asarray(pred_path, dtype=np.float32)

    # If 1D single-step vector (4,)
    if target.ndim == 1:
        target_H = target[np.newaxis, :]
        pred_H = pred[np.newaxis, :]
    elif target.ndim == 2:
        target_H = target[-1:, :]
        pred_H = pred[-1:, :]
    else:
        target_H = target[:, -1, :]
        pred_H = pred[:, -1, :]

    dx = target_H[:, 0] - pred_H[:, 0]
    dy = target_H[:, 1] - pred_H[:, 1]

    s_tgt, c_tgt = target_H[:, 2], target_H[:, 3]
    s_prd, c_prd = pred_H[:, 2], pred_H[:, 3]
    norm_prd = np.sqrt(s_prd**2 + c_prd**2 + 1e-6)
    s_p = s_prd / norm_prd
    c_p = c_prd / norm_prd

    sin_diff = s_tgt * c_p - c_tgt * s_p
    cos_diff = c_tgt * c_p + s_tgt * s_p
    eps = 1e-6 * np.where(cos_diff >= 0, np.ones_like(cos_diff), -np.ones_like(cos_diff))
    dtheta = np.arctan2(sin_diff, cos_diff + eps)

    fde = np.sqrt(w_xy * (dx**2 + dy**2) + w_theta * (dtheta**2))
    return float(np.mean(fde))


def compute_ade_numpy(
    target_path: np.ndarray,
    pred_path: np.ndarray,
    w_xy: float = 1.0,
    w_theta: float = 1.0,
) -> float:
    """Computes ADE for NumPy arrays."""
    if target_path.ndim == 2:
        target_path = target_path[np.newaxis, ...]
        pred_path = pred_path[np.newaxis, ...]

    dx = target_path[..., 0] - pred_path[..., 0]
    dy = target_path[..., 1] - pred_path[..., 1]

    s_tgt, c_tgt = target_path[..., 2], target_path[..., 3]
    s_prd, c_prd = pred_path[..., 2], pred_path[..., 3]
    norm_prd = np.sqrt(s_prd**2 + c_prd**2 + 1e-8)
    s_prd = s_prd / norm_prd
    c_prd = c_prd / norm_prd

    sin_diff = s_tgt * c_prd - c_tgt * s_prd
    cos_diff = c_tgt * c_prd + s_tgt * s_prd
    dtheta = np.arctan2(sin_diff, cos_diff)

    step_de = np.sqrt(w_xy * (dx**2 + dy**2) + w_theta * (dtheta**2))
    ade = np.mean(step_de, axis=-1)
    return float(np.mean(ade))


def create_tee_polygon(
    x: float,
    y: float,
    theta: float,
    scale: float = 30.0,
    length: float = 4.0,
) -> sg.Polygon:
    """Constructs a Shapely polygon representing the canonical Push-T Tee shape.

    Args:
        x: Body X position
        y: Body Y position
        theta: Body rotation angle in radians
        scale: Scaling factor (default 30.0)
        length: Tee length factor (default 4.0)

    Returns:
        Shapely Polygon in world coordinates
    """
    # Crossbar: [(-length * scale / 2, scale), (length * scale / 2, scale), (length * scale / 2, 0), (-length * scale / 2, 0)]
    crossbar = sg.box(-length * scale / 2.0, 0.0, length * scale / 2.0, scale)
    # Stem: [(-scale / 2, scale), (-scale / 2, length * scale), (scale / 2, length * scale), (scale / 2, scale)]
    stem = sg.box(-scale / 2.0, scale, scale / 2.0, length * scale)
    tee = crossbar.union(stem)

    # Rotate by theta (degrees for Shapely) around (0,0) then translate to (x, y)
    rotated = rotate(tee, theta * 180.0 / np.pi, origin=(0.0, 0.0))
    translated = translate(rotated, xoff=x, yoff=y)
    return translated


def compute_target_coverage_iou(
    block_pose: Union[np.ndarray, Tuple[float, float, float]],
    goal_pose: Union[np.ndarray, Tuple[float, float, float]] = (256.0, 256.0, np.pi / 4.0),
) -> float:
    """Computes exact Intersection over Union (IoU) coverage between T-block and target goal.

    Args:
        block_pose: (x, y, theta) of T-block
        goal_pose: (x, y, theta) of target goal

    Returns:
        IoU coverage in [0.0, 1.0]
    """
    bx, by, btheta = float(block_pose[0]), float(block_pose[1]), float(block_pose[2])
    gx, gy, gtheta = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])

    poly_block = create_tee_polygon(bx, by, btheta)
    poly_goal = create_tee_polygon(gx, gy, gtheta)

    intersection_area = poly_block.intersection(poly_goal).area
    union_area = poly_block.union(poly_goal).area

    if union_area <= 1e-8:
        return 0.0
    return float(intersection_area / union_area)
