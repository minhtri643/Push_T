"""Data Normalization and Coordinate Conversion Utilities.

Handles MinMax normalization to [-1, 1], continuous angular (sin, cos)
representations, and 4D continuous pose formatting for the Push-T environment.
"""

from typing import Tuple, Union
import numpy as np
import torch

# Coordinate bounds for Push-T environment
WORKSPACE_MIN: float = 0.0
WORKSPACE_MAX: float = 512.0


def normalize_pos(
    pos: Union[np.ndarray, torch.Tensor],
    min_val: float = WORKSPACE_MIN,
    max_val: float = WORKSPACE_MAX,
) -> Union[np.ndarray, torch.Tensor]:
    """Normalizes 2D coordinates from [min_val, max_val] to [-1, 1].

    Args:
        pos: Array or Tensor of shape (..., 2)
        min_val: Minimum spatial coordinate (default 0.0)
        max_val: Maximum spatial coordinate (default 512.0)

    Returns:
        Normalized coordinates in [-1, 1]
    """
    scale = (max_val - min_val) / 2.0
    mid = (max_val + min_val) / 2.0
    return (pos - mid) / scale


def unnormalize_pos(
    pos_norm: Union[np.ndarray, torch.Tensor],
    min_val: float = WORKSPACE_MIN,
    max_val: float = WORKSPACE_MAX,
) -> Union[np.ndarray, torch.Tensor]:
    """Unnormalizes coordinates from [-1, 1] back to [min_val, max_val].

    Args:
        pos_norm: Array or Tensor of shape (..., 2) in [-1, 1]
        min_val: Minimum spatial coordinate
        max_val: Maximum spatial coordinate

    Returns:
        Unnormalized coordinates in [min_val, max_val]
    """
    scale = (max_val - min_val) / 2.0
    mid = (max_val + min_val) / 2.0
    return pos_norm * scale + mid


def angle_to_sin_cos(
    theta: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Converts continuous angle theta in radians to (sin(theta), cos(theta)).

    Args:
        theta: Array or Tensor of angles in radians, shape (...)

    Returns:
        Tuple of (sin, cos) each of shape (...) in [-1, 1]
    """
    if isinstance(theta, torch.Tensor):
        return torch.sin(theta), torch.cos(theta)
    return np.sin(theta), np.cos(theta)


def sin_cos_to_angle(
    sin_t: Union[np.ndarray, torch.Tensor],
    cos_t: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Converts (sin, cos) back to angle theta in [-pi, pi] via atan2.

    Args:
        sin_t: Sin component, shape (...)
        cos_t: Cos component, shape (...)

    Returns:
        Angle theta in radians in [-pi, pi]
    """
    if isinstance(sin_t, torch.Tensor):
        return torch.atan2(sin_t, cos_t)
    return np.arctan2(sin_t, cos_t)


def pose_to_4d(
    x: Union[np.ndarray, torch.Tensor, float],
    y: Union[np.ndarray, torch.Tensor, float],
    theta: Union[np.ndarray, torch.Tensor, float],
    min_val: float = WORKSPACE_MIN,
    max_val: float = WORKSPACE_MAX,
) -> Union[np.ndarray, torch.Tensor]:
    """Converts (x, y, theta) into 4D continuous vector [x_norm, y_norm, sin(theta), cos(theta)] in [-1, 1]^4.

    This representation prevents angle wrap-around discontinuities at +-pi.

    Args:
        x: X-coordinate in workspace units
        y: Y-coordinate in workspace units
        theta: Angle in radians
        min_val: Workspace minimum
        max_val: Workspace maximum

    Returns:
        4D vector of shape (..., 4)
    """
    if isinstance(x, torch.Tensor):
        scale = (max_val - min_val) / 2.0
        mid = (max_val + min_val) / 2.0
        x_norm = (x - mid) / scale
        y_norm = (y - mid) / scale
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        return torch.stack([x_norm, y_norm, sin_t, cos_t], dim=-1)
    else:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        theta_arr = np.asarray(theta, dtype=np.float32)
        scale = (max_val - min_val) / 2.0
        mid = (max_val + min_val) / 2.0
        x_norm = (x_arr - mid) / scale
        y_norm = (y_arr - mid) / scale
        sin_t = np.sin(theta_arr)
        cos_t = np.cos(theta_arr)
        return np.stack([x_norm, y_norm, sin_t, cos_t], axis=-1)


def four_d_to_pose(
    p4: Union[np.ndarray, torch.Tensor],
    min_val: float = WORKSPACE_MIN,
    max_val: float = WORKSPACE_MAX,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Decodes 4D continuous vector [x_norm, y_norm, sin(theta), cos(theta)] back to (x, y, theta).

    Args:
        p4: Array or Tensor of shape (..., 4)
        min_val: Workspace minimum
        max_val: Workspace maximum

    Returns:
        Tuple of (x, y, theta) in original workspace units and radians
    """
    scale = (max_val - min_val) / 2.0
    mid = (max_val + min_val) / 2.0

    if isinstance(p4, torch.Tensor):
        x = p4[..., 0] * scale + mid
        y = p4[..., 1] * scale + mid
        sin_t = p4[..., 2]
        cos_t = p4[..., 3]
        theta = torch.atan2(sin_t, cos_t)
        return x, y, theta
    else:
        x = p4[..., 0] * scale + mid
        y = p4[..., 1] * scale + mid
        sin_t = p4[..., 2]
        cos_t = p4[..., 3]
        theta = np.arctan2(sin_t, cos_t)
        return x, y, theta


def normalize_image(
    img: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Normalizes image to float32 [0, 1] in Channel-First format (C, H, W).

    Args:
        img: Array of shape (H, W, C) or (B, H, W, C) or Tensor

    Returns:
        Normalized image of shape (C, H, W) or (B, C, H, W) in [0.0, 1.0]
    """
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        # Transpose from (..., H, W, C) to (..., C, H, W)
        if img.ndim == 3:
            return np.transpose(img, (2, 0, 1))
        elif img.ndim == 4:
            return np.transpose(img, (0, 3, 1, 2))
        return img
    elif isinstance(img, torch.Tensor):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.max() > 1.0:
            img = img.float() / 255.0
        else:
            img = img.float()
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            return img.permute(2, 0, 1)
        elif img.ndim == 4 and img.shape[-1] in (1, 3):
            return img.permute(0, 3, 1, 2)
        return img
    raise TypeError(f"Unsupported type {type(img)}")
