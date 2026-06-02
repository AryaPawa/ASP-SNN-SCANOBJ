"""
datasets/transforms.py — Point cloud augmentation functions.

Two main entry points:
    augment_slices()  — for classification (modifies slices only)
    augment_seg()     — for segmentation (shared transform on slices + pts)

All augmentations are numpy-based and applied in __getitem__.
"""

import numpy as np


def random_rotation_z() -> np.ndarray:
    """Random rotation matrix around z-axis. Returns [3, 3]."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], np.float32)


def random_so3_tilt(max_rad: float = 0.26) -> np.ndarray:
    """SO(3) tilt: random z-rotation + small x/y tilts. Returns [3, 3]."""
    Rz = random_rotation_z()
    ax = np.random.uniform(-max_rad, max_rad)
    ay = np.random.uniform(-max_rad, max_rad)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]], np.float32)
    Ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """
    Augment slices for classification tasks (ScanObjectNN).

    Args:
        slices: [M, K, C]
        cfg:    config with aug_* parameters

    Returns:
        augmented slices: [M, K, C]
    """
    M, K, C = slices.shape
    result = slices.copy()

    # Rotation
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', False):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    result[:, :, :3] = result[:, :, :3] @ R
    if C >= 6:
        result[:, :, 3:6] = result[:, :, 3:6] @ R

    # Anisotropic scale
    lo = getattr(cfg, 'aug_scale_lo', 0.8)
    hi = getattr(cfg, 'aug_scale_hi', 1.25)
    scale = np.random.uniform(lo, hi, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] *= scale

    # Translation
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] += trans

    # Jitter
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.01)
    clip = getattr(cfg, 'aug_jitter_clip', 0.05)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    # Per-slice point dropout
    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    # Global point dropout (ScanObjectNN)
    pdrop = getattr(cfg, 'aug_point_dropout', 0.0)
    if pdrop > 0:
        ratio = np.random.random() * pdrop
        for m in range(M):
            drop_idx = np.where(np.random.random(K) < ratio)[0]
            if len(drop_idx) > 0:
                result[m, drop_idx] = result[m, 0]

    return result.astype(np.float32)


def augment_seg(slices: np.ndarray, pts_features: np.ndarray,
                cfg) -> tuple:
    """
    Augment slices AND per-point features with SHARED transform.

    Critical: slices and pts must receive the same rotation/scale/translate
    otherwise the segmentation head learns to ignore xyz features.

    Args:
        slices:       [M, K, C]
        pts_features: [N, F]  (first 3 dims are xyz)
        cfg:          config

    Returns:
        (augmented_slices, augmented_pts_features)
    """
    M, K, C = slices.shape
    N, F = pts_features.shape

    # Shared rotation
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', True):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    # Shared scale
    lo = getattr(cfg, 'aug_scale_lo', 0.8)
    hi = getattr(cfg, 'aug_scale_hi', 1.25)
    scale = np.random.uniform(lo, hi, (1, 3)).astype(np.float32)

    # Shared translation
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 3)).astype(np.float32)

    # Apply to pts_features FIRST (xyz in first 3 dims)
    pts_aug = pts_features.copy()
    pts_aug[:, :3] = (pts_aug[:, :3] @ R) * scale + trans

    # Apply to slices
    result = slices.copy()
    result[:, :, :3] = (result[:, :, :3] @ R) * scale + trans
    if C >= 6:
        # Rotate normals / rgb stays untouched (normals are in 3:6 for ShapeNet)
        # For S3DIS, channels 3:6 are RGB — do NOT rotate them
        if not getattr(cfg, 'use_rgb', False):
            result[:, :, 3:6] = result[:, :, 3:6] @ R

    # Jitter (slices only)
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.01)
    clip = getattr(cfg, 'aug_jitter_clip', 0.05)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    # Per-slice dropout
    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    # Color augmentation (S3DIS)
    if getattr(cfg, 'use_rgb', False) and F > 3:
        # Color dropout: zero out RGB with probability
        cdrop = getattr(cfg, 'aug_color_drop', 0.0)
        if cdrop > 0 and np.random.random() < cdrop:
            pts_aug[:, 3:6] = 0.0
            result[:, :, 3:6] = 0.0

        # Color jitter
        cjit = getattr(cfg, 'aug_color_jitter', 0.0)
        if cjit > 0:
            noise = np.random.uniform(-cjit, cjit, (1, 3)).astype(np.float32)
            pts_aug[:, 3:6] = np.clip(pts_aug[:, 3:6] + noise, 0.0, 1.0)
            result[:, :, 3:6] = np.clip(result[:, :, 3:6] + noise.reshape(1,1,3), 0.0, 1.0)

    return result.astype(np.float32), pts_aug.astype(np.float32)
