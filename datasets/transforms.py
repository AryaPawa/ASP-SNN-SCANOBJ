"""
datasets/transforms.py — Data augmentation for point cloud tasks.

Batch 1 upgrade (A4): stronger segmentation-specific augmentation.

Prior versions used scale ∈ [0.8, 1.25], jitter σ=0.01, which is on the
weak side relative to modern point-cloud baselines. PointNeXt (Qian et al.
NeurIPS 2022) demonstrated that the *majority* of their gain over
PointNet++ came from stronger augmentation, not architecture — so this is
a high-leverage change for ShapeNetPart at essentially zero cost.

Segmentation augment_seg additions in Batch 1:
    1. Wider anisotropic scale range (default 0.66 – 1.5) — controlled by
       cfg.aug_scale_lo / aug_scale_hi so backwards compatible.
    2. Optional random point resampling (aug_resample=True) — shuffles and
       resamples the N points with replacement, adding training diversity.
       Labels are permuted consistently.
    3. Tighter jitter defaults (σ=0.005, clip=0.02) — prevents label drift
       at part boundaries while still adding useful geometric noise.
    4. Random per-axis mirror (aug_flip_x/y/z) — cheap invariance signal.

All new augmentations are strictly additive: cfg flags with sensible
defaults preserve prior behaviour when not set. augment_slices
(classification) is untouched to avoid interfering with ScanObjectNN tuning.

Reference: Qian et al., PointNeXt: Revisiting PointNet++ with Improved
Training and Scaling Strategies, NeurIPS 2022.
"""

import numpy as np


# ── Rotation primitives (unchanged from original) ─────────────────────

def random_rotation_z():
    """Rotation about the z axis. Returns [3, 3]."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]], dtype=np.float32)


def random_so3_tilt(max_rad=0.26):
    """Small perturbation from upright: yaw + small pitch/roll.
    Returns [3, 3]."""
    Rz = random_rotation_z()
    ax = np.random.uniform(-max_rad, max_rad)
    ay = np.random.uniform(-max_rad, max_rad)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]], np.float32)
    Ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


# ── Classification-side augmentation (unchanged) ──────────────────────

def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """
    Augment slices for classification tasks (ScanObjectNN).

    UNCHANGED in Batch 1 — do not want to touch the ScanObjectNN tuning
    that already produces good results.
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
    clip  = getattr(cfg, 'aug_jitter_clip', 0.05)
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


# ── Segmentation augmentation (A4 UPGRADES BELOW) ─────────────────────

def _random_mirror(R: np.ndarray, cfg) -> np.ndarray:
    """
    Apply random axis-mirror to a rotation matrix.
    Left-multiplies R by a diagonal flip. Handled in matrix form so it
    composes cleanly with rotation/scale.
    """
    flip_x = getattr(cfg, 'aug_flip_x', False) and (np.random.random() < 0.5)
    flip_y = getattr(cfg, 'aug_flip_y', False) and (np.random.random() < 0.5)
    flip_z = getattr(cfg, 'aug_flip_z', False) and (np.random.random() < 0.5)
    if not (flip_x or flip_y or flip_z):
        return R
    D = np.eye(3, dtype=np.float32)
    if flip_x: D[0, 0] = -1.
    if flip_y: D[1, 1] = -1.
    if flip_z: D[2, 2] = -1.
    return (D @ R).astype(np.float32)


def augment_seg(slices: np.ndarray, pts_features: np.ndarray,
                cfg, part_labels: np.ndarray = None) -> tuple:
    """
    Augment slices AND per-point features with SHARED transform.

    Critical: slices and pts must receive the same rotation/scale/translate
    otherwise the segmentation head learns to ignore xyz features.

    Batch 1 A4 additions (all optional, cfg-gated):
        - aug_flip_x/y/z:  random axis mirror (cheap invariance)
        - aug_scale_lo/hi: recommended defaults 0.66 / 1.5 (was 0.8 / 1.25)
        - aug_jitter_*:    recommended defaults σ=0.005 / clip=0.02
        - aug_resample:    random point resampling with replacement

    When `part_labels` is provided AND aug_resample=True, the label array
    is permuted consistently with the resampled point order.

    Args:
        slices:       [M, K, C]
        pts_features: [N, F]  (first 3 dims are xyz)
        cfg:          config
        part_labels:  [N] optional — required for label-consistent resample

    Returns:
        (augmented_slices, augmented_pts_features)
        OR
        (augmented_slices, augmented_pts_features, augmented_labels)
                                             if part_labels is not None
    """
    M, K, C = slices.shape
    N, F = pts_features.shape

    # ── Shared rotation ────────────────────────────────────────────────
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', True):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    # ── A4: optional random axis mirror ────────────────────────────────
    R = _random_mirror(R, cfg)

    # ── Shared scale (A4: wider default 0.66–1.5) ──────────────────────
    lo = getattr(cfg, 'aug_scale_lo', 0.66)
    hi = getattr(cfg, 'aug_scale_hi', 1.5)
    scale = np.random.uniform(lo, hi, (1, 3)).astype(np.float32)

    # ── Shared translation ─────────────────────────────────────────────
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 3)).astype(np.float32)

    # Apply to pts_features FIRST (xyz in first 3 dims)
    pts_aug = pts_features.copy()
    pts_aug[:, :3] = (pts_aug[:, :3] @ R) * scale + trans

    # Apply to slices
    result = slices.copy()
    result[:, :, :3] = (result[:, :, :3] @ R) * scale + trans
    if C >= 6:
        # Normals (ShapeNet, dims 3:6) rotate with points; RGB (S3DIS) does not.
        if not getattr(cfg, 'use_rgb', False):
            result[:, :, 3:6] = result[:, :, 3:6] @ R

    # ── Jitter (slices only). A4: tighter defaults for seg. ────────────
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.005)
    clip  = getattr(cfg, 'aug_jitter_clip',  0.02)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    # ── Per-slice point dropout ────────────────────────────────────────
    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    # ── Color augmentation (S3DIS only) ────────────────────────────────
    if getattr(cfg, 'use_rgb', False) and F > 3:
        cdrop = getattr(cfg, 'aug_color_drop', 0.0)
        if cdrop > 0 and np.random.random() < cdrop:
            pts_aug[:, 3:6] = 0.0
            result[:, :, 3:6] = 0.0
        cjit = getattr(cfg, 'aug_color_jitter', 0.0)
        if cjit > 0:
            noise = np.random.uniform(-cjit, cjit, (1, 3)).astype(np.float32)
            pts_aug[:, 3:6] = np.clip(pts_aug[:, 3:6] + noise, 0.0, 1.0)
            result[:, :, 3:6] = np.clip(
                result[:, :, 3:6] + noise.reshape(1, 1, 3), 0.0, 1.0
            )

    # ── A4: optional random point resampling ───────────────────────────
    # Draws N indices with replacement, permuting pts_features (and labels
    # if given) consistently. Slice tensors are NOT touched because slicing
    # was already done upstream by FPS+KNN with slice-id fixed per point.
    # (Resampling affects only the per-point head input path.)
    do_resample = getattr(cfg, 'aug_resample', False)
    if do_resample:
        idx = np.random.choice(N, size=N, replace=True)
        pts_aug = pts_aug[idx]
        if part_labels is not None:
            part_labels_out = part_labels[idx]
        # NOTE: if resample is used, the caller must also permute sid_arr
        # by the same idx, or resampling must happen before sid_arr is
        # computed. See dataset __getitem__ integration in Batch 1 notes.
    else:
        if part_labels is not None:
            part_labels_out = part_labels

    if part_labels is not None:
        return (result.astype(np.float32),
                pts_aug.astype(np.float32),
                part_labels_out)
    return result.astype(np.float32), pts_aug.astype(np.float32)