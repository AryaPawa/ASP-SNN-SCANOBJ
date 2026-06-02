"""
datasets/s3dis.py — S3DIS dataset for indoor scene segmentation.

Stanford Large-Scale 3D Indoor Spaces: 6 areas, 271 rooms, 13 classes.
Protocol: train on Areas 1,2,3,4,6 — test on Area 5.

Each room is stored as an .npy file with columns:
    [x, y, z, r, g, b, semantic_label]  (RGB in 0-255)

Training: random 1m x 1m blocks, N=4096 points per block.
Testing:  sliding-window blocks over full rooms, aggregate predictions.

Returns per sample:
    slices        [M, K, C]    C = in_channels from config
    geo           [M, 8]       geometry descriptors
    pts_features  [N, F]       per-point features for PerPointBranch
    sid_arr       [N]          slice assignment
    sem_labels    [N]          semantic labels (0-12)
    cat_id        0            dummy (no category conditioning)
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices
from .transforms import augment_seg


# 13 semantic classes
CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
    'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
]
NUM_CLASSES = 13

# Areas for train/test split
TRAIN_AREAS = [1, 2, 3, 4, 6]
TEST_AREA = 5


class S3DISDataset(Dataset):
    """
    S3DIS dataset with block-based sampling.

    During training: randomly sample blocks from rooms.
    During testing:  iterate over all blocks in test area rooms.
    """

    def __init__(self, data_dir: str, split: str, cfg=None):
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 4096)
        self.block_size = getattr(cfg, 'block_size', 1.0)
        self.use_rgb = getattr(cfg, 'use_rgb', True)
        self.use_height = getattr(cfg, 'use_height', True)

        test_area = getattr(cfg, 'test_area', 5)
        if split == 'train':
            areas = [a for a in [1, 2, 3, 4, 5, 6] if a != test_area]
        else:
            areas = [test_area]

        # Load all room files
        self.rooms = []
        for area in areas:
            area_dir = os.path.join(data_dir, f"Area_{area}")
            if not os.path.isdir(area_dir):
                raise FileNotFoundError(
                    f"S3DIS area directory not found: {area_dir}\n"
                    f"Run: python datasets/download.py --s3dis"
                )
            npy_files = sorted(glob.glob(os.path.join(area_dir, "*.npy")))
            for npy_path in npy_files:
                room_data = np.load(npy_path)  # [N, 7]: x,y,z,r,g,b,label
                self.rooms.append(room_data.astype(np.float32))

        if len(self.rooms) == 0:
            raise FileNotFoundError(
                f"No .npy room files found for areas {areas} in {data_dir}"
            )

        # For training: create a flat index of (room_idx, point_count)
        # so we can sample uniformly across rooms proportional to size
        self.room_sizes = [len(r) for r in self.rooms]
        self.total_points = sum(self.room_sizes)

        if split == 'train':
            # Each "sample" is one random block — we define epoch length
            # as total_points // n_points to see each point ~once per epoch
            self._len = self.total_points // self.n_points
        else:
            # For testing: pre-compute all block centres for sliding window
            self.test_blocks = self._precompute_test_blocks()
            self._len = len(self.test_blocks)

        print(f"[S3DIS] '{split}': {len(self.rooms)} rooms, "
              f"{self.total_points:,} points, {self._len} samples/epoch")

    def _precompute_test_blocks(self):
        """Pre-compute (room_idx, cx, cy) for sliding-window test blocks."""
        stride = self.block_size * 0.5  # 50% overlap
        blocks = []
        for ri, room in enumerate(self.rooms):
            xyz = room[:, :3]
            x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
            x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
            cx = x_min + self.block_size / 2
            while cx < x_max:
                cy = y_min + self.block_size / 2
                while cy < y_max:
                    blocks.append((ri, cx, cy))
                    cy += stride
                cx += stride
        return blocks

    def __len__(self):
        return self._len

    def _sample_block(self, room: np.ndarray,
                      cx: float = None, cy: float = None):
        """
        Extract a block of points from a room.

        Args:
            room: [N, 7] full room data
            cx, cy: block centre (None = random for training)

        Returns:
            block: [n_points, 7]
        """
        xyz = room[:, :3]
        half = self.block_size / 2

        if cx is None:
            # Random block centre (training)
            x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
            x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
            cx = np.random.uniform(x_min + half, max(x_min + half, x_max - half))
            cy = np.random.uniform(y_min + half, max(y_min + half, y_max - half))

        # Select points within block
        mask = (
            (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
            (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
        )
        block_pts = room[mask]

        if len(block_pts) == 0:
            # Fallback: take nearest n_points to centre
            dists = np.linalg.norm(xyz[:, :2] - np.array([cx, cy]), axis=1)
            idx = np.argsort(dists)[:self.n_points]
            block_pts = room[idx]

        # Sample to exact n_points
        if len(block_pts) >= self.n_points:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=False)
        else:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=True)
        return block_pts[choice]

    def _prepare_features(self, block: np.ndarray):
        """
        Prepare block data into sliceable point cloud and per-point features.

        Args:
            block: [N, 7]  x,y,z,r,g,b,label

        Returns:
            pts_for_slicing: [N, C]  for encoder (C matches in_channels)
            pts_features:    [N, F]  for PerPointBranch
            sem_labels:      [N]     int labels (0-12)
        """
        xyz = block[:, :3].copy()
        rgb = block[:, 3:6].copy() / 255.0  # normalise to [0,1]
        labels = block[:, 6].astype(np.int64)

        # Centre xyz within the block
        xyz = xyz - xyz.mean(axis=0)

        # Normalised height feature
        z_vals = block[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        if z_max - z_min > 1e-6:
            height = ((z_vals - z_min) / (z_max - z_min)).astype(np.float32)
        else:
            height = np.zeros(len(z_vals), dtype=np.float32)

        # Build slicing input: xyz + rgb + (optional height)
        # The encoder expects in_channels dimensions
        parts = [xyz]
        if self.use_rgb:
            parts.append(rgb)
        if self.use_height:
            parts.append(height.reshape(-1, 1))
        pts_for_slicing = np.concatenate(parts, axis=1).astype(np.float32)

        # Per-point features for PerPointBranch (same channels)
        pts_features = pts_for_slicing.copy()

        return pts_for_slicing, pts_features, labels

    def __getitem__(self, idx):
        if self.split == 'train':
            # Random room, random block
            room_idx = np.random.randint(0, len(self.rooms))
            block = self._sample_block(self.rooms[room_idx])
        else:
            # Deterministic test block
            room_idx, cx, cy = self.test_blocks[idx]
            block = self._sample_block(self.rooms[room_idx], cx, cy)

        pts_for_slicing, pts_features, sem_labels = self._prepare_features(block)

        # Slice
        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 256)
        fps_seed = idx if self.split == 'test' else None
        slices, geo, anchor_xyz = slice_point_cloud(pts_for_slicing, M, K, seed=fps_seed)

        # Assign points to slices (using xyz only)
        sid_arr = assign_points_to_slices(
            pts_for_slicing[:, :3], anchor_xyz
        )

        # Augment (training only)
        if self.split == 'train' and self.cfg is not None:
            slices, pts_features = augment_seg(slices, pts_features, self.cfg)

        return (
            slices.astype(np.float32),          # [M, K, C]
            geo.astype(np.float32),             # [M, 8]
            pts_features.astype(np.float32),    # [N, F]
            sid_arr.astype(np.int64),           # [N]
            sem_labels,                         # [N]
            0,                                  # dummy cat_id
        )


def compute_class_weights(data_dir: str, test_area: int = 5) -> np.ndarray:
    """
    Compute inverse-frequency class weights from training areas.

    Returns:
        weights: [13] float32 normalised so max = 1.0
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for area in [a for a in [1, 2, 3, 4, 5, 6] if a != test_area]:
        area_dir = os.path.join(data_dir, f"Area_{area}")
        for npy_path in glob.glob(os.path.join(area_dir, "*.npy")):
            room = np.load(npy_path)
            labels = room[:, 6].astype(int)
            for c in range(NUM_CLASSES):
                counts[c] += (labels == c).sum()

    # Inverse frequency, normalised
    total = counts.sum()
    freq = counts / total
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.max()  # normalise so max weight = 1.0
    return weights.astype(np.float32)