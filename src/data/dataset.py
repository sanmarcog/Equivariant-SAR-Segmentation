"""
src/data/dataset.py

Segmentation dataset for Phase 2.

Each sample is a dict:
    patch   : float32 [12, 64, 64]  — normalised 12-channel input
    mask    : float32 [1, 64, 64]   — binary GT mask (0/1)
    weight  : float   — copy-paste composite weight map (all-ones unless augmented)
    scene   : str     — scene name (for cross-region copy-paste guard)
    pos_i   : int     — top-left row of patch in scene
    pos_j   : int     — top-left col of patch in scene

Dataset split (FIXED):
    Train: Livigno_20240403, Livigno_20250129, Nuuk_20160413, Nuuk_20210411, Pish_20230221
    Val:   Livigno_20250318
    Test:  Tromso_20241220

Biased sampler (train only):
    BiasedPatchSampler returns indices with 50% positive patches per batch.
    "Positive" = patch contains ≥1 deposit pixel.

Copy-paste augmentation: see src/data/augment.py
    Applied to training batches AFTER loading, not at Dataset level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.preprocess import preprocess_scene, load_gt_mask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed splits
# ---------------------------------------------------------------------------

TRAIN_SCENES = [
    "Livigno_20240403",
    "Livigno_20250129",
    "Nuuk_20160413",
    "Nuuk_20210411",
    "Pish_20230221",
]
VAL_SCENES  = ["Livigno_20250318"]
TEST_SCENES = ["Tromso_20241220"]

# Region labels for copy-paste guard
_REGION = {
    "Livigno_20240403": "Livigno",
    "Livigno_20250129": "Livigno",
    "Livigno_20250318": "Livigno",
    "Nuuk_20160413":    "Nuuk",
    "Nuuk_20210411":    "Nuuk",
    "Pish_20230221":    "Pish",
    "Tromso_20241220":  "Tromso",
}

PATCH_SIZE = 64
PATCH_STRIDE_TRAIN = 32   # 50% overlap during training (patches built offline)
PATCH_STRIDE_INFER = 16   # 75% overlap at inference (in inference.py)


# ---------------------------------------------------------------------------
# Scene cache (load once, share across splits)
# ---------------------------------------------------------------------------

class _SceneCache:
    """Loads and caches preprocessed scene arrays + GT masks."""

    def __init__(self, data_dir: Path, scene_names: list[str], stats: dict):
        self._scenes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        mean = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array(stats["std"],  dtype=np.float32).reshape(-1, 1, 1)
        for name in scene_names:
            scene_dir = data_dir / name
            log.info("Loading scene %s ...", name)
            arr12 = preprocess_scene(scene_dir)          # [12, H, W]
            arr12 = (arr12 - mean) / std                  # z-score normalise
            mask  = load_gt_mask(scene_dir).astype(np.float32)   # [H, W]
            self._scenes[name] = (arr12, mask)
            log.info("  shape=%s  deposit_frac=%.4f", arr12.shape, mask.mean())

    def get(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._scenes[name]

    def names(self) -> list[str]:
        return list(self._scenes.keys())


# ---------------------------------------------------------------------------
# Patch index builder
# ---------------------------------------------------------------------------

def _build_patch_index(
    scene_cache: _SceneCache,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE_TRAIN,
) -> list[dict]:
    """
    Enumerate all valid patches across all scenes in the cache.

    Returns a list of dicts:
        { scene, pos_i, pos_j, is_positive }
    """
    records = []
    for scene_name in scene_cache.names():
        _, mask = scene_cache.get(scene_name)
        H, W = mask.shape
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch_mask = mask[i:i+patch_size, j:j+patch_size]
                is_pos = bool(patch_mask.sum() > 0)
                records.append({
                    "scene":       scene_name,
                    "pos_i":       i,
                    "pos_j":       j,
                    "is_positive": is_pos,
                })
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Patch-based segmentation dataset.

    Args:
        data_dir:    Root of AvalCD raw data directory.
        split:       One of 'train', 'val', 'test'.
        stats_path:  Path to norm_stats_12ch.json (must exist).
        transform:   Optional callable applied to the patch dict.
        patch_stride: Stride for patch extraction (default: 32 for train, 16 for val/test).
    """

    def __init__(
        self,
        data_dir:     str | Path,
        split:        str,
        stats_path:   str | Path,
        transform:    Callable | None = None,
        patch_stride: int | None = None,
    ) -> None:
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.transform = transform

        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        if split == "train":
            scene_names = TRAIN_SCENES
        elif split == "val":
            scene_names = VAL_SCENES
        else:
            scene_names = TEST_SCENES

        with open(stats_path) as f:
            stats = json.load(f)

        self.scene_cache = _SceneCache(self.data_dir, scene_names, stats)

        stride = patch_stride if patch_stride is not None else (
            PATCH_STRIDE_TRAIN if split == "train" else PATCH_STRIDE_INFER
        )
        self.records = _build_patch_index(self.scene_cache, PATCH_SIZE, stride)
        log.info(
            "Split '%s': %d patches (%d positive, %d negative)",
            split,
            len(self.records),
            sum(r["is_positive"] for r in self.records),
            sum(not r["is_positive"] for r in self.records),
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        arr12, mask = self.scene_cache.get(rec["scene"])

        i, j = rec["pos_i"], rec["pos_j"]
        p = PATCH_SIZE
        patch = arr12[:, i:i+p, j:j+p].copy()    # [12, 64, 64]
        pmask = mask[i:i+p, j:j+p].copy()         # [64, 64]

        sample = {
            "patch":  torch.from_numpy(patch),
            "mask":   torch.from_numpy(pmask).unsqueeze(0),   # [1, 64, 64]
            "scene":  rec["scene"],
            "region": _REGION.get(rec["scene"], "unknown"),
            "pos_i":  i,
            "pos_j":  j,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Helpers used by the biased sampler
    # ------------------------------------------------------------------

    @property
    def positive_indices(self) -> list[int]:
        return [i for i, r in enumerate(self.records) if r["is_positive"]]

    @property
    def negative_indices(self) -> list[int]:
        return [i for i, r in enumerate(self.records) if not r["is_positive"]]

    def get_scene_positive_indices(self, scene: str) -> list[int]:
        """All positive-patch indices from a given scene (for copy-paste)."""
        return [
            i for i, r in enumerate(self.records)
            if r["is_positive"] and r["scene"] == scene
        ]

    def get_region_positive_indices(self, region: str) -> list[int]:
        """All positive-patch indices from a given region (copy-paste source pool)."""
        return [
            i for i, r in enumerate(self.records)
            if r["is_positive"] and _REGION.get(r["scene"]) == region
        ]


# ---------------------------------------------------------------------------
# Biased sampler
# ---------------------------------------------------------------------------

class BiasedPatchSampler(Sampler):
    """
    Yields indices so that each batch contains ~pos_fraction positive patches.

    Uses reservoir sampling to draw positives and negatives independently,
    then interleaves them to build each batch. Works with DataLoader
    batch_size and drop_last.

    Args:
        dataset:      SegmentationDataset
        batch_size:   DataLoader batch_size
        pos_fraction: Target fraction of positives per batch (default 0.5)
        num_batches:  How many batches per epoch (default: len(pos)/pos_per_batch)
        seed:         RNG seed
    """

    def __init__(
        self,
        dataset:      SegmentationDataset,
        batch_size:   int,
        pos_fraction: float = 0.5,
        num_batches:  int | None = None,
        seed:         int = 0,
    ) -> None:
        self.pos_idx  = dataset.positive_indices
        self.neg_idx  = dataset.negative_indices
        self.bs       = batch_size
        self.pos_frac = pos_fraction
        self.rng      = np.random.default_rng(seed)

        self.n_pos_per_batch = max(1, round(batch_size * pos_fraction))
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch

        if num_batches is None:
            num_batches = len(self.pos_idx) // self.n_pos_per_batch
        self.num_batches = max(num_batches, 1)

    def __len__(self) -> int:
        return self.num_batches * self.bs

    def __iter__(self):
        for _ in range(self.num_batches):
            pos_sample = self.rng.choice(self.pos_idx, size=self.n_pos_per_batch, replace=True)
            neg_sample = self.rng.choice(self.neg_idx, size=self.n_neg_per_batch, replace=True)
            batch = np.concatenate([pos_sample, neg_sample])
            self.rng.shuffle(batch)
            yield from batch.tolist()
