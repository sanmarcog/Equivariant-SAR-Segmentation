"""
src/data/augment.py

Copy-paste augmentation for avalanche deposit patches.

Rules (from wiki/architecture.md):
  - Paste D1/D2-sized deposits within the same REGION only
    (Livigno→Livigno, Nuuk→Nuuk, Pish→Pish — no cross-region pasting).
  - Gaussian edge blending at paste boundary.
  - Cap at 30% of positive patches per batch (configurable).

This module provides:
    CopyPasteAugment — callable that modifies a batch dict in-place.

A "deposit patch" from a positive sample is pasted onto a randomly chosen
background patch. The paste location is chosen so the deposited pixels are
fully within the 64×64 target patch.

Blending uses a Gaussian alpha mask:
    alpha_map = 1 inside the deposit region, smooth falloff of sigma pixels
    blended   = alpha * src_patch + (1 - alpha) * dst_patch

Mask is OR'd (deposit pixels always labelled 1).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def _make_alpha_mask(
    binary_mask: np.ndarray,   # [H, W] bool
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Gaussian-blended alpha mask for a deposit region.
    Inside deposit ≈ 1, outside ≈ 0, smooth transition of width ~sigma px.
    """
    alpha = binary_mask.astype(np.float32)
    alpha = gaussian_filter(alpha, sigma=sigma)
    # Re-normalise so peak == 1 inside deposit
    peak = alpha.max()
    if peak > 1e-6:
        alpha = alpha / peak
    return alpha.clip(0.0, 1.0)


class CopyPasteAugment:
    """
    Applies copy-paste augmentation to a training batch.

    Args:
        dataset:     SegmentationDataset — used to fetch deposit patches by region.
        cap_frac:    Max fraction of positive patches in batch to augment (0.30).
        sigma:       Gaussian blend sigma in pixels (default 4).
        rng_seed:    Seed for reproducibility (None = non-deterministic).
    """

    def __init__(
        self,
        dataset,
        cap_frac: float = 0.30,
        sigma:    float = 4.0,
        rng_seed: int | None = None,
    ) -> None:
        self.dataset  = dataset
        self.cap_frac = cap_frac
        self.sigma    = sigma
        self.rng      = np.random.default_rng(rng_seed)

        # Pre-index region → positive indices for fast lookup
        from src.data.dataset import _REGION
        self._region_pos: dict[str, list[int]] = {}
        for scene in dataset.scene_cache.names():
            region = _REGION.get(scene, "unknown")
            idxs   = dataset.get_scene_positive_indices(scene)
            self._region_pos.setdefault(region, []).extend(idxs)

    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict]) -> list[dict]:
        """
        Apply copy-paste to some fraction of background patches in *batch*.

        Modifies batch in-place. Returns the (possibly modified) batch.
        """
        # Identify negative (background) patches by mask sum
        neg_indices = [
            k for k, s in enumerate(batch)
            if s["mask"].sum().item() == 0
        ]
        if not neg_indices:
            return batch

        # How many to augment (cap at cap_frac of total positives in batch)
        n_pos_in_batch = len(batch) - len(neg_indices)
        n_augment = min(
            len(neg_indices),
            max(1, round(n_pos_in_batch * self.cap_frac)),
        )
        targets = self.rng.choice(neg_indices, size=n_augment, replace=False)

        for tgt_k in targets:
            tgt = batch[tgt_k]
            region = tgt.get("region", "unknown")
            src_pool = self._region_pos.get(region, [])
            if not src_pool:
                continue

            # Draw a source deposit patch
            src_idx = int(self.rng.choice(src_pool))
            src = self.dataset[src_idx]

            src_mask = src["mask"][0].numpy().astype(bool)   # [64, 64]
            if src_mask.sum() == 0:
                continue

            # Paste at a random offset that keeps deposit in-bounds
            # Find deposit bounding box
            rows = np.where(src_mask.any(axis=1))[0]
            cols = np.where(src_mask.any(axis=0))[0]
            dep_h = rows[-1] - rows[0] + 1
            dep_w = cols[-1] - cols[0] + 1
            P = 64

            if dep_h >= P or dep_w >= P:
                continue   # deposit larger than patch — skip

            # Random paste offset so deposit fits inside target
            max_di = P - dep_h
            max_dj = P - dep_w
            di = int(self.rng.integers(0, max_di + 1))
            dj = int(self.rng.integers(0, max_dj + 1))

            # Translate mask to new position
            new_mask = np.zeros((P, P), dtype=bool)
            new_mask[
                di : di + dep_h,
                dj : dj + dep_w,
            ] = src_mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

            if new_mask.sum() == 0:
                continue

            # Build alpha mask (Gaussian blending)
            alpha = _make_alpha_mask(new_mask, sigma=self.sigma)   # [64, 64]
            alpha_t = torch.from_numpy(alpha).unsqueeze(0)          # [1, 64, 64]

            # Blend 12-channel patches
            # Align the source patch crop to the deposit bounding box
            src_crop = np.zeros((12, P, P), dtype=np.float32)
            src_crop[
                :,
                di : di + dep_h,
                dj : dj + dep_w,
            ] = src["patch"].numpy()[
                :,
                rows[0]:rows[-1]+1,
                cols[0]:cols[-1]+1,
            ]
            src_crop_t = torch.from_numpy(src_crop)  # [12, 64, 64]

            tgt["patch"] = alpha_t * src_crop_t + (1 - alpha_t) * tgt["patch"]
            # Hard-OR the mask (deposit pixels always labelled 1)
            tgt["mask"]  = (tgt["mask"] + torch.from_numpy(new_mask.astype(np.float32)).unsqueeze(0)).clamp(0, 1)

        return batch
