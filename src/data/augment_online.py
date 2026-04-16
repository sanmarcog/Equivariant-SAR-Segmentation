"""
src/data/augment_online.py

Online (per-batch) training augmentation tuned for the D4-equivariant CNN.

D4 (the dihedral group) makes the model invariant by construction to:
    horizontal flip, vertical flip, and 90°/180°/270° rotations
i.e. 4 of the 6 geometric augmentations Gatti et al. 2026 use are wasted compute
for our model — feeding it those transforms gives identical predictions.

We apply only the OFF-D4-LATTICE perturbations that genuinely diversify training:

    Geometric (synchronized across patch + mask):
        affine: ±7° rotation, ±5% scale, ±3° shear, ±2 px translation

    Radiometric (per-channel, no group symmetry to break):
        SAR Gaussian noise (σ=0.05 z-score units, channels 0,1,6,7)
        SAR intensity scaling (0.97–1.03 multiplicative, channels 0,1,6,7)
        Aux perturbation (0.97–1.03 scale + ±0.05 bias, channels 2,3,4,5)

Engineered channels (8–11: log_ratio_VH/VV, xpol_post/pre) are NOT directly
perturbed — they are derived from the raw SAR. Geometric transforms still
apply to them (they live in the same spatial grid).

Channel layout (must match SegmentationDataset patch ordering):
    0  VH_post     1  VV_post
    2  slope       3  sin_asp     4  cos_asp     5  LIA
    6  VH_pre      7  VV_pre
    8  log_ratio_VH  9  log_ratio_VV  10 xpol_post  11 xpol_pre
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


SAR_RAW_CHANNELS = [0, 1, 6, 7]      # VH/VV post + pre — eligible for noise & intensity scaling
TERRAIN_CHANNELS = [2, 3, 4, 5]      # slope, sin/cos aspect, LIA — eligible for aux perturbation
# Engineered channels 8–11 inherit the geometric transform but are not radiometrically perturbed
# (perturbing them independently would break their derivation from the raw SAR).


class OnlineAugment:
    """
    Apply geometric + radiometric augmentation to a single sample dict.

    Args:
        rot_deg:      rotation range in degrees (uniform [-rot_deg, +rot_deg])
        shear_deg:    shear range in degrees
        scale_range:  (min, max) multiplicative scale factor
        translate_px: max translation in pixels (uniform per axis)
        sar_noise_std: Gaussian noise std on SAR channels (in z-score units)
        sar_intensity_range: (min, max) multiplicative intensity factor for SAR
        aux_scale_range:    (min, max) multiplicative scale for aux/terrain
        aux_bias_range:     ±bias range for aux/terrain (z-score units)
        p:                  per-sample probability of applying augmentation
        rng_seed:           seed for the per-instance RNG
    """

    def __init__(
        self,
        rot_deg:             float = 7.0,
        shear_deg:           float = 3.0,
        scale_range:         tuple[float, float] = (0.95, 1.05),
        translate_px:        int   = 2,
        sar_noise_std:       float = 0.05,
        sar_intensity_range: tuple[float, float] = (0.97, 1.03),
        aux_scale_range:     tuple[float, float] = (0.97, 1.03),
        aux_bias_range:      float = 0.05,
        p:                   float = 1.0,
        rng_seed:            int   = 0,
    ) -> None:
        self.rot_deg             = rot_deg
        self.shear_deg           = shear_deg
        self.scale_range         = scale_range
        self.translate_px        = translate_px
        self.sar_noise_std       = sar_noise_std
        self.sar_intensity_range = sar_intensity_range
        self.aux_scale_range     = aux_scale_range
        self.aux_bias_range      = aux_bias_range
        self.p                   = p
        self.gen                 = torch.Generator().manual_seed(rng_seed)

    # ------------------------------------------------------------------

    def _u(self, lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=self.gen).item())

    def _affine_matrix(self, H: int, W: int) -> torch.Tensor:
        """
        Build a 2×3 affine matrix in normalized [-1,1] coordinates for grid_sample.
        Composition: scale → shear → rotation → translation.
        """
        rot   = math.radians(self._u(-self.rot_deg,   self.rot_deg))
        shear = math.radians(self._u(-self.shear_deg, self.shear_deg))
        scale = self._u(*self.scale_range)
        tx_px = self._u(-self.translate_px, self.translate_px)
        ty_px = self._u(-self.translate_px, self.translate_px)

        # affine_grid expects coordinates in [-1, 1]; convert pixel translation accordingly
        tx = (2.0 * tx_px) / W
        ty = (2.0 * ty_px) / H

        cos_r, sin_r = math.cos(rot), math.sin(rot)
        # Rotation × shear × scale (column-major intent: applied to (x,y))
        a = scale * cos_r
        b = scale * (cos_r * math.tan(shear) - sin_r)
        c = scale * sin_r
        d = scale * (sin_r * math.tan(shear) + cos_r)

        # NOTE: grid_sample's affine matrix maps OUTPUT coordinates to INPUT coordinates,
        # so the matrix here is the inverse of the visual transform — but uniform random
        # parameters mean the distribution of perturbations is identical either way.
        return torch.tensor([[a, b, tx], [c, d, ty]], dtype=torch.float32)

    # ------------------------------------------------------------------

    def _apply_geometric(
        self, patch: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """patch: [12, H, W], mask: [1, H, W]"""
        C, H, W = patch.shape
        theta = self._affine_matrix(H, W).unsqueeze(0)             # [1, 2, 3]

        # Stack into one batched tensor so we generate the grid once
        bundled = torch.cat([patch.unsqueeze(0), mask.unsqueeze(0)], dim=1)  # [1, 13, H, W]

        grid = F.affine_grid(theta, bundled.shape, align_corners=False)

        # Bilinear for patch (continuous), nearest for mask (binary)
        patch_warped = F.grid_sample(
            patch.unsqueeze(0), grid, mode="bilinear",
            padding_mode="reflection", align_corners=False,
        ).squeeze(0)
        mask_warped = F.grid_sample(
            mask.unsqueeze(0), grid, mode="nearest",
            padding_mode="zeros", align_corners=False,
        ).squeeze(0)
        return patch_warped, mask_warped

    def _apply_radiometric(self, patch: torch.Tensor) -> torch.Tensor:
        """In-place clone-then-modify; returns new tensor."""
        out = patch.clone()
        # SAR Gaussian noise (per-channel, zero-mean)
        if self.sar_noise_std > 0:
            for ch in SAR_RAW_CHANNELS:
                noise = torch.empty_like(out[ch]).normal_(
                    mean=0.0, std=self.sar_noise_std, generator=self.gen
                )
                out[ch] = out[ch] + noise
        # SAR intensity scaling (one factor per channel, multiplicative)
        for ch in SAR_RAW_CHANNELS:
            f = self._u(*self.sar_intensity_range)
            out[ch] = out[ch] * f
        # Aux/terrain scale + bias
        for ch in TERRAIN_CHANNELS:
            s = self._u(*self.aux_scale_range)
            b = self._u(-self.aux_bias_range, self.aux_bias_range)
            out[ch] = out[ch] * s + b
        return out

    # ------------------------------------------------------------------

    def __call__(self, sample: dict) -> dict:
        if self._u(0.0, 1.0) > self.p:
            return sample

        patch = sample["patch"]    # [12, H, W] float32
        mask  = sample["mask"]     # [1, H, W]  float32

        patch, mask = self._apply_geometric(patch, mask)
        patch       = self._apply_radiometric(patch)

        new = dict(sample)
        new["patch"] = patch
        new["mask"]  = mask
        return new
