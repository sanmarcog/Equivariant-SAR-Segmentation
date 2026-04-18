"""
src/losses.py

Loss functions for Phase 2 segmentation.

L = L_seg + λ_area × L_area

L_seg  = 0.5 × Focal(γ) + 0.5 × Tversky(α, β)
    Focal:   down-weights easy background; focuses on hard deposit pixels
    Tversky: FN penalised β/(α+β) relative to FP — optimises recall on rare deposits
    α=0.3, β=0.7 → FN weight 70%, FP weight 30%

L_area = L1( log(area_pred_m2 + 1), log(GT_area_m2 + 1) )
    Applied only when GT mask is non-empty (Tromsø samples with D-scale labels).
    λ_area = 0.1

Label smoothing: ε=0.05 — applied to the target masks before computing L_seg.
    Positive pixels: 1 - ε = 0.95
    Negative pixels: ε = 0.05

Hyperparameter grid (tuned on val F2 before full training):
    γ ∈ {1, 2, 3}
    α/β ∈ {0.3/0.7, 0.2/0.8}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Label smoothing helper
# ---------------------------------------------------------------------------

def smooth_labels(mask: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """
    Apply label smoothing to binary mask [B, 1, H, W].
    Positive (1) → 1 - eps.  Negative (0) → eps.
    """
    return mask * (1.0 - eps) + (1.0 - mask) * eps


# ---------------------------------------------------------------------------
# Focal loss (binary)
# ---------------------------------------------------------------------------

def focal_loss(
    logit:  torch.Tensor,    # [B, 1, H, W] or [B, 1]
    target: torch.Tensor,    # same shape, values in [0, 1]
    gamma:  float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary focal loss.

    F_L = − [ α_t × (1 − p_t)^γ × log(p_t) ]

    where p_t is the predicted probability for the true class.
    α balancing is omitted here (handled by biased sampler + pos_weight in BCE).
    """
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = torch.sigmoid(logit)
    # p_t for the true class
    pt  = p_t * target + (1.0 - p_t) * (1.0 - target)
    focal_weight = (1.0 - pt) ** gamma

    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Tversky loss
# ---------------------------------------------------------------------------

def tversky_loss(
    logit:  torch.Tensor,   # [B, 1, H, W]
    target: torch.Tensor,   # [B, 1, H, W], values in [0, 1]
    alpha:  float = 0.3,    # FP weight
    beta:   float = 0.7,    # FN weight  (α + β should equal 1)
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Tversky loss for imbalanced segmentation.

    TI = (TP + smooth) / (TP + α×FP + β×FN + smooth)
    Loss = 1 - TI

    α=0.3, β=0.7: penalises FN 2.3× more than FP → improves recall on rare deposits.

    Note on convention: Gatti et al. 2026 use α=0.7, β=0.3 (their notation
    swaps α and β). Our convention: α=FP weight, β=FN weight.
    """
    prob = torch.sigmoid(logit)

    # Flatten spatial dims
    prob_flat   = prob.view(prob.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    TP = (prob_flat * target_flat).sum(dim=1)
    FP = (prob_flat * (1.0 - target_flat)).sum(dim=1)
    FN = ((1.0 - prob_flat) * target_flat).sum(dim=1)

    TI   = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1.0 - TI
    return loss.mean()


# ---------------------------------------------------------------------------
# Dice loss (directly optimizes F1 → sharper predictions)
# ---------------------------------------------------------------------------

def dice_loss(
    logit:  torch.Tensor,   # [B, 1, H, W]
    target: torch.Tensor,   # [B, 1, H, W], values in [0, 1]
    smooth: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    prob_flat   = prob.view(prob.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    intersection = (prob_flat * target_flat).sum(dim=1)
    union = prob_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# BCE-only loss (ablation baseline, condition 1)
# ---------------------------------------------------------------------------

def bce_loss(
    logit:      torch.Tensor,
    target:     torch.Tensor,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """
    Weighted BCE with pos_weight (matches Gatti et al. 2026 baseline).
    Used for ablation condition 1 (baseline with BCE only).
    """
    pw = torch.tensor([pos_weight], device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)


# ---------------------------------------------------------------------------
# Combined segmentation loss
# ---------------------------------------------------------------------------

class SegLoss(nn.Module):
    """
    Combined segmentation loss: Focal + Tversky (equal weight).

    Args:
        gamma:   Focal loss γ (default 2.0; grid: {1, 2, 3})
        alpha:   Tversky FP weight (default 0.3; grid: {0.3, 0.2})
        beta:    Tversky FN weight (default 0.7; grid: {0.7, 0.8})
        eps:     Label smoothing ε (default 0.05)
        mode:    'focal_tversky' (conditions 3–5) | 'bce' (condition 1–2)
    """

    def __init__(
        self,
        gamma:      float = 2.0,
        alpha:      float = 0.3,
        beta:       float = 0.7,
        eps:        float = 0.05,
        mode:       str   = "focal_tversky",
        pos_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha
        self.beta       = beta
        self.eps        = eps
        self.mode       = mode
        self.pos_weight = pos_weight

    def forward(
        self,
        logit:  torch.Tensor,   # [B, 1, H, W]
        target: torch.Tensor,   # [B, 1, H, W] raw binary mask (0/1)
    ) -> torch.Tensor:
        target_smooth = smooth_labels(target, self.eps)

        if self.mode == "bce":
            return bce_loss(logit, target_smooth, pos_weight=self.pos_weight)

        if self.mode == "dice":
            return dice_loss(logit, target_smooth)

        if self.mode == "bce_dice":
            return 0.5 * bce_loss(logit, target_smooth, pos_weight=self.pos_weight) + 0.5 * dice_loss(logit, target_smooth)

        fl = focal_loss(logit, target_smooth, gamma=self.gamma)
        tl = tversky_loss(logit, target_smooth, alpha=self.alpha, beta=self.beta)
        return 0.5 * fl + 0.5 * tl


# ---------------------------------------------------------------------------
# Area loss (supplementary, Tromsø only)
# ---------------------------------------------------------------------------

class AreaLoss(nn.Module):
    """
    L1 loss on log(area + 1) between predicted and GT area.

    Args:
        pixel_m2: Area of one pixel in m² (10m × 10m = 100.0).
    """

    def __init__(self, pixel_m2: float = 100.0) -> None:
        super().__init__()
        self.pixel_m2 = pixel_m2

    def forward(
        self,
        area_pred_m2: torch.Tensor,   # [B, 1] from model
        mask_gt:      torch.Tensor,   # [B, 1, H, W] binary mask
    ) -> torch.Tensor:
        """
        Computes area loss only for samples with non-empty GT masks.
        Returns 0.0 if no valid samples in batch.
        """
        # GT area from mask
        area_gt_m2 = mask_gt.sum(dim=[2, 3]) * self.pixel_m2   # [B, 1]

        # Only compute loss for patches with actual deposits
        valid = (area_gt_m2 > 0).squeeze(1)   # [B] bool
        if not valid.any():
            return area_pred_m2.sum() * 0.0   # zero, differentiable

        pred_log = torch.log(area_pred_m2[valid] + 1.0)
        gt_log   = torch.log(area_gt_m2[valid]   + 1.0)
        return F.l1_loss(pred_log, gt_log)


# ---------------------------------------------------------------------------
# Full combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Full training loss: L = L_seg + λ_area × L_area

    Args:
        gamma, alpha, beta, eps, mode: passed to SegLoss
        lambda_area: weight on area loss (default 0.1)
    """

    def __init__(
        self,
        gamma:       float = 2.0,
        alpha:       float = 0.3,
        beta:        float = 0.7,
        eps:         float = 0.05,
        mode:        str   = "focal_tversky",
        lambda_area: float = 0.1,
        pos_weight:  float = 3.0,
    ) -> None:
        super().__init__()
        self.seg_loss  = SegLoss(
            gamma=gamma, alpha=alpha, beta=beta, eps=eps, mode=mode, pos_weight=pos_weight,
        )
        self.area_loss = AreaLoss()
        self.lambda_area = lambda_area

    def forward(
        self,
        logit:       torch.Tensor,    # [B, 1, H, W]
        target:      torch.Tensor,    # [B, 1, H, W] binary mask
        area_m2:     torch.Tensor,    # [B, 1] predicted area from model
    ) -> dict[str, torch.Tensor]:
        l_seg  = self.seg_loss(logit, target)
        l_area = self.area_loss(area_m2, target)
        total  = l_seg + self.lambda_area * l_area

        return {
            "loss":       total,
            "loss_seg":   l_seg,
            "loss_area":  l_area,
        }
