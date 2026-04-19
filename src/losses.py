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
# Per-component IoU loss
# ---------------------------------------------------------------------------

class ComponentIoULoss(nn.Module):
    """
    Per-instance soft-Dice/IoU loss computed independently for each connected
    component in the ground truth mask. Background gets standard BCE.

    L = BCE(all pixels) + λ * mean(per_component_soft_dice)

    The per-component term naturally handles scale: small components get high
    Dice from partial overlap, large components need precise boundaries.

    Args:
        lam:        weight of per-component IoU term relative to BCE
        pos_weight: BCE pos_weight
        eps:        label smoothing for BCE
        bbox_dilation: pixels to dilate component bbox for prediction crop
    """

    def __init__(
        self,
        lam:            float = 1.0,
        pos_weight:     float = 1.0,
        eps:            float = 0.05,
        bbox_dilation:  int   = 3,
        use_dice_global: bool = False,
    ) -> None:
        super().__init__()
        self.lam             = lam
        self.pos_weight      = pos_weight
        self.eps             = eps
        self.bbox_dilation   = bbox_dilation
        self.use_dice_global = use_dice_global

    def _per_component_dice(
        self, logit: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        from scipy.ndimage import label as _ndi_label

        prob = torch.sigmoid(logit)
        B = logit.shape[0]
        H, W = logit.shape[-2], logit.shape[-1]
        d = self.bbox_dilation
        dice_losses = []

        for b in range(B):
            m = mask[b, 0]
            p = prob[b, 0]

            pos_np = (m > 0.5).cpu().numpy()
            if not pos_np.any():
                continue

            labeled, n_comps = _ndi_label(pos_np)
            labeled_t = torch.from_numpy(labeled).to(logit.device)

            for comp_id in range(1, n_comps + 1):
                comp_pixels = labeled_t == comp_id
                rows = torch.where(comp_pixels.any(dim=1))[0]
                cols = torch.where(comp_pixels.any(dim=0))[0]
                if len(rows) == 0:
                    continue

                r0 = max(0, rows[0].item() - d)
                r1 = min(H, rows[-1].item() + 1 + d)
                c0 = max(0, cols[0].item() - d)
                c1 = min(W, cols[-1].item() + 1 + d)

                gt_crop = m[r0:r1, c0:c1]
                pred_crop = p[r0:r1, c0:c1]

                intersection = (pred_crop * gt_crop).sum()
                union = pred_crop.sum() + gt_crop.sum()
                dice = (2.0 * intersection + 1.0) / (union + 1.0)
                dice_losses.append(1.0 - dice)

        if not dice_losses:
            return logit.sum() * 0.0
        return torch.stack(dice_losses).mean()

    def forward(
        self,
        logit:     torch.Tensor,   # [B, 1, H, W]
        target:    torch.Tensor,   # [B, 1, H, W] binary 0/1
        comp_size: torch.Tensor | None = None,  # unused, kept for interface compat
    ) -> torch.Tensor:
        target_smooth = smooth_labels(target, self.eps)
        if self.use_dice_global:
            l_global = dice_loss(logit, target_smooth)
        else:
            pw = torch.tensor([self.pos_weight], device=logit.device)
            l_global = F.binary_cross_entropy_with_logits(logit, target_smooth, pos_weight=pw)
        l_comp = self._per_component_dice(logit, target)
        return l_global + self.lam * l_comp


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
        mode:    'focal_tversky' | 'bce' | 'dice' | 'bce_dice' | 'size_adaptive'
    """

    def __init__(
        self,
        gamma:      float = 2.0,
        alpha:      float = 0.3,
        beta:       float = 0.7,
        eps:        float = 0.05,
        mode:       str   = "focal_tversky",
        pos_weight: float = 3.0,
        small_thr:  int   = 10,
        large_thr:  int   = 30,
        boundary_weight: float = 0.4,
        balance_weight:  float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha
        self.beta       = beta
        self.eps        = eps
        self.mode       = mode
        self.pos_weight = pos_weight

        if mode in ("component_iou", "component_iou_dice"):
            self.component_iou = ComponentIoULoss(
                lam=balance_weight, pos_weight=pos_weight, eps=eps,
                use_dice_global=(mode == "component_iou_dice"),
            )

    def forward(
        self,
        logit:     torch.Tensor,   # [B, 1, H, W]
        target:    torch.Tensor,   # [B, 1, H, W] raw binary mask (0/1)
        comp_size: torch.Tensor | None = None,  # [B, 1, H, W] component sizes
    ) -> torch.Tensor:
        if self.mode in ("component_iou", "component_iou_dice"):
            return self.component_iou(logit, target)

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
        small_thr:   int   = 10,
        large_thr:   int   = 30,
        boundary_weight: float = 0.4,
        balance_weight:  float = 1.0,
    ) -> None:
        super().__init__()
        self.seg_loss  = SegLoss(
            gamma=gamma, alpha=alpha, beta=beta, eps=eps, mode=mode, pos_weight=pos_weight,
            small_thr=small_thr, large_thr=large_thr,
            boundary_weight=boundary_weight, balance_weight=balance_weight,
        )
        self.area_loss = AreaLoss()
        self.lambda_area = lambda_area

    def forward(
        self,
        logit:       torch.Tensor,    # [B, 1, H, W]
        target:      torch.Tensor,    # [B, 1, H, W] binary mask
        area_m2:     torch.Tensor,    # [B, 1] predicted area from model
        comp_size:   torch.Tensor | None = None,  # [B, 1, H, W]
    ) -> dict[str, torch.Tensor]:
        l_seg  = self.seg_loss(logit, target, comp_size=comp_size)
        l_area = self.area_loss(area_m2, target)
        total  = l_seg + self.lambda_area * l_area

        return {
            "loss":       total,
            "loss_seg":   l_seg,
            "loss_area":  l_area,
        }
