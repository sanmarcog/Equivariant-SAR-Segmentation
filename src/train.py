"""
src/train.py

Training loop for Phase 2 segmentation.

Ablation conditions (in order, each with 3 seeds):
    1  Baseline: 12ch, random sampling, BCE, no skip connections
    2  + biased patch sampling (pos_fraction=0.5)
    3  + Focal + Tversky loss
    4  + U-Net skip connections
    5  + copy-paste augmentation (full system)

Usage:
    python -m src.train \\
        --data-dir /path/to/avalcd/raw \\
        --stats    data/norm_stats_12ch.json \\
        --out-dir  checkpoints/ \\
        --condition 5 \\
        --seed 0 \\
        --gamma 2 --alpha 0.3 --beta 0.7 --pos-frac 0.5 \\
        [--epochs 110] [--batch-size 32] [--lr 1e-4] [--wd 1e-4]

For hyperparameter grid search (run before full training):
    python -m src.train --grid-search \\
        --data-dir ... --stats ... --out-dir grid/ --condition 5 --seed 0

W&B logging: set WANDB_PROJECT env var.  Set --no-wandb to disable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset, BiasedPatchSampler, PATCH_STRIDE_TRAIN
from src.data.augment import CopyPasteAugment
from src.losses import CombinedLoss
from src.models.segnet import build_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = {
    1: dict(use_skip=False, mode="bce",           biased=False, copy_paste=False),
    2: dict(use_skip=False, mode="bce",           biased=True,  copy_paste=False),
    3: dict(use_skip=False, mode="focal_tversky", biased=True,  copy_paste=False),
    4: dict(use_skip=True,  mode="focal_tversky", biased=True,  copy_paste=False),
    5: dict(use_skip=True,  mode="focal_tversky", biased=True,  copy_paste=True),
}

# Hyperparameter grid (swept on val pixel F2 before full training)
HPARAM_GRID = {
    "gamma":    [1, 2, 3],
    "alpha_beta": [(0.3, 0.7), (0.2, 0.8)],
    "pos_frac": [0.4, 0.5, 0.6],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_fn(batch: list[dict]) -> dict:
    """Stack list of sample dicts into batched tensors."""
    return {
        "patch":  torch.stack([s["patch"]  for s in batch]),
        "mask":   torch.stack([s["mask"]   for s in batch]),
        "scene":  [s["scene"]  for s in batch],
        "region": [s["region"] for s in batch],
        "pos_i":  [s["pos_i"]  for s in batch],
        "pos_j":  [s["pos_j"]  for s in batch],
    }


def build_loaders(
    data_dir:     Path,
    stats_path:   Path,
    batch_size:   int,
    condition:    int,
    pos_frac:     float,
    seed:         int,
    num_workers:  int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for a given ablation condition."""
    cfg = ABLATION_CONDITIONS[condition]

    train_ds = SegmentationDataset(
        data_dir=data_dir,
        split="train",
        stats_path=stats_path,
        patch_stride=PATCH_STRIDE_TRAIN,
    )
    val_ds = SegmentationDataset(
        data_dir=data_dir,
        split="val",
        stats_path=stats_path,
        patch_stride=16,
    )

    if cfg["biased"]:
        sampler = BiasedPatchSampler(
            train_ds, batch_size=batch_size, pos_fraction=pos_frac, seed=seed
        )
        train_loader = DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  CombinedLoss,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    copy_paste: CopyPasteAugment | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = total_seg = total_area = 0.0
    n_batches  = 0

    for batch in loader:
        if copy_paste is not None:
            # Apply copy-paste per batch (list of dicts)
            samples = [
                {k: (v[i] if isinstance(v, torch.Tensor) else v[i])
                 for k, v in batch.items()}
                for i in range(batch["patch"].shape[0])
            ]
            samples = copy_paste(samples)
            # Re-stack
            batch = {
                "patch":  torch.stack([s["patch"] for s in samples]),
                "mask":   torch.stack([s["mask"]  for s in samples]),
                "scene":  [s["scene"]  for s in samples],
                "region": [s["region"] for s in samples],
                "pos_i":  [s["pos_i"]  for s in samples],
                "pos_j":  [s["pos_j"]  for s in samples],
            }

        patch = batch["patch"].to(device, non_blocking=True)   # [B, 12, 64, 64]
        mask  = batch["mask"].to(device, non_blocking=True)    # [B, 1, 64, 64]

        optimizer.zero_grad()
        out    = model(patch)
        losses = criterion(out["logit"], mask, out["area_m2"])
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        total_seg  += losses["loss_seg"].item()
        total_area += losses["loss_area"].item()
        n_batches  += 1

    return {
        "loss":      total_loss / max(n_batches, 1),
        "loss_seg":  total_seg  / max(n_batches, 1),
        "loss_area": total_area / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Validation — pixel F2 (for early stopping and hparam selection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    max_pixels: int = 10_000_000,
) -> dict[str, float]:
    """
    Compute pixel-level F1, F2 (threshold sweep) and AUPRC on val set.

    Subsamples to max_pixels pixels for training-time metrics — full val set
    is 131M pixels and sweep+auprc would take ~90s per epoch × 360 epochs = 9h.
    Subsampled metrics preserve rankings for early stopping and checkpoint
    selection. Final test eval still uses full resolution.
    """
    model.eval()
    all_prob = []
    all_gt   = []

    for batch in loader:
        patch = batch["patch"].to(device)
        mask  = batch["mask"].to(device)
        out   = model(patch)
        prob  = torch.sigmoid(out["logit"])
        all_prob.append(prob.cpu().float())
        all_gt.append(mask.cpu().float())

    prob_all = torch.cat(all_prob).view(-1).numpy()
    gt_all   = torch.cat(all_gt).view(-1).numpy()

    if prob_all.size > max_pixels:
        rng = np.random.default_rng(0)
        idx = rng.choice(prob_all.size, size=max_pixels, replace=False)
        prob_all = prob_all[idx]
        gt_all   = gt_all[idx]

    from src.evaluate import sweep_thresholds, auprc
    results = sweep_thresholds(prob_all, gt_all)
    results["auprc"] = auprc(prob_all, gt_all)
    return results


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(
    data_dir:   Path,
    stats_path: Path,
    out_dir:    Path,
    condition:  int,
    seed:       int,
    gamma:      float,
    alpha:      float,
    beta:       float,
    pos_frac:   float,
    epochs:     int     = 110,
    batch_size: int     = 32,
    lr:         float   = 1e-4,
    wd:         float   = 1e-4,
    warmup_epochs: int  = 10,
    num_workers: int    = 4,
    use_wandb:  bool    = False,
    run_name:   str | None = None,
    pos_weight: float   = 3.0,
) -> dict:
    """
    Train one model (one ablation condition + one seed).

    Returns the best val metrics dict.
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ABLATION_CONDITIONS[condition]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Condition %d | seed %d | device %s", condition, seed, device)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, train_ds = build_loaders(
        data_dir, stats_path, batch_size, condition, pos_frac, seed, num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(use_skip=cfg["use_skip"]).to(device)
    log.info("Model params: %d", count_parameters(model))

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        gamma=gamma, alpha=alpha, beta=beta,
        mode=cfg["mode"],
        pos_weight=pos_weight,
    ).to(device)

    # ── Copy-paste augmentation ────────────────────────────────────────
    copy_paste = None
    if cfg["copy_paste"]:
        copy_paste = CopyPasteAugment(train_ds, cap_frac=0.30, sigma=4.0, rng_seed=seed)

    # ── Optimiser + LR schedule ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ── W&B ───────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        run_name = run_name or f"cond{condition}_seed{seed}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "equivariant-sar-seg"),
            name=run_name,
            config=dict(
                condition=condition, seed=seed, gamma=gamma,
                alpha=alpha, beta=beta, pos_frac=pos_frac,
                epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                use_skip=cfg["use_skip"], mode=cfg["mode"],
                biased=cfg["biased"], copy_paste=cfg["copy_paste"],
                n_params=count_parameters(model),
            ),
        )

    # ── Training loop ─────────────────────────────────────────────────
    best_auprc  = -1.0
    no_improve  = 0
    patience    = 20
    best_ckpt   = out_dir / f"best_cond{condition}_seed{seed}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, copy_paste
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | loss=%.4f seg=%.4f area=%.4f | "
            "val F1=%.4f F2=%.4f AUPRC=%.4f | lr=%.2e | %.1fs",
            epoch, epochs,
            train_metrics["loss"], train_metrics["loss_seg"], train_metrics["loss_area"],
            val_metrics["best_f1"], val_metrics["best_f2"], val_metrics["auprc"],
            scheduler.get_last_lr()[0], elapsed,
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch":         epoch,
                "train/loss":    train_metrics["loss"],
                "train/seg":     train_metrics["loss_seg"],
                "train/area":    train_metrics["loss_area"],
                "val/f1":        val_metrics["best_f1"],
                "val/f2":        val_metrics["best_f2"],
                "val/auprc":     val_metrics["auprc"],
                "val/thr_f1":    val_metrics["thr_f1"],
                "val/thr_f2":    val_metrics["thr_f2"],
                "lr":            scheduler.get_last_lr()[0],
            })

        # Checkpoint on best AUPRC
        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "val_metrics": val_metrics,
                "cfg":         cfg,
                "hyperparams": dict(gamma=gamma, alpha=alpha, beta=beta, pos_frac=pos_frac),
            }, best_ckpt)
            log.info("  ↑ new best val AUPRC=%.4f  saved → %s", best_auprc, best_ckpt)
        else:
            # Only count non-improvement after warmup
            if epoch > warmup_epochs:
                no_improve += 1
                if no_improve >= patience:
                    log.info(
                        "Early stopping at epoch %d (no AUPRC improvement for %d epochs)",
                        epoch, patience,
                    )
                    break

    if use_wandb:
        import wandb
        wandb.finish()

    # Load best val metrics from checkpoint to get F2 (for grid search ranking)
    best_val_metrics: dict = {}
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        best_val_metrics = ckpt.get("val_metrics", {})

    log.info(
        "Training complete. Best val AUPRC=%.4f  F2=%.4f",
        best_auprc, best_val_metrics.get("best_f2", float("nan")),
    )
    return {
        "best_auprc": best_auprc,
        "best_f2":    best_val_metrics.get("best_f2", float("nan")),
        "ckpt":       str(best_ckpt),
    }


# ---------------------------------------------------------------------------
# Hyperparameter grid search
# ---------------------------------------------------------------------------

def grid_search(
    data_dir:  Path,
    stats_path: Path,
    out_dir:   Path,
    condition: int,
    seed:      int,
    epochs:    int = 20,     # short runs for grid search
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    """
    Grid search over (γ, α/β, pos_frac). Uses val F2 as selection criterion.
    Only runs focal_tversky mode (conditions 3–5).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for (gamma, (alpha, beta), pos_frac) in itertools.product(
        HPARAM_GRID["gamma"],
        HPARAM_GRID["alpha_beta"],
        HPARAM_GRID["pos_frac"],
    ):
        key = f"g{gamma}_a{alpha}_b{beta}_pf{pos_frac}"
        log.info("Grid run: %s", key)
        try:
            metrics = train(
                data_dir=data_dir, stats_path=stats_path,
                out_dir=out_dir / key,
                condition=condition, seed=seed,
                gamma=gamma, alpha=alpha, beta=beta, pos_frac=pos_frac,
                epochs=epochs, batch_size=batch_size, num_workers=num_workers,
            )
            results.append({"key": key, "gamma": gamma, "alpha": alpha,
                            "beta": beta, "pos_frac": pos_frac, **metrics})
        except Exception as e:
            log.error("Grid run %s failed: %s", key, e)

    # Rank by best val pixel F2 — grid search asks "which hparams give the best result?"
    # (AUPRC is used for early stopping/checkpointing within each run, not for selection)
    results.sort(key=lambda r: -r.get("best_f2", 0.0))
    grid_path = out_dir / "grid_results.json"
    with open(grid_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Grid search complete. Best: %s  (F2=%.4f)", results[0]["key"], results[0].get("best_f2", float("nan")))
    log.info("Results saved → %s", grid_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 training")
    p.add_argument("--data-dir",   required=True,  type=Path)
    p.add_argument("--stats",      required=True,  type=Path)
    p.add_argument("--out-dir",    required=True,  type=Path)
    p.add_argument("--condition",  required=True,  type=int, choices=[1,2,3,4,5])
    p.add_argument("--seed",       default=0,      type=int)
    p.add_argument("--gamma",      default=2.0,    type=float)
    p.add_argument("--alpha",      default=0.3,    type=float)
    p.add_argument("--beta",       default=0.7,    type=float)
    p.add_argument("--pos-frac",   default=0.5,    type=float)
    p.add_argument("--pos-weight", default=3.0,    type=float,
                   help="BCE pos_weight (only used when cfg mode == 'bce')")
    p.add_argument("--epochs",     default=110,    type=int)
    p.add_argument("--batch-size", default=32,     type=int)
    p.add_argument("--lr",         default=1e-4,   type=float)
    p.add_argument("--wd",         default=1e-4,   type=float)
    p.add_argument("--warmup-epochs", default=10,  type=int)
    p.add_argument("--num-workers", default=4,     type=int)
    p.add_argument("--wandb",       action="store_true", dest="use_wandb")
    p.add_argument("--no-wandb",    action="store_false", dest="use_wandb")
    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--grid-epochs", default=20, type=int)
    p.set_defaults(use_wandb=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.grid_search:
        grid_search(
            data_dir=args.data_dir,
            stats_path=args.stats,
            out_dir=args.out_dir,
            condition=args.condition,
            seed=args.seed,
            epochs=args.grid_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train(
            data_dir=args.data_dir,
            stats_path=args.stats,
            out_dir=args.out_dir,
            condition=args.condition,
            seed=args.seed,
            gamma=args.gamma,
            alpha=args.alpha,
            beta=args.beta,
            pos_frac=args.pos_frac,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            warmup_epochs=args.warmup_epochs,
            num_workers=args.num_workers,
            use_wandb=args.use_wandb,
            pos_weight=args.pos_weight,
        )
