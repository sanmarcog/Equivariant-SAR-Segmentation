"""
src/data/norm_stats.py

Compute per-channel mean and std over all pixels in the TRAIN split,
using Welford's online algorithm (single pass, no full array in RAM).

The 12-channel stats are saved to data/norm_stats_12ch.json and
used by the dataset to z-score normalise patches at load time.

IMPORTANT: stats are computed on train split only. Never touch val/test.

Usage:
    python -m src.data.norm_stats \\
        --data-dir /path/to/avalcd/raw \\
        --out      data/norm_stats_12ch.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.preprocess import preprocess_scene, CHANNEL_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Fixed train split (must never change)
TRAIN_SCENES = [
    "Livigno_20240403",
    "Livigno_20250129",
    "Nuuk_20160413",
    "Nuuk_20210411",
    "Pish_20230221",
]


def compute_stats(data_dir: Path, out_path: Path) -> dict:
    """
    Preprocess every train scene and accumulate per-channel Welford stats.

    Args:
        data_dir: Root of AvalCD raw data directory containing one subdir per scene.
        out_path: Where to write the JSON stats file.

    Returns:
        stats dict with keys: channels, mean, std
    """
    n_ch = len(CHANNEL_NAMES)
    count = np.zeros(n_ch, dtype=np.float64)
    mean  = np.zeros(n_ch, dtype=np.float64)
    M2    = np.zeros(n_ch, dtype=np.float64)

    for scene_name in TRAIN_SCENES:
        scene_dir = data_dir / scene_name
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {scene_dir}")

        log.info("Processing scene %s ...", scene_name)
        arr = preprocess_scene(scene_dir)  # [12, H, W]
        H, W = arr.shape[1], arr.shape[2]

        for c in range(n_ch):
            pixels = arr[c].ravel().astype(np.float64)
            # Welford batch update
            batch_n    = len(pixels)
            batch_mean = pixels.mean()
            batch_M2   = ((pixels - batch_mean) ** 2).sum()

            # Combine with running accumulators
            if count[c] == 0:
                count[c] = batch_n
                mean[c]  = batch_mean
                M2[c]    = batch_M2
            else:
                combined_n  = count[c] + batch_n
                delta       = batch_mean - mean[c]
                mean[c]     = (count[c] * mean[c] + batch_n * batch_mean) / combined_n
                M2[c]      += batch_M2 + delta ** 2 * count[c] * batch_n / combined_n
                count[c]    = combined_n

        log.info("  processed %d×%d = %d pixels", H, W, H * W)

    std = np.sqrt(M2 / count)
    # Guard against zero / near-zero std (constant channels)
    std = np.where(std < 1e-6, 1.0, std)

    stats = {
        "channels": CHANNEL_NAMES,
        "mean":     mean.tolist(),
        "std":      std.tolist(),
        "n_pixels": count.tolist(),
        "scenes":   TRAIN_SCENES,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats saved to %s", out_path)

    for c, name in enumerate(CHANNEL_NAMES):
        log.info("  %12s  mean=%+7.3f  std=%6.3f", name, mean[c], std[c])

    return stats


def load_stats(stats_path: str | Path) -> dict:
    with open(stats_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute 12-ch normalisation stats on train split")
    p.add_argument("--data-dir", required=True, type=Path, help="AvalCD raw data root")
    p.add_argument("--out",      required=True, type=Path, help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_stats(args.data_dir, args.out)
