"""
scripts/stacking_toy_experiment.py

Toy experiment: does multi-temporal pre-event stacking improve the SAR
change signal quality for D2 avalanche deposits?

Compares two change images over the Tromsø OOD test scene:
  - change_single : postVH_dB − preVH_avalcd_dB  (AvalCD single pre-event)
  - change_stack  : postVH_dB − preVH_stack_dB   (median of 3 descending pre-event scenes)

For each GT polygon (117 total, D1–D4), computes:
  contrast = mean(change within polygon) − mean(change in 100 m buffer)

Reports contrast by D-scale class and saves two figures:
  figures/toy_change_images.png   — side-by-side change maps
  figures/toy_contrast_by_dscale.png — boxplot: single vs stack

Usage:
    python scripts/stacking_toy_experiment.py

Requirements:
    pip install rasterio numpy geopandas matplotlib scipy tqdm
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy import ndimage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AVALCD_DIR   = Path("/Users/sanmarco/Documents/GitHub/Equivariant-CNN-SAR/data/raw/Tromso_20241220")
PREEVENTS_DIR = Path("/Users/sanmarco/Documents/GitHub/Equivariant-SAR-Segmentation/data/tromso_preevents")
FIGURES_DIR  = Path("/Users/sanmarco/Documents/GitHub/Equivariant-SAR-Segmentation/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

POST_VH  = AVALCD_DIR / "Tromso_20241220_postVH.tif"
PRE_VH   = AVALCD_DIR / "Tromso_20241220_preVH.tif"
GT_GPKG  = AVALCD_DIR / "Tromso_20241220_GT.gpkg"

# Descending scenes only (T16xxxx acquisition time = ~16:00 UTC = descending for Tromsø)
# Post-event is 20241220T161604 — descending. Stack must match.
# Note: 20241208 zip (1.0 GB) was truncated during download — excluded.
# Re-download with: python scripts/download_tromso_preevents.py (will skip completed files)
DESCENDING_ZIPS = [
    "S1A_IW_GRDH_1SDV_20241121T160755_20241121T160820_056655_06F34F_5F82.SAFE.zip",
    "S1A_IW_GRDH_1SDV_20241203T160754_20241203T160819_056830_06FA40_34A9.SAFE.zip",
]

BUFFER_M = 100   # background buffer width in metres
DSCALE_LABELS = {1: "D1", 2: "D2", 3: "D3", 4: "D4"}


# ---------------------------------------------------------------------------
# Step 1: load AvalCD reference grid
# ---------------------------------------------------------------------------
def load_reference_grid() -> tuple[dict, np.ndarray, np.ndarray]:
    """Return profile, postVH_dB, preVH_dB arrays."""
    with rasterio.open(POST_VH) as src:
        profile = src.profile.copy()
        post = src.read(1).astype(np.float32)
    with rasterio.open(PRE_VH) as src:
        pre = src.read(1).astype(np.float32)
    return profile, post, pre


# ---------------------------------------------------------------------------
# Step 2: extract and reproject VH from one SAFE zip
# ---------------------------------------------------------------------------
def safe_vh_to_db(zip_path: Path, ref_profile: dict) -> np.ndarray:
    """
    Extract VH measurement tiff from a SAFE zip, convert DN→dB,
    reproject to ref_profile grid. Returns float32 array (H, W).

    S1 GRD measurement tiffs have no affine CRS — georef is stored as
    GCPs in WGS84. We use rasterio.warp.reproject with gcps= kwarg.
    """
    with zipfile.ZipFile(zip_path) as z:
        vh_members = [f for f in z.namelist()
                      if "measurement" in f and "-vh-" in f and f.endswith(".tiff")]
        if not vh_members:
            raise FileNotFoundError(f"No VH measurement tiff in {zip_path.name}")
        vh_name = vh_members[0]

        with tempfile.TemporaryDirectory() as tmp:
            z.extract(vh_name, tmp)
            tiff_path = Path(tmp) / vh_name

            with rasterio.open(tiff_path) as src:
                # Raw uint16 DN → linear → dB
                dn = src.read(1).astype(np.float32)
                valid = dn > 0
                linear = np.where(valid, dn ** 2, np.nan)
                db = np.where(valid, 10.0 * np.log10(linear + 1e-10), np.nan)

                # S1 GRD tiffs use GCPs for georeferencing (no affine CRS)
                gcps, gcp_crs = src.gcps
                if not gcps:
                    raise ValueError(f"No GCPs found in {zip_path.name}")

                # Reproject to AvalCD grid using GCPs
                dst_shape = (ref_profile["height"], ref_profile["width"])
                dst_arr = np.full(dst_shape, np.nan, dtype=np.float32)

                rasterio.warp.reproject(
                    source=db,
                    destination=dst_arr,
                    gcps=gcps,
                    src_crs=gcp_crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=rasterio.warp.Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )

    return dst_arr


# ---------------------------------------------------------------------------
# Step 3: build median stack
# ---------------------------------------------------------------------------
def build_stack(ref_profile: dict) -> np.ndarray:
    """Warp all descending scenes and return pixel-wise median."""
    scenes = []
    for name in tqdm(DESCENDING_ZIPS, desc="Processing SAFE scenes"):
        path = PREEVENTS_DIR / name
        arr = safe_vh_to_db(path, ref_profile)
        scenes.append(arr)

    stack = np.stack(scenes, axis=0)   # [3, H, W]
    median = np.nanmedian(stack, axis=0).astype(np.float32)

    # Align absolute level to AvalCD pre-event in background pixels
    # (removes the ~90 dB offset from missing calibration constant)
    with rasterio.open(PRE_VH) as src:
        avalcd_pre = src.read(1).astype(np.float32)

    valid = np.isfinite(median) & np.isfinite(avalcd_pre)
    offset = float(np.nanmean(avalcd_pre[valid] - median[valid]))
    median += offset

    return median


# ---------------------------------------------------------------------------
# Step 4: compute change images
# ---------------------------------------------------------------------------
def compute_change(post: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """Change image: post − pre (dB). Positive = backscatter increased."""
    return post - pre


# ---------------------------------------------------------------------------
# Step 5: per-polygon contrast
# ---------------------------------------------------------------------------
def polygon_contrast(
    change: np.ndarray,
    gdf: gpd.GeoDataFrame,
    transform,
    crs,
    buffer_m: float = BUFFER_M,
) -> list[dict]:
    """
    For each polygon: contrast = mean(change inside) − mean(change in buffer).
    Returns list of dicts with size, area, signal, background, contrast.
    """
    results = []
    gdf_proj = gdf.to_crs(crs)
    H, W = change.shape

    for _, row in gdf_proj.iterrows():
        geom = row.geometry

        # Rasterize polygon
        poly_mask = rasterio.features.rasterize(
            [(geom, 1)], out_shape=(H, W), transform=transform,
            fill=0, dtype=np.uint8,
        )

        # Rasterize buffer ring
        buf_geom  = geom.buffer(buffer_m)
        ring_mask = rasterio.features.rasterize(
            [(buf_geom, 1)], out_shape=(H, W), transform=transform,
            fill=0, dtype=np.uint8,
        )
        ring_mask = np.maximum(ring_mask - poly_mask, 0)  # exclude polygon interior

        poly_vals = change[poly_mask == 1]
        ring_vals = change[ring_mask == 1]

        # Need at least 1 pixel in polygon and 4 in ring
        if len(poly_vals) == 0 or len(ring_vals) < 4:
            continue

        poly_vals = poly_vals[np.isfinite(poly_vals)]
        ring_vals = ring_vals[np.isfinite(ring_vals)]

        if len(poly_vals) == 0 or len(ring_vals) == 0:
            continue

        results.append({
            "size":       int(row["size"]),
            "area_m2":    float(row["area"]),
            "signal":     float(np.mean(poly_vals)),
            "background": float(np.mean(ring_vals)),
            "contrast":   float(np.mean(poly_vals) - np.mean(ring_vals)),
            "n_pixels":   len(poly_vals),
        })

    return results


# ---------------------------------------------------------------------------
# Step 6: figures
# ---------------------------------------------------------------------------
def plot_change_images(
    change_single: np.ndarray,
    change_stack: np.ndarray,
    gdf: gpd.GeoDataFrame,
    transform,
    crs: str,
) -> None:
    from rasterio.plot import show
    import matplotlib.colors as mcolors

    vmin, vmax = -10, 10
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Single pre-event (AvalCD)", "Stacked pre-event (median 3 scenes)"]
    arrays = [change_single, change_stack]

    gdf_proj = gdf.to_crs(crs)

    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=rasterio.transform.array_bounds(
                           arr.shape[0], arr.shape[1], transform))
        ax.set_title(title, fontsize=13)

        # Overlay GT polygons coloured by D-scale
        colors = {1: "cyan", 2: "yellow", 3: "orange", 4: "red"}
        for dscale, color in colors.items():
            subset = gdf_proj[gdf_proj["size"] == dscale]
            if not subset.empty:
                subset.boundary.plot(ax=ax, color=color, linewidth=0.8)

        plt.colorbar(im, ax=ax, label="Change (dB)", shrink=0.7)

    # Legend
    patches = [mpatches.Patch(color=c, label=f"D{d}")
               for d, c in {1:"cyan",2:"yellow",3:"orange",4:"red"}.items()]
    axes[1].legend(handles=patches, loc="lower right", fontsize=9)

    plt.suptitle("Change image: post − pre  (dB)\nTromsø OOD scene, Dec 2024",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "toy_change_images.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_contrast_boxplot(
    results_single: list[dict],
    results_stack: list[dict],
) -> None:
    import pandas as pd

    df_s = pd.DataFrame(results_single)
    df_s["method"] = "Single pre-event"
    df_k = pd.DataFrame(results_stack)
    df_k["method"] = "Stacked pre-event"
    df = pd.concat([df_s, df_k], ignore_index=True)

    dscales = sorted(df["size"].unique())
    fig, axes = plt.subplots(1, len(dscales), figsize=(4 * len(dscales), 5), sharey=True)
    if len(dscales) == 1:
        axes = [axes]

    colors = {"Single pre-event": "#4878CF", "Stacked pre-event": "#D65F5F"}

    for ax, d in zip(axes, dscales):
        subset = df[df["size"] == d]
        data = [subset[subset["method"] == m]["contrast"].values
                for m in ["Single pre-event", "Stacked pre-event"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, method in zip(bp["boxes"], colors):
            patch.set_facecolor(colors[method])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        n_s = len(data[0])
        n_k = len(data[1])
        ax.set_title(f"D{d}  (n={n_s})", fontsize=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Single", "Stack"], fontsize=9)
        ax.set_ylabel("Contrast (dB)" if d == dscales[0] else "")

    patches = [mpatches.Patch(color=c, label=m) for m, c in colors.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    plt.suptitle("Polygon contrast: signal − background (dB)\nHigher = cleaner change signal",
                 fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "toy_contrast_by_dscale.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Toy experiment: single vs stacked pre-event reference")
    print("=" * 60)

    print("\n[1/5] Loading AvalCD reference grid...")
    profile, post_dB, pre_avalcd_dB = load_reference_grid()
    print(f"      Scene shape: {post_dB.shape}, CRS: {profile['crs']}")

    print("\n[2/5] Building median stack from 3 descending scenes...")
    pre_stack_dB = build_stack(profile)

    print("\n[3/5] Computing change images...")
    change_single = compute_change(post_dB, pre_avalcd_dB)
    change_stack  = compute_change(post_dB, pre_stack_dB)

    print(f"      change_single: mean={np.nanmean(change_single):.3f} dB  "
          f"std={np.nanstd(change_single):.3f}")
    print(f"      change_stack:  mean={np.nanmean(change_stack):.3f} dB  "
          f"std={np.nanstd(change_stack):.3f}")

    print("\n[4/5] Computing per-polygon contrast...")
    gdf = gpd.read_file(GT_GPKG)
    results_single = polygon_contrast(change_single, gdf, profile["transform"], profile["crs"])
    results_stack  = polygon_contrast(change_stack,  gdf, profile["transform"], profile["crs"])
    print(f"      Polygons evaluated: {len(results_single)} / {len(gdf)}")

    # Summary table
    import pandas as pd
    df_s = pd.DataFrame(results_single)
    df_k = pd.DataFrame(results_stack)

    print("\n[5/5] Results: mean contrast (dB) by D-scale\n")
    print(f"  {'D-scale':<10} {'n':>4}  {'Single (dB)':>12}  {'Stack (dB)':>12}  {'Δ (dB)':>10}")
    print("  " + "-" * 55)
    for d in sorted(df_s["size"].unique()):
        s_vals = df_s[df_s["size"] == d]["contrast"]
        k_vals = df_k[df_k["size"] == d]["contrast"]
        delta = k_vals.mean() - s_vals.mean()
        marker = " ◀ stack better" if delta > 0.5 else (" ▶ single better" if delta < -0.5 else "")
        print(f"  D{d:<9} {len(s_vals):>4}  {s_vals.mean():>12.2f}  {k_vals.mean():>12.2f}  "
              f"{delta:>+10.2f}{marker}")

    print("\nSaving figures...")
    plot_change_images(change_single, change_stack, gdf, profile["transform"], profile["crs"])
    plot_contrast_boxplot(results_single, results_stack)

    print("\nDone.")
    print("  figures/toy_change_images.png")
    print("  figures/toy_contrast_by_dscale.png")


if __name__ == "__main__":
    main()
