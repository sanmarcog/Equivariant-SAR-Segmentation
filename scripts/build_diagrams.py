"""Render diagram PNGs for the presentation.

Outputs to figures/:
  - d4_on_debris.png       D4 group acting on an avalanche patch
  - equivariance_flow.png  Encoder equivariance diagram
  - architecture.png       Full model block diagram
  - journey_timeline.png   Experimental sweep timeline

Regenerate: python scripts/build_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

NAVY = "#1A365D"
BLUE = "#2C5282"
GREEN = "#2F855A"
RED = "#C53030"
GREY = "#4A5568"
LIGHT = "#EDF2F7"
GOLD = "#EDC05E"
LIGHTBLUE = "#BEE3F8"
LIGHTGREEN = "#C6F6D5"


# ───────────────────────────────────────────────────────────────────────
# 1. D4 group acting on an avalanche deposit
# ───────────────────────────────────────────────────────────────────────

def _find_deposit_patch(size=160):
    """Find a patch with a clear deposit for the D4 viz."""
    vh = np.load(ROOT / "results_final/vh_tromso.npy")
    gt = np.load(ROOT / "results_final/gt_tromso.npy")
    H, W = gt.shape
    # Find connected components (quick: scan windows)
    best = None
    best_score = 0
    for cy in range(size // 2, H - size // 2, size // 2):
        for cx in range(size // 2, W - size // 2, size // 2):
            gt_patch = gt[cy - size // 2:cy + size // 2,
                          cx - size // 2:cx + size // 2]
            s = gt_patch.sum()
            # Reward patches that have a deposit but aren't all deposit
            # and are centered (deposit mass near middle)
            area = gt_patch.size
            if 0.03 * area < s < 0.35 * area:
                # Centre-of-mass penalty
                ys, xs = np.where(gt_patch > 0.5)
                if len(ys) == 0:
                    continue
                cmy, cmx = ys.mean(), xs.mean()
                off = ((cmy - size / 2) ** 2 + (cmx - size / 2) ** 2) ** 0.5
                score = s / (1 + off / 30)
                if score > best_score:
                    best_score = score
                    best = (cy, cx)
    if best is None:
        # Fallback: centre
        cy, cx = H // 2, W // 2
    else:
        cy, cx = best
    vh_patch = vh[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
    gt_patch = gt[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
    return vh_patch, gt_patch


def d4_transforms(img):
    """Return list of (label, transformed image) for all 8 D4 elements."""
    return [
        ("id",            img),
        ("r90",           np.rot90(img, 1)),
        ("r180",          np.rot90(img, 2)),
        ("r270",          np.rot90(img, 3)),
        ("flip-h",        np.flip(img, axis=1)),
        ("flip-v",        np.flip(img, axis=0)),
        ("flip-diag",     img.T),
        ("flip-antidiag", np.rot90(img.T, 2)),
    ]


def render_d4_on_debris():
    vh, gt = _find_deposit_patch(size=180)

    # VH to display units
    vh_vis = np.clip(vh, -25, -5)
    vh_vis = (vh_vis - vh_vis.min()) / (vh_vis.max() - vh_vis.min() + 1e-8)

    transforms_vh = d4_transforms(vh_vis)
    transforms_gt = d4_transforms(gt)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.08})
    fig.patch.set_facecolor("white")

    for i, ((label, vh_t), (_, gt_t)) in enumerate(zip(transforms_vh, transforms_gt)):
        r, c = i // 4, i % 4
        ax = axes[r, c]
        ax.imshow(vh_t, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(gt_t < 0.5, gt_t),
                  cmap="autumn", alpha=0.55, interpolation="nearest")
        ax.set_title(label, fontsize=14, fontweight="bold",
                     color=NAVY, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GOLD if i == 0 else GREY)
            spine.set_linewidth(2.5 if i == 0 else 1)

    fig.suptitle("The 8 symmetries of  D4  acting on an avalanche deposit",
                 fontsize=20, fontweight="bold", color=NAVY, y=0.98)
    fig.text(0.5, 0.02,
             "All eight are physically equivalent.  The equivariant encoder "
             "produces the same features (up to a group action) for every one.",
             ha="center", fontsize=13, color=GREY, style="italic")

    out = FIG / "d4_on_debris.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ───────────────────────────────────────────────────────────────────────
# 2. Equivariance flow diagram
# ───────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fontsize=12,
         text_color=NAVY, bold=True):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.8, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=text_color)


def _arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2, label=None,
           label_color=None, label_offset=(0.1, 0)):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="->", mutation_scale=18,
                          linewidth=lw, color=color)
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, fontsize=11, color=label_color or color,
                fontweight="bold", ha="left", va="center")


def render_equivariance_flow():
    """Two parallel pipelines; horizontal equivalence arrows placed between
    rows (not cutting through boxes)."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.5, 7.2, "Encoder is exactly D4-equivariant   ·   full model is approximately equivariant",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    # Two columns
    xL, xR = 2.5, 8.5
    box_w = 2.2
    cL, cR = xL + box_w / 2, xR + box_w / 2   # column centres

    # Row y-centres (top → bottom): input → encoder → invariant → decoder → logit
    rows = [
        ("Input",              6.55, None),
        ("D4-equivariant\nEncoder", 5.55, LIGHTBLUE),
        ("GroupPooling",       4.25, LIGHTGREEN),
        ("Conv2d Decoder",     2.75, "#FED7D7"),
        ("Logit",              1.35, None),
    ]

    # Header labels
    ax.text(cL, 7.0, "Input  x", ha="center", fontsize=14,
            color=NAVY, fontweight="bold")
    ax.text(cR, 7.0, "Transformed input  g · x", ha="center", fontsize=14,
            color=NAVY, fontweight="bold")

    # Draw boxes for encoder / pool / decoder
    box_specs = [
        ("D4-equivariant Encoder", 5.25, LIGHTBLUE,   0.8),
        ("GroupPooling",           4.05, LIGHTGREEN,  0.6),
        ("Conv2d Decoder",         2.55, "#FED7D7",   0.6),
    ]
    for text, y, color, h in box_specs:
        _box(ax, xL, y, box_w, h, text, fc=color, fontsize=12)
        _box(ax, xR, y, box_w, h, text, fc=color, fontsize=12)

    # Input / output text
    ax.text(cL, 6.4, "x", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(cR, 6.4, "g · x", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(cL, 1.3, "logit(x)", ha="center", fontsize=14,
            fontweight="bold", color=NAVY)
    ax.text(cR, 1.3, "logit(g · x)", ha="center", fontsize=14,
            fontweight="bold", color=NAVY)

    # Vertical flow arrows inside each column
    col_y_pairs = [
        (6.3, 6.05),   # x → encoder
        (5.25, 4.65),  # encoder → grouppool
        (4.05, 3.15),  # grouppool → decoder
        (2.55, 1.55),  # decoder → logit
    ]
    for y_start, y_end in col_y_pairs:
        _arrow(ax, cL, y_start, cL, y_end, color=NAVY, lw=2)
        _arrow(ax, cR, y_start, cR, y_end, color=NAVY, lw=2)

    # Horizontal equivalence arrows placed in GAPS between row boxes
    # (so they don't overlap text)
    gap_rows = [
        (5.9,  "equivariant", GREEN,  "exact"),   # between input and encoder (label: encoder maps g·)
        (4.85, "equivariant", GREEN,  "exact"),   # between encoder out and GroupPool
        (3.5,  "= invariant", GREEN,  "exact"),   # between invariant tensor rows (under group pool)
        (2.05, "≈ approx",    RED,    "approximate"),  # between decoder rows
        (0.9,  "≈ approx",    RED,    "approximate"),  # between logit outputs
    ]

    # Place only three, visually clean arrows:
    #  1) encoder output row (y = 4.85): feature-level EXACT
    #  2) grouppool output row (y = 3.5): invariant EXACT
    #  3) final logit row (y = 0.9): approximate
    horiz = [
        (4.85, "g · acts equivariantly", GREEN, "E X A C T"),
        (3.50, "invariant (no g on output)", GREEN, "E X A C T"),
        (0.90, "decoder breaks exact symmetry", RED, "A P P R O X"),
    ]
    for y, label, color, tag in horiz:
        _arrow(ax, xL + box_w + 0.15, y, xR - 0.15, y,
               color=color, lw=2.5)
        ax.text((xL + box_w + xR) / 2, y + 0.22, label,
                ha="center", fontsize=11, color=color, fontweight="bold",
                style="italic")
        ax.text((xL + box_w + xR) / 2, y - 0.32, tag,
                ha="center", fontsize=11, color=color, fontweight="bold")

    # Side labels
    ax.text(0.3, 5.65, "equivariant\nfeatures", fontsize=11,
            color=GREY, style="italic", va="center")
    ax.text(0.3, 4.35, "invariant\ntensor", fontsize=11,
            color=GREY, style="italic", va="center")
    ax.text(0.3, 2.85, "standard\ndecoder", fontsize=11,
            color=GREY, style="italic", va="center")

    plt.tight_layout()
    out = FIG / "equivariance_flow.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ───────────────────────────────────────────────────────────────────────
# 3. Architecture diagram
# ───────────────────────────────────────────────────────────────────────

def render_architecture():
    """Full-width architecture: two encoder branches + extras sidebar."""
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.5, 8.15, "D4SegNet architecture",
            ha="center", fontsize=20, fontweight="bold", color=NAVY)
    ax.text(7.5, 7.75,
            "12-channel bi-temporal input   ·   625,617 parameters   ·   ~4× fewer than Swin-UNet",
            ha="center", fontsize=13, color=GREY, style="italic")

    # ── Input channels row ─────────────────────────────────────────
    _box(ax, 0.8, 6.4, 3.4, 0.9,
         "post  channels 0–5\n(VH, VV, log-ratio, RGD, LIA, ...)",
         fc=LIGHTBLUE, fontsize=11)
    _box(ax, 4.8, 6.4, 3.4, 0.9,
         "pre  channels 6,7,2–5\n(VH, VV + shared derived)",
         fc=LIGHTBLUE, fontsize=11)
    _box(ax, 8.8, 6.4, 3.4, 0.9,
         "extra  channels 8–11\n(DEM, slope, aspect, ...)",
         fc="#FEEBC8", fontsize=11)

    # ── Encoders ───────────────────────────────────────────────────
    _box(ax, 0.8, 5.0, 3.4, 0.9,
         "D4-equivariant encoder\n5 blocks  ·  escnn N=4 + flips",
         fc=GREEN, text_color="white", fontsize=11)
    _box(ax, 4.8, 5.0, 3.4, 0.9,
         "D4-equivariant encoder\n(shared weights)",
         fc=GREEN, text_color="white", fontsize=11)

    # arrows input → encoder
    _arrow(ax, 2.5, 6.4, 2.5, 5.9, lw=2)
    _arrow(ax, 6.5, 6.4, 6.5, 5.9, lw=2)

    # shared weights indicator
    ax.annotate("", xy=(4.8, 5.45), xytext=(4.2, 5.45),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.5))
    ax.text(4.5, 5.65, "shared", ha="center", fontsize=10,
            color=GOLD, fontweight="bold")

    # ── Difference ────────────────────────────────────────────────
    _box(ax, 2.8, 3.7, 3.4, 0.8,
         "post features  −  pre features\n(equivariant difference)",
         fc=LIGHT, fontsize=11)
    # arrows encoder → difference
    _arrow(ax, 2.5, 5.0, 3.8, 4.5, lw=2)
    _arrow(ax, 6.5, 5.0, 5.2, 4.5, lw=2)

    # ── GroupPooling ──────────────────────────────────────────────
    _box(ax, 2.8, 2.4, 3.4, 0.8,
         "GroupPooling  →  invariant tensor",
         fc=GREEN, text_color="white", fontsize=11)
    _arrow(ax, 4.5, 3.7, 4.5, 3.2, lw=2)

    # ── Decoder ───────────────────────────────────────────────────
    _box(ax, 7.2, 2.4, 3.4, 0.8,
         "Conv2d decoder\n4 stages  ·  skip connections",
         fc="#FED7D7", fontsize=11)
    _arrow(ax, 6.2, 2.8, 7.2, 2.8, lw=2)

    # extras → decoder injection (thick gold arrow bending in)
    _arrow(ax, 10.5, 6.4, 10.5, 2.8, color=GOLD, lw=2.5)
    _arrow(ax, 10.5, 2.8, 10.6, 2.8, color=GOLD, lw=2.5)
    ax.text(10.6, 4.6,
            "AdaptiveAvgPool\ninjection @ each\ndecoder stage",
            fontsize=9, color=GOLD, style="italic", va="center",
            fontweight="bold", ha="left")

    # ── Heads ─────────────────────────────────────────────────────
    _box(ax, 6.3, 0.9, 2.5, 0.8,
         "seg head\nlogit  (B, 1, H, W)",
         fc=NAVY, text_color="white", fontsize=11)
    _box(ax, 9.0, 0.9, 2.5, 0.8,
         "area head\narea  (B, 1)",
         fc=NAVY, text_color="white", fontsize=11)
    _arrow(ax, 8.0, 2.4, 7.55, 1.7, lw=2)
    _arrow(ax, 9.8, 2.4, 10.25, 1.7, lw=2)

    # ── Sidebar: param count ──────────────────────────────────────
    _box(ax, 13.1, 4.7, 1.7, 1.2,
         "0.63 M\nparams",
         fc=GOLD, text_color=NAVY, fontsize=18)
    ax.text(13.95, 4.2, "Swin-UNet: 2.39 M",
            ha="center", fontsize=10, color=GREY)
    ax.text(13.95, 3.8, "~4× fewer",
            ha="center", fontsize=12, color=NAVY, fontweight="bold")

    plt.tight_layout()
    out = FIG / "architecture.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ───────────────────────────────────────────────────────────────────────
# 4. Experimental journey timeline
# ───────────────────────────────────────────────────────────────────────

def render_journey():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.5, 7.2, "Experimental journey  ·  every step is a SLURM sweep on Hyak",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    steps = [
        ("gatti_mirror.sh",  "cond1, no aug, no skip",             "F1 ≈ 0.75", LIGHTBLUE, NAVY),
        ("aug_experiment.sh","aug 1× / 3× / 5×",                   "aug3× wins", LIGHTBLUE, NAVY),
        ("reg_sweep 1/2/3/3b","depth · wd · focal-tversky · ...",  "bigger hurts\nwarm-restarts hurt", "#FED7D7", RED),
        ("lambda_sweep.sh",  "component-IoU λ grid",               "bimodal trade-off", "#FED7D7", RED),
        ("calibration_sweep","T-scale / Platt / isotonic",         "marginal gain", LIGHT, GREY),
        ("★  FINAL  ★",      "cond1, no-skip, aug3×, seed 1",      "pixel F1 = 0.7938", GOLD, NAVY),
    ]

    y = 6.2
    dy = 0.95
    for i, (name, detail, outcome, color, tcolor) in enumerate(steps):
        top = y - i * dy
        _box(ax, 0.5, top, 3.5, 0.7, name, fc=color, text_color=tcolor, fontsize=13)
        ax.text(4.3, top + 0.35, detail, fontsize=12, color=NAVY, va="center")
        ax.text(9.0, top + 0.35, "→", fontsize=18, color=GREY, va="center")
        ax.text(9.5, top + 0.35, outcome, fontsize=12, color=tcolor,
                va="center", fontweight="bold")
        if i < len(steps) - 1:
            _arrow(ax, 2.25, top, 2.25, top - dy + 0.7, color=GREY, lw=1.5)

    plt.tight_layout()
    out = FIG / "journey_timeline.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ───────────────────────────────────────────────────────────────────────
# 5. Dataset EDA
# ───────────────────────────────────────────────────────────────────────

def render_eda():
    """Two-panel EDA chart: D-scale histogram + deposit area distribution.
    The text annotations (data split, class imbalance) are native PPT
    elements — see build_slides.py."""
    from scipy import ndimage

    # Use Patagonia-leaning palette for the chart so it matches slide bg
    CREAM_BG = "#EFE8DB"
    INK_C   = "#2D3142"
    RUST_C  = "#C1440E"
    FOREST_C = "#4A6741"
    SAND_C  = "#D4A574"
    MUTED_C = "#8B8374"
    LIGHT_BG = "#F7F2E8"

    gt = np.load(ROOT / "results_final/gt_tromso.npy")
    lab, n = ndimage.label(gt > 0.5)
    sizes_px = np.array([(lab == i).sum() for i in range(1, n + 1)])
    sizes_m2 = sizes_px * 100  # 10m SAR pixel ~ 100 m² per pixel

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    fig.patch.set_facecolor(CREAM_BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.28,
                        wspace=0.28)

    # ── Panel 1: D-scale histogram ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CREAM_BG)
    dscales = ["D1", "D2", "D3", "D4"]
    counts = [5, 25, 72, 16]
    bar_colors = [SAND_C, SAND_C, RUST_C, INK_C]  # D3 in rust (our highlight)
    xs = np.arange(len(dscales))
    bars = ax.bar(xs, counts, color=bar_colors,
                  edgecolor=INK_C, linewidth=1.2, width=0.7)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.5, str(c),
                ha="center", fontsize=14, fontweight="bold", color=INK_C)
    ax.set_title("Test deposits by EAWS D-scale  (destructive size)",
                 fontsize=14, fontweight="bold", color=INK_C, loc="left",
                 pad=12)
    ax.set_ylabel("count", fontsize=10, color=MUTED_C)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", alpha=0.3, color=MUTED_C)
    ax.set_axisbelow(True)

    # Two-line x-labels: D-code (bold) + short size word (muted italic)
    labels = [
        "D1\nsmall",
        "D2\nmedium",
        "D3\nlarge",
        "D4\nvery large",
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11, color=INK_C)
    # Caption below the x-labels explaining what each D-level means
    ax.text(0.5, -0.30,
            "D1 knocks down a person  ·  D2 can bury  ·  D3 can destroy a car  ·  D4 can destroy a building",
            transform=ax.transAxes, fontsize=9, color=MUTED_C,
            style="italic", va="top", ha="center")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED_C)
    ax.tick_params(axis="y", colors=MUTED_C)
    ax.tick_params(axis="x", length=0)

    # ── Panel 2: deposit area distribution ────────────────────────
    ax = axes[1]
    ax.set_facecolor(CREAM_BG)
    ax.hist(sizes_m2, bins=np.logspace(2, 5.5, 28),
            color=FOREST_C, edgecolor=INK_C, alpha=0.85, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("deposit area  (m²)", fontsize=10, color=MUTED_C)
    ax.set_ylabel("count", fontsize=10, color=MUTED_C)
    ax.set_title("Deposit area distribution  (log scale)",
                 fontsize=14, fontweight="bold", color=INK_C, loc="left",
                 pad=12)
    med = np.median(sizes_m2)
    ax.axvline(med, color=RUST_C, lw=2, ls="--",
               label=f"median: {int(med):,} m²")
    ax.grid(alpha=0.3, color=MUTED_C); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=False,
              labelcolor=INK_C)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED_C)
    ax.tick_params(colors=MUTED_C)

    out = FIG / "eda.png"
    plt.savefig(out, dpi=180, facecolor=CREAM_BG, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print(f"  → {out}")


# ───────────────────────────────────────────────────────────────────────
# 6. Head-to-head example (case id=28)
# ───────────────────────────────────────────────────────────────────────

def render_head_to_head():
    """Specific D3-ish example where we detect and Gatti nearly misses.
    Case id=28 at (cy=501, cx=347): our IoU 0.77, Gatti IoU 0.20."""
    vh = np.load(ROOT / "results_final/vh_tromso.npy")
    gt = np.load(ROOT / "results_final/gt_tromso.npy")
    p_ours = np.load(ROOT / "results_final/prob_p2_tromso.npy")
    p_gat = np.load(ROOT / "results_final/prob_gatti_tromso.npy")

    cy, cx = 501, 347
    R = 80
    y0, y1 = max(0, cy - R), min(vh.shape[0], cy + R)
    x0, x1 = max(0, cx - R), min(vh.shape[1], cx + R)

    vh_c   = vh[y0:y1, x0:x1]
    gt_c   = gt[y0:y1, x0:x1]
    po_c   = p_ours[y0:y1, x0:x1]
    pg_c   = p_gat[y0:y1, x0:x1]

    vh_vis = np.clip(vh_c, -25, -5)
    vh_vis = (vh_vis - vh_vis.min()) / (vh_vis.max() - vh_vis.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4),
                             gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("white")

    titles = [
        ("VH backscatter", None, None),
        ("Ground truth (D3-sized)", None, None),
        ("Our prediction", "IoU = 0.77", GREEN),
        ("Swin-UNet (retrain)", "IoU = 0.20", RED),
    ]

    # Panel 1: VH
    axes[0].imshow(vh_vis, cmap="gray", vmin=0, vmax=1)

    # Panel 2: VH + GT overlay
    axes[1].imshow(vh_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(gt_c < 0.5, gt_c),
                   cmap="autumn", alpha=0.55)

    # Panel 3: VH + our prob + GT outline
    axes[2].imshow(vh_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(po_c < 0.15, po_c),
                   cmap="viridis", alpha=0.80, vmin=0.15, vmax=1.0)
    axes[2].contour(gt_c, levels=[0.5], colors="cyan", linewidths=1.5)

    # Panel 4: VH + gatti prob + GT outline
    axes[3].imshow(vh_vis, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(pg_c < 0.15, pg_c),
                   cmap="viridis", alpha=0.80, vmin=0.15, vmax=1.0)
    axes[3].contour(gt_c, levels=[0.5], colors="cyan", linewidths=1.5)

    for ax, (t, sub, color) in zip(axes, titles):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=14, fontweight="bold", color=NAVY, pad=4)
        if sub:
            ax.text(0.5, -0.08, sub, transform=ax.transAxes,
                    ha="center", fontsize=14, fontweight="bold", color=color)

    fig.suptitle("One deposit, two models  ·  Tromsø test scene",
                 fontsize=18, fontweight="bold", color=NAVY, y=1.02)
    fig.text(0.5, -0.08,
             "Same ground truth (cyan outline).  Our model covers the deposit; "
             "the Swin-UNet barely touches it.  This is the 10.7 pp instance-F1 "
             "gap in one picture.",
             ha="center", fontsize=12, color=GREY, style="italic")

    out = FIG / "head_to_head.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    print("Building diagram PNGs...")
    render_d4_on_debris()
    render_equivariance_flow()
    render_architecture()
    render_journey()
    render_eda()
    render_head_to_head()
    print("Done.")
