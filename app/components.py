"""Reusable UI bits: model card, four-panel image grid, live metrics row."""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage

from app.instance_data import (
    best_iou_against, index_deposits, instance_metrics,
)

# Patagonia palette — keep in sync with scripts/build_slides.py
INK    = "#2D3142"
RUST   = "#C1440E"
FOREST = "#4A6741"
SAND   = "#D4A574"
MUTED  = "#8B8374"


# ─── Hero headline at the top ─────────────────────────────────────────

def hero_card():
    """Two-card hero strip: pixel F1 (tied) vs instance F1 (+10.7 pp)."""
    st.markdown(
        f"""
        <div style="display:flex; gap:14px; margin: 4px 0 28px 0;">
          <div style="flex:1; background:#F7F2E8; border-left:5px solid {MUTED};
                      padding:16px 20px; border-radius:2px;">
            <div style="font-family:Futura,sans-serif; font-size:0.78rem;
                        letter-spacing:0.10em; color:{MUTED};">PIXEL  F1</div>
            <div style="font-family:Futura,sans-serif; font-size:1.5rem;
                        font-weight:bold; color:{INK}; margin-top:6px;">
              0.794  /  0.795
            </div>
            <div style="color:{MUTED}; font-size:0.95rem; margin-top:2px;">
              ours / Swin-UNet — <b>tied</b>
            </div>
          </div>
          <div style="flex:1; background:#F7F2E8; border-left:5px solid {RUST};
                      padding:16px 20px; border-radius:2px;">
            <div style="font-family:Futura,sans-serif; font-size:0.78rem;
                        letter-spacing:0.10em; color:{RUST};">INSTANCE  F1</div>
            <div style="font-family:Futura,sans-serif; font-size:1.5rem;
                        font-weight:bold; color:{INK}; margin-top:6px;">
              0.637  /  0.530
            </div>
            <div style="color:{RUST}; font-size:0.95rem; margin-top:2px;
                        font-weight:bold;">
              +10.7 pp — the gap pixel F1 hides
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Model card sidebar ───────────────────────────────────────────────

def model_card() -> None:
    st.markdown("### D4-EquiCNN")
    st.markdown(
        f"<small style='color:{MUTED}'>Phase 2  ·  pixel segmentation</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.metric("Pixel F1",                       "0.794")
    st.metric("Instance F1  (excl D1)",         "0.637", "+10.7 pp vs Swin-UNet")
    st.metric("Parameters",                     "0.63 M", "−1.76 M vs Swin-UNet")
    st.markdown("---")
    st.markdown(
        f"<small>"
        f"<a style='color:{RUST}' href='https://github.com/sanmarcog/Equivariant-SAR-Segmentation'>Phase 2 repo</a>  ·  "
        f"<a style='color:{RUST}' href='https://github.com/sanmarcog/Equivariant-CNN-SAR'>Phase 1</a>"
        f"</small>",
        unsafe_allow_html=True,
    )


# ─── Image rendering ──────────────────────────────────────────────────

def _vh_to_norm(vh: np.ndarray) -> np.ndarray:
    """Tone-map VH backscatter (dB) to [0, 1] for display."""
    vis = np.clip(vh, -25, -5)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    return vis


def _disk(radius: int) -> np.ndarray:
    L = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_panel(
    vh_norm: np.ndarray,
    mask: np.ndarray | None = None,
    color_hex: str = RUST,
    alpha: int = 130,
) -> Image.Image:
    """VH grayscale base + optional tinted mask overlay → PIL Image."""
    h, w = vh_norm.shape
    base = (vh_norm * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)

    if mask is None:
        return Image.fromarray(rgb, mode="RGB")

    r, g, b = _hex_to_rgb(color_hex)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask] = (r, g, b, alpha)

    img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    img = Image.alpha_composite(img, Image.fromarray(overlay, mode="RGBA"))
    return img.convert("RGB")




# ─── 4-panel grid ──────────────────────────────────────────────────────

def _binarize(prob: np.ndarray, threshold: float) -> np.ndarray:
    return prob > threshold


def four_panel(scene: dict, threshold: float, show_gatti: bool):
    """Three panels: GT  ·  our binary mask @ threshold  ·  Gatti binary mask."""
    vh_norm   = _vh_to_norm(scene["vh"])
    gt_mask   = scene["gt"] > 0.5
    pred_ours = _binarize(scene["prob_ours"], threshold)

    panels = [
        ("Ground truth",                              render_panel(vh_norm, gt_mask,    color_hex=SAND)),
        ("Our model  @ threshold",                    render_panel(vh_norm, pred_ours,  color_hex=RUST)),
    ]
    if show_gatti:
        pred_gatti = _binarize(scene["prob_gatti"], threshold)
        panels.append(
            ("Swin-UNet (retrain)  @ threshold",
             render_panel(vh_norm, pred_gatti, color_hex=FOREST))
        )

    cols = st.columns(len(panels))
    for col, (title, img) in zip(cols, panels):
        col.image(img, caption=title, use_container_width=True)


# ─── Probability distribution diagnostic ──────────────────────────────

@st.cache_data(show_spinner=False)
def _hist_counts(prob: np.ndarray, bins: int = 60):
    """Cached histogram counts so the chart isn't recomputed every rerun."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(prob.ravel(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def _draw_histogram(centers, counts, threshold, opt_thr, color_hex, label):
    """Build a small PIL histogram with rust threshold line + dashed F1-opt."""
    from PIL import ImageDraw, ImageFont

    W, H = 520, 180
    margin_l, margin_r, margin_t, margin_b = 50, 14, 22, 26
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    img = Image.new("RGB", (W, H), (0xF7, 0xF2, 0xE8))
    draw = ImageDraw.Draw(img)

    # Log scale (so the near-zero spike doesn't crush the tails)
    log_counts = np.log10(np.maximum(counts.astype(float), 1.0))
    max_log = log_counts.max() if log_counts.max() > 0 else 1.0

    # Bars
    bar_color = _hex_to_rgb(color_hex)
    n = len(centers)
    bar_w = plot_w / n
    for i, h in enumerate(log_counts):
        x0 = margin_l + i * bar_w
        bar_h = (h / max_log) * plot_h
        y0 = margin_t + plot_h - bar_h
        draw.rectangle([x0, y0, x0 + bar_w * 0.92, margin_t + plot_h],
                       fill=bar_color)

    # Axis baseline
    draw.line([margin_l, margin_t + plot_h,
               W - margin_r, margin_t + plot_h], fill=_hex_to_rgb(MUTED), width=1)

    # F1-optimal threshold (dashed, muted)
    opt_x = margin_l + opt_thr * plot_w
    for y in range(margin_t, margin_t + plot_h, 6):
        draw.line([opt_x, y, opt_x, y + 3],
                  fill=_hex_to_rgb(MUTED), width=2)

    # User's chosen threshold (solid rust)
    thr_x = margin_l + threshold * plot_w
    draw.line([thr_x, margin_t, thr_x, margin_t + plot_h],
              fill=_hex_to_rgb(RUST), width=3)

    # Axis labels (probability 0..1)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 11)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_bold = font
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = margin_l + v * plot_w
        draw.text((x - 8, H - margin_b + 4), f"{v:.2f}",
                  fill=_hex_to_rgb(MUTED), font=font)
    draw.text((margin_l, 4), label, fill=_hex_to_rgb(INK), font=font_bold)
    draw.text((W - margin_r - 80, 4), f"F1-opt @ {opt_thr:.3f}",
              fill=_hex_to_rgb(MUTED), font=font)
    return img


def probability_histogram(scene: dict, threshold: float, show_gatti: bool):
    """Side-by-side probability histograms with threshold markers."""
    st.markdown("#### Where the probabilities live")
    st.caption(
        "Log-scale histograms of every pixel's predicted probability.  "
        f"Solid line = your chosen threshold ({threshold:.3f}).  Dashed = "
        "each model's F1-optimal threshold."
    )

    centers_o, counts_o = _hist_counts(scene["prob_ours"])
    img_o = _draw_histogram(
        centers_o, counts_o, threshold,
        scene["meta"]["thr_ours"], RUST, "OUR MODEL",
    )

    if show_gatti:
        centers_g, counts_g = _hist_counts(scene["prob_gatti"])
        img_g = _draw_histogram(
            centers_g, counts_g, threshold,
            scene["meta"]["thr_gatti"], FOREST, "SWIN-UNET (RETRAIN)",
        )
        cols = st.columns(2)
        cols[0].image(img_o, use_container_width=True)
        cols[1].image(img_g, use_container_width=True)
        st.caption(
            "Notice how Gatti's mass is mostly < 0.05 — saturated near zero.  "
            "Drag the threshold across 0.1–0.6 and very few of his pixels "
            "actually flip class.  That's the calibration asymmetry: same F1, "
            "very different confidence distributions."
        )
    else:
        st.image(img_o, use_container_width=False)


# ─── Live metrics row ─────────────────────────────────────────────────

def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt   = (gt > 0.5).astype(bool)
    tp   = np.logical_and(pred, gt).sum()
    fp   = np.logical_and(pred, ~gt).sum()
    fn   = np.logical_and(~pred, gt).sum()
    eps  = 1e-8
    p    = tp / (tp + fp + eps)
    r    = tp / (tp + fn + eps)
    f1   = 2 * p * r / (p + r + eps)
    iou  = tp / (tp + fp + fn + eps)
    return {"f1": float(f1), "precision": float(p),
            "recall": float(r), "iou": float(iou)}


def metrics_row(scene: dict, threshold: float, show_gatti: bool):
    pred_ours = _binarize(scene["prob_ours"], threshold)
    m_ours    = _pixel_metrics(pred_ours, scene["gt"])

    st.markdown(f"#### Pixel-level metrics  ·  threshold = {threshold:.3f}")

    cols = st.columns(4)
    cols[0].metric("F1",        f"{m_ours['f1']:.3f}")
    cols[1].metric("Precision", f"{m_ours['precision']:.3f}")
    cols[2].metric("Recall",    f"{m_ours['recall']:.3f}")
    cols[3].metric("IoU",       f"{m_ours['iou']:.3f}")

    if show_gatti:
        pred_gatti = _binarize(scene["prob_gatti"], threshold)
        m_gatti    = _pixel_metrics(pred_gatti, scene["gt"])
        st.caption(
            f"Swin-UNet at the same threshold:  "
            f"F1={m_gatti['f1']:.3f}  ·  P={m_gatti['precision']:.3f}  ·  "
            f"R={m_gatti['recall']:.3f}  ·  IoU={m_gatti['iou']:.3f}"
        )

    # Instance-level live readout — the core methodological story
    gt_lab, _ = index_deposits(
        scene["gt"], scene["prob_ours"], scene["prob_gatti"],
    )
    im_o = instance_metrics(gt_lab, pred_ours, iou_thr=0.3)

    st.markdown("#### Instance-level F1  ·  IoU > 0.3, greedy 1-1 matching")

    def _instance_card(im: dict, label: str, accent: str) -> str:
        return (
            f"<div style='background:#F7F2E8; border-left:4px solid {accent}; "
            f"padding:12px 16px; border-radius:2px;'>"
            f"<div style='font-family:Futura,sans-serif; font-size:0.78rem; "
            f"letter-spacing:0.10em; color:{accent};'>{label}</div>"
            f"<div style='font-size:2.0rem; font-weight:bold; color:{INK}; "
            f"font-family:Futura,sans-serif;'>"
            f"{im['f1']:.3f}</div>"
            f"<div style='color:{MUTED}; font-size:0.88rem; margin-top:2px;'>"
            f"<b style='color:{INK}'>{im['tp']}</b> deposits found  ·  "
            f"<b style='color:{INK}'>{im['fp']}</b> false positives  ·  "
            f"<b style='color:{INK}'>{im['fn']}</b> missed</div>"
            f"</div>"
        )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(_instance_card(im_o, "OUR MODEL", RUST),
                    unsafe_allow_html=True)
    if show_gatti:
        im_g = instance_metrics(
            gt_lab, _binarize(scene["prob_gatti"], threshold), iou_thr=0.3,
        )
        with cols[1]:
            st.markdown(_instance_card(im_g, "SWIN-UNET", FOREST),
                        unsafe_allow_html=True)

    thr_o = scene["meta"]["thr_ours"]
    thr_g = scene["meta"]["thr_gatti"]
    st.caption(
        f"F1-optimal thresholds — ours **{thr_o}**, Swin-UNet **{thr_g}**.  "
        f"Both peak at ≈ 0.79 pixel F1 but at opposite ends of the threshold "
        f"range. This is the calibration asymmetry discussed in the README."
    )


# ─── Scene meta card ──────────────────────────────────────────────────

def scene_meta(scene: dict):
    m = scene["meta"]
    cols = st.columns(5)
    cols[0].markdown("**Polygons**");   cols[0].markdown(f"{m['n_polygons']}")
    for col, (k, v) in zip(cols[1:], m["d_breakdown"].items()):
        col.markdown(f"**{k}**")
        col.markdown(f"{v}")


# ─── Change signal (log-ratio of post / pre VH) ───────────────────────

def _render_diverging(vh_norm: np.ndarray, signal: np.ndarray,
                      vmax: float = 4.0,
                      neg_color: str = "#2D3142",   # ink / dark slate
                      zero_color: str = "#EFE8DB",  # cream
                      pos_color: str = "#C1440E",   # rust
                      alpha_max: int = 215) -> Image.Image:
    """VH grayscale + diverging colormap for a signed change signal."""
    h, w = vh_norm.shape
    base = (vh_norm * 255).astype(np.uint8)
    rgb_base = np.stack([base, base, base], axis=-1).astype(np.uint8)

    neg  = np.array(_hex_to_rgb(neg_color),  dtype=np.float32)
    zero = np.array(_hex_to_rgb(zero_color), dtype=np.float32)
    pos  = np.array(_hex_to_rgb(pos_color),  dtype=np.float32)

    # Normalise to [-1, 1]
    s = np.clip(signal / vmax, -1.0, 1.0)

    overlay_rgb = np.zeros((h, w, 3), dtype=np.float32)
    neg_pix = s < 0
    t_neg = (-s[neg_pix])[..., None]
    overlay_rgb[neg_pix] = zero * (1 - t_neg) + neg * t_neg
    pos_pix = ~neg_pix
    t_pos = (s[pos_pix])[..., None]
    overlay_rgb[pos_pix] = zero * (1 - t_pos) + pos * t_pos

    # Alpha based on magnitude — near-zero stays transparent
    alpha = (np.abs(s) * alpha_max).astype(np.uint8)
    overlay = np.concatenate(
        [overlay_rgb.astype(np.uint8), alpha[..., None]], axis=-1
    )

    img = Image.fromarray(rgb_base, mode="RGB").convert("RGBA")
    img = Image.alpha_composite(img, Image.fromarray(overlay, mode="RGBA"))
    return img.convert("RGB")


def change_signal_section(scene: dict) -> bool:
    """Show post-pre VH log-ratio if pre-event VH is bundled.
    Returns True if the section rendered, False if it silently no-op'd."""
    from app.scenes import DATA  # local import to avoid circular

    pre_path = DATA / "vh_pre_tromso.npy"
    if not pre_path.exists():
        return False

    vh_post = scene["vh"]
    vh_pre  = np.load(pre_path)
    if vh_pre.shape != vh_post.shape:
        return False

    st.markdown("### What the model actually sees: the SAR change signal")
    st.caption(
        "Sentinel-1 VH change between pre-event and post-event acquisitions.  "
        "Rust = scene got brighter (rough avalanche debris reflects more); "
        "slate = scene got darker.  This — not the absolute backscatter — is "
        "the primary input the model latches onto."
    )

    log_ratio = vh_post - vh_pre   # dB → dB difference == log ratio
    vh_norm   = _vh_to_norm(vh_post)
    img = _render_diverging(vh_norm, log_ratio, vmax=4.0)
    st.image(img, caption="post − pre  (dB)",
             use_container_width=True)
    return True


# ─── Interactive deposit inspector ────────────────────────────────────

def _draw_outline(img: Image.Image, bbox: tuple[int, int, int, int],
                  color: str = RUST, width: int = 4) -> Image.Image:
    """Draw a rust-colored bounding box on a copy of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    y0, y1, x0, x1 = bbox
    draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4],
                   outline=color, width=width)
    return out


def _detection_status(our_iou: float, gatti_iou: float, iou_thr: float = 0.3):
    """Plain-language match status at IoU > 0.3 (the README convention)."""
    ours = our_iou >= iou_thr
    gat  = gatti_iou >= iou_thr
    if ours and gat:
        return ("Detected by both",          FOREST)
    if ours and not gat:
        return ("Found only by our model",   RUST)
    if gat and not ours:
        return ("Found only by Swin-UNet",   SAND)
    return ("Missed by both",                MUTED)


def _deposit_card(deposit, our_iou, gatti_iou, threshold):
    status, status_color = _detection_status(our_iou, gatti_iou)
    st.markdown(
        f"<div style='background:#F7F2E8; padding:14px 18px; "
        f"border-left:4px solid {status_color}; border-radius:2px;'>"
        f"<div style='color:{MUTED}; font-size:0.85rem; "
        f"font-family:Futura,sans-serif; letter-spacing:0.06em;'>"
        f"DEPOSIT  #{deposit['id']:02d}</div>"
        f"<div style='font-size:1.4rem; color:{INK}; "
        f"font-weight:bold; margin-bottom:4px;'>{status}</div>"
        f"<div style='color:{MUTED}; font-size:0.95rem;'>"
        f"area: <b style='color:{INK}'>{deposit['area_m2']:,} m²</b>  ·  "
        f"size: {deposit['size_px']} px"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Our model**")
        st.metric("IoU @ chosen thr",   f"{our_iou:.2f}")
        st.metric("max probability",    f"{deposit['our_max_prob']:.2f}")
        st.metric("mean probability",   f"{deposit['our_mean_prob']:.2f}")
    with cols[1]:
        st.markdown("**Swin-UNet**")
        st.metric("IoU @ chosen thr",   f"{gatti_iou:.2f}")
        st.metric("max probability",    f"{deposit['gatti_max_prob']:.2f}")
        st.metric("mean probability",   f"{deposit['gatti_mean_prob']:.2f}")


def interactive_inspector(scene: dict, threshold: float):
    """Click-to-inspect: large GT-overlaid VH image; click a deposit
    → side-by-side info card with per-deposit stats and IoU."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.warning(
            "streamlit-image-coordinates is not installed.  Run "
            "`pip install streamlit-image-coordinates` to enable the "
            "click-to-inspect feature."
        )
        return

    st.markdown("### Click a deposit to inspect it")
    st.caption(
        "The colored regions below are the ground-truth deposits.  "
        "Click any one to see its size, our model's confidence, and how "
        "the two models agree at the threshold you've chosen."
    )

    # Index deposits once (cached by Streamlit)
    lab, deposits = index_deposits(
        scene["gt"], scene["prob_ours"], scene["prob_gatti"],
    )

    # Build the clickable image: VH + GT overlay
    vh_norm = _vh_to_norm(scene["vh"])
    gt_mask = scene["gt"] > 0.5
    base_img = render_panel(vh_norm, gt_mask, color_hex=SAND, alpha=170)

    # First-time visitors land on a pre-selected showcase deposit (case 28 —
    # our model hits, Gatti barely touches it, the head-to-head from the deck).
    if "selected_deposit" not in st.session_state:
        showcase_id = 28
        if any(d["id"] == showcase_id for d in deposits):
            st.session_state["selected_deposit"] = showcase_id

    # Highlight the previously clicked deposit, if any
    selected = st.session_state.get("selected_deposit")
    selected_dep = None
    if selected is not None:
        selected_dep = next(
            (d for d in deposits if d["id"] == selected), None
        )
    if selected_dep is not None:
        base_img = _draw_outline(base_img, selected_dep["bbox"],
                                 color=RUST, width=5)

    layout_cols = st.columns([3, 2])
    with layout_cols[0]:
        # Constrain display size for the clickable widget
        H, W = scene["gt"].shape
        DISPLAY_W = 720
        scale = W / DISPLAY_W
        result = streamlit_image_coordinates(
            base_img, key="deposit_clicker", width=DISPLAY_W,
        )
        if result is not None:
            cx = int(result["x"] * scale)
            cy = int(result["y"] * scale)
            if 0 <= cy < H and 0 <= cx < W:
                clicked_id = int(lab[cy, cx])
                if clicked_id > 0:
                    if st.session_state.get("selected_deposit") != clicked_id:
                        st.session_state["selected_deposit"] = clicked_id
                        st.rerun()

    with layout_cols[1]:
        if selected_dep is None:
            st.info(
                "👈  Click a colored region in the image to inspect a "
                "deposit.  The card here will fill in with size, model "
                "confidences, and per-deposit IoU."
            )
            return

        # Live IoU at user's current threshold
        pred_ours_lab, _  = ndimage.label(
            _binarize(scene["prob_ours"], threshold)
        )
        pred_gatti_lab, _ = ndimage.label(
            _binarize(scene["prob_gatti"], threshold)
        )
        gt_this = (lab == selected_dep["id"])
        our_iou   = best_iou_against(gt_this, pred_ours_lab)
        gatti_iou = best_iou_against(gt_this, pred_gatti_lab)

        _deposit_card(selected_dep, our_iou, gatti_iou, threshold)
