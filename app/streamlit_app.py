"""Main Streamlit entry — sidebar controls + main panel of results."""
from __future__ import annotations

import streamlit as st

from app.components import (
    change_signal_section, four_panel, hero_card, interactive_inspector,
    metrics_row, model_card, probability_histogram, scene_meta,
)
from app.scenes import list_scenes, load_scene

# ── Patagonia palette (cream / ink / rust) ──────────────────────────
CREAM  = "#EFE8DB"
INK    = "#2D3142"
RUST   = "#C1440E"
MUTED  = "#8B8374"


def main() -> None:
    st.set_page_config(
        page_title="D4-Equivariant SAR Avalanche Segmentation",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS for cream background, ink type, rust accents ──
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {CREAM};
        }}
        [data-testid="stSidebar"] {{
            background-color: #F7F2E8;
        }}
        h1, h2, h3, h4 {{ color: {INK}; font-family: 'Futura', 'Avenir Next', sans-serif; }}
        p, .stMarkdown {{ color: {INK}; }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
        [data-testid="stMetricValue"] {{ color: {INK}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        model_card()

        st.markdown("---")
        st.markdown("### Settings")

        scene_name = st.selectbox("Scene", list_scenes(), index=0)
        threshold  = st.slider(
            "Decision threshold", 0.0, 1.0, 0.225, 0.025,
            help="Pixels above this probability are predicted as deposit.",
        )
        st.caption(
            "Try **0.05** (sees everything) → **0.225** (our optimum) → "
            "**0.775** (Gatti's optimum). Watch the binary masks change."
        )
        show_gatti = st.checkbox(
            "Compare with Swin-UNet baseline", value=True,
        )

    # ── Main ──────────────────────────────────────────────────────
    st.title("Detecting avalanches in SAR")
    st.markdown(
        f"<p style='color:{MUTED}; font-size:1.05rem'>"
        f"A small D4-equivariant CNN finds avalanche debris in bi-temporal "
        f"Sentinel-1 SAR.  ~4× fewer parameters than Swin-UNet, +10.7 pp on "
        f"instance-level F1.  Move the threshold and watch what changes."
        f"</p>",
        unsafe_allow_html=True,
    )

    # Hero card — the headline finding, before any interaction
    hero_card()

    scene = load_scene(scene_name)

    # 3-panel grid
    four_panel(scene, threshold, show_gatti)

    # Live pixel metrics at chosen threshold
    metrics_row(scene, threshold, show_gatti)

    # Probability distribution diagnostic — explains calibration asymmetry
    probability_histogram(scene, threshold, show_gatti)

    st.markdown("---")

    # Interactive deposit inspector (click a deposit → per-deposit info card)
    interactive_inspector(scene, threshold)

    # Change signal — renders only if pre-event VH is bundled.
    # Silently no-op when the file is missing.
    change_signal_section(scene)

    st.markdown("---")

    # Scene metadata
    with st.expander("Scene composition  (EAWS D-scale breakdown)"):
        scene_meta(scene)
        st.caption(
            "D1 small (knocks down a person)  ·  D2 medium (can bury)  ·  "
            "D3 large (can destroy a car)  ·  D4 very large (can destroy a building)"
        )

    # About
    with st.expander("What is this?"):
        st.markdown(
            """
**D4-equivariant CNN for SAR avalanche debris segmentation** — a research
artefact, not an operational tool.

The model is a 5-block D4-equivariant encoder built with [escnn](
https://github.com/QUVA-Lab/escnn), a Conv2d decoder, and an auxiliary
deposit-area regressor. Trained on the public AvalCD benchmark
([Gatti et al. 2026](https://github.com/mattiagatti/avalanche-deep-change-detection))
on 5 scenes; this Tromsø scene is held out as the OOD test set.

**The methodological finding.** Pixel F1 says we tie the published Swin-UNet
baseline (0.794 vs 0.795 on retrain). Instance F1 — same predictions, scored
by which deposits got found — says we're ahead by 10.7 percentage points on
the deposits practitioners actually act on (D3 size, +13 pp).

**Limitations of this demo.** One held-out scene. Predictions are
pre-computed and committed under `results_final/`; this app interactively
re-thresholds and post-processes them. A v2 with live model inference on a
fresh preprocessed patch is in scope but not yet built.
            """
        )

    st.markdown("---")
    st.caption(
        "Research preview — not for safety-critical decisions.  "
        f"Source: github.com/sanmarcog/Equivariant-SAR-Segmentation"
    )


if __name__ == "__main__":
    main()
