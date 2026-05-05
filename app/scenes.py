"""Cached SAR scenes available in the demo.

For v1 we ship one scene — the Tromsø out-of-distribution test scene used
for the headline numbers in the README. Probability maps for both our model
and the retrained Swin-UNet baseline are pre-computed and committed under
results_final/.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "results_final"


class SceneArrays(TypedDict):
    vh: np.ndarray         # (H, W)  VH backscatter, dB
    gt: np.ndarray         # (H, W)  binary ground truth
    prob_ours: np.ndarray  # (H, W)  our D4-EquiCNN probability map
    prob_gatti: np.ndarray # (H, W)  Swin-UNet (retrained) probability map
    meta: dict


SCENE_CATALOG = {
    "Tromsø  ·  2024-12-20  (test, OOD)": {
        "vh":          "vh_tromso.npy",
        "gt":          "gt_tromso.npy",
        "prob_ours":   "prob_p2_tromso.npy",
        "prob_gatti":  "prob_gatti_tromso.npy",
        "n_polygons":  117,
        "d_breakdown": {"D1": 5, "D2": 25, "D3": 72, "D4": 16},
        "thr_ours":    0.225,
        "thr_gatti":   0.775,
    },
}


def list_scenes() -> list[str]:
    return list(SCENE_CATALOG.keys())


@st.cache_data(show_spinner="Loading scene...")
def load_scene(name: str) -> SceneArrays:
    """Memory-mapped load of the four arrays for a scene."""
    spec = SCENE_CATALOG[name]
    return {
        "vh":         np.load(DATA / spec["vh"]),
        "gt":         np.load(DATA / spec["gt"]),
        "prob_ours":  np.load(DATA / spec["prob_ours"]),
        "prob_gatti": np.load(DATA / spec["prob_gatti"]),
        "meta":       spec,
    }
