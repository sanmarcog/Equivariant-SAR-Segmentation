"""Build presentation.pptx from project content.

Regenerates the deck: `python scripts/build_slides.py`
Output: presentation.pptx at repo root.
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

NAVY = RGBColor(0x1A, 0x36, 0x5D)
BLUE = RGBColor(0x2C, 0x55, 0x82)
GREEN = RGBColor(0x2F, 0x85, 0x5A)
RED = RGBColor(0xC5, 0x30, 0x30)
GREY = RGBColor(0x4A, 0x55, 0x68)
LIGHT = RGBColor(0xED, 0xF2, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _set_slide_size(prs: Presentation, width_in=13.333, height_in=7.5):
    prs.slide_width = Inches(width_in)
    prs.slide_height = Inches(height_in)


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rect(slide, left, top, width, height, fill=None, line_color=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.shadow.inherit = False
    if fill is not None:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
    return shp


def _textbox(slide, left, top, width, height, text, *, size=18, bold=False,
             color=None, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color
    return tb


def _mono(slide, left, top, width, height, text, *, size=14, color=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line if line else " "
        run.font.name = "Menlo"
        run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    return tb


def _header(slide, title, *, subtitle=None, bar_color=BLUE, slide_w=Inches(13.333)):
    _rect(slide, Emu(0), Emu(0), slide_w, Inches(0.08), fill=bar_color)
    _textbox(slide, Inches(0.5), Inches(0.25), slide_w - Inches(1.0), Inches(0.7),
             title, size=32, bold=True, color=NAVY)
    if subtitle:
        _textbox(slide, Inches(0.5), Inches(0.95), slide_w - Inches(1.0), Inches(0.5),
                 subtitle, size=16, color=GREY)


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def _footer(slide, page_no, total, slide_w=Inches(13.333), slide_h=Inches(7.5)):
    _textbox(slide, Inches(0.5), slide_h - Inches(0.4), Inches(6), Inches(0.3),
             "D4-Equivariant SAR Avalanche Segmentation", size=10, color=GREY)
    _textbox(slide, slide_w - Inches(1.5), slide_h - Inches(0.4),
             Inches(1), Inches(0.3), f"{page_no} / {total}",
             size=10, color=GREY, align=PP_ALIGN.RIGHT)


def _bullets(slide, left, top, width, height, items, *, size=18, color=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = f"•  {item}"
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color


def build():
    prs = Presentation()
    _set_slide_size(prs)
    SW, SH = Inches(13.333), Inches(7.5)

    slides = []

    # ─────────────────────── SLIDE 1 — TITLE ─────────────────────────
    s = _blank(prs); slides.append(s)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, Inches(0.5), Inches(2.5), Inches(0.15), Inches(2.5), fill=RGBColor(0xED, 0xC0, 0x5E))
    _textbox(s, Inches(1.0), Inches(2.4), Inches(11), Inches(1.2),
             "D4-Equivariant CNN", size=60, bold=True, color=WHITE)
    _textbox(s, Inches(1.0), Inches(3.3), Inches(11), Inches(1.2),
             "for SAR Avalanche Segmentation", size=44, bold=True, color=WHITE)
    _textbox(s, Inches(1.0), Inches(4.6), Inches(11), Inches(0.6),
             "100× fewer parameters.  Better instance detection.",
             size=24, color=RGBColor(0xED, 0xC0, 0x5E))
    _textbox(s, Inches(1.0), Inches(6.2), Inches(11), Inches(0.4),
             "Guillermo San Marco  ·  April 2026",
             size=16, color=WHITE)
    _notes(s, """
We built a tiny D4-equivariant CNN for avalanche deposit segmentation in bi-temporal
Sentinel-1 SAR. Headline: we tie the published Gatti 2024 Swin-UNet on pixel F1
using about 1% of the parameters, and we show that the standard pixel metric hides a
10.7-point gap on instance-level F1. I'll walk through the architecture, the
experimental journey, the negative results, and the metric finding. ~15 minutes.
""")

    # ─────────────────────── SLIDE 2 — THE PROBLEM ───────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "The problem",
            subtitle="Sentinel-1 bi-temporal SAR · tiny, arbitrarily-oriented deposits")
    _bullets(s, Inches(0.5), Inches(1.7), Inches(7), Inches(5), [
        "Avalanche deposits are small — often <1% of scene pixels",
        "Only visible as a difference between pre-event / post-event acquisitions",
        "Orientation w.r.t. the radar frame has no physical meaning",
        "SAR is noisy — speckle, incidence-angle effects, terrain shadowing",
        "Public baseline: Gatti 2024 Swin-UNet, pixel F1 ≈ 0.79",
    ], size=20)
    s.shapes.add_picture(str(FIG / "pair16_overlay.png"),
                         Inches(7.8), Inches(1.8), height=Inches(5.2))
    _notes(s, """
SAR is noisy and deposits are tiny. The only reliable signal is the difference
between a pre-event and post-event scene. Critically, a deposit's orientation w.r.t.
the radar frame has no physical meaning — the satellite flies a specific track but
the avalanche doesn't know that. Gatti et al. 2024 published a Swin-UNet baseline
at pixel F1 around 0.79 — that's our reference point. The image on the right is a
representative scene pair with the deposit mask overlaid.
""")

    # ─────────────────────── SLIDE 3 — THE RESEARCH QUESTION ─────────
    s = _blank(prs); slides.append(s)
    _header(s, "The research question")
    _rect(s, Inches(0.5), Inches(1.7), Inches(6), Inches(5), fill=LIGHT)
    _textbox(s, Inches(0.8), Inches(1.9), Inches(5.5), Inches(0.5),
             "Published baseline", size=22, bold=True, color=GREY)
    _textbox(s, Inches(0.8), Inches(2.4), Inches(5.5), Inches(0.5),
             "Swin-UNet  (Gatti 2024)", size=26, bold=True, color=NAVY)
    _bullets(s, Inches(0.8), Inches(3.2), Inches(5.5), Inches(3), [
        "~60 M parameters",
        "Learns rotation invariance from augmentation",
        "Test pixel F1 ≈ 0.79",
    ], size=18)

    _rect(s, Inches(6.8), Inches(1.7), Inches(6), Inches(5), fill=RGBColor(0xE6, 0xF4, 0xEA))
    _textbox(s, Inches(7.1), Inches(1.9), Inches(5.5), Inches(0.5),
             "Our proposal", size=22, bold=True, color=GREEN)
    _textbox(s, Inches(7.1), Inches(2.4), Inches(5.5), Inches(0.5),
             "D4-equivariant CNN", size=26, bold=True, color=NAVY)
    _bullets(s, Inches(7.1), Inches(3.2), Inches(5.5), Inches(3), [
        "Rotation / reflection symmetry baked into the architecture",
        "How small can we go?",
        "8 symmetries: {id, r90, r180, r270, 4× flips}",
    ], size=18)
    _notes(s, """
The Swin-UNet has 60 million parameters and learns rotation/reflection invariance
implicitly through augmentation. Our question: if that invariance is a known physical
property of the problem, what happens when we bake it into the architecture instead
of learning it? We use D4 — the dihedral group of order 8, the eight symmetries of
the image square.
""")

    # ─────────────────────── SLIDE 4 — EQUIVARIANCE DIAGRAM ──────────
    s = _blank(prs); slides.append(s)
    _header(s, "What D4 equivariance buys us",
            subtitle="Encoder exactly equivariant  ·  full model approximately equivariant")
    diagram = """          Input x                       g · x  (rotated/flipped)
             │                              │
             ▼                              ▼
       ┌─────────┐                   ┌─────────┐
       │ Encoder │                   │ Encoder │
       └─────────┘                   └─────────┘
             │                              │
             ▼                              ▼
         Encoder(x)    ── g · ─▶     Encoder(g·x)      ← EXACT
             │                              │
             ▼                              ▼
        GroupPool                       GroupPool
             │                              │
             ▼                              ▼
       invariant(x)   ─────── = ─▶    invariant(g·x)   ← EXACT
             │                              │
             ▼                              ▼
       Conv2d Decoder                  Conv2d Decoder  ← approximate
             │                              │
             ▼                              ▼
           logit(x)   ≈  g⁻¹ ·         logit(g·x)      ← approximate"""
    _mono(s, Inches(0.7), Inches(1.7), Inches(12), Inches(5), diagram, size=14, color=NAVY)
    _notes(s, """
Equivariance means: rotate the input, the output rotates the same way — by construction.
Our encoder is exactly equivariant thanks to escnn — verified in CI on every push.
After GroupPooling we have an invariant representation. The decoder is standard
Conv2d, so the full model is only approximately equivariant. This is a deliberate
trade-off: the decoder keeps the parameter count tiny and allows skip connections.
""")

    # ─────────────────────── SLIDE 5 — ARCHITECTURE ─────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Architecture")
    arch = """ 12-channel bi-temporal SAR input   (B, 12, H, W)
 ├── post  = [VH, VV, log-ratio, RGD, LIA, ...]   idx 0–5
 ├── pre   = [VH, VV, <shared derived 2–5>]       idx 6,7,2–5
 └── extra = [DEM, slope, ...]                    idx 8–11
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  D4-equivariant encoder   (escnn, N=4, with flips)      │
 │  shared weights across pre / post branches              │
 │                                                         │
 │     post ─► enc(post)        pre ─► enc(pre)            │
 │                │                      │                 │
 │                └────── difference ────┘                 │
 │                         │                               │
 │                  GroupPooling ─► invariant tensor       │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
    Conv2d decoder  (4 stages, skip from invariant trunk)
        │
        ├── seg  head ─► logit     (B, 1, H, W)
        └── area head ─► area_m²   (B, 1)"""
    _mono(s, Inches(0.6), Inches(1.5), Inches(10), Inches(5), arch, size=12, color=NAVY)
    _rect(s, Inches(10.8), Inches(2.8), Inches(2.2), Inches(2.2), fill=NAVY)
    _textbox(s, Inches(10.8), Inches(3.0), Inches(2.2), Inches(0.4),
             "PARAMS", size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(10.8), Inches(3.4), Inches(2.2), Inches(1.0),
             "625,617", size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(10.8), Inches(4.3), Inches(2.2), Inches(0.6),
             "~1% of\nSwin-UNet", size=14, color=WHITE, align=PP_ALIGN.CENTER)
    _notes(s, """
Twelve input channels — pre and post SAR plus derived features: log-ratio, incidence
angle, DEM. The encoder is five equivariant blocks with weights shared across pre
and post branches. We take the difference in the regular representation — still
equivariant — then GroupPool to get an invariant tensor. The decoder is plain Conv2d
with skip connections from the invariant trunk. Total parameter count: 625k — about
1% of Swin-UNet. This number is locked by a CI test called test_parameter_count.
""")

    # ─────────────────────── SLIDE 6 — JOURNEY ──────────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "The experimental journey", subtitle="Every arrow is a SLURM sweep on Hyak")
    journey = """  gatti_mirror.sh       cond1, no aug, no skip          F1 ~0.75
           │
           ▼
  aug_experiment.sh     aug 1× / 3× / 5×                aug3× wins
           │
           ▼
  reg_sweep 1/2/3/3b    depth, wd, focal-tversky, ...   bigger hurts
           │                                            warm restarts hurt
           ▼
  lambda_sweep.sh       component-IoU λ grid            bimodal trade-off
           │
           ▼
  calibration_sweep     T-scale / Platt / isotonic      marginal gain
           │
           ▼
  ★  cond1, no-skip, aug3×, seed 1                      F1 = 0.7938  ★"""
    _mono(s, Inches(0.5), Inches(1.7), Inches(12.5), Inches(5.2), journey, size=15, color=NAVY)
    _notes(s, """
Every arrow here is a SLURM sweep on Hyak. Started by mirroring Gatti's setup —
no augmentation, no skip — F1 around 0.75. Augmentation sweep: 3x rotation/flip is
optimal. Regularization sweeps said: bigger models hurt, warm restarts hurt, heavy
regularization hurts. That's evidence for the inductive-bias story — we don't need
more capacity. Lambda sweep on component-IoU loss showed a bimodal trade-off — next
slide. Final config: condition 1 loss, no-skip variant, aug 3x, seed 1 — pixel
F1 0.7938.
""")

    # ─────────────────────── SLIDE 7 — LAMBDA SWEEP ─────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Negative result: λ sweep trade-off",
            subtitle="Component-IoU loss weight has no single optimum")
    _bullets(s, Inches(0.5), Inches(1.8), Inches(5.5), Inches(4), [
        "Low λ → better pixel F1, fragmented small deposits",
        "High λ → better small-instance recall, pixel F1 drops",
        "No single λ wins both axes",
        "The loss is a dial, not a setting — which end depends on use case",
    ], size=18)
    s.shapes.add_picture(str(FIG / "lambda_sweep.png"),
                         Inches(6.5), Inches(1.7), height=Inches(5.2))
    _notes(s, """
We built a component-IoU variant from Kofler et al.'s blob-loss family. The lambda
sweep showed a bimodal trade-off: low lambda favors pixel F1, high lambda favors
detecting small deposits. No single value dominates. Calling this out because the
literature usually reports a single number; we're saying the loss is a dial, not a
setting — which end you want depends on your downstream use case.
""")

    # ─────────────────────── SLIDE 8 — PUNCHLINE TABLES ─────────────
    s = _blank(prs); slides.append(s)
    _header(s, "The methodological punchline",
            subtitle="Same predictions, same ground truth — different metric, different story")

    # Left card: pixel F1
    _rect(s, Inches(0.7), Inches(1.9), Inches(5.8), Inches(4.5), fill=LIGHT)
    _textbox(s, Inches(0.7), Inches(2.0), Inches(5.8), Inches(0.6),
             "Pixel F1", size=28, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(0.7), Inches(2.8), Inches(5.8), Inches(0.5),
             "ours    0.794", size=26, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(0.7), Inches(3.4), Inches(5.8), Inches(0.5),
             "Gatti   0.791", size=26, color=GREY, align=PP_ALIGN.CENTER)
    _rect(s, Inches(1.5), Inches(4.1), Inches(4.2), Inches(0.05), fill=GREY)
    _textbox(s, Inches(0.7), Inches(4.2), Inches(5.8), Inches(0.6),
             "Δ = +0.003", size=22, color=GREY, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(0.7), Inches(5.2), Inches(5.8), Inches(0.8),
             "a tie.", size=34, bold=True, color=GREY, align=PP_ALIGN.CENTER)

    # Right card: instance F1
    _rect(s, Inches(6.85), Inches(1.9), Inches(5.8), Inches(4.5),
          fill=RGBColor(0xE6, 0xF4, 0xEA))
    _textbox(s, Inches(6.85), Inches(2.0), Inches(5.8), Inches(0.6),
             "Instance F1", size=28, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(6.85), Inches(2.8), Inches(5.8), Inches(0.5),
             "ours    0.620", size=26, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(6.85), Inches(3.4), Inches(5.8), Inches(0.5),
             "Gatti   0.513", size=26, color=GREY, align=PP_ALIGN.CENTER)
    _rect(s, Inches(7.6), Inches(4.1), Inches(4.2), Inches(0.05), fill=GREEN)
    _textbox(s, Inches(6.85), Inches(4.2), Inches(5.8), Inches(0.6),
             "Δ = +0.107", size=22, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(6.85), Inches(5.2), Inches(5.8), Inches(0.8),
             "a real gap.", size=34, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _notes(s, """
Here's the finding. On pixel F1 — the standard metric everyone reports — we tie
Gatti. Three thousandths of a point. That's noise. But on instance F1, with a
center-point matching rule, we're ahead by 10.7 percentage points. Same predictions,
same ground truth, different metric, completely different story. This is the core
methodological contribution of the work.
""")

    # ─────────────────────── SLIDE 9 — METRIC DISAGREEMENT FIG ──────
    s = _blank(prs); slides.append(s)
    _header(s, "Metric disagreement — the picture")
    s.shapes.add_picture(str(FIG / "metric_disagreement.png"),
                         Inches(2.3), Inches(1.5), height=Inches(5.6))
    _notes(s, """
This is the slide to remember. Left bar group: pixel F1 — bars the same height.
Right bar group: instance F1 — clear gap. We're not better at labeling pixels.
We're better at producing one clean prediction per deposit instead of several
fragmented ones.
""")

    # ─────────────────────── SLIDE 10 — D-SCALE BREAKDOWN ───────────
    s = _blank(prs); slides.append(s)
    _header(s, "Where the gap comes from — D-scale breakdown",
            subtitle="EAWS destructive-size classes  ·  larger deposits = operational relevance")

    # Table-ish visualization
    rows = [
        ("D-scale",     "ours",  "Gatti", "Δ",    NAVY),
        ("D1 (small)",  "0.31",  "0.27",  "+0.04", GREY),
        ("D2",          "0.58",  "0.49",  "+0.09", GREY),
        ("D3",          "0.71",  "0.58",  "+0.13", GREEN),
    ]
    y0 = Inches(1.9)
    row_h = Inches(0.95)
    col_x = [Inches(1.5), Inches(5.0), Inches(7.5), Inches(10.0)]
    col_w = Inches(3.0)
    for i, row in enumerate(rows):
        y = y0 + row_h * i
        bold = (i == 0 or i == 3)
        color = row[4]
        size = 24 if i == 0 else (30 if i == 3 else 24)
        if i == 0:
            _rect(s, Inches(1.2), y, Inches(11), row_h - Emu(20000),
                  fill=LIGHT)
        if i == 3:
            _rect(s, Inches(1.2), y, Inches(11), row_h - Emu(20000),
                  fill=RGBColor(0xE6, 0xF4, 0xEA))
        for j, val in enumerate(row[:4]):
            _textbox(s, col_x[j], y + Inches(0.15), col_w, row_h,
                     val, size=size, bold=bold, color=color, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(0.5), Inches(6.0), Inches(12), Inches(1),
             "D3 is where the gap is largest — and D3 is what practitioners act on.",
             size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _notes(s, """
The instance-F1 gap isn't uniform across deposit sizes. Broken down by EAWS D-scale —
the European avalanche standard — the biggest gap is at D3, destructive-size deposits.
13 percentage points. That matters operationally. D3 is what triggers road closures
and rescue dispatch. Our symmetry-aware model produces a single clean D3 deposit
where Gatti's fragments into 3-4 high-confidence blobs.
""")

    # ─────────────────────── SLIDE 11 — FRAGMENTATION ───────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Why: fragmentation", subtitle="Gatti's big deposits split into multiple high-confidence blobs")
    _bullets(s, Inches(0.5), Inches(1.8), Inches(6), Inches(5), [
        "Gatti's predictions fragment large deposits",
        "Pixel F1 doesn't care — pixels still mostly right",
        "Instance F1 penalizes fragmentation once per deposit",
        "Equivariant encoder → smoother orientation-invariant features, fewer spurious local peaks",
    ], size=18)
    s.shapes.add_picture(str(FIG / "failure_modes.png"),
                         Inches(6.8), Inches(1.7), height=Inches(5.2))
    _notes(s, """
Look at the failure-mode panels. Ground truth: one deposit. Gatti's prediction
breaks it into three blobs with gaps. Pixel-wise still mostly right. But counted
as instances, that's false positives plus a missed true positive. Our model produces
one contiguous prediction. The equivariant encoder gives a smoother
orientation-invariant feature map — fewer local spurious peaks.
""")

    # ─────────────────────── SLIDE 12 — PARAM FRONTIER ──────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Parameter efficiency",
            subtitle="Pareto-dominant on  params × F1")
    s.shapes.add_picture(str(FIG / "param_frontier.png"),
                         Inches(2.8), Inches(1.5), height=Inches(5.0))
    _textbox(s, Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.6),
             "0.63 M parameters   ·   two orders of magnitude below Swin-UNet   ·   same pixel F1",
             size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _notes(s, """
Pareto plot. X-axis: parameter count. Y-axis: pixel F1. Our point sits alone in the
bottom-right — same F1, two orders of magnitude fewer parameters. This matters for:
(1) edge / near-real-time deployment, (2) training data efficiency, (3) environmental
cost of the research workflow itself.
""")

    # ─────────────────────── SLIDE 13 — CALIBRATION ─────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Calibration",
            subtitle="Both models asymmetrically miscalibrated  ·  temperature scaling doesn't change F1 ranking")
    _bullets(s, Inches(0.5), Inches(1.9), Inches(6), Inches(4.5), [
        "Over-confident at high probabilities",
        "Under-confident at low probabilities",
        "Temperature scaling helps marginally",
        "F1 ranking invariant to calibration choice",
        "Ship with T-scaling on; document the residual asymmetry",
    ], size=18)
    s.shapes.add_picture(str(FIG / "calibration_histograms.png"),
                         Inches(6.8), Inches(1.7), height=Inches(5.2))
    _notes(s, """
Quick slide on calibration. Both models over-confident at the high end and
under-confident at the low end. Temperature scaling gives a marginal improvement
but doesn't change the F1 ranking. We ship with T-scaling on and document the
residual asymmetry so downstream users know what they're getting.
""")

    # ─────────────────────── SLIDE 14 — NEGATIVE RESULTS ────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Honest negative results")
    items = [
        ("Bigger models hurt",      "confirms inductive-bias story, not scale"),
        ("Heavy regularization hurt", "equivariance already regularizes"),
        ("Warm restarts hurt",      "loss landscape isn't the blocker"),
        ("DPR instance head failed","150 crops vs 30k patches  →  catastrophic forgetting"),
    ]
    y = Inches(1.9)
    for i, (headline, detail) in enumerate(items):
        yy = y + Inches(1.05) * i
        _rect(s, Inches(0.6), yy, Inches(0.25), Inches(0.85), fill=RED)
        _textbox(s, Inches(1.0), yy + Inches(0.05), Inches(5), Inches(0.45),
                 headline, size=22, bold=True, color=NAVY)
        _textbox(s, Inches(1.0), yy + Inches(0.48), Inches(11), Inches(0.4),
                 detail, size=16, color=GREY)
    _notes(s, """
Every one of these cost us a week. Bigger models hurt — evidence for inductive bias
over scale. Heavy regularization hurt — the equivariance already regularizes, don't
pile on. Warm restarts hurt — the loss landscape isn't the blocker. And the DPR
instance-segmentation head never converged: 150 instance crops couldn't compete with
30k pixel patches, and we saw catastrophic forgetting of the pre-trained encoder.
We're calling this out because the community has a bias toward success stories.
""")

    # ─────────────────────── SLIDE 15 — REPRO ───────────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "Reproducibility")
    _bullets(s, Inches(0.5), Inches(1.8), Inches(6), Inches(5), [
        "CI green — 7 tests on every push",
        "  encoder equivariance (escnn check_equivariance)",
        "  GroupPooling invariance  ·  determinism",
        "  shape + parameter-count locked",
        "Final checkpoint + eval JSON committed",
        "norm_stats_12ch.json committed (force-added)",
        "SLURM scripts env-var templated — no hardcoded cluster paths",
    ], size=16)
    repro_tree = """tests/test_equivariance.py      7 tests ✓
data/norm_stats_12ch.json       committed
results_final/
  eval_cond1_seed1_s16_tta.json    F1=0.7938
  prob_p2_tromso.npy               our probs
  prob_gatti_tromso.npy            baseline probs
  gt_tromso.npy                    GT
  vh_tromso.npy                    VH viz
.env.example  ← REPO  DATA_DIR  SIF  BIND_ROOT"""
    _mono(s, Inches(7), Inches(1.8), Inches(6), Inches(5.2), repro_tree, size=12, color=NAVY)
    _notes(s, """
Repro matters. Seven CI tests on every push — encoder equivariance via escnn's own
check, GroupPooling invariance, determinism, shape, parameter count locked. Final
checkpoint and eval JSON both committed. Norm stats file committed — it was in
.gitignore and I force-added it. SLURM scripts use environment variables so nothing
is hardcoded to my Hyak account. Anyone can reproduce this in a fresh clone.
""")

    # ─────────────────────── SLIDE 16 — FILE MAP ────────────────────
    s = _blank(prs); slides.append(s)
    _header(s, "File-by-file map")
    filemap = """src/models/segnet.py        D4SegNet + D4SegNetNoSkip, _eq_block
src/data/                   dataset, preprocess, norm_stats, augment
src/losses.py               bce / focal_tversky / dice / component_iou / area
src/train.py                loop, AMP, val-F1 checkpoint selection
src/inference.py            sliding window, 4 blending modes, 4-fold TTA
src/evaluate.py             bootstrap CIs, permutation test, F1-opt threshold
src/aggregate.py            roll up run JSONs → comparison tables
src/slurm/*.sh              12 parameterised sweep scripts
tests/test_equivariance.py  CI sanity (7 tests)
wiki/                       living lab notebook (goal, arch, eval, log, ...)
figures/                    hero, metric_disagreement, param_frontier, ...
results_final/              final numbers + probs for the Tromsø scene"""
    _mono(s, Inches(0.7), Inches(1.7), Inches(12), Inches(5.5), filemap,
          size=15, color=NAVY)
    _notes(s, """
Quick tour. Model lives in segnet.py. Data handling has its own subpackage.
Losses are modular — pick one with a flag. train.py does the loop with AMP and
F1-based checkpointing. inference.py handles sliding-window with four blending
modes and 4-fold TTA. evaluate.py does bootstrap confidence intervals and
permutation tests against the baseline. SLURM scripts are env-var templated.
The wiki is a living lab notebook — if you want to know why we made a decision,
it's there, dated.
""")

    # ─────────────────────── SLIDE 17 — TAKE-HOMES ──────────────────
    s = _blank(prs); slides.append(s)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _textbox(s, Inches(1), Inches(0.6), Inches(11), Inches(0.8),
             "Take-homes", size=44, bold=True, color=WHITE)

    items = [
        ("1", "Symmetry-aware inductive bias beats scale",
              "at 1% the parameters, same pixel F1, better instance F1"),
        ("2", "Pixel F1 hides fragmentation",
              "the metric you report is a scientific choice, not a formatting choice"),
        ("3", "Don't ship instance heads without instance data",
              "DPR failed on 150 crops — data scarcity beats architectural cleverness"),
    ]
    y = Inches(1.9)
    for i, (num, head, sub) in enumerate(items):
        yy = y + Inches(1.6) * i
        _rect(s, Inches(1), yy, Inches(1.2), Inches(1.3),
              fill=RGBColor(0xED, 0xC0, 0x5E))
        _textbox(s, Inches(1), yy, Inches(1.2), Inches(1.3),
                 num, size=60, bold=True, color=NAVY,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        _textbox(s, Inches(2.5), yy + Inches(0.1), Inches(10), Inches(0.6),
                 head, size=26, bold=True, color=WHITE)
        _textbox(s, Inches(2.5), yy + Inches(0.75), Inches(10), Inches(0.5),
                 sub, size=16, color=RGBColor(0xED, 0xC0, 0x5E))
    _notes(s, """
Three take-homes. One: for a problem with known symmetry, encoding it beats learning
it — at a hundredth of the parameters. Two: the metric you report is a scientific
choice, not a formatting choice. Pixel F1 was structurally blind to the fragmentation
gap. Three: the DPR negative result says data scarcity beats architectural cleverness
every time. If you're about to add an instance head, count your instance labels first.
""")

    # ─────────────────────── SLIDE 18 — THANKS ──────────────────────
    s = _blank(prs); slides.append(s)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _textbox(s, Inches(1), Inches(2.5), Inches(11), Inches(1.5),
             "Thanks.", size=80, bold=True, color=WHITE)
    _textbox(s, Inches(1), Inches(3.8), Inches(11), Inches(0.8),
             "Questions?", size=40, color=RGBColor(0xED, 0xC0, 0x5E))
    _textbox(s, Inches(1), Inches(5.3), Inches(11), Inches(0.4),
             "github.com/gsanmarco/Equivariant-SAR-Segmentation",
             size=18, color=WHITE)
    _textbox(s, Inches(1), Inches(5.8), Inches(11), Inches(0.4),
             "CI green · Checkpoint released · Figures in /figures · Wiki in /wiki",
             size=14, color=RGBColor(0xA0, 0xAE, 0xC0))
    _notes(s, """
Happy to go deeper on: escnn group representation details, the instance-matching
rule (center-point with IoU tie-break), copy-paste augmentation ablation, or the
DPR catastrophic-forgetting analysis. All numbers are bootstrap CIs plus permutation
tested — nothing is cherry-picked.
""")

    # ── page footers ────────────────────────────────────────────────
    total = len(slides)
    # Skip footers on first and last slides (full-bleed)
    for i, sl in enumerate(slides):
        if i == 0 or i == total - 1:
            continue
        _footer(sl, i + 1, total)

    out = ROOT / "presentation.pptx"
    prs.save(str(out))
    print(f"Saved: {out}  ({total} slides)")


if __name__ == "__main__":
    build()
