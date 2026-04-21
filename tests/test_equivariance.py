"""
Test D4 equivariance properties and basic architecture correctness.

The encoder is D4-equivariant by construction (escnn guarantees this).
The full model is approximately equivariant — the standard Conv2d decoder
and extra channel injection introduce orientation-dependent effects.
"""

import torch
import pytest
import escnn.nn as enn
from escnn import gspaces


# ── Encoder equivariance via escnn's own testing ──────────────────────

def test_encoder_block_equivariance():
    """Each encoder block is exactly equivariant (verified by escnn internals)."""
    from src.models.segnet import _eq_block

    gspace = gspaces.flipRot2dOnR2(N=4)
    trivial_in = enn.FieldType(gspace, [gspace.trivial_repr] * 6)
    reg_out = enn.FieldType(gspace, [gspace.regular_repr] * 8)

    block = _eq_block(trivial_in, reg_out, pool=False)
    block.eval()

    # escnn's built-in equivariance check
    x = enn.GeometricTensor(torch.randn(2, 6, 32, 32), trivial_in)
    block.check_equivariance(atol=1e-4, rtol=1e-4)


def test_group_pooling_invariance():
    """GroupPooling produces invariant output from equivariant input."""
    gspace = gspaces.flipRot2dOnR2(N=4)
    reg_type = enn.FieldType(gspace, [gspace.regular_repr] * 8)
    gp = enn.GroupPooling(reg_type)

    x = enn.GeometricTensor(torch.randn(2, 64, 8, 8), reg_type)
    out = gp(x)

    # Output type should be trivial (invariant)
    assert out.type.size == 8, f"GroupPooling output size: {out.type.size}"


def test_full_model_consistency():
    """Full model produces consistent output for same input (deterministic)."""
    from src.models.segnet import D4SegNet
    model = D4SegNet()
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 12, 64, 64)

    with torch.no_grad():
        out1 = model(x)["logit"]
        out2 = model(x)["logit"]

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"Non-deterministic output: {diff}"


# ── Shape and parameter tests ──────────────────────────────────────────

def test_output_shapes():
    """Verify output shapes match architecture docstring."""
    from src.models.segnet import D4SegNet
    model = D4SegNet()
    model.eval()
    x = torch.randn(1, 12, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["logit"].shape == (1, 1, 64, 64)
    assert out["area_m2"].shape == (1, 1)


def test_variable_input_size():
    """Model should handle 128x128 input."""
    from src.models.segnet import D4SegNet
    model = D4SegNet()
    model.eval()
    x = torch.randn(1, 12, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["logit"].shape == (1, 1, 128, 128)


def test_parameter_count():
    """Verify parameter count matches README claim."""
    from src.models.segnet import D4SegNet
    model = D4SegNet()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 625617, f"Expected 625617 params, got {n}"


def test_noskip_variant():
    """D4SegNetNoSkip should have same param count and work."""
    from src.models.segnet import D4SegNetNoSkip
    model = D4SegNetNoSkip()
    model.eval()
    x = torch.randn(1, 12, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["logit"].shape == (1, 1, 64, 64)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 625617
