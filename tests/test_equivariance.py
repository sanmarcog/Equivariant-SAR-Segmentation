"""
Test D4 equivariance properties of the architecture.

The encoder is exactly D4-equivariant. The full model (encoder + decoder)
is approximately equivariant — the standard Conv2d decoder and extra channel
injection introduce small orientation-dependent effects.
"""

import torch
import pytest


@pytest.fixture(scope="module")
def model():
    from src.models.segnet import D4SegNet
    m = D4SegNet()
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(1, 12, 64, 64)


# ── Encoder equivariance (exact) ──────────────────────────────────────

def test_encoder_equivariance_90(model, input_tensor):
    """Encoder change features are exactly invariant under 90-degree rotation."""
    import escnn.nn as enn
    x = input_tensor
    x_rot = torch.rot90(x, k=1, dims=[-2, -1])

    with torch.no_grad():
        # Run encoder on both
        x_post = x[:, model.POST_IDX]
        x_pre = x[:, model.PRE_IDX]
        fp = model._encode_branch(x_post)
        fr = model._encode_branch(x_pre)
        bt = model._group_diff(fp[4], fr[4], model.gp5)

        x_post_r = x_rot[:, model.POST_IDX]
        x_pre_r = x_rot[:, model.PRE_IDX]
        fp_r = model._encode_branch(x_post_r)
        fr_r = model._encode_branch(x_pre_r)
        bt_r = model._group_diff(fp_r[4], fr_r[4], model.gp5)

    # GroupPooled bottleneck should be INVARIANT (same regardless of rotation)
    diff = (bt - bt_r).abs().max().item()
    assert diff < 1e-4, f"Encoder invariance violated: max diff = {diff}"


def test_encoder_equivariance_flip(model, input_tensor):
    """Encoder change features are exactly invariant under horizontal flip."""
    x = input_tensor
    x_flip = torch.flip(x, dims=[-1])

    with torch.no_grad():
        x_post = x[:, model.POST_IDX]
        x_pre = x[:, model.PRE_IDX]
        fp = model._encode_branch(x_post)
        fr = model._encode_branch(x_pre)
        bt = model._group_diff(fp[4], fr[4], model.gp5)

        x_post_f = x_flip[:, model.POST_IDX]
        x_pre_f = x_flip[:, model.PRE_IDX]
        fp_f = model._encode_branch(x_post_f)
        fr_f = model._encode_branch(x_pre_f)
        bt_f = model._group_diff(fp_f[4], fr_f[4], model.gp5)

    diff = (bt - bt_f).abs().max().item()
    assert diff < 1e-4, f"Encoder invariance under flip violated: max diff = {diff}"


# ── Full model approximate equivariance ────────────────────────────────

def test_full_model_approximate_equivariance_90(model, input_tensor):
    """Full model output is approximately equivariant to 90-degree rotation.
    Not exact due to standard Conv2d decoder and extra channel injection."""
    with torch.no_grad():
        out = model(input_tensor)["logit"]
        x_rot = torch.rot90(input_tensor, k=1, dims=[-2, -1])
        out_rot = model(x_rot)["logit"]
        out_rot_back = torch.rot90(out_rot, k=-1, dims=[-2, -1])

    diff = (out - out_rot_back).abs().max().item()
    # Approximate: decoder breaks exact equivariance
    assert diff < 2.0, f"Full model equivariance error unexpectedly large: {diff}"


# ── Shape and parameter tests ──────────────────────────────────────────

def test_output_shapes(model, input_tensor):
    """Verify output shapes match architecture docstring."""
    with torch.no_grad():
        out = model(input_tensor)
    assert out["logit"].shape == (1, 1, 64, 64)
    assert out["area_m2"].shape == (1, 1)


def test_variable_input_size(model):
    """Model should handle 128x128 input."""
    x = torch.randn(1, 12, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["logit"].shape == (1, 1, 128, 128)


def test_parameter_count(model):
    """Verify parameter count matches README claim."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 625617, f"Expected 625617 params, got {n}"
