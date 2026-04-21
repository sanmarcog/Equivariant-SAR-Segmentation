"""
Test that D4SegNet is exactly equivariant to 90-degree rotations and flips.
This is the central architectural claim of the project.
"""

import torch
import pytest


def _build_model():
    from src.models.segnet import D4SegNet
    model = D4SegNet()
    model.eval()
    return model


@pytest.fixture(scope="module")
def model():
    return _build_model()


@pytest.fixture(scope="module")
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(1, 12, 64, 64)


def test_90_rotation_equivariance(model, input_tensor):
    """Rotating input by 90 degrees should rotate output by 90 degrees."""
    with torch.no_grad():
        out_orig = model(input_tensor)["logit"]
        # Rotate input 90 degrees counterclockwise
        x_rot = torch.rot90(input_tensor, k=1, dims=[-2, -1])
        out_rot = model(x_rot)["logit"]
        # Rotate output back
        out_rot_back = torch.rot90(out_rot, k=-1, dims=[-2, -1])

    diff = (out_orig - out_rot_back).abs().max().item()
    assert diff < 1e-4, f"90-degree rotation equivariance violated: max diff = {diff}"


def test_180_rotation_equivariance(model, input_tensor):
    """Rotating input by 180 degrees should rotate output by 180 degrees."""
    with torch.no_grad():
        out_orig = model(input_tensor)["logit"]
        x_rot = torch.rot90(input_tensor, k=2, dims=[-2, -1])
        out_rot = model(x_rot)["logit"]
        out_rot_back = torch.rot90(out_rot, k=-2, dims=[-2, -1])

    diff = (out_orig - out_rot_back).abs().max().item()
    assert diff < 1e-4, f"180-degree rotation equivariance violated: max diff = {diff}"


def test_horizontal_flip_equivariance(model, input_tensor):
    """Flipping input horizontally should flip output horizontally."""
    with torch.no_grad():
        out_orig = model(input_tensor)["logit"]
        x_flip = torch.flip(input_tensor, dims=[-1])
        out_flip = model(x_flip)["logit"]
        out_flip_back = torch.flip(out_flip, dims=[-1])

    diff = (out_orig - out_flip_back).abs().max().item()
    assert diff < 1e-4, f"Horizontal flip equivariance violated: max diff = {diff}"


def test_vertical_flip_equivariance(model, input_tensor):
    """Flipping input vertically should flip output vertically."""
    with torch.no_grad():
        out_orig = model(input_tensor)["logit"]
        x_flip = torch.flip(input_tensor, dims=[-2])
        out_flip = model(x_flip)["logit"]
        out_flip_back = torch.flip(out_flip, dims=[-2])

    diff = (out_orig - out_flip_back).abs().max().item()
    assert diff < 1e-4, f"Vertical flip equivariance violated: max diff = {diff}"


def test_output_shapes(model, input_tensor):
    """Verify output shapes match architecture docstring."""
    with torch.no_grad():
        out = model(input_tensor)
    assert out["logit"].shape == (1, 1, 64, 64), f"logit shape: {out['logit'].shape}"
    assert out["area_m2"].shape == (1, 1), f"area_m2 shape: {out['area_m2'].shape}"


def test_variable_input_size(model):
    """Model should handle 128x128 input (used in Gatti-mirror experiments)."""
    x = torch.randn(1, 12, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["logit"].shape == (1, 1, 128, 128)


def test_parameter_count(model):
    """Verify parameter count matches README claim."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 625617, f"Expected 625617 params, got {n_params}"
