import sys

import pytest
import torch

sys.path.append("..")
from architectures import GeometryConditionedLinearOperator, zero_mean_imposition

device = torch.device("cpu")


def make_model(device=device):
    return GeometryConditionedLinearOperator(
        n_dofs=16,
        n_control_points=10,
        hidden_dim=8,
        n_heads=2,
        n_layers_geo=1,
        dropout_rate=0.0,
        activation_str="gelu",
        zero_mean=True,
        device=device,
    )


def test_forward_shape_and_zero_mean():
    torch.manual_seed(0)
    model = make_model(device=device)
    batch = 4
    d = 16

    # Create inputs
    f = torch.randn(batch, d, device=device)
    f = zero_mean_imposition(f)
    g = torch.randn(batch, 10, 4, device=device)

    u = model((f, g))

    assert u.shape == (batch, d)

    # Check zero-mean constraint per sample (allow small numerical tol)
    means = u.mean(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)


def test_construct_matrix_and_apply_matches_forward():
    torch.manual_seed(1)
    model = make_model(device=device)
    batch = 3
    d = 16

    f = torch.randn(batch, d, device=device)
    f = zero_mean_imposition(f)
    g = torch.randn(batch, 10, 4, device=device)

    A = model.construct_matrix(g)
    assert A.shape == (batch, d, d)

    # Manual apply of A to f should match model forward (before and after post-processing)
    u_manual = zero_mean_imposition(torch.bmm(A, f.unsqueeze(-1)).squeeze(-1))
    u_model = model((f, g))

    # Both tensors should be equal after the same post-processing
    assert torch.allclose(
        u_manual.mean(dim=1), torch.zeros(batch, device=device), atol=1e-4
    )
    assert torch.allclose(u_model, u_manual, atol=1e-5)
    assert torch.allclose(u_model, u_manual, atol=1e-5)
