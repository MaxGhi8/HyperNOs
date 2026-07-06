import torch

from hypernos.architectures import GeometryConditionedLinearOperator_mp_afieti

device = torch.device("cpu")


def make_model(device=device, n_heads_A=1):
    return GeometryConditionedLinearOperator_mp_afieti(
        n_dofs=16,
        n_control_points=10,
        hidden_dim=8,
        n_heads=2,
        n_heads_A=n_heads_A,
        n_layers_geo=1,
        dropout_rate=0.0,
        activation_str="gelu",
        zero_mean=True,
        device=device,
    )


def dense_matrix_from_components(model, Q, K_scaled, epsilon):
    eye = torch.eye(model.n_dofs, device=Q.device).unsqueeze(0)
    return torch.bmm(Q, K_scaled.transpose(-2, -1)) + eye * epsilon


def test_forward_shape():
    torch.manual_seed(0)
    model = make_model()
    batch = 4
    d = 16

    f = torch.randn(batch, d, device=device)
    g = torch.randn(batch, 10, 4, device=device)

    u = model((f, g))

    assert u.shape == (batch, d)


def test_compute_operator_components_shapes():
    torch.manual_seed(2)
    model = make_model()
    batch = 3
    d = 16
    hidden_dim = 8

    g = torch.randn(batch, 10, 4, device=device)

    Q, K_scaled, epsilon = model.compute_operator_components(g)

    assert Q.shape == (batch, d, hidden_dim)
    assert K_scaled.shape == (batch, d, hidden_dim)
    assert epsilon.dim() == 0
    assert epsilon.item() > 0


def test_apply_operator_matches_dense_apply_and_forward():
    torch.manual_seed(1)
    model = make_model()
    batch = 3
    d = 16

    f = torch.randn(batch, d, device=device)
    g = torch.randn(batch, 10, 4, device=device)

    Q, K_scaled, epsilon = model.compute_operator_components(g)
    A = dense_matrix_from_components(model, Q, K_scaled, epsilon)

    u_dense = torch.bmm(A, f.unsqueeze(-1)).squeeze(-1)
    u_efficient = model.apply_operator(f, Q, K_scaled, epsilon)
    u_model = model((f, g))

    assert torch.allclose(u_efficient, u_dense, atol=1e-5)
    assert torch.allclose(u_model, u_dense, atol=1e-5)


def test_apply_operator_matches_dense_apply_multiple_heads_A():
    torch.manual_seed(3)
    model = make_model(n_heads_A=4)
    batch = 3
    d = 16

    f = torch.randn(batch, d, device=device)
    g = torch.randn(batch, 10, 4, device=device)

    Q, K_scaled, epsilon = model.compute_operator_components(g)
    A = dense_matrix_from_components(model, Q, K_scaled, epsilon)

    u_dense = torch.bmm(A, f.unsqueeze(-1)).squeeze(-1)
    u_efficient = model.apply_operator(f, Q, K_scaled, epsilon)

    assert torch.allclose(u_efficient, u_dense, atol=1e-5)
