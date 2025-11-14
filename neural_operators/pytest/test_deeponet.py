import sys

import pytest
import torch

sys.path.append("..")

from architectures import DeepONet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters for 1D case
branch_hyperparams_1d = {
    "n_inputs": 100,
    "hidden_layer": [64, 50, 20, 64],
    "n_blocks": 3,
    "act_fun": "relu",
    "residual": True,
    "layer_norm": True,
    "dropout_rate": 0.1,
}

trunk_hyperparams_1d = {
    "n_inputs": 3,
    "hidden_layer": [64, 64, 64],
    "n_blocks": 2,
    "act_fun": "relu",
    "residual": True,
    "layer_norm": False,
    "dropout_rate": 0.0,
}

# Default hyperparameters for 2D case
branch_hyperparams_2d = {
    "n_inputs": 1,
    "n_basis": 100,
    "channels_conv": [16, 32, 48],
    "hidden_layer": [128, 64],
    "stride": 2,
    "padding": 0,
    "kernel_size": 3,
    "act_fun": "relu",
    "residual": False,
    "include_grid": True,
    "normalization": "batch",
    "dropout_rate": 0.0,
    "output_dim_conv": 512,
}

trunck_hyperparams_2d = {
    "n_inputs": 3,
    "hidden_layer": [64, 64, 64],
    "n_blocks": 2,
    "act_fun": "relu",
    "residual": True,
    "layer_norm": False,
    "dropout_rate": 0.0,
}


#########################################
# 1D DeepONet Tests
#########################################
def test_deeponet_1d_single_output_initialization():
    """Test 1D DeepONet initialization with single output"""
    n_basis = 64
    n_output = 1
    dim = 1

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=dim,
    )

    assert model.n_basis == n_basis
    assert model.n_output == n_output
    assert model.dim == dim
    assert hasattr(model, "branch_NN")
    assert hasattr(model, "trunk_NN")


def test_deeponet_1d_multi_output_initialization():
    """Test 1D DeepONet initialization with multiple outputs"""
    n_basis = 64
    n_output = 4
    dim = 1

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=dim,
    )

    assert model.n_basis == n_basis
    assert model.n_output == n_output
    assert model.output_division == n_basis // n_output


def test_deeponet_1d_forward_single_output_FNN():
    """Test 1D DeepONet forward pass with single output"""
    batch_size = 4
    n_points_trunk = 10  # Number of evaluation points for trunk network
    n_basis = 32
    n_output = 1

    branch_hyperparams_1d["residual"] = False
    trunk_hyperparams_1d["residual"] = False

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1,
    )

    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape
    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_1d_forward_multi_output_FNN():
    """Test 1D DeepONet forward pass with multiple outputs"""
    batch_size = 4
    n_points_trunk = 10  # Number of evaluation points for trunk network
    n_basis = 64
    n_output = 4

    branch_hyperparams_1d["residual"] = False
    trunk_hyperparams_1d["residual"] = False

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1,
    )

    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape
    expected_shape = (batch_size, n_points_trunk, n_output)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_1d_forward_single_output_ResNet():
    """Test 1D DeepONet forward pass with single output"""
    batch_size = 4
    n_points_trunk = 10  # Number of evaluation points for trunk network
    n_basis = 32
    n_output = 1

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1,
    )

    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape
    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_1d_forward_multi_output_ResNet():
    """Test 1D DeepONet forward pass with multiple outputs"""
    batch_size = 4
    n_points_trunk = 10  # Number of evaluation points for trunk network
    n_basis = 64
    n_output = 4

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1,
    )

    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape
    expected_shape = (batch_size, n_points_trunk, n_output)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_residual_networks_1d():
    """Test DeepONet with residual branch and trunk networks 1D"""
    # Test with residual=True
    branch_params = branch_hyperparams_1d.copy()
    trunk_params = trunk_hyperparams_1d.copy()
    branch_params["residual"] = True
    trunk_params["residual"] = True

    model = DeepONet(
        branch_hyperparameters=branch_params,
        trunk_hyperparameters=trunk_params,
        n_basis=32,
        n_output=1,
        dim=1,
    )

    batch_size = 4
    branch_input = torch.randn(batch_size, branch_params["n_inputs"])
    n_points_trunk = 200
    trunk_input = torch.randn(n_points_trunk, trunk_params["n_inputs"])

    output = model((branch_input, trunk_input))
    assert output.shape == (batch_size, n_points_trunk)


def test_different_activation_functions():
    """Test DeepONet with different activation functions"""
    activations = ["relu", "tanh", "leaky_relu", "silu"]

    for act_fun in activations:
        branch_params = branch_hyperparams_1d.copy()
        trunk_params = trunk_hyperparams_1d.copy()
        branch_params["act_fun"] = act_fun
        trunk_params["act_fun"] = act_fun

        model = DeepONet(
            branch_hyperparameters=branch_params,
            trunk_hyperparameters=trunk_params,
            n_basis=16,
            n_output=1,
            dim=1,
        )

        batch_size = 2
        branch_input = torch.randn(batch_size, branch_params["n_inputs"])
        n_trunk_points = 50
        trunk_input = torch.randn(n_trunk_points, trunk_params["n_inputs"])

        output = model((branch_input, trunk_input))
        assert not torch.isnan(output).any(), f"NaN values with {act_fun} activation"


def test_n_basis_divisibility_assertion():
    """Test that n_basis must be divisible by n_output"""
    with pytest.raises(AssertionError, match="n_basis must be divisible by n_output"):
        DeepONet(
            branch_hyperparameters=branch_hyperparams_1d,
            trunk_hyperparameters=trunk_hyperparams_1d,
            n_basis=31,  # Not divisible by n_output=4
            n_output=4,
            dim=1,
        )


def test_gradient_flow():
    """Test that gradients flow through the network"""
    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=32,
        n_output=1,
        dim=1,
    )

    batch_size = 4
    branch_input = torch.randn(
        batch_size, branch_hyperparams_1d["n_inputs"], requires_grad=True
    )
    n_trunk_points = 20
    trunk_input = torch.randn(
        n_trunk_points, trunk_hyperparams_1d["n_inputs"], requires_grad=True
    )
    target = torch.randn(batch_size, n_trunk_points)

    output = model((branch_input, trunk_input))
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check that gradients exist
    assert branch_input.grad is not None
    assert trunk_input.grad is not None
    assert branch_input.grad.abs().sum() > 0
    assert trunk_input.grad.abs().sum() > 0


def test_device_consistency():
    """Test that all model components are on the correct device"""
    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=32,
        n_output=1,
        dim=1,
        device=device,
    )

    # Check that model is on expected device
    model_device = next(model.parameters()).device.type
    assert model_device == device.type


def test_different_network_sizes():
    """Test with different network architectures"""
    architectures = [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]

    for hidden_layers in architectures:
        branch_params = branch_hyperparams_1d.copy()
        trunk_params = trunk_hyperparams_1d.copy()
        branch_params["hidden_layer"] = hidden_layers
        trunk_params["hidden_layer"] = hidden_layers

        model = DeepONet(
            branch_hyperparameters=branch_params,
            trunk_hyperparameters=trunk_params,
            n_basis=32,
            n_output=1,
            dim=1,
            device=device,
        )

        batch_size = 2
        branch_input = torch.randn(batch_size, branch_params["n_inputs"]).to(device)
        n_trunk_points = 50
        trunk_input = torch.randn(n_trunk_points, trunk_params["n_inputs"]).to(device)

        output = model((branch_input, trunk_input))
        assert output.shape == (batch_size, n_trunk_points)


#########################################
# 2D DeepONet Tests
#########################################
def test_deeponet_2d_single_output():
    """Test 2D DeepONet forward pass with single output"""
    batch_size = 2
    n_basis = 32
    n_output = 1

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    # Create input data
    # Branch input: 2D field data
    h, w = 50, 50
    branch_input = torch.randn(batch_size, h, w, branch_hyperparams_2d["n_inputs"]).to(
        device
    )

    # Trunk input: 2D coordinates
    n_points_trunk = 200
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )
    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape for 2D single output
    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_2d_multi_output():
    """Test 2D DeepONet forward pass with multiple outputs"""
    batch_size = 2
    n_basis = 64
    n_output = 2

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    # Create input data
    h, w = 50, 50
    branch_input = torch.randn(batch_size, h, w, branch_hyperparams_2d["n_inputs"]).to(
        device
    )
    n_trunk_points = 300
    trunk_input = torch.randn(n_trunk_points, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape for 2D multi output
    expected_shape = (batch_size, n_trunk_points, n_output)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_2d_with_cnn_branch():
    """Test 2D DeepONet with CNN2D_Branch network"""
    batch_size = 4
    n_basis = 100
    n_output = 1
    h, w = 64, 64

    # Use CNN-based branch network
    branch_hyperparams_cnn = {
        "n_inputs": 1,
        "channels_conv": [16, 32, 64],
        "hidden_layer": [128, 128],
        "kernel_size": 3,
        "act_fun": "relu",
        "padding": 0,
        "stride": 2,
        "residual": False,
        "include_grid": True,
        "normalization": "batch",
        "dropout_rate": 0.0,
    }

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_cnn,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    # Create input data
    branch_input = torch.randn(batch_size, h, w, branch_hyperparams_cnn["n_inputs"]).to(
        device
    )
    n_points_trunk = 256
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    # Forward pass
    output = model((branch_input, trunk_input))

    # Check output shape
    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_deeponet_2d_different_spatial_resolutions():
    """Test 2D DeepONet handles different spatial resolutions"""
    batch_size = 2
    n_basis = 64
    n_output = 1

    # Test with different spatial resolutions
    spatial_sizes = [(32, 32), (48, 48), (64, 64), (128, 128)]
    n_points_trunk = 100
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    for h, w in spatial_sizes:
        model = DeepONet(
            branch_hyperparameters=branch_hyperparams_2d,
            trunk_hyperparameters=trunck_hyperparams_2d,
            n_basis=n_basis,
            n_output=n_output,
            dim=2,
            device=device,
        )
        branch_input = torch.randn(
            batch_size, h, w, branch_hyperparams_2d["n_inputs"]
        ).to(device)
        output = model((branch_input, trunk_input))

        expected_shape = (batch_size, n_points_trunk)
        assert output.shape == expected_shape, f"Failed for spatial size {h}x{w}"
        assert not torch.isnan(output).any()


def test_deeponet_2d_multi_channel_input():
    """Test 2D DeepONet with multi-channel input (e.g., vector fields)"""
    batch_size = 2
    n_basis = 64
    n_output = 1
    h, w = 64, 64
    n_channels = 3  # Multi-channel input (e.g., RGB or 3D vector field)

    # Modify branch hyperparameters for multi-channel input
    branch_hyperparams_multi = branch_hyperparams_2d.copy()
    branch_hyperparams_multi["n_inputs"] = n_channels

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_multi,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    # Create multi-channel input
    branch_input = torch.randn(batch_size, h, w, n_channels).to(device)
    n_points_trunk = 150
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    # Forward pass
    output = model((branch_input, trunk_input))

    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_deeponet_2d_gradient_flow():
    """Test gradient flow through 2D DeepONet"""
    batch_size = 2
    n_basis = 64
    n_output = 1
    h, w = 32, 32

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    model.train()

    # Create input data with gradient tracking
    branch_input = torch.randn(
        batch_size, h, w, branch_hyperparams_2d["n_inputs"], requires_grad=True
    ).to(device)
    n_points_trunk = 50
    trunk_input = torch.randn(
        n_points_trunk, trunck_hyperparams_2d["n_inputs"], requires_grad=True
    ).to(device)

    # Forward and backward pass
    output = model((branch_input, trunk_input))
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed
    has_branch_grad = any(
        p.grad is not None for p in model.branch_NN.parameters() if p.requires_grad
    )
    has_trunk_grad = any(
        p.grad is not None for p in model.trunk_NN.parameters() if p.requires_grad
    )

    assert has_branch_grad, "No gradients for branch network"
    assert has_trunk_grad, "No gradients for trunk network"


def test_deeponet_2d_batch_sizes():
    """Test 2D DeepONet with different batch sizes"""
    n_basis = 64
    n_output = 1
    h, w = 64, 64

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    n_points_trunk = 100
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    batch_sizes = [2, 4, 8]

    for batch_size in batch_sizes:
        branch_input = torch.randn(
            batch_size, h, w, branch_hyperparams_2d["n_inputs"]
        ).to(device)
        output = model((branch_input, trunk_input))

        expected_shape = (batch_size, n_points_trunk)
        assert output.shape == expected_shape, f"Failed for batch size {batch_size}"


def test_deeponet_2d_trunk_evaluation_points():
    """Test 2D DeepONet with different numbers of trunk evaluation points"""
    batch_size = 2
    n_basis = 64
    n_output = 1
    h, w = 64, 64

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    branch_input = torch.randn(batch_size, h, w, branch_hyperparams_2d["n_inputs"]).to(
        device
    )

    trunk_point_counts = [10, 50, 100, 256, 512]

    for n_points_trunk in trunk_point_counts:
        trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
            device
        )
        output = model((branch_input, trunk_input))

        expected_shape = (batch_size, n_points_trunk)
        assert (
            output.shape == expected_shape
        ), f"Failed for {n_points_trunk} trunk points"


def test_deeponet_2d_multi_output_different_basis():
    """Test 2D DeepONet with multiple outputs and different basis sizes"""
    batch_size = 2
    h, w = 64, 64

    # Test different combinations of n_basis and n_output
    configs = [
        (64, 2),
        (128, 4),
        (100, 5),
        (60, 3),
    ]

    for n_basis, n_output in configs:
        model = DeepONet(
            branch_hyperparameters=branch_hyperparams_2d,
            trunk_hyperparameters=trunck_hyperparams_2d,
            n_basis=n_basis,
            n_output=n_output,
            dim=2,
            device=device,
        )

        branch_input = torch.randn(
            batch_size, h, w, branch_hyperparams_2d["n_inputs"]
        ).to(device)
        n_points_trunk = 100
        trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
            device
        )

        output = model((branch_input, trunk_input))

        expected_shape = (batch_size, n_points_trunk, n_output)
        assert (
            output.shape == expected_shape
        ), f"Failed for n_basis={n_basis}, n_output={n_output}"


def test_deeponet_2d_with_normalization():
    """Test 2D DeepONet with different normalization strategies"""
    batch_size = 2
    n_basis = 64
    n_output = 1
    h, w = 64, 64

    normalizations = ["none", "batch", "layer"]

    for norm in normalizations:
        branch_params = branch_hyperparams_2d.copy()
        branch_params["normalization"] = norm

        model = DeepONet(
            branch_hyperparameters=branch_params,
            trunk_hyperparameters=trunck_hyperparams_2d,
            n_basis=n_basis,
            n_output=n_output,
            dim=2,
            device=device,
        )

        branch_input = torch.randn(batch_size, h, w, branch_params["n_inputs"]).to(
            device
        )
        n_points_trunk = 100
        trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
            device
        )

        output = model((branch_input, trunk_input))

        expected_shape = (batch_size, n_points_trunk)
        assert output.shape == expected_shape, f"Failed for normalization={norm}"
        assert not torch.isnan(output).any()


def test_deeponet_2d_output_consistency():
    """Test that 2D DeepONet produces consistent outputs for same input"""
    batch_size = 2
    n_basis = 64
    n_output = 1
    h, w = 64, 64

    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_2d,
        trunk_hyperparameters=trunck_hyperparams_2d,
        n_basis=n_basis,
        n_output=n_output,
        dim=2,
        device=device,
    )

    model.eval()  # Set to evaluation mode

    # Create input data
    branch_input = torch.randn(batch_size, h, w, branch_hyperparams_2d["n_inputs"]).to(
        device
    )
    n_points_trunk = 100
    trunk_input = torch.randn(n_points_trunk, trunck_hyperparams_2d["n_inputs"]).to(
        device
    )

    # Multiple forward passes
    with torch.no_grad():
        output1 = model((branch_input, trunk_input))
        output2 = model((branch_input, trunk_input))

    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)
