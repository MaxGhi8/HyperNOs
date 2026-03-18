"""
Comprehensive pytest suite for CNN2D_DON network.
"""

import sys

import pytest
import torch

sys.path.append("../")

from architectures import CNN2D_DON

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################
# Fixtures
#########################################
@pytest.fixture
def default_params():
    """Default parameters for CNN2D_DON."""
    return {
        "batch_size": 4,
        "height": 64,
        "width": 64,
        "in_channels": 1,
        "n_basis": 100,
        "hidden_channels": [16, 32, 64],
        "hidden_layers": [128, 128],
        "kernel_size": 3,
        "activation_str": "relu",
        "padding": 0,
        "stride": 2,
        "device": device,
    }


@pytest.fixture
def basic_branch_net(default_params):
    """Create a basic CNN2D_DON network."""
    return CNN2D_DON(
        in_channels=default_params["in_channels"],
        n_basis=default_params["n_basis"],
        hidden_channels=default_params["hidden_channels"],
        hidden_layers=default_params["hidden_layers"],
        kernel_size=default_params["kernel_size"],
        activation_str=default_params["activation_str"],
        padding=default_params["padding"],
        stride=default_params["stride"],
        include_grid=False,
        device=default_params["device"],
        normalization="batch",
        dropout_rate=0.1,
    )


@pytest.fixture
def sample_input(default_params):
    """Create sample input tensor."""
    return torch.randn(
        default_params["batch_size"],
        default_params["height"],
        default_params["width"],
        default_params["in_channels"],
    ).to(device)


#########################################
# Tests
#########################################
class TestBasicFunctionality:
    """Test basic forward pass and output shape."""

    def test_forward_pass(self, basic_branch_net, sample_input, default_params):
        """Test basic forward pass."""
        with torch.no_grad():
            output = basic_branch_net(sample_input)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        ), f"Expected shape ({default_params['batch_size']}, {default_params['n_basis']}), got {output.shape}"

    def test_output_type(self, basic_branch_net, sample_input):
        """Test output is a torch tensor."""
        with torch.no_grad():
            output = basic_branch_net(sample_input)

        assert isinstance(output, torch.Tensor)

    def test_output_device(self, basic_branch_net, sample_input):
        """Test output is on correct device."""
        with torch.no_grad():
            output = basic_branch_net(sample_input)

        assert output.device == sample_input.device


class TestSpatialDimensions:
    """Test the network handles different input spatial dimensions."""

    @pytest.mark.parametrize(
        "height,width",
        [
            (32, 32),
            (64, 64),
            (128, 128),
            (96, 96),
            (48, 48),
        ],
    )
    def test_different_spatial_sizes(
        self, basic_branch_net, default_params, height, width
    ):
        """Test network with different spatial dimensions."""
        x = torch.randn(
            default_params["batch_size"],
            height,
            width,
            default_params["in_channels"],
        ).to(device)

        with torch.no_grad():
            output = basic_branch_net(x)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        ), f"Failed for input size {height}x{width}"


class TestGridCoordinates:
    """Test the network with spatial grid coordinates included."""

    def test_with_grid_coordinates(self, default_params):
        """Test network with grid coordinates included."""
        branch_net_with_grid = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=default_params["hidden_channels"],
            hidden_layers=default_params["hidden_layers"],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            include_grid=True,
            device=default_params["device"],
            normalization="batch",
            dropout_rate=0.1,
        )

        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
        ).to(device)

        with torch.no_grad():
            output = branch_net_with_grid(x)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        )


class TestActivationFunctions:
    """Test the network with different activation functions."""

    @pytest.mark.parametrize(
        "activation",
        [
            "relu",
            "tanh",
            "leaky_relu",
            "sigmoid",
        ],
    )
    def test_different_activations(self, default_params, activation):
        """Test network with different activation functions."""
        branch_net = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=activation,
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
        )

        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
        ).to(device)

        with torch.no_grad():
            output = branch_net(x)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        )


class TestNormalization:
    """Test the network with different normalization strategies."""

    @pytest.mark.parametrize(
        "normalization",
        [
            "none",
            "batch",
            "layer",
        ],
    )
    def test_different_normalizations(self, default_params, normalization):
        """Test network with different normalization strategies."""
        branch_net = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
            normalization=normalization,
        )

        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
        ).to(device)

        with torch.no_grad():
            output = branch_net(x)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        )


class TestGradientFlow:
    """Test gradient flow through the network."""

    def test_backward_pass(self, default_params):
        """Test gradient flow through the network."""
        branch_net = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
        )

        branch_net.train()
        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
            requires_grad=True,
        ).to(device)

        output = branch_net(x)
        loss = output.sum()
        loss.backward()

        # Check if gradients were computed for model parameters
        has_gradients = any(
            p.grad is not None for p in branch_net.parameters() if p.requires_grad
        )
        assert has_gradients, "No gradients computed for model parameters"

    def test_gradient_magnitudes(self, default_params):
        """Test gradient magnitudes are reasonable."""
        branch_net = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
        )

        branch_net.train()
        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
        ).to(device)

        output = branch_net(x)
        loss = output.sum()
        loss.backward()

        # Check gradient magnitudes
        grad_norms = [
            p.grad.norm().item() for p in branch_net.parameters() if p.grad is not None
        ]

        assert len(grad_norms) > 0, "No gradients computed"
        assert all(
            not torch.isnan(torch.tensor(g)) for g in grad_norms
        ), "NaN in gradients"
        assert all(
            not torch.isinf(torch.tensor(g)) for g in grad_norms
        ), "Inf in gradients"


class TestMultiChannelInput:
    """Test the network with multi-channel input (e.g., vector fields)."""

    @pytest.mark.parametrize("in_channels", [1, 2, 3, 4])
    def test_different_input_channels(self, default_params, in_channels):
        """Test network with different numbers of input channels."""
        branch_net = CNN2D_DON(
            in_channels=in_channels,
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
        )

        x = torch.randn(
            default_params["batch_size"],
            default_params["height"],
            default_params["width"],
            in_channels,
        ).to(device)

        with torch.no_grad():
            output = branch_net(x)

        assert output.shape == (
            default_params["batch_size"],
            default_params["n_basis"],
        )


class TestBatchSizes:
    """Test the network with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32])
    def test_different_batch_sizes(self, default_params, batch_size):
        """Test network with different batch sizes."""
        branch_net = CNN2D_DON(
            in_channels=default_params["in_channels"],
            n_basis=default_params["n_basis"],
            hidden_channels=[16, 32],
            hidden_layers=[64],
            kernel_size=default_params["kernel_size"],
            activation_str=default_params["activation_str"],
            padding=default_params["padding"],
            stride=default_params["stride"],
            device=default_params["device"],
        )

        x = torch.randn(
            batch_size,
            default_params["height"],
            default_params["width"],
            default_params["in_channels"],
        ).to(device)

        with torch.no_grad():
            output = branch_net(x)

        assert output.shape == (batch_size, default_params["n_basis"])
