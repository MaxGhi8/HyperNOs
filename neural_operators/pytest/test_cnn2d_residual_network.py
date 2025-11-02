import sys

sys.path.append("..")

import torch
from architectures import CNN2DResidualNetwork, Conv2DResidualBlock


def test_conv2d_residual_block_forward():
    """Test forward pass of Conv2D residual block"""
    batch_size, channels, height, width = 2, [32, 42, 32], 64, 64

    block = Conv2DResidualBlock(
        channels=channels,
        kernel_size=3,
        activation_str="relu",
        normalization="none",
        dropout_rate=0.1,
    )

    x = torch.randn(batch_size, channels[0], height, width)
    output = block(x)

    assert output.shape == (batch_size, channels[-1], height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_cnn2d_residual_network_forward():
    """Test forward pass of CNN2D residual network"""
    batch_size, in_channels, out_channels = 2, 3, 1
    height, width = 64, 64

    model = CNN2DResidualNetwork(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=[16, 32, 16],
        kernel_size=3,
        activation_str="relu",
        n_blocks=4,
        normalization="batch",
        dropout_rate=0.1,
    )

    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)

    assert output.shape == (batch_size, out_channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_different_input_sizes():
    """Test that the model works with different input sizes"""
    model = CNN2DResidualNetwork(
        in_channels=1,
        out_channels=1,
        hidden_channels=[16, 32, 16],
        n_blocks=2,
        kernel_size=5,
        activation_str="relu",
    )

    # Test different sizes
    sizes = [(32, 32), (64, 64), (128, 128), (48, 96)]

    for height, width in sizes:
        x = torch.randn(1, 1, height, width)
        output = model(x)
        assert output.shape == (1, 1, height, width)


def test_different_activations():
    """Test different activation functions"""
    activations = ["relu", "tanh", "leaky_relu", "sigmoid"]

    for activation in activations:
        model = CNN2DResidualNetwork(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 8, 16],
            activation_str=activation,
            n_blocks=2,
            kernel_size=7,
        )

        x = torch.randn(1, 1, 32, 32)
        output = model(x)
        assert output.shape == (1, 1, 32, 32)
        assert not torch.isnan(output).any()


def test_parameter_count():
    """Test that the model has reasonable parameter count"""
    model = CNN2DResidualNetwork(
        in_channels=3,
        out_channels=1,
        hidden_channels=[64, 32, 64],
        n_blocks=4,
        kernel_size=3,
        activation_str="relu",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params == total_params


def test_gradient_flow():
    """Test that gradients flow through the network"""
    model = CNN2DResidualNetwork(
        in_channels=1,
        out_channels=1,
        hidden_channels=[32, 16, 32],
        n_blocks=2,
        kernel_size=3,
        activation_str="relu",
    )

    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check that gradients exist and are not zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_layer_norm():
    """Test model with layer normalization"""
    model = CNN2DResidualNetwork(
        in_channels=1,
        out_channels=1,
        hidden_channels=[16, 8, 16],
        n_blocks=2,
        kernel_size=3,
        activation_str="relu",
        normalization="layer",
    )

    x = torch.randn(2, 1, 32, 32)
    output = model(x)
    assert output.shape == (2, 1, 32, 32)


def test_device_placement():
    """Test model device placement"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = CNN2DResidualNetwork(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 8, 16],
            n_blocks=2,
            kernel_size=3,
            activation_str="relu",
            device=device,
        )

        x = torch.randn(1, 1, 32, 32, device=device)
        output = model(x)
        assert output.device == x.device
