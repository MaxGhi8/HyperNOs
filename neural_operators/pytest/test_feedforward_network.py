import torch
import pytest

import sys
sys.path.append('..')

from architectures import (
    FeedForwardNetwork,
    activation_fun,
    centered_softmax,
    zero_mean_imposition,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestFeedForwardNetwork:
    """Test cases for FeedForward Neural Network"""

    def test_device_placement(self):
        """Test model device placement"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[32],
            activation_str="relu",
            device=device
        )
        
        x = torch.randn(4, 10, device=device)
        output = model(x)
        
        # Use .type to compare device types (cuda vs cpu) without comparing indices
        assert output.device.type == device.type == x.device.type
        assert model.device.type == device.type == x.device.type

    def test_simple_fnn_initialization(self):
        """Test basic FNN initialization"""
        in_channels = 10
        out_channels = 5
        hidden_channels = [64, 32]
        
        model = FeedForwardNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation_str="relu",
            device=device
        )
        
        assert model.in_channels == in_channels
        assert model.out_channels == out_channels
        assert model.hidden_channels == hidden_channels

    def test_fnn_forward_pass(self):
        """Test forward pass through the network"""
        batch_size = 32
        in_channels = 10
        out_channels = 5
        hidden_channels = [64, 32]
        
        model = FeedForwardNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation_str="relu",
            device=device
        )
        
        x = torch.randn(batch_size, in_channels).to(device)
        output = model(x)
        
        assert output.shape == (batch_size, out_channels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_activations(self):
        """Test different activation functions"""
        activations = ["relu", "gelu", "tanh", "leaky_relu", "sigmoid", "silu"]
        
        for activation in activations:
            model = FeedForwardNetwork(
                in_channels=10,
                out_channels=5,
                hidden_channels=[32, 16],
                activation_str=activation,
                device=device
            )

            x = torch.randn(8, 10).to(device)
            output = model(x)
            assert output.shape == (8, 5)
            assert not torch.isnan(output).any()

    def test_with_layer_norm(self):
        """Test FNN with layer normalization"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[64, 32],
            activation_str="relu",
            layer_norm=True,
            device=device
        )
        
        x = torch.randn(16, 10).to(device)
        output = model(x)
        assert output.shape == (16, 5)

    def test_with_dropout(self):
        """Test FNN with dropout"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[64, 32],
            activation_str="relu",
            dropout_rate=0.2,
            device=device
        )

        x = torch.randn(16, 10).to(device)
        output = model(x)
        assert output.shape == (16, 5)

    def test_activation_on_output(self):
        """Test FNN with activation on output layer"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[32],
            activation_str="relu",
            activation_on_output=True,
            device=device
        )
        
        x = torch.randn(8, 10).to(device)
        output = model(x)
        assert output.shape == (8, 5)
        # With ReLU on output, all values should be non-negative
        assert (output >= 0).all()

    def test_zero_mean_output(self):
        """Test zero mean constraint on output"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[32],
            activation_str="relu",
            zero_mean=True,
            device=device
        )

        x = torch.randn(16, 10).to(device)
        output = model(x)
        
        # Check that mean is close to zero for each sample
        means = output.mean(dim=1)
        assert torch.allclose(means, torch.zeros_like(means).to(device), atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[32, 16],
            activation_str="relu",
            device=device
        )
        
        x = torch.randn(8, 10, requires_grad=True).to(device)
        target = torch.randn(8, 5).to(device)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check that gradients exist and are not zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_different_architectures(self):
        """Test with different network architectures"""
        architectures = [
            [32],
            [64, 32],
            [128, 64, 32],
            [256, 128, 64, 32]
        ]
        
        for hidden in architectures:
            model = FeedForwardNetwork(
                in_channels=10,
                out_channels=5,
                hidden_channels=hidden,
                activation_str="relu",
                device=device
            )
            
            x = torch.randn(4, 10).to(device)
            output = model(x)
            assert output.shape == (4, 5)

    def test_parameter_count(self):
        """Test parameter counting"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[64, 32],
            activation_str="relu",
            device=device
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params


    def test_activation_functions(self):
        """Test all available activation functions"""
        for act_str in ["relu", "gelu", "tanh", "leaky_relu", "sigmoid", "silu"]:
            act = activation_fun(act_str)
            assert act is not None

    def test_centered_softmax(self):
        """Test centered softmax function"""
        x = torch.randn(8, 10)
        output = centered_softmax(x)
        
        # Check that sum is close to zero for each sample
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-6)

    def test_zero_mean_imposition(self):
        """Test zero mean imposition function"""
        x = torch.randn(8, 10)
        output = zero_mean_imposition(x)
        
        # Check that mean is close to zero for each sample
        means = output.mean(dim=1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)

    def test_batch_independence(self):
        """Test that batch samples are processed independently"""
        model = FeedForwardNetwork(
            in_channels=10,
            out_channels=5,
            hidden_channels=[32],
            activation_str="relu",
            device=device
        )

        x1 = torch.randn(1, 10).to(device)
        x2 = torch.randn(1, 10).to(device)
        x_batch = torch.cat([x1, x2], dim=0)
        
        # Forward pass for individual samples
        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)
        
        # Check that batch output matches individual outputs
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
