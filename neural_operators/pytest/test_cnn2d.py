import torch
import pytest
import sys
sys.path.append("..")

from architectures import CNN2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCNN2D:
    """Test cases for standard 2D CNN (without residual connections)"""

    def test_cnn2d_initialization(self):
        """Test basic CNN2D initialization"""
        in_channels = 3
        out_channels = 1
        hidden_channels = [16, 32]
        
        model = CNN2D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            activation_str="relu",
            device=device
        )
        
        assert model.in_channels == in_channels
        assert model.out_channels == out_channels
        assert model.hidden_channels == hidden_channels

    def test_cnn2d_forward_pass(self):
        """Test forward pass through CNN2D"""
        batch_size, in_channels, out_channels = 4, 3, 1
        height, width = 64, 64
        
        model = CNN2D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=[16, 32],
            kernel_size=3,
            activation_str="relu",
            device=device
        )

        x = torch.randn(batch_size, height, width, in_channels).to(device)
        output = model(x)

        assert output.shape == (batch_size, height, width, out_channels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_input_sizes(self):
        """Test that the model works with different input sizes"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            kernel_size=3,
            device=device
        )
        
        # Test different sizes
        sizes = [(32, 32), (64, 64), (128, 128), (48, 96)]
        
        for height, width in sizes:
            x = torch.randn(1, height, width, 1).to(device)
            output = model(x)
            assert output.shape == (1, height, width, 1)

    def test_different_activations(self):
        """Test different activation functions"""
        activations = ["relu", "tanh", "leaky_relu", "sigmoid"]
        
        for activation in activations:
            model = CNN2D(
                in_channels=2,
                out_channels=2,
                hidden_channels=[16, 32],
                activation_str=activation,
                device=device
            )
            
            x = torch.randn(1, 32, 32, 2).to(device)
            output = model(x)
            assert output.shape == (1, 32, 32, 2)
            assert not torch.isnan(output).any()

    def test_batch_normalization(self):
        """Test CNN2D with batch normalization"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            normalization="batch",
            device=device
        )
        
        x = torch.randn(4, 32, 32, 1).to(device)
        output = model(x)
        assert output.shape == (4, 32, 32, 1)

    def test_layer_normalization(self):
        """Test CNN2D with layer normalization (GroupNorm)"""
        model = CNN2D(
            in_channels=3,
            out_channels=2,
            hidden_channels=[16, 32],
            normalization="layer",
            device=device
        )

        x = torch.randn(4, 32, 32, 3).to(device)
        output = model(x)
        assert output.shape == (4, 32, 32, 2)

    def test_with_dropout(self):
        """Test CNN2D with dropout"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            dropout_rate=0.2,
            device=device
        )

        x = torch.randn(4, 32, 32, 1).to(device)
        output = model(x)
        assert output.shape == (4, 32, 32, 1)

    def test_with_grid(self):
        """Test CNN2D with spatial grid inclusion"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            include_grid=True,
            device=device
        )

        x = torch.randn(4, 32, 32, 1).to(device)
        output = model(x)
        assert output.shape == (4, 32, 32, 1)

    def test_grid_generation(self):
        """Test that grid is generated correctly"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16],
            include_grid=True,
            device=device
        )
        
        batch_size, height, width = 2, 32, 32
        x = torch.randn(batch_size, height, width, 1).to(device)
        
        # Get grid
        grid = model.get_grid_2d(x.shape)

        # Check grid shape: (batch_size, height, width, 2)
        assert grid.shape == (batch_size, height, width, 2)
        
        # Check that grid values are in [0, 1]
        assert grid.min() >= 0
        assert grid.max() <= 1

    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            device=device
        )
        
        x = torch.randn(2, 32, 32, 1).to(device)
        target = torch.randn(2, 32, 32, 1).to(device)

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
            [16],
            [32, 16],
            [64, 32, 16],
            [128, 64, 32, 16]
        ]
        
        for hidden in architectures:
            model = CNN2D(
                in_channels=6,
                out_channels=16,
                hidden_channels=hidden,
                device=device
            )

            x = torch.randn(2, 32, 32, 6).to(device)
            output = model(x)
            assert output.shape == (2, 32, 32, 16)

    def test_device_consistency(self):
        """Test that all model components are on the correct device"""
        model = CNN2D(
            in_channels=1,
            out_channels=1,
            hidden_channels=[16, 32],
            device=device
        )
        
        # Check that model parameters are on expected device
        model_device = next(model.parameters()).device
        assert model_device.type == device.type

    def test_kernel_size_variations(self):
        """Test different kernel sizes"""
        kernel_sizes = [3, 5, 7]
        
        for k in kernel_sizes:
            model = CNN2D(
                in_channels=1,
                out_channels=1,
                hidden_channels=[16, 32],
                kernel_size=k,
                device=device
            )

            x = torch.randn(2, 32, 32, 1).to(device)
            output = model(x)
            assert output.shape == (2, 32, 32, 1)
