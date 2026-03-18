"""
Test suite for LazyKernelConv2d implementation in CNN2D_DON model.

This module tests that the lazy kernel size initialization allows proper
saving and loading of model state dictionaries.
"""

import os
import tempfile

import pytest
import torch
import sys

sys.path.append("..")
from architectures import CNN2D_DON


@pytest.fixture
def model_config():
    """Fixture providing default model configuration."""
    return {
        "in_channels": 3,
        "n_basis": 100,
        "hidden_channels": [16, 32, 64],
        "hidden_layers": [128, 128],
        "kernel_size": 3,
        "activation_str": "relu",
        "padding": 1,
        "stride": 2,
        "include_grid": False,
        "device": torch.device("cpu"),
        "normalization": "none",
        "dropout_rate": 0.0,
    }


@pytest.fixture
def dummy_input():
    """Fixture providing dummy input tensor."""
    batch_size = 4
    height = 64
    width = 64
    channels = 3
    return torch.randn(batch_size, height, width, channels)


@pytest.fixture
def initialized_model(model_config, dummy_input):
    """Fixture providing an initialized model (after first forward pass)."""
    model = CNN2D_DON(**model_config)
    # Run forward pass to initialize lazy layer
    _ = model(dummy_input)
    return model


class TestLazyKernelConv2d:
    """Test suite for LazyKernelConv2d functionality."""

    def test_lazy_layer_initialization(self, model_config, dummy_input):
        """Test that the lazy layer initializes correctly on first forward pass."""
        model = CNN2D_DON(**model_config)
        
        # Before forward pass, layer should have uninitialized parameters
        assert model.spatial_reduction_layer.has_uninitialized_params(), \
            "Spatial reduction layer should have uninitialized parameters before forward pass"
        assert model.spatial_reduction_layer._kernel_size is None, \
            "Kernel size should be None before initialization"
        
        # Run forward pass
        output = model(dummy_input)
        
        # After forward pass, layer should be initialized
        assert not model.spatial_reduction_layer.has_uninitialized_params(), \
            "Spatial reduction layer should be initialized after forward pass"
        assert model.spatial_reduction_layer._kernel_size is not None, \
            "Kernel size should be set after initialization"
        assert output.shape == (dummy_input.shape[0], model_config["n_basis"]), \
            f"Output shape should be (batch_size, n_basis), got {output.shape}"

    def test_kernel_size_matches_input(self, initialized_model, dummy_input):
        """Test that the kernel size matches the spatial dimensions after conv layers."""
        # Get the spatial dimensions after conv_network
        with torch.no_grad():
            x = dummy_input.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
            x = initialized_model.conv_network(x)
            expected_height, expected_width = x.shape[2], x.shape[3]
        
        kernel_size = initialized_model.spatial_reduction_layer._kernel_size
        assert kernel_size == (expected_height, expected_width), \
            f"Kernel size {kernel_size} should match spatial dimensions ({expected_height}, {expected_width})"

    def test_save_and_load(self, initialized_model, model_config, dummy_input):
        """Test that the model can be saved and loaded correctly."""
        # Get output from original model
        output1 = initialized_model(dummy_input)
        
        # Save model state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            temp_path = tmp_file.name
            torch.save(initialized_model.state_dict(), temp_path)
        
        try:
            # Create new model instance
            model2 = CNN2D_DON(**model_config)
            
            # Load saved state
            model2.load_state_dict(torch.load(temp_path, weights_only=True))
            
            # Verify the loaded model is initialized
            assert not model2.spatial_reduction_layer.has_uninitialized_params(), \
                "Loaded model should have initialized parameters"
            assert model2.spatial_reduction_layer._kernel_size is not None, \
                "Loaded model should have kernel size set"
            
            # Get output from loaded model
            output2 = model2(dummy_input)
            
            # Compare outputs
            assert torch.allclose(output1, output2, atol=1e-5), \
                "Outputs from original and loaded models should match"
            
        finally:
            # Clean up temporary file
            os.remove(temp_path)

    def test_output_shape_consistency(self, initialized_model, model_config):
        """Test that output shape is consistent across different input sizes."""
        # Test with different spatial dimensions
        test_inputs = [
            torch.randn(2, 32, 32, 3),
            torch.randn(2, 64, 64, 3),
            torch.randn(2, 128, 128, 3),
        ]
        
        for test_input in test_inputs:
            # Create a fresh model for each input size
            model = CNN2D_DON(**model_config)
            output = model(test_input)
            
            assert output.shape == (test_input.shape[0], model_config["n_basis"]), \
                f"Output shape should be (batch_size, n_basis) for input shape {test_input.shape}"

    def test_state_dict_contains_lazy_layer(self, initialized_model):
        """Test that the state dict contains the lazy layer parameters."""
        state_dict = initialized_model.state_dict()
        
        # Check that spatial reduction layer parameters are in state dict
        assert "spatial_reduction_layer.weight" in state_dict, \
            "State dict should contain spatial_reduction_layer.weight"
        assert "spatial_reduction_layer.bias" in state_dict, \
            "State dict should contain spatial_reduction_layer.bias"
        
        # Check that parameters have the correct shape
        weight = state_dict["spatial_reduction_layer.weight"]
        kernel_size = initialized_model.spatial_reduction_layer._kernel_size
        expected_shape = (
            initialized_model.hidden_channels[-1],
            initialized_model.hidden_channels[-1],
            kernel_size[0],
            kernel_size[1],
        )
        assert weight.shape == expected_shape, \
            f"Weight shape {weight.shape} should match expected shape {expected_shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
