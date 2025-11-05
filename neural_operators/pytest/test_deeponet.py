import torch
import pytest
import sys
sys.path.append('..')

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
    "dropout_rate": 0.1
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
# branch_hyperparams_2d = {
#     "n_inputs": 1,
#     "n_points": [32, 32],
#     "channels_conv": [16, 32],
#     "stride": [2, 2],
#     "kernel_size": [3, 3],
#     "hidden_layer": [128, 64],
#     "act_fun": "relu",
#     "residual": False,
#     "output_dim_conv": 512
# }

# trunck_hyperparams_2d = {
#     "n_inputs": 2,
#     "hidden_layer": [64, 64],
#     "act_fun": "relu",
#     "residual": False
# }

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
        dim=dim
    )
    
    assert model.n_basis == n_basis
    assert model.n_output == n_output
    assert model.dim == dim
    assert hasattr(model, 'branch_NN')
    assert hasattr(model, 'trunk_NN')

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
        dim=dim
    )
    
    assert model.n_basis == n_basis
    assert model.n_output == n_output
    assert model.output_division == n_basis // n_output

def test_deeponet_1d_forward_single_output():
    """Test 1D DeepONet forward pass with single output"""
    batch_size = 4
    n_points_trunk = 10 # Number of evaluation points for trunk network
    n_basis = 32
    n_output = 1
    
    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1
    )
    
    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])
    
    # Forward pass
    output = model(branch_input, trunk_input)
    
    # Check output shape
    expected_shape = (batch_size, n_points_trunk)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_deeponet_1d_forward_multi_output():
    """Test 1D DeepONet forward pass with multiple outputs"""
    batch_size = 4
    n_points_trunk = 10 # Number of evaluation points for trunk network
    n_basis = 64
    n_output = 4
    
    model = DeepONet(
        branch_hyperparameters=branch_hyperparams_1d,
        trunk_hyperparameters=trunk_hyperparams_1d,
        n_basis=n_basis,
        n_output=n_output,
        dim=1
    )
    
    # Create input data
    branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    trunk_input = torch.randn(n_points_trunk, trunk_hyperparams_1d["n_inputs"])

    # Forward pass
    output = model(branch_input, trunk_input)
    
    # Check output shape
    expected_shape = (batch_size, n_points_trunk, n_output)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

# def test_deeponet_2d_single_output():
#     """Test 2D DeepONet forward pass with single output"""
#     batch_size = 2
#     n_basis = 32
#     n_output = 1
    
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_2d,
#         trunk_hyperparameters=trunck_hyperparams_2d,
#         n_basis=n_basis,
#         n_output=n_output,
#         dim=2
#     )
    
#     # Create input data
#     # Branch input: 2D field data
#     h, w = branch_hyperparams_2d["n_points"]
#     branch_input = torch.randn(batch_size, branch_hyperparams_2d["n_inputs"], h, w)
    
#     # Trunk input: 2D coordinates
#     trunk_input = torch.randn(batch_size, trunck_hyperparams_2d["n_inputs"])
    
#     # Forward pass
#     output = model(branch_input, trunk_input)
    
#     # Check output shape for 2D single output
#     expected_shape = (batch_size, h, w)
#     assert output.shape == expected_shape
#     assert not torch.isnan(output).any()
#     assert not torch.isinf(output).any()

# def test_deeponet_2d_multi_output():
#     """Test 2D DeepONet forward pass with multiple outputs"""
#     batch_size = 2
#     n_basis = 64
#     n_output = 2
    
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_2d,
#         trunk_hyperparameters=trunck_hyperparams_2d,
#         n_basis=n_basis,
#         n_output=n_output,
#         dim=2
#     )
    
#     # Create input data
#     h, w = branch_hyperparams_2d["n_points"]
#     branch_input = torch.randn(batch_size, branch_hyperparams_2d["n_inputs"], h, w)
#     trunk_input = torch.randn(batch_size, trunck_hyperparams_2d["n_inputs"])
    
#     # Forward pass
#     output = model(branch_input, trunk_input)
    
#     # Check output shape for 2D multi output
#     expected_shape = (batch_size, h, w, n_output)
#     assert output.shape == expected_shape
#     assert not torch.isnan(output).any()
#     assert not torch.isinf(output).any()

# def test_residual_networks():
#     """Test DeepONet with residual branch and trunk networks"""
#     # Test with residual=True
#     branch_params = branch_hyperparams_1d.copy()
#     trunk_params = trunk_hyperparams_1d.copy()
#     branch_params["residual"] = True
#     trunk_params["residual"] = True
    
#     model = DeepONet(
#         branch_hyperparameters=branch_params,
#         trunk_hyperparameters=trunk_params,
#         n_basis=32,
#         n_output=1,
#         dim=1
#     )
    
#     batch_size = 4
#     branch_input = torch.randn(batch_size, branch_params["n_inputs"])
#     trunk_input = torch.randn(batch_size, trunk_params["n_inputs"])
    
#     output = model(branch_input, trunk_input)
#     assert output.shape == (batch_size, branch_params["n_inputs"])

# def test_different_activation_functions():
#     """Test DeepONet with different activation functions"""
#     activations = ["relu", "gelu", "tanh", "leaky_relu", "silu"]
    
#     for act_fun in activations:
#         branch_params = branch_hyperparams_1d.copy()
#         trunk_params = trunk_hyperparams_1d.copy()
#         branch_params["act_fun"] = act_fun
#         trunk_params["act_fun"] = act_fun
        
#         model = DeepONet(
#             branch_hyperparameters=branch_params,
#             trunk_hyperparameters=trunk_params,
#             n_basis=16,
#             n_output=1,
#             dim=1
#         )
        
#         batch_size = 2
#         branch_input = torch.randn(batch_size, branch_params["n_inputs"])
#         trunk_input = torch.randn(batch_size, trunk_params["n_inputs"])
        
#         output = model(branch_input, trunk_input)
#         assert not torch.isnan(output).any(), f"NaN values with {act_fun} activation"

# def test_n_basis_divisibility_assertion():
#     """Test that n_basis must be divisible by n_output"""
#     with pytest.raises(AssertionError, match="n_basis must be divisible by n_output"):
#         DeepONet(
#             branch_hyperparameters=branch_hyperparams_1d,
#             trunk_hyperparameters=trunk_hyperparams_1d,
#             n_basis=31,  # Not divisible by n_output=4
#             n_output=4,
#             dim=1
#         )

# def test_gradient_flow():
#     """Test that gradients flow through the network"""
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_1d,
#         trunk_hyperparameters=trunk_hyperparams_1d,
#         n_basis=32,
#         n_output=1,
#         dim=1
#     )
    
#     batch_size = 4
#     branch_input = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"], requires_grad=True)
#     trunk_input = torch.randn(batch_size, trunk_hyperparams_1d["n_inputs"], requires_grad=True)
#     target = torch.randn(batch_size, branch_hyperparams_1d["n_inputs"])
    
#     output = model(branch_input, trunk_input)
#     loss = torch.nn.functional.mse_loss(output, target)
#     loss.backward()
    
#     # Check that gradients exist
#     assert branch_input.grad is not None
#     assert trunk_input.grad is not None
#     assert branch_input.grad.abs().sum() > 0
#     assert trunk_input.grad.abs().sum() > 0

# def test_parameter_count():
#     """Test parameter counting"""
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_1d,
#         trunk_hyperparameters=trunk_hyperparams_1d,
#         n_basis=64,
#         n_output=1,
#         dim=1
#     )
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     assert total_params > 0
#     assert trainable_params == total_params
#     print(f"Total parameters: {total_params:,}")

# def test_device_consistency():
#     """Test that all model components are on the correct device"""
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_1d,
#         trunk_hyperparameters=trunk_hyperparams_1d,
#         n_basis=32,
#         n_output=1,
#         dim=1
#     )
    
#     # Check that model is on expected device
#     model_device = next(model.parameters()).device
#     assert model_device == device

# def test_output_shape_computation():
#     """Test the _compute_output_shape method"""
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_1d,
#         trunk_hyperparameters=trunk_hyperparams_1d,
#         n_basis=32,
#         n_output=2,
#         dim=1
#     )
    
#     batch_size = 8
#     expected_shape = (batch_size, branch_hyperparams_1d["n_inputs"], 2)
#     computed_shape = model._compute_output_shape(batch_size)
#     assert computed_shape == expected_shape

# def test_unsupported_dimension_error():
#     """Test that unsupported dimensions raise an error"""
#     model = DeepONet(
#         branch_hyperparameters=branch_hyperparams_1d,
#         trunk_hyperparameters=trunk_hyperparams_1d,
#         n_basis=32,
#         n_output=1,
#         dim=3  # Unsupported dimension
#     )
    
#     # This should work for initialization, but fail during shape computation
#     with pytest.raises(ValueError, match="Unsupported dimension: 3"):
#         model._compute_output_shape(4)

# def test_different_network_sizes():
#     """Test with different network architectures"""
#     architectures = [
#         [32],
#         [64, 32],
#         [128, 64, 32],
#         [256, 128, 64, 32]
#     ]
    
#     for hidden_layers in architectures:
#         branch_params = branch_hyperparams_1d.copy()
#         trunk_params = trunk_hyperparams_1d.copy()
#         branch_params["hidden_layer"] = hidden_layers
#         trunk_params["hidden_layer"] = hidden_layers
        
#         model = DeepONet(
#             branch_hyperparameters=branch_params,
#             trunk_hyperparameters=trunk_params,
#             n_basis=32,
#             n_output=1,
#             dim=1
#         )
        
#         batch_size = 2
#         branch_input = torch.randn(batch_size, branch_params["n_inputs"])
#         trunk_input = torch.randn(batch_size, trunk_params["n_inputs"])
        
#         output = model(branch_input, trunk_input)
#         assert output.shape == (batch_size, branch_params["n_inputs"])

