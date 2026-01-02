from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from ..FNN.FeedForwardNetwork import FeedForwardNetwork

torch.set_default_dtype(torch.float32)


#########################################
# activation function
#########################################
def activation_fun(activation_str):
    """
    Activation function to be used within the network.
    The function is the same throughout the network.
    """
    if activation_str == "relu":
        return nn.ReLU()
    # elif activation_str == "gelu": # todo: unsupported
    #     return nn.GELU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_str == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Not implemented activation function")


#########################################
# Lazy Kernel Size Conv2d Layer
#########################################
class LazyKernelConv2d(nn.Module):
    """
    A Conv2d layer where the kernel size is determined lazily during the first forward pass.
    
    This is similar to PyTorch's LazyConv2d, but instead of deferring the number of input channels,
    we defer the kernel size which will be set to match the spatial dimensions of the input.
    
    This allows the layer to adapt to the spatial dimensions of the input during the first forward pass,
    and properly save/load the state dict after initialization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        device: torch.device = None,
        nonlinearity: str = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.nonlinearity = nonlinearity
        
        # Use UninitializedParameter for weights and bias
        # These will be properly initialized during the first forward pass
        self.weight = UninitializedParameter(device=device)
        self.bias = UninitializedParameter(device=device)
        
        # Track whether the layer has been initialized
        self._kernel_size = None
        
    def initialize_parameters(self, spatial_height: int, spatial_width: int):
        """Initialize the weight and bias parameters based on the spatial dimensions."""
        if self.has_uninitialized_params():
            self._kernel_size = (spatial_height, spatial_width)
            
            # Initialize weight with proper shape
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    spatial_height,
                    spatial_width,
                    device=self.weight.device,
                )
            )
            
            # Initialize bias
            self.bias = nn.Parameter(
                torch.zeros(self.out_channels, device=self.bias.device)
            )
            
            # Apply Kaiming initialization
            nn.init.kaiming_normal_(self.weight, nonlinearity=self.nonlinearity)
            nn.init.zeros_(self.bias)
    
    def has_uninitialized_params(self) -> bool:
        """Check if the layer has uninitialized parameters."""
        return isinstance(self.weight, UninitializedParameter) or isinstance(
            self.bias, UninitializedParameter
        )
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom state dict loading to handle uninitialized parameters.
        
        If the layer is uninitialized and we're loading a state dict with initialized
        parameters, we need to materialize the parameters first.
        """
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        
        # Initialize the parameters if they are not
        if self.has_uninitialized_params() and weight_key in state_dict:
            loaded_weight = state_dict[weight_key]
            kernel_height, kernel_width = loaded_weight.shape[2], loaded_weight.shape[3]
            
            self._kernel_size = (kernel_height, kernel_width)
            self.weight = nn.Parameter(
                torch.empty_like(loaded_weight, device=self.weight.device)
            )
            self.bias = nn.Parameter(
                torch.empty_like(state_dict[bias_key], device=self.bias.device)
            )
        
        # Now call the parent's _load_from_state_dict to actually load the values
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. Initializes parameters on first call.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        if self.has_uninitialized_params():
            # Initialize based on input spatial dimensions
            _, _, spatial_height, spatial_width = x.shape
            self.initialize_parameters(spatial_height, spatial_width)
        
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        if self._kernel_size is not None:
            return (
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self._kernel_size}, stride={self.stride}, padding={self.padding}"
            )
        else:
            return (
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size=uninitialized, stride={self.stride}, padding={self.padding}"
            )


#########################################
# 2D CNN Branch Network for DeepONet
#########################################
class CNN2D_DON(nn.Module):
    """
    2D CNN Branch Network for DeepONet.

    This network processes 2D input functions and outputs basis coefficients.
    The final convolutional layer uses a kernel size equal to the spatial dimensions
    of the feature maps, reducing them to 1x1 before flattening to produce the
    basis coefficients for the DeepONet.
    """

    def __init__(
        self,
        in_channels: int,
        n_basis: int,
        hidden_channels: list[int],
        hidden_layers: list[int],
        kernel_size: int | list[int] = 3,
        activation_str: str = "relu",
        padding: int | list[int] = 0,
        stride: int | list[int] = 2,
        include_grid: bool = False,
        device: torch.device = torch.device("cpu"),
        normalization: str = "none",
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize 2D CNN Branch Network for DeepONet.

        Args:
            in_channels (int): Number of input channels
            n_basis (int): Number of basis functions (output dimension)
            hidden_channels (list[int]): List of hidden channel dimensions for each CNN layer
            hidden_layers (list[int]): List of hidden layer sizes for final FNN after flattening
            kernel_size (int | list[int]): Convolution kernel size for hidden layers
            activation_str (str): Activation function name
            padding (int | list[int]): Padding for hidden layers
            stride (int | list[int]): Stride length for hidden layers
            include_grid (bool): Whether to include spatial grid as input
            device (torch.device): Device to run the model on
            normalization (str): Normalization type ('batch', 'layer', or 'none')
            dropout_rate (float): Dropout rate
        """
        super(CNN2D_DON, self).__init__()

        assert (
            len(hidden_channels) >= 1
        ), "Hidden channels must have at least one element"
        assert normalization in [
            "batch",
            "layer",
            "none",
        ], "normalization must be 'batch', 'layer', or 'none'"

        self.in_channels = in_channels
        self.n_basis = n_basis
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, list)
            else [kernel_size] * len(hidden_channels)
        )
        self.stride = (
            stride if isinstance(stride, list) else [stride] * len(hidden_channels)
        )
        self.padding = (
            padding if isinstance(padding, list) else [padding] * len(hidden_channels)
        )
        self.include_grid = include_grid
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels + 2 if include_grid else in_channels
        self._activation_str = activation_str
        self.activation = activation_fun(activation_str)
        modules = []

        ## Input layer
        modules.append(
            nn.Conv2d(
                self.in_channels,
                hidden_channels[0],
                self.kernel_size[0],
                padding=self.padding[0],
                stride=self.stride[0],
            )
        )

        if normalization == "batch":
            modules.append(nn.BatchNorm2d(hidden_channels[0]))
        elif normalization == "layer":
            modules.append(nn.GroupNorm(1, hidden_channels[0]))

        modules.append(self.activation)

        # todo: check if is useful add dropout here
        # if dropout_rate > 0:
        #     modules.append(nn.Dropout2d(dropout_rate))

        ## Hidden layers
        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Conv2d(
                    hidden_channels[i],
                    hidden_channels[i + 1],
                    kernel_size=self.kernel_size[i + 1],
                    padding=self.padding[i + 1],
                    stride=self.stride[i + 1],
                    bias=(normalization != "batch"),
                )
            )

            if normalization == "batch":
                modules.append(nn.BatchNorm2d(hidden_channels[i + 1]))
            elif normalization == "layer":
                modules.append(nn.GroupNorm(1, hidden_channels[i + 1]))

            modules.append(self.activation)

            if dropout_rate > 0:
                modules.append(nn.Dropout2d(dropout_rate))

        self.conv_network = nn.Sequential(*modules)

        ## Final convolutional layer with dynamic kernel size (reduce dim to 1x1)
        # Kernel size will be determined in first forward pass using LazyKernelConv2d
        self.spatial_reduction_layer = LazyKernelConv2d(
            in_channels=self.hidden_channels[-1],
            out_channels=self.hidden_channels[-1],
            stride=1,
            padding=0,
            device=device,
            nonlinearity=self._activation_str,
        )

        ## Final FNN layers after flattening
        self.fnn = FeedForwardNetwork(
            in_channels=self.hidden_channels[-1],
            out_channels=self.n_basis,
            hidden_channels=self.hidden_layers,
            activation_str=self._activation_str,
            device=device,
            layer_norm=(self.normalization == "layer"),
            dropout_rate=self.dropout_rate,
            activation_on_output=False,
            zero_mean=False,
            example_input_normalizer=None,
            example_output_normalizer=None,
        )

        self.to(device)
        self.device = device



    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch height width {self.in_channels-2*self.include_grid}"],
    ) -> Float[Tensor, "batch {self.n_basis}"]:
        """
        Forward pass through the CNN branch network.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Basis coefficients of shape (batch, n_basis)
        """
        # Add grid if requested
        if self.include_grid:
            grid = self.get_grid_2d(x.shape).to(x.device)
            x = torch.cat((grid, x), dim=-1)

        # Permute to (batch, channels, height, width) for Conv2d
        x = x.permute(0, 3, 1, 2)

        # Pass through convolutional layers
        x = self.conv_network(x)

        # Apply spatial reduction layer (reduce to 1x1)
        # The LazyKernelConv2d will initialize on first forward pass
        x = self.spatial_reduction_layer(x)
        x = self.activation(x)
        x = x.flatten(start_dim=1)

        # Pass through FeedForward network
        x = self.fnn(x)

        return x

    @cache
    def get_grid_2d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # grid for x
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # grid for y
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)
