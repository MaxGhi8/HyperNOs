from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

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
# 2D CNN Branch Network for DeepONet
#########################################
class CNN2D_Branch(nn.Module):
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
        super(CNN2D_Branch, self).__init__()

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
        # Kernel size will be determined in first forward pass
        self.spatial_reduction_layers = None

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

    def _get_or_create_spatial_reduction_layer(
        self, spatial_height: int, spatial_width: int, num_channels: int
    ) -> nn.Conv2d:
        """
        Get or create the spatial reduction layer for the given spatial dimensions.
        I suppose there is only one spatial reduction possibility, i.e. along the dataset there is the same value for spatial dimensions.
        """
        if self.spatial_reduction_layers is None:
            layer = nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=(spatial_height, spatial_width),
                stride=1,
                padding=0,
            )

            nn.init.kaiming_normal_(layer.weight, nonlinearity=self._activation_str)
            nn.init.zeros_(layer.bias)

            layer = layer.to(self.device)

            self.spatial_reduction_layers = layer

        return self.spatial_reduction_layers

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
        x = self.input_normalizer(x)

        # Add grid if requested
        if self.include_grid:
            grid = self.get_grid_2d(x.shape).to(x.device)
            x = torch.cat((grid, x), dim=-1)

        # Permute to (batch, channels, height, width) for Conv2d
        x = x.permute(0, 3, 1, 2)

        # Pass through convolutional layers
        x = self.conv_network(x)

        # Get or create spatial reduction layer for this input size (reduce to 1x1)
        _, num_channels, spatial_height, spatial_width = x.shape
        spatial_reduction_layer = self._get_or_create_spatial_reduction_layer(
            spatial_height, spatial_width, num_channels
        )
        x = spatial_reduction_layer(x)
        x = self.activation(x)
        x = x.squeeze()

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
