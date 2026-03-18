from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

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
# Input/Output Normalizer Classes
#########################################
class input_normalizer_class(nn.Module):
    def __init__(self, input_normalizer) -> None:
        super(input_normalizer_class, self).__init__()
        self.input_normalizer = input_normalizer

    def forward(self, x):
        return self.input_normalizer.encode(x)


class output_denormalizer_class(nn.Module):
    def __init__(self, output_normalizer) -> None:
        super(output_denormalizer_class, self).__init__()
        self.output_normalizer = output_normalizer

    def forward(self, x):
        return self.output_normalizer.decode(x)


#########################################
# 2D CNN Network (Standard, No Residual Connections)
#########################################
class CNN2D(nn.Module):
    """
    Standard 2D Convolutional Neural Network .
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
        activation_str: str = "relu",
        padding: str = "same",
        include_grid: bool = False,
        device: torch.device = torch.device("cpu"),
        normalization: str = "none",  # "batch", "layer", or "none"
        dropout_rate: float = 0.0,
        example_input_normalizer=None,
        example_output_normalizer=None,
    ) -> None:
        """
        Initialize standard 2D CNN.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            hidden_channels (list[int]): List of hidden channel dimensions for each layer
            kernel_size (int): Convolution kernel size
            activation_str (str): Activation function name
            padding (str): Padding type
            include_grid (bool): Whether to include spatial grid as input
            device (torch.device): Device to run the model on
            normalization (str): Normalization type ('batch', 'layer', or 'none')
            dropout_rate (float): Dropout rate
            example_input_normalizer: Optional input normalizer
            example_output_normalizer: Optional output normalizer
        """
        super(CNN2D, self).__init__()

        assert len(hidden_channels) >= 1, "Hidden channels must have at least one element"
        assert normalization in [
            "batch",
            "layer",
            "none",
        ], "normalization must be 'batch', 'layer', or 'none'"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.padding = padding
        self.include_grid = include_grid
        
        # Adjust input channels if including grid
        self.in_channels = in_channels + 2 if include_grid else in_channels

        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else input_normalizer_class(example_input_normalizer)
        )

        # Build the convolutional network layer by layer
        modules = []

        # Input layer
        modules.append(
            nn.Conv2d(self.in_channels, hidden_channels[0], kernel_size, padding=padding)
        )

        if normalization == "batch":
            modules.append(nn.BatchNorm2d(hidden_channels[0]))
        elif normalization == "layer":
            modules.append(nn.GroupNorm(1, hidden_channels[0]))
        
        modules.append(activation_fun(activation_str))
        
        # todo: check if is useful add dropout here
        # if dropout_rate > 0:
        #     modules.append(nn.Dropout2d(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Conv2d(
                    hidden_channels[i],
                    hidden_channels[i + 1],
                    kernel_size,
                    padding=padding,
                    bias=(normalization != "batch"),
                )
            )
            
            if normalization == "batch":
                modules.append(nn.BatchNorm2d(hidden_channels[i + 1]))
            elif normalization == "layer":
                modules.append(nn.GroupNorm(1, hidden_channels[i + 1]))
            
            modules.append(activation_fun(activation_str))
            
            if dropout_rate > 0:
                modules.append(nn.Dropout2d(dropout_rate))

        self.conv_network = nn.Sequential(*modules)

        # Output projection layer
        self.output_layer = nn.Conv2d(
            hidden_channels[-1], out_channels, kernel_size, padding=padding
        )

        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        # Kaiming initialization
        self._init_weights(activation_str)

        # Move the model to the specified device
        self.to(device)

        # Enable JIT compilation for better performance if PyTorch version supports it
        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            self._enable_compilation()


    def _init_weights(self, activation_str) -> None:
        """
        Initialize weights using Kaiming initialization for better training dynamics.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation_str)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _enable_compilation(self) -> None:
        """Enable PyTorch 2.0+ compilation for performance if available."""
        try:
            self = torch.compile(self)
            print("PyTorch compilation enabled for better performance")
        except Exception as e:
            print(f"Could not enable PyTorch compilation: {e}")

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch height width {self.in_channels-2*self.include_grid}"]
    ) -> Float[Tensor, "batch height width {self.out_channels}"]:
        """
        Forward pass through the standard CNN.
        """
        x = self.input_normalizer(x)
        
        # Add grid if requested
        if self.include_grid:
            grid = self.get_grid_2d(x.shape).to(x.device)
            x = torch.cat((grid, x), dim=-1)

        x = x.permute(0, 3, 1, 2)  # (n_samples)*(channels)*(*n_x)
        x = self.conv_network(x)
        x = self.output_layer(x)
        x = x.permute(0, 2, 3, 1)  # (n_samples)*(*n_x)*(channels)

        return self.output_denormalizer(x)

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