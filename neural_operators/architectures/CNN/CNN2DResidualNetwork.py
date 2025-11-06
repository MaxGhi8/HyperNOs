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
    # elif activation_str == "gelu":
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
# 2D Convolutional Residual Block
#########################################
class Conv2DResidualBlock(nn.Module):
    """
    2D Convolutional Residual block for the CNN ResNet: x -> x + F(x)
    Where F is a convolutional neural network, with optional normalization and dropout.
    """

    def __init__(
        self,
        channels: list[int],
        kernel_size: int,
        activation_str: str,
        normalization: str = "none",  # "batch", "layer", or "none"
        dropout_rate: float = 0.0,
        padding: str = "same",
    ) -> None:
        super(Conv2DResidualBlock, self).__init__()

        assert normalization in [
            "batch",
            "layer",
            "none",
        ], "normalization must be 'batch', 'layer', or 'none'"
        assert (
            padding == "same"
        ), "Only 'same' padding is supported currently, in order to maintain spatial dimensions."
        assert (
            channels[0] == channels[-1]
        ), "Input and output channels must be the same for residual connection."

        self.channels = channels

        ## Construct the residual block
        modules = []
        for i in range(len(self.channels) - 1):
            # Convolutional layer
            modules.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size,
                    padding=padding,
                    bias=(normalization != "batch" and i < len(self.channels) - 2),
                )
            )

            # Activation function
            if i < len(self.channels) - 2:
                # Normalization layer
                if normalization == "batch":
                    modules.append(nn.BatchNorm2d(self.channels[i + 1]))
                elif normalization == "layer":
                    modules.append(
                        nn.GroupNorm(1, self.channels[i + 1])
                    )  # LayerNorm equivalent for 2D
                else:
                    modules.append(nn.Identity())

                modules.append(activation_fun(activation_str))

                # Add dropout
                if dropout_rate > 0:
                    modules.append(nn.Dropout2d(dropout_rate))

        self.res_block = nn.Sequential(*modules)

        # Kaiming initialization
        self._init_weights(activation_str)

    def _init_weights(self, activation_str) -> None:
        """
        Initialize weights using Kaiming initialization for better training dynamics.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation_str)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch {self.channels[0]} height width"]
    ) -> Float[Tensor, "batch {self.channels[-1]} height width"]:
        return x + self.res_block(x)


#########################################
# 2D CNN Residual Network
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


class CNN2DResidualNetwork(nn.Module):
    """
    2D Convolutional Residual Network for image-based neural operators

    Args:
        normalization (str): Type of normalization to use:
            - "batch": BatchNorm2d (standard for CNNs, normalizes across batch and spatial dims)
            - "layer": GroupNorm with 1 group (equivalent to LayerNorm for 2D, normalizes across channels)
            - "none": No normalization (uses bias in conv layers)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        kernel_size: int,
        activation_str: str,
        n_blocks: int,
        padding: str = "same",
        include_grid: bool = False,
        device: torch.device = torch.device("cpu"),
        normalization: str = "none",  # "batch", "layer", or "none"
        dropout_rate: float = 0.0,
        example_input_normalizer=None,
        example_output_normalizer=None,
    ) -> None:
        super(CNN2DResidualNetwork, self).__init__()

        assert n_blocks >= 0, "Number of blocks must be greater or equal to 0"
        assert normalization in [
            "batch",
            "layer",
            "none",
        ], "normalization must be 'batch', 'layer', or 'none'"
        assert padding == "same", "Only 'same' padding is supported currently."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.padding = padding
        self.include_grid = include_grid
        if self.include_grid:
            self.in_channels += 2  # for 2D grid

        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else input_normalizer_class(example_input_normalizer)
        )

        # Input projection layer normalization
        if normalization == "batch":
            input_norm = nn.BatchNorm2d(hidden_channels[0])
        elif normalization == "layer":
            input_norm = nn.GroupNorm(
                1, hidden_channels[0]
            )  # LayerNorm equivalent for 2D
        else:  # normalization == "none"
            input_norm = nn.Identity()

        # Input projection layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channels, hidden_channels[0], kernel_size, padding=self.padding
            ),
            input_norm,
            activation_fun(activation_str),
            # nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(), # todo: check if is useful add DropOut here
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                Conv2DResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    activation_str=activation_str,
                    normalization=normalization,
                    dropout_rate=dropout_rate,
                    padding=padding,
                )
                for _ in range(n_blocks)
            ]
        )

        # Output projection layer
        self.output_layer = nn.Conv2d(
            hidden_channels[-1], out_channels, kernel_size, padding="same"
        )

        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        # Kaiming initialization for input and output layers
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
        self,
        x: Float[Tensor, "batch height width {self.in_channels-2*self.include_grid}"],
    ) -> Float[Tensor, "batch height width {self.out_channels}"]:

        x = self.input_normalizer(x)

        if self.include_grid:
            grid = self.get_grid_2d(x.shape).to(x.device)
            x = torch.cat((grid, x), dim=-1)  # concatenate last dimension

        x = x.permute(0, 3, 1, 2)  # (n_samples)*(channels)*(*n_x)
        x = self.input_layer(x)
        x = self.residual_blocks(x)
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
