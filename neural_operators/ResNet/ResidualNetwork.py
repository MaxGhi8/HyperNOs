"""
This file contains all the core architectures and modules of the Fourier Neural Operator (FNO) for 1D case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

torch.set_default_dtype(torch.float32)  # default tensor dtype


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
    elif activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_str == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Not implemented activation function")


@jaxtyped(typechecker=beartype)
def centered_softmax(x: Float[Tensor, "n_samples n"]) -> Float[Tensor, "n_samples n"]:
    """
    Centered softmax function to be used within the network.
    """
    return F.softmax(x, dim=1) - 1 / x.shape[1]


@jaxtyped(typechecker=beartype)
def zero_mean_imposition(
    x: Float[Tensor, "n_samples n"],
) -> Float[Tensor, "n_samples n"]:
    """
    Take a vector and impose the zero mean constraint.
    """
    return x - x.mean(dim=1, keepdim=True)


#########################################
# Residual Block
#########################################
class ResidualBlock(nn.Module):
    """
    Residual block for the ResNet
    """

    def __init__(
        self,
        hidden_channels: list[int],
        activation_str: str,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.hidden_channels = hidden_channels

        modules = []
        for i in range(len(self.hidden_channels) - 1):
            # Affine layer
            modules.append(
                nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1])
            )

            # Activation function
            modules.append(activation_fun(activation_str))

        self.res_block = nn.Sequential(*modules)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.hidden_channels[0]}"]
    ) -> Float[Tensor, "n_samples {self.hidden_channels[-1]}"]:
        return self.res_block(x)


#########################################
# Residual Network
#########################################
class ResidualNetwork(nn.Module):
    """
    Residual Network for the Fourier Neural Operator
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        activation_str: str,
        n_blocks: int,
        device: torch.device = torch.device("cpu"),
        zero_mean: bool = False,
        input_normalizer=nn.Identity(),
        output_denormalizer=nn.Identity(),
    ) -> None:
        super(ResidualNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert (
            hidden_channels[-1] == hidden_channels[0]
        ), "The input and output dimensions must be the same for being concatenated"
        assert n_blocks >= 0, "Number of layers must be greater or equal to 0"

        self.input_normalizer = lambda x: input_normalizer(x)

        self.input_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0]), activation_fun(activation_str)
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels, activation_str) for _ in range(n_blocks)]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels[-1], out_channels), activation_fun(activation_str)
        )

        self.output_denormalizer = lambda x: output_denormalizer(x)

        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity

        self.to(device)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.in_channels}"]
    ) -> Float[Tensor, "n_samples {self.out_channels}"]:

        x = self.input_layer(self.input_normalizer(x))

        for res_block in self.residual_blocks:
            x = x + res_block(x)

        x = self.output_layer(x)

        return zero_mean_imposition(self.output_denormalizer(x))
