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
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.hidden_channels = hidden_channels

        modules = []
        for i in range(len(self.hidden_channels) - 1):
            # Affine layer
            modules.append(
                nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1])
            )

            # Layer normalization
            if layer_norm:
                modules.append(nn.LayerNorm(self.hidden_channels[i + 1]))

            # Activation function (except for the last layer)
            if i < len(self.hidden_channels) - 2:
                modules.append(activation_fun(activation_str))

                # Add dropout if specified
                if dropout_rate > 0:
                    modules.append(nn.Dropout(dropout_rate))

        self.res_block = nn.Sequential(*modules)

        # Kaiming initialization
        try:
            self._init_weights(activation_str)
        except Exception as e:
            print(
                "Warning: Kaiming initialization failed. Using default initialization."
            )

    def _init_weights(self, activation_str) -> None:
        """
        Initialize weights using Kaiming initialization for better training dynamics.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation_str)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.hidden_channels[0]}"]
    ) -> Float[Tensor, "n_samples {self.hidden_channels[-1]}"]:
        return x + self.res_block(x)


#########################################
# Residual Network
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
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        zero_mean: bool = False,
        example=None,
    ) -> None:
        super(ResidualNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert (
            hidden_channels[-1] == hidden_channels[0]
        ), "The input and output dimensions must be the same for being concatenated"
        assert n_blocks >= 0, "Number of layers must be greater or equal to 0"

        self.input_normalizer = (
            nn.Identity()
            if example is None
            else input_normalizer_class(example.input_normalizer)
        )

        self.input_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]) if layer_norm else nn.Identity(),
            activation_fun(activation_str),
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_channels,
                    activation_str,
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_blocks)
            ]
        )

        self.output_layer = nn.Linear(hidden_channels[-1], out_channels)

        self.output_denormalizer = (
            nn.Identity()
            if example is None
            else output_denormalizer_class(example.output_normalizer)
        )

        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity

        # Kaiming initialization
        # try:
        #     self._init_weights(activation_str)
        # except Exception as e:
        #     print(
        #         "Warning: Kaiming initialization failed. Using default initialization."
        #     )

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
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation_str)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _enable_compilation(self) -> None:
        """Enable PyTorch 2.0+ compilation for performance if available."""
        try:
            # This is a PyTorch 2.0+ feature
            self = torch.compile(self)
            print("PyTorch compilation enabled for better performance")
        except Exception as e:
            print(f"Could not enable PyTorch compilation: {e}")

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.in_channels}"]
    ) -> Float[Tensor, "n_samples {self.out_channels}"]:

        x = self.input_layer(self.input_normalizer(x))
        x = self.residual_blocks(x)
        x = self.output_layer(x)

        return self.post_processing(self.output_denormalizer(x))
