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
def activation_fun(activation_str:str) -> nn.Module:
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
    elif activation_str == "silu":
        return nn.SiLU()
    else:
        raise ValueError("Not implemented activation function")


#########################################
# centered softmax and zero mean imposition
#########################################
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
# FeedForward Neural Network
#########################################
class FeedForwardNetwork(nn.Module):
    """
    Standard Feedforward Neural Network (Multi-Layer Perceptron)
    Similar structure to ResidualNetwork but without residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        activation_str: str,
        device: torch.device = torch.device("cpu"),
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        activation_on_output: bool = False,
        zero_mean: bool = False,
        example_input_normalizer=None,
        example_output_normalizer=None,
    ) -> None:
        """
        Initialize FeedForward Neural Network.

        Args:
            in_channels (int): Number of input features
            out_channels (int): Number of output features
            hidden_channels (list[int]): List of hidden layer dimensions
            activation_str (str): Activation function name
            device (torch.device): Device to run the model on
            layer_norm (bool): Whether to use layer normalization
            dropout_rate (float): Dropout rate (0 means no dropout)
            activation_on_output (bool): Whether to apply activation on output layer
            zero_mean (bool): Whether to impose zero mean constraint on output
            example_input_normalizer: Optional input normalizer
            example_output_normalizer: Optional output normalizer
        """
        super(FeedForwardNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        assert len(hidden_channels) >= 1, "Hidden channels must have at least one element"

        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else input_normalizer_class(example_input_normalizer)
        )

        # Build the network layer by layer
        modules = []

        # Input layer
        modules.append(nn.Linear(in_channels, hidden_channels[0]))
        if layer_norm:
            modules.append(nn.LayerNorm(hidden_channels[0]))
        modules.append(activation_fun(activation_str))
        if dropout_rate > 0:
            modules.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            modules.append(nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_channels[i + 1]))
            modules.append(activation_fun(activation_str))
            if dropout_rate > 0:
                modules.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*modules)

        # Output layer
        self.output_layer = nn.Linear(hidden_channels[-1], out_channels)
        self.output_layer_activation = (
            activation_fun(activation_str) if activation_on_output else nn.Identity()
        )

        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity()

        # Kaiming initialization
        try:
            self._init_weights(activation_str)
        except Exception as e:
            print(
                "Warning: Kaiming initialization failed. Using default initialization."
            )

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
        """
        Forward pass through the feedforward network.

        Args:
            x: Input tensor of shape (n_samples, in_channels)

        Returns:
            Output tensor of shape (n_samples, out_channels)
        """
        x = self.input_normalizer(x)
        x = self.network(x)
        x = self.output_layer_activation(self.output_layer(x))

        return self.post_processing(self.output_denormalizer(x))
