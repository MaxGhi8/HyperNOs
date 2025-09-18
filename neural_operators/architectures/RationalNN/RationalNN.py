import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

torch.set_default_dtype(torch.float32)  # default tensor dtype


#########################################
# Rational Activation Function
#########################################
class RationalActivation(nn.Module):
    """
    Rational activation function: f(x) = P(x) / Q(x)
    where P(x) and Q(x) are polynomials.
    """

    def __init__(
        self,
        activation_str: str = "relu",
        degrees: tuple[int, int] = (3, 3),
        cuda=True,
    ):
        """
        Initialize rational activation function.

        Args:
            activation_str: Function to approximate initially ("relu", "gelu", "swish", "tanh")
            degrees: Tuple of (numerator_degree, denominator_degree)
            cuda: Whether to use CUDA
        """
        super(RationalActivation, self).__init__()

        self.degrees = degrees
        assert self.degrees[0] >= 3, "Numerator degree must be at least 3"
        assert self.degrees[1] >= 2, "Denominator degree must be at least 2"
        self.numerator_degree, self.denominator_degree = degrees

        # Initialize coefficients
        self.numerator = nn.Parameter(torch.randn(self.numerator_degree + 1))
        self.denominator = nn.Parameter(torch.randn(self.denominator_degree + 1))

        # Initialize with approximation to common functions
        self._init_coefficients(activation_str)

        if cuda:
            self.cuda()

    def _init_coefficients(self, activation_str: str) -> None:
        """Initialize coefficients to approximate common activation functions."""
        with torch.no_grad():
            if activation_str == "relu":
                self.numerator.data = torch.tensor(
                    [0.0, 1.0] + [0.0] * (self.numerator_degree - 1)
                )[: self.numerator_degree + 1]
                self.denominator.data = torch.tensor(
                    [2.0, 0.0, 1.0] + [0.0] * (self.denominator_degree - 2)
                )[: self.denominator_degree + 1]

            elif activation_str == "gelu":
                self.numerator.data = torch.tensor(
                    [0.0, 0.797, 0.0, 0.044] + [0.0] * (self.numerator_degree - 3)
                )[: self.numerator_degree + 1]
                self.denominator.data = torch.tensor(
                    [1.0, 0.0, 0.372] + [0.0] * (self.denominator_degree - 2)
                )[: self.denominator_degree + 1]

            elif activation_str == "swish":
                self.numerator.data = torch.tensor(
                    [0.0, 1.0] + [0.0] * (self.numerator_degree - 1)
                )[: self.numerator_degree + 1]
                self.denominator.data = torch.tensor(
                    [1.78, 0.0, 1.0] + [0.0] * (self.denominator_degree - 2)
                )[: self.denominator_degree + 1]

            elif activation_str == "tanh":
                self.numerator.data = torch.tensor(
                    [0.0, 1.0, 0.0, -0.33] + [0.0] * (self.numerator_degree - 3)
                )[: self.numerator_degree + 1]
                self.denominator.data = torch.tensor(
                    [1.0, 0.0, 0.33] + [0.0] * (self.denominator_degree - 2)
                )[: self.denominator_degree + 1]

            else:
                nn.init.normal_(self.numerator, mean=0.0, std=1.0)
                nn.init.normal_(self.denominator, mean=0.0, std=1.0)
                self.denominator.data[0] = torch.abs(self.denominator.data[0]) + 1e-6

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "n_samples n"]) -> Float[Tensor, "n_samples n"]:
        """
        Forward pass of rational activation function.
        """
        # Compute polynomial powers from x**0 to x**(max(self.numerator_degree, self.denominator_degree))
        x_powers = [torch.ones_like(x)]
        for i in range(1, max(self.numerator_degree, self.denominator_degree) + 1):
            x_powers.append(x_powers[-1] * x)

        # Compute numerator
        numerator = torch.zeros_like(x)
        for i, coeff in enumerate(self.numerator):
            numerator += coeff * x_powers[i]

        # Compute denominator
        denominator = torch.zeros_like(x)
        for i, coeff in enumerate(self.denominator):
            denominator += coeff * x_powers[i]

        # Prevent division by zero
        denominator = denominator + 1e-8 * torch.sign(denominator)

        return numerator / denominator


#########################################
# Post processing functions
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
# Normalizer classes
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
# Standard Rational Neural Network
#########################################
class RationalStandardNetwork(nn.Module):
    """
    Standard Neural Network using rational activation functions
    Each layer has its own independent rational activation function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        activation_str: str = "rational_relu",
        device: torch.device = torch.device("cpu"),
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        zero_mean: bool = False,
        rational_degrees: tuple = (5, 4),
        example_input_normalizer=None,
        example_output_normalizer=None,
    ) -> None:
        super(RationalStandardNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rational_degrees = rational_degrees
        self.hidden_channels = hidden_channels

        # Input normalizer
        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else input_normalizer_class(example_input_normalizer)
        )

        # Build the network layers
        self.layers = nn.ModuleList()
        self.rational_activations = nn.ModuleList()
        all_dims = [in_channels] + hidden_channels + [out_channels]

        for i in range(len(all_dims) - 1):
            self.layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))

            if layer_norm and i < len(all_dims) - 2:
                self.layers.append(nn.LayerNorm(all_dims[i + 1]))

            if i < len(all_dims) - 2:
                self.layers.append(RationalActivation(activation_str, rational_degrees))
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*self.layers)

        # Output normalizer
        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity()

        # Initialize weights
        self._init_weights(activation_str)

        # Move the model to the specified device
        self.to(device)

        # Enable JIT compilation for better performance if PyTorch version supports it
        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            self._enable_compilation()

    def _init_weights(self, activation_str) -> None:
        """
        Initialize weights. Use Xavier for rational activations, Kaiming for others.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation_str.startswith("rational"):
                    # Xavier initialization works better with rational activations
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, nonlinearity=activation_str.replace("rational_", "")
                    )
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

    def count_rational_parameters(self):
        """Count the number of learnable parameters in rational activations."""
        rational_params = 0
        for module in self.modules():
            if isinstance(module, RationalActivation):
                rational_params += sum(p.numel() for p in module.parameters())
        return rational_params

    def count_rational_functions(self):
        """Count the total number of rational activation functions in the network."""
        count = 0
        for module in self.modules():
            if isinstance(module, RationalActivation):
                count += 1
        return count

    def get_rational_functions_info(self):
        """Get comprehensive information about all rational activation functions in the network."""
        rational_info = []

        rational_idx = 0
        for i, module in enumerate(self.network):
            if isinstance(module, RationalActivation):
                rational_info.append(
                    {
                        "location": "standard_network",
                        "layer_id": rational_idx,
                        "module_id": i,
                        "degrees": module.degrees,
                        "numerator_coeffs": module.numerator.data.cpu().numpy(),
                        "denominator_coeffs": module.denominator.data.cpu().numpy(),
                    }
                )
                rational_idx += 1

        return rational_info

    def get_rational_diversity_stats(self):
        """Get statistics about the diversity of learned rational functions."""
        all_rational_info = self.get_rational_functions_info()

        if not all_rational_info:
            return {"message": "No rational activation functions found"}

        # Calculate coefficient statistics
        all_num_coeffs = []
        all_den_coeffs = []

        for info in all_rational_info:
            all_num_coeffs.append(info["numerator_coeffs"])
            all_den_coeffs.append(info["denominator_coeffs"])

        # Convert to numpy arrays for easier computation
        num_coeffs_array = np.array(all_num_coeffs)
        den_coeffs_array = np.array(all_den_coeffs)

        stats = {
            "total_rational_functions": len(all_rational_info),
            "numerator_coeffs_mean": np.mean(num_coeffs_array, axis=0),
            "numerator_coeffs_std": np.std(num_coeffs_array, axis=0),
            "denominator_coeffs_mean": np.mean(den_coeffs_array, axis=0),
            "denominator_coeffs_std": np.std(den_coeffs_array, axis=0),
            "coefficient_diversity": {
                "numerator_pairwise_distances": (
                    np.mean(
                        [
                            np.linalg.norm(num_coeffs_array[i] - num_coeffs_array[j])
                            for i in range(len(num_coeffs_array))
                            for j in range(i + 1, len(num_coeffs_array))
                        ]
                    )
                    if len(num_coeffs_array) > 1
                    else 0
                ),
                "denominator_pairwise_distances": (
                    np.mean(
                        [
                            np.linalg.norm(den_coeffs_array[i] - den_coeffs_array[j])
                            for i in range(len(den_coeffs_array))
                            for j in range(i + 1, len(den_coeffs_array))
                        ]
                    )
                    if len(den_coeffs_array) > 1
                    else 0
                ),
            },
        }

        return stats

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.in_channels}"]
    ) -> Float[Tensor, "n_samples {self.out_channels}"]:

        x = self.input_normalizer(x)
        x = self.network(x)
        return self.post_processing(self.output_denormalizer(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("RATIONAL STANDARD NEURAL NETWORK")
    print("=" * 60)

    # Create a rational standard neural network
    standard_model = RationalStandardNetwork(
        in_channels=10,
        out_channels=5,
        hidden_channels=[64, 32, 16],
        activation_str="rational_gelu",
        device=device,
        layer_norm=True,
        dropout_rate=0.1,
        rational_degrees=(4, 3),
    )

    # Test forward pass
    x = torch.randn(32, 10).to(device)
    y_standard = standard_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_standard.shape}")
    print(f"Total parameters: {sum(p.numel() for p in standard_model.parameters())}")
    print(f"Rational parameters: {standard_model.count_rational_parameters()}")
    print(f"Number of rational functions: {standard_model.count_rational_functions()}")

    # Display rational function information
    print("\nRational activation functions:")
    for info in standard_model.get_rational_functions_info():
        print(
            f"  Layer {info['layer_id']} (Module {info['module_id']}): degrees {info['degrees']}"
        )

    # Show diversity statistics
    print("\nRational function diversity:")
    diversity_stats_standard = standard_model.get_rational_diversity_stats()
    print(
        f"  Total rational functions: {diversity_stats_standard['total_rational_functions']}"
    )
    print(
        f"  Numerator diversity (avg pairwise distance): {diversity_stats_standard['coefficient_diversity']['numerator_pairwise_distances']:.4f}"
    )
    print(
        f"  Denominator diversity (avg pairwise distance): {diversity_stats_standard['coefficient_diversity']['denominator_pairwise_distances']:.4f}"
    )
