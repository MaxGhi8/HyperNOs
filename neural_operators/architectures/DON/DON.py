import torch
import torch.nn as nn

from ..FNN.FeedForwardNetwork import FeedForwardNetwork
from ..ResNet.ResidualNetwork import ResidualNetwork
from .CNN2D_DON import CNN2D_DON

torch.set_default_dtype(torch.float32)

activation_function = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "silu": nn.SiLU(),
}


class DeepONet(nn.Module):
    def __init__(
        self,
        branch_hyperparameters: dict,
        trunk_hyperparameters: dict,
        n_basis: int,
        n_output: int,
        dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.n_basis = n_basis
        self.n_output = n_output
        self.dim = dim
        self.device = device

        assert (
            self.n_basis % self.n_output == 0
        ), "n_basis must be divisible by n_output"

        self.output_division = self.n_basis // self.n_output

        self._initialize_branch_network(branch_hyperparameters)
        self._initialize_trunk_network(trunk_hyperparameters)

        self.to(self.device)

    def _initialize_trunk_network(self, trunk_hyperparameters: dict) -> None:
        if trunk_hyperparameters["residual"]:
            self.trunk_NN = ResidualNetwork(
                in_channels=trunk_hyperparameters["n_inputs"],
                out_channels=self.n_basis,
                hidden_channels=trunk_hyperparameters["hidden_layer"],
                activation_str=trunk_hyperparameters["act_fun"],
                n_blocks=trunk_hyperparameters["n_blocks"],
                device=self.device,
                layer_norm=trunk_hyperparameters["layer_norm"],
                dropout_rate=trunk_hyperparameters["dropout_rate"],
                activation_on_output=True,
                zero_mean=False,
                example_input_normalizer=None,
                example_output_normalizer=None,
            )
        else:
            self.trunk_NN = FeedForwardNetwork(
                in_channels=trunk_hyperparameters["n_inputs"],
                out_channels=self.n_basis,
                hidden_channels=trunk_hyperparameters["hidden_layer"],
                activation_str=trunk_hyperparameters["act_fun"],
                device=self.device,
                layer_norm=trunk_hyperparameters["layer_norm"],
                dropout_rate=trunk_hyperparameters["dropout_rate"],
                activation_on_output=True,
                zero_mean=False,
                example_input_normalizer=None,
                example_output_normalizer=None,
            )

    def _initialize_branch_network(self, branch_hyperparameters: dict) -> None:
        # Store parameters needed for forward pass
        self.n_input_branch = branch_hyperparameters["n_inputs"]

        # Initialization of the network
        if self.dim == 1:
            if branch_hyperparameters["residual"]:
                self.branch_NN = ResidualNetwork(
                    in_channels=branch_hyperparameters["n_inputs"],
                    out_channels=self.n_basis,
                    hidden_channels=branch_hyperparameters["hidden_layer"],
                    activation_str=branch_hyperparameters["act_fun"],
                    n_blocks=branch_hyperparameters["n_blocks"],
                    device=self.device,
                    layer_norm=branch_hyperparameters["layer_norm"],
                    dropout_rate=branch_hyperparameters["dropout_rate"],
                    activation_on_output=False,
                    zero_mean=False,
                    example_input_normalizer=None,
                    example_output_normalizer=None,
                )

            else:
                self.branch_NN = FeedForwardNetwork(
                    in_channels=branch_hyperparameters["n_inputs"],
                    out_channels=self.n_basis,
                    hidden_channels=branch_hyperparameters["hidden_layer"],
                    activation_str=branch_hyperparameters["act_fun"],
                    device=self.device,
                    layer_norm=branch_hyperparameters["layer_norm"],
                    dropout_rate=branch_hyperparameters["dropout_rate"],
                    activation_on_output=False,
                    zero_mean=False,
                    example_input_normalizer=None,
                    example_output_normalizer=None,
                )

        elif self.dim == 2:
            if branch_hyperparameters["residual"]:
                # todo, not implemented yet
                raise NotImplementedError(
                    "Residual 2D CNN Branch Network not implemented yet."
                )

            else:
                self.branch_NN = CNN2D_DON(
                    in_channels=branch_hyperparameters["n_inputs"],
                    n_basis=self.n_basis,
                    hidden_channels=branch_hyperparameters["channels_conv"],
                    hidden_layers=branch_hyperparameters["hidden_layer"],
                    kernel_size=branch_hyperparameters["kernel_size"],
                    activation_str=branch_hyperparameters["act_fun"],
                    padding=branch_hyperparameters["padding"],
                    stride=branch_hyperparameters["stride"],
                    include_grid=branch_hyperparameters["include_grid"],
                    device=self.device,
                    normalization=branch_hyperparameters["normalization"],
                    dropout_rate=branch_hyperparameters["dropout_rate"],
                )

    def forward(self, input_don) -> torch.Tensor:
        """
        Forward pass of DeepONet.
        """
        input_branch, input_trunk = input_don
        branch_output = self.branch_NN(input_branch)
        trunk_output = self.trunk_NN(input_trunk)

        if self.n_output == 1:
            return branch_output @ trunk_output.T
        else:
            return self._compute_multi_output(branch_output, trunk_output)

    def _compute_multi_output(
        self, branch_output: torch.Tensor, trunk_output: torch.Tensor
    ) -> torch.Tensor:
        """Compute output for multiple output case (n_output>1)."""

        batch_size = branch_output.shape[0]
        n_points_trunk = trunk_output.shape[0]
        output_shape = (batch_size, n_points_trunk, self.n_output)

        # Initialize output tensor
        output = torch.zeros(
            output_shape, device=branch_output.device, dtype=branch_output.dtype
        )

        # Computation for all output channels
        for i in range(self.n_output):
            start_idx = i * self.output_division
            end_idx = (i + 1) * self.output_division
            output[:, :, i] = (
                branch_output[:, start_idx:end_idx]
                @ trunk_output[:, start_idx:end_idx].T
            )

        return output
