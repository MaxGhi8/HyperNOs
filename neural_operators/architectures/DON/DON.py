import torch
import torch.nn as nn
from FNN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
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
        self, branch_hyperparameters, trunk_hyperparameters, DON_hyperparameters
    ):
        super().__init__()

        self.n_basis = DON_hyperparameters["n_basis"]
        self.n_output = DON_hyperparameters["n_output"]
        self.dim = DON_hyperparameters["dim"]

        self.n_input_branch = branch_hyperparameters["n_inputs"]
        self.hidden_branch = branch_hyperparameters["hidden_layer"]
        self.branch_residual = branch_hyperparameters["residual"]
        self.act_fun_branch = activation_function[branch_hyperparameters["act_fun"]]

        if self.dim == 1:
            if self.branch_residual:
                self.branch_NN = Residual_FeedForward(
                    self.n_input_branch,
                    self.n_basis,
                    self.hidden_branch,
                    self.act_fun_branch,
                )
            else:
                self.branch_NN = FeedForward(
                    self.n_input_branch,
                    self.n_basis,
                    self.hidden_branch,
                    self.act_fun_branch,
                )

        elif self.dim == 2:
            self.n_points = branch_hyperparameters["n_points"]
            self.conv_layers = branch_hyperparameters["channels_conv"]
            self.stride = branch_hyperparameters["stride"]
            self.kernel_size = branch_hyperparameters["kernel_size"]
            self.output_dim_conv = branch_hyperparameters["output_dim_conv"]
            if self.branch_residual:

                self.branch_NN = residual_branch_2D(
                    self.n_input_branch,
                    self.conv_layers,
                    self.stride,
                    self.kernel_size,
                    self.hidden_branch,
                    self.n_basis,
                    self.act_fun_branch,
                    self.output_dim_conv,
                    self.n_points[0],
                )
            else:
                self.branch_NN = branch_2D(
                    self.n_input_branch,
                    self.conv_layers,
                    self.stride,
                    self.kernel_size,
                    self.hidden_branch,
                    self.n_basis,
                    self.act_fun_branch,
                    self.output_dim_conv,
                    self.n_points[0],
                )

        self.n_input_trunk = trunk_hyperparameters["n_inputs"]
        self.hidden_trunk = trunk_hyperparameters["hidden_layer"]
        self.act_fun_trunk = activation_function[trunk_hyperparameters["act_fun"]]
        self.residual = trunk_hyperparameters["residual"]
        if self.residual:
            self.trunk_NN = Residual_FeedForward(
                self.n_input_trunk,
                self.n_basis,
                self.hidden_trunk,
                self.act_fun_trunk,
                True,
            )
        else:
            self.trunk_NN = FeedForward(
                self.n_input_trunk,
                self.n_basis,
                self.hidden_trunk,
                self.act_fun_trunk,
                True,
            )

    def forward(self, input_branch, input_trunk):
        branch_output = self.branch_NN(input_branch)
        trunk_output = self.trunk_NN(input_trunk)

        if self.n_output != 1:
            assert self.n_basis % self.n_output == 0
            output_division = self.n_basis // self.n_output
            if self.dim == 2:
                output = torch.zeros(
                    [
                        branch_output.shape[0],
                        self.n_points[0] * self.n_points[1],
                        self.n_output,
                    ],
                    device=device,
                )

            elif self.dim == 1:
                output = torch.zeros(
                    [branch_output.shape[0], self.n_input_branch, self.n_output],
                    device=device,
                )
            for i in range(self.n_output):
                output[:, :, i] = (
                    branch_output[:, i * output_division : (i + 1) * output_division]
                    @ trunk_output[:, i * output_division : (i + 1) * output_division].T
                )

            if self.dim == 2:
                output = output.view(
                    -1, self.n_points[0], self.n_points[1], self.n_output
                )
        else:
            output = branch_output @ trunk_output.T
            if self.dim == 2:
                output = output.view(-1, self.n_points[0], self.n_points[1])
        return output
