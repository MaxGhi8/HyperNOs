"""
This file contains all the core architectures and modules of the
Boundary Adapted Multi Patch Neural Operator (BAMPNO) for solving PDEs in 2D.
"""

import os
import sys
from functools import cache

sys.path.append("..")
import architectures.BAMPNO.chebyshev_utilities as cheb
import mat73
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from utilities import find_file

#########################################
# default values
#########################################
torch.set_default_dtype(torch.float32)  # default tensor dtype


#########################################
# activation function
#########################################
@jaxtyped(typechecker=beartype)
def activation(
    x: Float[Tensor, "n_samples n_patch *n d"], activation_str: str
) -> Float[Tensor, "n_samples n_patch *n d"]:
    """
    Activation function to be used within the network.
    The function is the same throughout the network.
    """
    if activation_str == "relu":
        return F.relu(x)
    elif activation_str == "gelu":
        return F.gelu(x)
    elif activation_str == "tanh":
        return F.tanh(x)
    elif activation_str == "leaky_relu":
        return F.leaky_relu(x)
    else:
        raise ValueError("Not implemented activation function")


#########################################
# MLP
#########################################
class MLP(nn.Module):
    """Shallow neural network with one hidden layer"""

    def __init__(
        self,
        problem_dim: int,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        fun_act: str,
        bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super(MLP, self).__init__()
        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.mlp1 = torch.nn.Linear(
            in_channels, mid_channels, bias=self.bias, device=device
        )
        self.mlp2 = torch.nn.Linear(
            mid_channels, out_channels, bias=self.bias, device=device
        )
        self.fun_act = fun_act

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples n_patch *d_x {self.in_channels}"]
    ) -> Float[Tensor, "n_samples n_patch *d_x {self.out_channels}"]:
        x = self.mlp1(x)  # affine transformation
        x = activation(x, self.fun_act)  # activation function
        x = self.mlp2(x)  # affine transformation
        return x


#########################################
# Chebyshev multi-patch Layer
#########################################
class ChebyshevLayer(nn.Module):
    def __init__(
        self,
        n_patch: int,
        continuity_condition: dict,
        in_channels: int,
        out_channels: int,
        modes: int,
        M: Tensor,
        M_1: Tensor,
        fun_act: str,
        device: torch.device,
        zero_BC: dict = None,
        weights_norm: str = None,
        same_params: bool = False,
    ):
        """
        2D Integral layer with boundary adapted Chebyshev basis

        n_patch : int
            number of patches

        continuity_condition : dict
            dictionary that informs the continuity condition to be applied.
            if None, no continuity condition is applied.

        in_channels : int
            input dimension (d_v in the theory)

        out_channels : int
            output dimension (d_{v+1} in the theory)

        modes : int
            number of modes

        M : torch.tensor
            Matrix for the change of basis

        M_1 : torch.tensor
            Matrix for the inverse change of basis

        weights_norm : str
            string for selecting the weights normalization to use

        fun_act : str
            string for selecting the activation function to use, needed for the weights normalization

        zero_BC: dict
            Dictionary that informs the boundary of each patch that needs to be set to zero.

        same_params : bool
            if True we use the same parameters for all the patches
        """
        super(ChebyshevLayer, self).__init__()
        self.n_patch = n_patch
        self.continuity_condition = continuity_condition
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.M = M
        self.M_1 = M_1
        self.weights_norm = weights_norm
        self.fun_act = fun_act
        self.zero_BC = zero_BC
        self.same_params = same_params

        # same parameters for all the patches
        if self.same_params:
            if self.weights_norm == "Xavier":
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(
                        torch.empty(
                            self.modes,
                            self.modes,
                            self.in_channels,
                            self.out_channels,
                        )
                    ),
                    gain=1 / (self.in_channels * self.out_channels),
                )

            elif self.weights_norm == "Kaiming":
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(
                        torch.empty(
                            self.modes,
                            self.modes,
                            self.in_channels,
                            self.out_channels,
                        )
                    ),
                    a=0,
                    mode="fan_in",
                    nonlinearity=self.fun_act,
                )

            elif self.weights_norm == None:
                self.scale = 1 / (in_channels * out_channels)
                self.weights = nn.Parameter(
                    self.scale
                    * torch.rand(
                        self.modes,
                        self.modes,
                        self.in_channels,
                        self.out_channels,
                    )
                )

            else:
                raise ValueError("weights_norm does not exist, check for typo.")

        # differents parameters in each patch
        else:
            if self.weights_norm == "Xavier":
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(
                        torch.empty(
                            self.n_patch,
                            self.modes,
                            self.modes,
                            self.in_channels,
                            self.out_channels,
                        )
                    ),
                    gain=1 / (self.in_channels * self.out_channels),
                )

            elif self.weights_norm == "Kaiming":
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(
                        torch.empty(
                            self.n_patch,
                            self.modes,
                            self.modes,
                            self.in_channels,
                            self.out_channels,
                        )
                    ),
                    a=0,
                    mode="fan_in",
                    nonlinearity=self.fun_act,
                )

            elif self.weights_norm == None:
                self.scale = 1 / (in_channels * out_channels)
                self.weights = nn.Parameter(
                    self.scale
                    * torch.rand(
                        self.n_patch,
                        self.modes,
                        self.modes,
                        self.in_channels,
                        self.out_channels,
                    )
                )

            else:
                raise ValueError("weights_norm does not exist, check for typo.")

        ## Move to device
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def tensor_mul(
        self,
        input: Float[
            Tensor,
            "n_batch {self.n_patch} {self.modes} {self.modes} {self.in_channels}",
        ],
        weights: Float[
            Tensor,
            "{self.n_patch} {self.modes} {self.modes} {self.in_channels} {self.out_channels}",
        ],
    ) -> Float[
        Tensor, "n_batch {self.n_patch} {self.modes} {self.modes} {self.out_channels}"
    ]:
        """Multiplication between tensors when the parameters are different for each patch"""
        return torch.einsum("bpxyi,pxyio->bpxyo", input, weights)

    @jaxtyped(typechecker=beartype)
    def tensor_mul_same_params(
        self,
        input: Float[
            Tensor,
            "n_batch {self.n_patch} {self.modes} {self.modes} {self.in_channels}",
        ],
        weights: Float[
            Tensor, "{self.modes} {self.modes} {self.in_channels} {self.out_channels}"
        ],
    ) -> Float[
        Tensor, "n_batch {self.n_patch} {self.modes} {self.modes} {self.out_channels}"
    ]:
        """Multiplication between tensors when the parameters are equals for each patch"""
        return torch.einsum("bpxyi,xyio->bpxyo", input, weights)

    # @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_batch {self.n_patch} *n_x {self.in_channels}"]
    ) -> Float[Tensor, "n_batch {self.n_patch} *n_x {self.out_channels}"]:
        """
        input --> CFT --> boundary adapted + continuity --> parameters --> inverse boundary adapted --> ICFT --> output
        Total computation cost is equal to O(n log(n))

        input: torch.tensor
            the input 'x' is a tensor of shape (n_samples, n_patch, *n_x, in_channels)

        output: torch.tensor
            return a tensor of shape (n_samples, n_patch, *n_x, out_channels)
        """
        batch_size, n_x, n_y = x.shape[0], x.shape[2], x.shape[3]

        # print(
        #     "x_in",
        #     sum(abs(x[:, 0, :, -1, :] - x[:, 1, :, 0, :]).reshape(-1, 1)),
        # )
        # print(
        #     "x_in",
        #     sum(abs(x[:, 1, -1, :, :] - x[:, 2, 0, :, :]).reshape(-1, 1)),
        # )

        # CFT
        x = cheb.patched_values_to_coefficients(x)

        # Multiply relevant Cheb modes
        out_ft = torch.zeros(
            batch_size, self.n_patch, n_x, n_y, self.out_channels, device=x.device
        )
        if self.same_params:
            out_ft[:, :, : self.modes, : self.modes, :] = self.tensor_mul_same_params(
                x[:, :, : self.modes, : self.modes, :], self.weights
            )
        else:
            out_ft[:, :, : self.modes, : self.modes, :] = self.tensor_mul(
                x[:, :, : self.modes, : self.modes, :], self.weights
            )

        # Boundary adapted transform
        out_ft = cheb.patched_change_basis(out_ft, self.M_1)

        # Boundary condition
        if self.zero_BC:
            zero_vector = torch.zeros(batch_size, self.modes, self.out_channels)
            for patch in self.zero_BC:
                out_ft[:, patch, :2, :2, :] = torch.zeros(
                    batch_size, 2, 2, self.out_channels
                )
                for edge in self.zero_BC[patch]:
                    if edge == 1:  # left
                        out_ft[:, patch, 0, : self.modes, :] = zero_vector
                    elif edge == 2:  # right
                        out_ft[:, patch, 1, : self.modes, :] = zero_vector
                    elif edge == 3:  # bottom
                        out_ft[:, patch, : self.modes, 0, :] = zero_vector
                    elif edge == 4:  # top
                        out_ft[:, patch, : self.modes, 1, :] = zero_vector

        # Continuity condition
        for p1, p2 in self.continuity_condition["horizontal"]:
            tmp = (
                out_ft[:, p1, 0, 2 : self.modes, :]
                + out_ft[:, p2, 1, 2 : self.modes, :]
            ) / 2
            out_ft[:, p1, 0, 2 : self.modes, :] = tmp
            out_ft[:, p2, 1, 2 : self.modes, :] = tmp

        for p1, p2 in self.continuity_condition["vertical"]:
            tmp = (
                out_ft[:, p1, 2 : self.modes, 0, :]
                + out_ft[:, p2, 2 : self.modes, 1, :]
            ) / 2
            out_ft[:, p1, 2 : self.modes, 0, :] = tmp
            out_ft[:, p2, 2 : self.modes, 1, :] = tmp

        # Inverse boundary adapted transform
        out_ft = cheb.patched_change_basis(out_ft, self.M)

        # ICFT
        out_ft = cheb.patched_coefficients_to_values(out_ft)

        return out_ft


#########################################
# Boundary Adapted Multi Patch Neural Operator (BAMPNO)
#########################################
class output_denormalizer_class(nn.Module):
    def __init__(self, output_normalizer) -> None:
        super(output_denormalizer_class, self).__init__()
        self.output_normalizer = output_normalizer

    def forward(self, x):
        return self.output_normalizer.decode(x)


class BAMPNO(nn.Module):
    """
    Boundary Adapted Multi Patch Neural Operator (BAMPNO) for solving PDEs in 2D.
    """

    def __init__(
        self,
        problem_dim: int,
        n_patch: int,
        continuity_condition: dict,
        n_pts: int,
        grid_filename: str,
        in_dim: int,
        d_v: int,
        out_dim: int,
        L: int,
        modes: int,
        fun_act: str,
        weights_norm: str,
        zero_BC: dict = None,
        arc: str = "Classic",
        RNN: bool = False,
        same_params: bool = False,
        FFTnorm=None,
        device: torch.device = torch.device("cpu"),
        example_output_normalizer=None,
        retrain_seed: int = -1,
    ):
        """
        problem_dim : int
            dimension of the problem (1D, 2D or 3D), in this case 2D

        n_patch : int
            number of patches in the domain

        continuity_condition : dict
            dictionary that informs the continuity condition to be applied across the patches.

        grid_filename: str
            name of the file that contains the grid of the physical domain where the data is evaluated.

        n_pts : int
            number of points in the grid in each direction

        in_dim : int
            dimension of the input space

        d_v : int
            dimension of the space in the integral operator

        out_dim : int
            dimension of the output space

        L: int
            number of integral operators layers to perform

        modes : int
            equal to k_{max, i}

        fun_act: str
            string for selecting the activation function to use throughout the architecture

        weights_norm: str
            string for selecting the weights normalization --> 'Xavier' or 'Kaiming'

        zero_BC: dict
            Dictionary that informs the boundary of each patch that needs to be set to zero.
            if None, no homogeneous boundary condition is applied.

        arc: str
            string for selecting the architecture of the network --> 'Residual', 'Tran', 'Classic', 'Zongyi'

        RNN: bool
            if True we use the RNN architecture, otherwise the classic one

        same_params: bool
            if True we use the same parameters for all the patches, otherwise we use different parameters for each patch

        FFTnorm: str
            string for selecting the normalization for the FFT --> None or 'ortho'

        device: torch.device
            device for the model

        example_output_normalizer: callable
            function to normalize the output of the model

        retrain_seed: int
            seed for retraining (if is equal to -1, no retraining is performed)
        """
        super(BAMPNO, self).__init__()
        self.problem_dim = problem_dim
        self.n_patch = n_patch
        self.continuity_condition = continuity_condition
        self.n_pts = n_pts
        self.grid_filename = grid_filename
        self.in_dim = in_dim + self.problem_dim
        self.d_v = d_v
        self.out_dim = out_dim
        self.L = L
        self.modes = modes
        self.fun_act = fun_act
        self.weights_norm = weights_norm
        self.zero_BC = zero_BC
        self.arc = arc
        self.RNN = RNN
        self.same_params = same_params
        self.FFTnorm = FFTnorm
        self.device = device
        self.retrain_seed = retrain_seed

        self.bias = False if self.zero_BC is not None else True

        self.M = self.get_M(n_pts).to(device)
        self.M_1 = self.get_M_1(n_pts).to(device)

        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        assert self.problem_dim == 2

        ## Lifting
        # self.p = torch.nn.Linear(self.in_dim, self.d_v, bias=self.bias)
        self.p = MLP(
            self.problem_dim,
            self.in_dim,
            self.d_v,
            128,
            self.fun_act,
            self.bias,
            self.device,
        )

        ## Integral layer
        if self.arc == "Tran":  # residual form
            if self.RNN:
                self.ws1 = torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                self.ws2 = torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                self.integrals = ChebyshevLayer(
                    self.n_patch,
                    self.continuity_condition,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.M,
                    self.M_1,
                    self.fun_act,
                    self.device,
                    self.zero_BC,
                    self.weights_norm,
                    self.same_params,
                )
            else:
                self.ws1 = nn.ModuleList(
                    [
                        torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                        for _ in range(self.L)
                    ]
                )
                self.ws2 = nn.ModuleList(
                    [
                        torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                        for _ in range(self.L)
                    ]
                )

                self.integrals = nn.ModuleList(
                    [
                        ChebyshevLayer(
                            self.n_patch,
                            self.continuity_condition,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.M,
                            self.M_1,
                            self.fun_act,
                            self.device,
                            self.zero_BC,
                            self.weights_norm,
                            self.same_params,
                        )
                        for _ in range(self.L)
                    ]
                )

        elif self.arc == "Zongyi":
            if self.RNN:
                self.ws = torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                self.mlps = MLP(
                    self.problem_dim,
                    self.d_v,
                    self.d_v,
                    self.d_v,
                    self.fun_act,
                    self.bias,
                    self.deivce,
                )
                self.integrals = ChebyshevLayer(
                    self.n_patch,
                    self.continuity_condition,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.M,
                    self.M_1,
                    self.fun_act,
                    self.device,
                    self.zero_BC,
                    self.weights_norm,
                    self.same_params,
                )

            else:
                self.ws = nn.ModuleList(
                    [
                        torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                        for _ in range(self.L)
                    ]
                )

                self.mlps = nn.ModuleList(
                    [
                        MLP(
                            self.problem_dim,
                            self.d_v,
                            self.d_v,
                            self.d_v,
                            self.fun_act,
                            self.bias,
                            self.device,
                        )
                        for _ in range(self.L)
                    ]
                )
                self.integrals = nn.ModuleList(
                    [
                        ChebyshevLayer(
                            self.n_patch,
                            self.continuity_condition,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.M,
                            self.M_1,
                            self.fun_act,
                            self.device,
                            self.zero_BC,
                            self.weights_norm,
                            self.same_params,
                        )
                        for _ in range(self.L)
                    ]
                )

        elif self.arc == "Classic" or self.arc == "Residual":
            if self.RNN:
                self.ws = torch.nn.Linear(self.d_v, self.d_v, bias=self.bias)
                self.integrals = ChebyshevLayer(
                    self.n_patch,
                    self.continuity_condition,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.M,
                    self.M_1,
                    self.fun_act,
                    self.device,
                    self.zero_BC,
                    self.weights_norm,
                    self.same_params,
                )

            else:
                self.ws = nn.ModuleList(
                    [
                        nn.Linear(self.d_v, self.d_v, bias=self.bias)
                        for _ in range(self.L)
                    ]
                )

                self.integrals = nn.ModuleList(
                    [
                        ChebyshevLayer(
                            self.n_patch,
                            self.continuity_condition,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.M,
                            self.M_1,
                            self.fun_act,
                            self.device,
                            self.zero_BC,
                            self.weights_norm,
                            self.same_params,
                        )
                        for _ in range(self.L)
                    ]
                )

        ## Projection
        # self.q = torch.nn.Linear(self.d_v, self.out_dim, bias=self.bias)
        self.q = MLP(
            self.problem_dim,
            self.d_v,
            self.out_dim,
            128,
            self.fun_act,
            self.bias,
            self.device,
        )

        ## Move to device
        self.to(self.device)

        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            self._enable_compilation()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "n_batch {self.n_patch} *n_x {self.in_dim-self.problem_dim}"],
    ) -> Float[Tensor, "n_batch {self.n_patch} *n_x {self.out_dim}"]:
        # x = x.to(self.device)

        ## Grid and initialization
        grid = self.get_grid(
            x.size(0),
            self.grid_filename,
            x.device,
            s=1,
            search_path=os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            ),
        )
        x = torch.cat(
            (grid, x), dim=-1
        )  # concatenate last dimension --> (n_samples)*(n_patch)*(*n_x)*(in_dim+problem_dim)

        ## Perform lifting operator P
        x = self.p(x)  # shape = (n_samples)*(n_patch)*(*n_x)*(d_v)

        ## Integral Layers
        for i in range(self.L):
            if self.arc == "Tran":
                if self.RNN:
                    x_1 = self.integrals(x)
                    x_1 = activation(self.ws1(x_1), self.fun_act)
                    x_1 = self.ws2(x_1)
                else:
                    x_1 = self.integrals[i](x)
                    x_1 = activation(self.ws1[i](x_1), self.fun_act)
                    x_1 = self.ws2[i](x_1)
                x_1 = activation(x_1, self.fun_act)  # (?)
                x = x + x_1

            elif self.arc == "Classic":
                if self.RNN:
                    x1 = self.integrals(x)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                    x = activation(x, self.fun_act)

            elif self.arc == "Zongyi":
                if self.RNN:
                    x1 = self.integrals(x)
                    x1 = self.mlps(x1)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x1 = self.mlps[i](x1)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                    x = activation(x, self.fun_act)

            elif self.arc == "Residual":
                if self.RNN:
                    x1 = self.integrals(x)
                    x2 = self.ws(x)
                else:
                    x2 = self.integrals[i](x)
                    x1 = self.ws[i](x)
                if i < self.L - 1:
                    x = x + activation(x1 + x2, self.fun_act)

        return self.output_denormalizer(
            self.q(x)
        )  # shape --> (n_samples)*(n_patch)*(*n_x)*(out_dim)

    @cache
    def get_grid(
        self,
        batch_size: int,
        grid_filename: str,
        device: torch.device,
        s: int = 1,
        search_path: str = "/",
    ) -> Float[Tensor, "{batch_size} {self.n_patch} *n_x {self.problem_dim}"]:
        """
        Function to get the grid of the physical domain where the data is evaluated.

        batch_size: int
            batch_size is the number of samples in the batch.
        grid_filename: str
            grid_filename is the name of the file that contains the grid of the physical domain where the data is evaluated.
        """
        grid_path = find_file(grid_filename, search_path)
        reader = mat73.loadmat(grid_path)
        X_phys = torch.from_numpy(reader["X_phys"]).type(torch.float32)
        Y_phys = torch.from_numpy(reader["Y_phys"]).type(torch.float32)
        X_phys = X_phys[:, ::s, ::s]
        Y_phys = Y_phys[:, ::s, ::s]

        n_patch = X_phys.shape[0]
        n_x = X_phys.shape[1]
        n_y = X_phys.shape[2]

        assert X_phys.shape == Y_phys.shape, "X and Y must have the same shape"
        assert (
            n_patch == self.n_patch
        ), "n_patch must be the same in the model and in the data"

        X_phys = X_phys.reshape(1, n_patch, n_x, n_y, 1).repeat(
            [batch_size, 1, 1, 1, 1]
        )
        Y_phys = Y_phys.reshape(1, n_patch, n_x, n_y, 1).repeat(
            [batch_size, 1, 1, 1, 1]
        )
        return torch.cat((X_phys, Y_phys), dim=-1).to(
            device
        )  # shape = (batch_size)*(n_patch)*(*n_x)*(problem_dim)

    @cache
    def get_M(self, n: int) -> Float[Tensor, "{self.modes} {self.modes}"]:
        """
        Function to get the matrix for the change of basis, B = M*T.
        n: int
            n is the dimension of the matrix.
        """
        M = torch.eye(n)
        M[0, 0] = 1 / 2
        M[0, 1] = -1 / 2
        M[1, 0] = 1 / 2
        M[1, 1] = 1 / 2
        for i in range(2, n):
            if i % 2 == 0:
                M[i, 0] = -1
            else:
                M[i, 1] = -1
        return M

    @cache
    def get_M_1(self, n: int) -> Float[Tensor, "{self.modes} {self.modes}"]:
        """
        Function to get the inverse of the matrix for the change of basis, T = M_1*B.
        n: int
            n is the dimension of the matrix.
        """
        M_1 = torch.eye(n)
        M_1[0, 1] = 1
        M_1[1, 0] = -1
        for i in range(2, n):
            if i % 2 == 0:
                M_1[i, 0] = 1
                M_1[i, 1] = 1
            else:
                M_1[i, 0] = -1
                M_1[i, 1] = 1
        return M_1

    def _enable_compilation(self) -> None:
        """Enable PyTorch 2.0+ compilation for performance if available."""
        try:
            # This is a PyTorch 2.0+ feature
            self = torch.compile(self)
            print("PyTorch compilation enabled for better performance")
        except Exception as e:
            print(f"Could not enable PyTorch compilation: {e}")
