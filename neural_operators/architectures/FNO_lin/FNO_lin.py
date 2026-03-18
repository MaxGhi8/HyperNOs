"""
This file contains all the core architectures and modules of the FNO_lin modification.
"""

from functools import cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Tuple, Union
from jaxtyping import Complex, Float, jaxtyped
from torch import Tensor

#########################################
# default values
#########################################
torch.set_default_dtype(torch.float32)  # default tensor dtype


#########################################
# activation function
#########################################
@jaxtyped(typechecker=beartype)
def activation(
    x: Float[Tensor, "n_samples *n d"], activation_str: str
) -> Float[Tensor, "n_samples *n d"]:
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
    ):
        super(MLP, self).__init__()
        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp1 = torch.nn.Linear(in_channels, mid_channels)
        self.mlp2 = torch.nn.Linear(mid_channels, out_channels)
        self.fun_act = fun_act

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples *d_x {self.in_channels}"]
    ) -> Float[Tensor, "n_samples *d_x {self.out_channels}"]:
        x = self.mlp1(x)  # affine transformation
        x = activation(x, self.fun_act)  # activation function
        x = self.mlp2(x)  # affine transformation
        return x


class MLP_conv(nn.Module):
    """As MLP but with convolutional layers"""

    def __init__(
        self,
        problem_dim: int,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        fun_act: str,
    ):
        super(MLP_conv, self).__init__()
        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.problem_dim == 1:
            self.mlp1 = torch.nn.Conv1d(in_channels, mid_channels, 1)
            self.mlp2 = torch.nn.Conv1d(mid_channels, out_channels, 1)
        elif self.problem_dim == 2:
            self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
            self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)
        elif self.problem_dim == 3:
            self.mlp1 = torch.nn.Conv3d(in_channels, mid_channels, 1)
            self.mlp2 = torch.nn.Conv3d(mid_channels, out_channels, 1)

        self.fun_act = fun_act

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_samples {self.in_channels} *d_x"]
    ) -> Float[Tensor, "n_samples {self.out_channels} *d_x"]:
        x = self.mlp1(x)  # affine transformation
        x = activation(x, self.fun_act)  # activation function
        x = self.mlp2(x)  # affine transformation
        return x


#########################################
# Fourier layer
#########################################
class FourierLayer(nn.Module):
    """
    Integral layer with Fourier basis
    """

    def __init__(
        self,
        problem_dim: int,
        in_channels: int,
        out_channels: int,
        modes: int,
        weights_norm: str,
        fun_act: str,
        FFTnorm,
    ):
        super(FourierLayer, self).__init__()

        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        self.weights_norm = weights_norm
        self.fun_act = fun_act
        self.FFTnorm = FFTnorm

        if self.problem_dim == 1:
            self.scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                self.scale
                * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
            )

            # if self.weights_norm == 'Xavier':
            #     # Xavier normalization
            #     self.weights = nn.init.xavier_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat)),
            #         gain = 1/(self.in_channels*self.out_channels))
            # elif self.weights_norm == 'Kaiming':
            #     # Kaiming normalization
            #     self.weights = torch.nn.init.kaiming_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat)),
            #         a = 0, mode = 'fan_in', nonlinearity = self.fun_act)

        elif self.problem_dim == 2:
            self.scale = 1 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )
            self.weights2 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )

            # if self.weights_norm == 'Xavier':
            #     # Xavier normalization
            #     self.weights1 = nn.init.xavier_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat)),
            #         gain = 1/(self.in_channels*self.out_channels))
            #     self.weights2 = nn.init.xavier_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat)),
            #         gain = 1/(self.in_channels*self.out_channels))
            # elif self.weights_norm == 'Kaiming':
            #     # Kaiming normalization
            #     self.weights1 = torch.nn.init.kaiming_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat)),
            #         a = 0, mode = 'fan_in', nonlinearity = self.fun_act)
            #     self.weights2 = torch.nn.init.kaiming_normal_(
            #         nn.Parameter(torch.empty(self.in_channels, self.out_channels, self.modes, self.modes, dtype=torch.cfloat)),
            #         a = 0, mode = 'fan_in', nonlinearity = self.fun_act)

        elif self.problem_dim == 3:
            self.scale = 1 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )
            self.weights2 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )
            self.weights3 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )
            self.weights4 = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    self.modes,
                    self.modes,
                    dtype=torch.cfloat,
                )
            )

    @jaxtyped(typechecker=beartype)
    def tensor_mul_1d(
        self,
        input: Complex[Tensor, "n_batch {self.in_channels} {self.modes}"],
        weights: Complex[Tensor, "{self.in_channels} {self.out_channels} {self.modes}"],
    ) -> Complex[Tensor, "n_batch {self.out_channels} {self.modes}"]:
        """Multiplication between complex numbers"""
        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        return torch.einsum("bim,iom->bom", input, weights)

    @jaxtyped(typechecker=beartype)
    def tensor_mul_2d(
        self,
        input: Complex[Tensor, "n_batch {self.in_channels} {self.modes} {self.modes}"],
        weights: Complex[
            Tensor, "{self.in_channels} {self.out_channels} {self.modes} {self.modes}"
        ],
    ) -> Complex[Tensor, "n_batch {self.out_channels} {self.modes} {self.modes}"]:
        """Multiplication between complex numbers"""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    @jaxtyped(typechecker=beartype)
    def tensor_mul_3d(
        self,
        input: Complex[
            Tensor, "n_batch {self.in_channels} {self.modes} {self.modes} {self.modes}"
        ],
        weights: Complex[
            Tensor,
            "{self.in_channels} {self.out_channels} {self.modes} {self.modes} {self.modes}",
        ],
    ) -> Complex[
        Tensor, "n_batch {self.out_channels} {self.modes} {self.modes} {self.modes}"
    ]:
        """Multiplication between complex numbers"""
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_batch {self.in_channels} *n_x"]
    ) -> Float[Tensor, "n_batch {self.out_channels} *n_x"]:
        """
        input --> FFT --> parameters --> IFFT --> output
        Total computation cost is equal to O(n log(n))

        input: torch.tensor
            the input 'x' is a tensor of shape (n_samples, in_channels, *n_x)

        output: torch.tensor
            return a tensor of shape (n_samples, out_channels, *n_x)
        """
        batchsize = x.shape[0]

        if self.problem_dim == 1:
            # Fourier transform
            x_ft = torch.fft.rfft(x, norm=self.FFTnorm)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[:, :, : self.modes] = self.tensor_mul_1d(
                x_ft[:, :, : self.modes], self.weights
            )

            # Inverse Fourier transform
            x = torch.fft.irfft(out_ft, n=x.size(-1), norm=self.FFTnorm)

        elif self.problem_dim == 2:
            # Fourier transform
            x_ft = torch.fft.rfft2(x, norm=self.FFTnorm)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[:, :, : self.modes, : self.modes] = self.tensor_mul_2d(
                x_ft[:, :, : self.modes, : self.modes], self.weights1
            )
            out_ft[:, :, -self.modes :, : self.modes] = self.tensor_mul_2d(
                x_ft[:, :, -self.modes :, : self.modes], self.weights2
            )

            # Inverse Fourier transform
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=self.FFTnorm)

        elif self.problem_dim == 3:
            # Fourier transform
            x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=self.FFTnorm)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-3),
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )

            # first corner
            out_ft[:, :, : self.modes, : self.modes, : self.modes] = self.tensor_mul_3d(
                x_ft[:, :, : self.modes, : self.modes, : self.modes], self.weights1
            )
            # second corner
            out_ft[:, :, -self.modes :, : self.modes, : self.modes] = (
                self.tensor_mul_3d(
                    x_ft[:, :, -self.modes :, : self.modes, : self.modes], self.weights2
                )
            )
            # third corner
            out_ft[:, :, : self.modes, -self.modes :, : self.modes] = (
                self.tensor_mul_3d(
                    x_ft[:, :, : self.modes, -self.modes :, : self.modes], self.weights3
                )
            )
            # fourth corner
            out_ft[:, :, -self.modes :, -self.modes :, : self.modes] = (
                self.tensor_mul_3d(
                    x_ft[:, :, -self.modes :, -self.modes :, : self.modes],
                    self.weights4,
                )
            )

            # Inverse Fourier transform
            x = torch.fft.irfftn(
                out_ft,
                dim=(-3, -2, -1),
                s=(x.size(-3), x.size(-2), x.size(-1)),
                norm=self.FFTnorm,
            )

        return x


#########################################
# Fourier Neural Operator
#########################################
class output_denormalizer_class(nn.Module):
    def __init__(self, output_normalizer) -> None:
        super(output_denormalizer_class, self).__init__()
        self.output_normalizer = output_normalizer

    def forward(self, x):
        return self.output_normalizer.decode(x)


class FNO_lin(nn.Module):
    """
    Fourier Neural Operator for a problem on square domain
    """

    def __init__(
        self,
        problem_dim: int,
        in_dim: int,
        d_v: int,
        out_dim: int,
        L: int,
        modes: int,
        fun_act: str,
        weights_norm: str,
        arc: str = "Classic",
        RNN: bool = False,
        FFTnorm=None,
        padding: int = 4,
        device: torch.device = torch.device("cpu"),
        example_output_normalizer=None,
        retrain_fno=-1,
    ):
        """
        in_dim : int
            dimension of the input space

        d_v : int
            dimension of the space in the integral Fourier operator

        out_dim : int
            dimension of the output space

        L: int
            number of integral operators Fourier to perform

        modes : int
            equal to k_{max, i}

        fun_act: str
            string for selecting the activation function to use throughout the architecture

        weights_norm: str
            string for selecting the weights normalization --> 'Xavier' or 'Kaiming'

        arc: str
            string for selecting the architecture of the network --> 'Residual', 'Tran', 'Classic', 'Zongyi'

        RNN: bool
            if True we use the RNN architecture, otherwise the classic one

        FFTnorm: str
            string for selecting the normalization for the FFT --> None or 'ortho'

        padding: int
            number of points for padding, only for Fourier architecture

        device: torch.device
            device for the model

        retrain_fno: int
            seed for retraining (if is equal to -1, no retraining is performed)
        """
        super(FNO_lin, self).__init__()
        self.problem_dim = problem_dim
        self.in_dim = in_dim + self.problem_dim
        self.d_v = d_v
        self.out_dim = out_dim
        self.L = L
        self.modes = modes
        self.fun_act = fun_act
        self.weights_norm = weights_norm
        self.arc = arc
        self.RNN = RNN
        self.FFTnorm = FFTnorm
        self.padding = padding
        self.retrain_fno = retrain_fno
        self.device = device

        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        ## Lifting
        # self.p = torch.nn.Linear(self.in_dim, self.d_v)
        self.p = MLP(self.problem_dim, self.in_dim, self.d_v, 128, self.fun_act)

        ## Fourier layer
        if self.arc == "Tran":  # residual form
            if self.RNN:
                if self.problem_dim == 1:
                    self.ws1 = torch.nn.Conv1d(self.d_v, self.d_v, 1)
                    self.ws2 = torch.nn.Conv1d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 2:
                    self.ws1 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                    self.ws2 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 3:
                    self.ws1 = torch.nn.Conv3d(self.d_v, self.d_v, 1)
                    self.ws2 = torch.nn.Conv3d(self.d_v, self.d_v, 1)
                self.integrals = FourierLayer(
                    self.problem_dim,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.weights_norm,
                    self.fun_act,
                    self.FFTnorm,
                )
            else:
                if self.problem_dim == 1:
                    self.ws1 = nn.ModuleList(
                        [torch.nn.Conv1d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                    self.ws2 = nn.ModuleList(
                        [torch.nn.Conv1d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 2:
                    self.ws1 = nn.ModuleList(
                        [torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                    self.ws2 = nn.ModuleList(
                        [torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 3:
                    self.ws1 = nn.ModuleList(
                        [torch.nn.Conv3d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                    self.ws2 = nn.ModuleList(
                        [torch.nn.Conv3d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )

                self.integrals = nn.ModuleList(
                    [
                        FourierLayer(
                            self.problem_dim,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.weights_norm,
                            self.fun_act,
                            self.FFTnorm,
                        )
                        for _ in range(self.L)
                    ]
                )

        elif self.arc == "Zongyi":
            if self.RNN:
                if self.problem_dim == 1:
                    self.ws = torch.nn.Conv1d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 2:
                    self.ws = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 3:
                    self.ws = torch.nn.Conv3d(self.d_v, self.d_v, 1)
                self.mlps = MLP_conv(
                    self.problem_dim, self.d_v, self.d_v, self.d_v, self.fun_act
                )
                self.integrals = FourierLayer(
                    self.problem_dim,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.weights_norm,
                    self.fun_act,
                    self.FFTnorm,
                )
            else:
                if self.problem_dim == 1:
                    self.ws = nn.ModuleList(
                        [torch.nn.Conv1d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 2:
                    self.ws = nn.ModuleList(
                        [torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 3:
                    self.ws = nn.ModuleList(
                        [torch.nn.Conv3d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )

                self.mlps = nn.ModuleList(
                    [
                        MLP_conv(
                            self.problem_dim, self.d_v, self.d_v, self.d_v, self.fun_act
                        )
                        for _ in range(self.L)
                    ]
                )
                self.integrals = nn.ModuleList(
                    [
                        FourierLayer(
                            self.problem_dim,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.weights_norm,
                            self.fun_act,
                            self.FFTnorm,
                        )
                        for _ in range(self.L)
                    ]
                )

        elif self.arc == "Classic" or self.arc == "Residual":
            if self.RNN:
                if self.problem_dim == 1:
                    self.ws = torch.nn.Conv1d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 2:
                    self.ws = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                elif self.problem_dim == 3:
                    self.ws = torch.nn.Conv3d(self.d_v, self.d_v, 1)

                self.integrals = FourierLayer(
                    self.problem_dim,
                    self.d_v,
                    self.d_v,
                    self.modes,
                    self.weights_norm,
                    self.fun_act,
                    self.FFTnorm,
                )
            else:
                if self.problem_dim == 1:
                    self.ws = nn.ModuleList(
                        [nn.Conv1d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 2:
                    self.ws = nn.ModuleList(
                        [nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )
                elif self.problem_dim == 3:
                    self.ws = nn.ModuleList(
                        [nn.Conv3d(self.d_v, self.d_v, 1) for _ in range(self.L)]
                    )

                self.integrals = nn.ModuleList(
                    [
                        FourierLayer(
                            self.problem_dim,
                            self.d_v,
                            self.d_v,
                            self.modes,
                            self.weights_norm,
                            self.fun_act,
                            self.FFTnorm,
                        )
                        for _ in range(self.L)
                    ]
                )

        ## Projection
        # self.q = torch.nn.Linear(self.d_v, self.out_dim)
        self.q = MLP(self.problem_dim, self.d_v, self.out_dim, 128, self.fun_act)

        ## eigenvalues
        # scale = 1 / self.d_v
        # self.eigenvalues = nn.Parameter(
        #     scale * torch.rand(self.out_dim, dtype=torch.float)
        # )
        # for a == 1
        self.eigenvalues = torch.tensor(
            [1 / i**2 for i in range(1, self.out_dim + 1)], device=device
        )

        ## Move to device
        self.to(device)

        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            self._enable_compilation()

    # @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_batch *n_x {self.in_dim+1}"]
    ) -> Float[Tensor, "n_batch *n_x {1}"]:

        rhs = x[..., [1]]
        x = x[..., [0]]

        ## Grid and initialization
        if self.problem_dim == 1:
            grid = self.get_grid_1d(x.shape).to(x.device)
        elif self.problem_dim == 2:
            grid = self.get_grid_2d(x.shape).to(x.device)
        elif self.problem_dim == 3:
            grid = self.get_grid_3d(x.shape).to(x.device)

        x = torch.cat(
            (grid, x), dim=-1
        )  # concatenate last dimension --> (n_samples)*(*n_x)*(in_dim+problem_dim)

        ## Perform lifting operator P
        x = self.p(x)  # shape = (n_samples)*(*n_x)*(in_dim+problem_dim)

        ## Reshaping
        if self.problem_dim == 1:
            x = x.permute(0, 2, 1)  # (n_samples)*(d_v)*(*n_x)
        elif self.problem_dim == 2:
            x = x.permute(0, 3, 1, 2)  # (n_samples)*(d_v)*(*n_x)
        elif self.problem_dim == 3:
            x = x.permute(0, 4, 1, 2, 3)  # (n_samples)*(d_v)*(*n_x)

        ## Padding
        if self.padding > 0 and self.problem_dim == 1:
            x = F.pad(x, [0, self.padding])
        elif self.padding > 0 and self.problem_dim == 2:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        elif self.padding > 0 and self.problem_dim == 3:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

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

        ## Padding
        if self.padding > 0 and self.problem_dim == 1:
            x = x[..., : -self.padding]
        elif self.padding > 0 and self.problem_dim == 2:
            x = x[..., : -self.padding, : -self.padding]
        elif self.padding > 0 and self.problem_dim == 3:
            x = x[..., : -self.padding, : -self.padding, : -self.padding]

        ## Perform projection (Q)
        if self.problem_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.problem_dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif self.problem_dim == 3:
            x = x.permute(0, 2, 3, 4, 1)

        x = self.q(x)  # shape --> (n_samples)*(*n_x)*(out_dim)

        x = self.orthonormalize_functions(
            x,
            torch.div(1, torch.tensor(x.shape)[1:-1]),
            method="svd",
        )

        # Check orthonormalization
        # Gram, ortho = self.verify_orthonormality_batch(
        #         x, torch.div(1, torch.tensor(x.shape)[1:-1])
        #     )
        # print("Orthonormality check for each  function of the batch:", ortho)

        # Compute <f, v_j> for every batch
        inner_products = self.batch_inner_product(
            x, rhs, torch.div(1, torch.tensor(x.shape)[1:-1])
        )

        # Compute \sum_{j=1}^{n_{eig}} c_j <f, v_j> v_j
        if self.problem_dim == 1:
            x = torch.einsum("bxv,bv,v->bx", x, inner_products, self.eigenvalues)
        elif self.problem_dim == 2:
            x = torch.einsum("bxyv,bv,v->bxyv", x, inner_products, self.eigenvalues)
        elif self.problem_dim == 3:
            x = torch.einsum("bxyzv,bv,v->bxyzv", x, inner_products, self.eigenvalues)

        return self.output_denormalizer(x.unsqueeze(-1))

    @jaxtyped(typechecker=beartype)
    def orthonormalize_functions(
        self,
        functions: Float[Tensor, "n_samples *n_x {self.out_dim}"],
        spacing: Union[float, List[float], torch.Tensor],
        method: str = "svd",
        device: Union[torch.device, None] = None,
    ) -> Float[Tensor, "n_samples *n_x {self.out_dim}"]:
        """
        Orthonormalizes functions in L^2 space using PyTorch.

        Args:
            functions: Tensor with batch dimension first, then spatial dims, then functions:
                    - 1D: (batch_size, n_points, n_functions)
                    - 2D: (batch_size, n_x, n_y, n_functions)
                    - 3D: (batch_size, n_x, n_y, n_z, n_functions)

            spacing: Grid spacing(s). Can be:
                    - float: uniform spacing for all dimensions
                    - List[float]: spacing for each dimension [dx, dy, dz]
                    - torch.Tensor: spacing tensor

            method (optional): 'svd' or 'gram_schmidt'

            device (optional): torch device

        Returns:
            Orthonormal functions tensor of same shape as input
        """
        if device is None:
            device = functions.device

        functions = functions.to(device)

        # Get dimensions
        batch_size, *spatial_dims, n_functions = functions.shape
        n_points = torch.prod(torch.tensor(spatial_dims)).item()

        # Handle spacing
        if isinstance(spacing, (int, float)):
            volume_element = spacing ** len(spatial_dims)
        elif isinstance(spacing, (list, tuple)):
            volume_element = torch.prod(torch.tensor(spacing, device=device))
        else:
            volume_element = torch.prod(spacing)

        # Flatten spatial dimensions: (batch_size, n_points, n_functions)
        functions_flat = functions.view(batch_size, n_points, n_functions)

        if method == "svd":
            return self._orthonormalize_svd_batch(
                functions_flat, volume_element.item(), spatial_dims
            )
        elif method == "gram_schmidt":
            return self._orthonormalize_gram_schmidt_batch(
                functions_flat, volume_element.item(), spatial_dims
            )
        else:
            raise ValueError("method must be 'svd' or 'gram_schmidt'")

    @jaxtyped(typechecker=beartype)
    def _orthonormalize_svd_batch(
        self,
        functions_flat: Float[Tensor, "batch_size n_points {self.out_dim}"],
        volume_element: float,
        spatial_dims: List[int],
    ) -> Float[Tensor, "batch_size *n_x {self.out_dim}"]:
        """SVD-based orthonormalization for batched input (numerically stable)"""
        batch_size, n_points, n_functions = functions_flat.shape

        # Weight matrix for L^2 inner product
        sqrt_vol = torch.sqrt(
            torch.tensor(volume_element, device=functions_flat.device)
        )
        weighted_functions = sqrt_vol * functions_flat

        # Batch reduced-SVD decomposition
        U, S, _ = torch.linalg.svd(weighted_functions, full_matrices=False)

        # Remove numerically zero singular values for each batch
        tol = 1e-12
        orthonormal_flat = torch.zeros_like(functions_flat)

        for b in range(batch_size):
            # Find rank for this batch
            if len(S[b]) > 0:
                batch_tol = tol * S[b, 0]
                rank = torch.sum(S[b] > batch_tol).item()
            else:
                rank = 0

            if rank > 0:
                # Orthonormal basis in original space for this batch
                orthonormal_flat[b, :, :rank] = U[b, :, :rank] / sqrt_vol

        # Reshape back to original spatial dimensions
        return orthonormal_flat.view(batch_size, *spatial_dims, n_functions)

    def _orthonormalize_gram_schmidt_batch(
        self,
        functions_flat: torch.Tensor,
        volume_element: float,
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """Gram-Schmidt orthonormalization for batched input"""
        batch_size, n_points, n_functions = functions_flat.shape
        Q = torch.zeros_like(functions_flat)

        for b in range(batch_size):
            for j in range(n_functions):
                # Start with j-th function in batch b
                v = functions_flat[b, :, j].clone()

                # Subtract projections onto previous orthonormal functions
                for i in range(j):
                    # L^2 inner product: <v, q_i> = \int v(x) q_i(x) dx \approx \delta x * \sum v(x_k) q_i(x_k)
                    inner_product = torch.sum(v * Q[b, :, i]) * volume_element
                    v = v - inner_product * Q[b, :, i]

                # Normalize: ||v||_L^2 = sqrt(int |v(x)|^2 dx)
                norm_squared = torch.sum(v * v) * volume_element
                norm = torch.sqrt(norm_squared)

                if norm > 1e-12:
                    Q[b, :, j] = v / norm
                else:
                    # Function is linearly dependent, set to zero
                    Q[b, :, j] = 0

        # Reshape back to original spatial dimensions
        return Q.view(batch_size, *spatial_dims, n_functions)

    def verify_orthonormality_batch(
        self,
        functions: torch.Tensor,
        spacing: Union[float, List[float], torch.Tensor],
        tol: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify that functions form an orthonormal set in L^2 for each batch.

        Returns:
            gram_matrices: Gram matrices for each batch (batch_size, n_functions, n_functions)
            is_orthonormal: Boolean tensor indicating orthonormality for each batch (batch_size,)
        """
        batch_size, *spatial_dims, n_functions = functions.shape
        n_points = torch.prod(torch.tensor(spatial_dims)).item()

        # Handle spacing
        if isinstance(spacing, (int, float)):
            volume_element = spacing ** len(spatial_dims)
        elif isinstance(spacing, (list, tuple)):
            volume_element = torch.prod(torch.tensor(spacing, device=functions.device))
        else:
            volume_element = torch.prod(spacing)

        # Flatten for easier computation
        functions_flat = functions.view(batch_size, n_points, n_functions)

        # Compute Gram matrices for all batches
        gram_matrices = torch.zeros(
            batch_size, n_functions, n_functions, device=functions.device
        )

        for b in range(batch_size):
            for i in range(n_functions):
                for j in range(n_functions):
                    gram_matrices[b, i, j] = (
                        torch.sum(functions_flat[b, :, i] * functions_flat[b, :, j])
                        * volume_element
                    )

        # Check if each is approximately identity
        identity = (
            torch.eye(n_functions, device=functions.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        is_orthonormal = torch.all(
            torch.all(torch.abs(gram_matrices - identity) < tol, dim=-1), dim=-1
        )

        return gram_matrices, is_orthonormal

    def batch_inner_product(
        self,
        functions1: torch.Tensor,
        functions2: torch.Tensor,
        spacing: Union[float, List[float], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute L^2 inner products between corresponding functions in two batches.

        Args:
            functions1, functions2: Tensors of shape (batch_size, *spatial_dims, n_functions)
            spacing: Grid spacing(s)

        Returns:
            Inner products tensor of shape (batch_size, n_functions, n_functions)
            where result[b, i, j] = ⟨functions1[b, :, i], functions2[b, :, j]⟩
        """
        batch_size, *spatial_dims, n_functions_1 = functions1.shape
        n_functions_2 = functions2.shape[-1]
        n_points = torch.prod(torch.tensor(spatial_dims)).item()

        # Handle spacing
        if isinstance(spacing, (int, float)):
            volume_element = spacing ** len(spatial_dims)
        elif isinstance(spacing, (list, tuple)):
            volume_element = torch.prod(torch.tensor(spacing, device=functions1.device))
        else:
            volume_element = torch.prod(spacing)

        # Flatten spatial dimensions
        f1_flat = functions1.view(batch_size, n_points, n_functions_1)
        f2_flat = functions2.view(batch_size, n_points, n_functions_2)

        # Compute inner products: (batch_size, n_functions, n_functions)
        inner_products = torch.einsum("bpi,bpj->bij", f1_flat, f2_flat) * volume_element

        return inner_products.squeeze()

    @cache
    def get_grid_3d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        # grid for y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        # grid for z
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        return torch.cat((gridx, gridy, gridz), dim=-1)

    @cache
    def get_grid_2d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # grid for y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    @cache
    def get_grid_1d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x = shape[0], shape[1]
        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx

    def _enable_compilation(self) -> None:
        """Enable PyTorch 2.0+ compilation for performance if available."""
        try:
            # This is a PyTorch 2.0+ feature
            self = torch.compile(self)
            print("PyTorch compilation enabled for better performance")
        except Exception as e:
            print(f"Could not enable PyTorch compilation: {e}")
