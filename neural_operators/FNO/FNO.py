"""
This file contains all the core architectures and modules of the Fourier Neural Operator (FNO) for 1D and 2D cases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
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
    def forward(
        self, x: Float[Tensor, "n_batch {self.in_channels} *n_x"]
    ) -> Float[Tensor, "n_batch {self.out_channels} *n_x"]:
        """
        input --> FFT --> parameters --> IFFT --> output
        Total computation cost is equal to O(n log(n))

        input: torch.tensor
            the input 'x' is a tensor of shape (n_samples, in_channels, n_x, n_y)

        output: torch.tensor
            return a tensor of shape (n_samples, out_channels, n_x, n_y)
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

        return x


#########################################
# Fourier Neural Operator
#########################################
class FNO(nn.Module):
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
        super(FNO, self).__init__()
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

        ## Move to device
        self.to(device)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_batch *n_x {self.in_dim-self.problem_dim}"]
    ) -> Float[Tensor, "n_batch *n_x {self.out_dim}"]:

        ## Grid and initialization
        if self.problem_dim == 1:
            grid = self.get_grid_1d(x.shape).to(x.device)
        elif self.problem_dim == 2:
            grid = self.get_grid_2d(x.shape).to(x.device)
        x = torch.cat(
            (grid, x), dim=-1
        )  # concatenate last dimension --> (n_samples)*(*n_x)*(in_dim+problem_dim)

        ## Perform lifting operator P
        x = self.p(x)  # shape = (n_samples)*(*n_x)*(in_dim+problem_dim)
        if self.problem_dim == 1:
            x = x.permute(0, 2, 1)  # (n_samples)*(d_v)*(*n_x)
        elif self.problem_dim == 2:
            x = x.permute(0, 3, 1, 2)  # (n_samples)*(d_v)*(*n_x)

        ## Padding
        if self.padding > 0 and self.problem_dim == 1:
            x = F.pad(x, [0, self.padding])
        elif self.padding > 0 and self.problem_dim == 2:
            x = F.pad(x, [0, self.padding, 0, self.padding])

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

        ## Perform projection (Q)
        if self.problem_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.problem_dim == 2:
            x = x.permute(0, 2, 3, 1)  # (n_samples)*(*n_x)*(d_v)
        x = self.q(x)  # shape --> (n_samples)*(*n_x)*(out_dim)

        return x

    def get_grid_2d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        # grid for y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        # concatenate along the last dimension
        grid = torch.cat((gridy, gridx), dim=-1)
        return grid

    def get_grid_1d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x = shape[0], shape[1]
        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx
