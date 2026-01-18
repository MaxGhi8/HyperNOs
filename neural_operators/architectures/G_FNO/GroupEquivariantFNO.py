"""
This file contains the group equivariant architectures and modules of the G-FNO for 2D and 3D cases,
refactored to match the structure of the standard FNO implementation.
"""

import math
from functools import cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped
from torch import Tensor


#########################################
# Activation Functions
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
# GConv2d
#########################################
class GConv2d(nn.Module):
    """
    Group equivariant 2D convolutional layer.
    This is needed both for the lifting and the projection, to implement G-equivariance convolution with kernel size 1;
    both for the kernel integral layer to multiply the Fourier transform.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        first_layer: bool = False,
        last_layer: bool = False,
        spectral: bool = False,
        Hermitian: bool = False,  #!
        reflection: bool = False,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel size must be odd"

        dtype = torch.cfloat if spectral else torch.float

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.Hermitian = Hermitian
        self.first_layer = first_layer
        self.last_layer = last_layer

        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)

        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size

        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None

        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    dtype=dtype,
                )
            )

        else:
            if self.Hermitian:
                self.W = nn.ParameterDict(
                    {
                        "00_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                1,
                                1,
                                dtype=torch.float,
                            )
                        ),
                        "y0_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_X - 1,
                                1,
                                dtype=dtype,
                            )
                        ),
                        "yposx_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_Y,
                                self.kernel_size_X - 1,
                                dtype=dtype,
                            )
                        ),
                    }
                )

            else:
                self.W = nn.Parameter(
                    torch.empty(
                        out_channels,
                        1,
                        in_channels,
                        self.group_size,
                        self.kernel_size_Y,
                        self.kernel_size_X,
                        dtype=dtype,
                    )
                )

        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        # Initialize weights W
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        # Initialize bias B
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        # Impose Hermitian symmetry and construct the final weights
        # After that the weights have the shape (out_channels, 1, in_channels, *group_size, kernel_size, kernel_size)
        # Moreover self.kernel_size_Y = kernel_size
        if self.Hermitian:
            self.weights = torch.cat(
                [
                    self.W["y0_modes"].conj().flip(dims=[-2]),
                    self.W["00_modes"],
                    self.W["y0_modes"],
                ],
                dim=-2,
            )
            self.weights = torch.cat(
                [
                    self.W["yposx_modes"].conj().rot90(k=2, dims=[-2, -1]),
                    self.weights,
                    self.W["yposx_modes"],
                ],
                dim=-1,
            )
        else:
            self.weights = self.W[:]

        # Adjust weights for first and last layer
        if self.first_layer or self.last_layer:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)

            # Rotate weights for p4 group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])

            # Reflect weights for p4m group
            if self.reflection:
                # self.rt_group_size = 4
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(dims=[-2])

            if self.first_layer:
                self.weights = self.weights.view(
                    -1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            elif self.last_layer:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y
                )
                self.bias = self.B

        # Adjust weights for intermediate layers
        else:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # Rotate weights for p4 group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                # Reflect weights for p4m group
                if self.reflection:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : (self.rt_group_size - 1)],
                            self.weights[:, k, :, (self.rt_group_size + 1) :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )

            if self.reflection:
                self.weights[:, self.rt_group_size :] = torch.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip([-2])

            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        # Hermitian symmetry
        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X :]

    def forward(self, x):
        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        if self.B is not None:
            x = x + self.bias

        return x


#########################################
# GConv3d
#########################################
class GConv3d(nn.Module):
    """
    Group equivariant 3D convolutional layer.
    Similar to GConv2d, but operates on (batch, channels, x, y, t).
    Used for lifting (scalar->group), projection (group->scalar), and group convolutions (group->group).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernel_size_T,
        bias=True,
        first_layer=False,
        last_layer=False,
        spectral=False,
        Hermitian=False,
        reflection=False,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel size must be odd"

        dtype = torch.cfloat if spectral else torch.float

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.Hermitian = Hermitian
        self.first_layer = first_layer
        self.last_layer = last_layer

        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)

        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.kernel_size_T_full = kernel_size_T
        self.kernel_size_T = kernel_size_T // 2 + 1 if Hermitian else kernel_size_T

        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    self.kernel_size_T,
                    dtype=dtype,
                )
            )
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict(
                    {
                        "y00_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_X - 1,
                                1,
                                1,
                                dtype=torch.cfloat,
                            )
                        ),
                        "yposx0_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_Y,
                                self.kernel_size_X - 1,
                                1,
                                dtype=torch.cfloat,
                            )
                        ),
                        "000_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels, 1, in_channels, self.group_size, 1, 1, 1
                            )
                        ),
                        "yxpost_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_Y,
                                self.kernel_size_Y,
                                self.kernel_size_T - 1,
                                dtype=torch.cfloat,
                            )
                        ),
                    }
                )
            else:
                self.W = nn.Parameter(
                    torch.empty(
                        out_channels,
                        1,
                        in_channels,
                        self.group_size,
                        self.kernel_size_Y,
                        self.kernel_size_X,
                        self.kernel_size_T,
                        dtype=dtype,
                    )
                )

        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1, 1)) if bias else None

        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        # Impose Hermitian symmetry and construct the final weights
        # After that the weights have the shape (out_channels, 1, in_channels, *group_size, kernel_size, kernel_size, kernel_size)
        # Moreover self.kernel_size_Y = kernel_size
        if self.Hermitian:
            self.weights = torch.cat(
                [
                    self.W["y00_modes"].conj().flip((-3,)),
                    self.W["000_modes"],
                    self.W["y00_modes"],
                ],
                dim=-3,
            )
            self.weights = torch.cat(
                [
                    self.W["yposx0_modes"].conj().rot90(k=2, dims=[-3, -2]),
                    self.weights,
                    self.W["yposx0_modes"],
                ],
                dim=-2,
            )
            self.weights = torch.cat(
                [
                    self.W["yxpost_modes"].conj().rot90(k=2, dims=[-3, -2]).flip((-1,)),
                    self.weights,
                    self.W["yxpost_modes"],
                ],
                dim=-1,
            )
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:
            # For lifting (first_layer) or projection (last_layer), we deal with scalar <-> group transitions.
            # We first expand the weights to match the group size.
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # Rotate weights for p4 group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-3, -2])

            # Reflect weights for p4m group
            if self.reflection:
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(dims=[-3])

            if self.first_layer:
                self.weights = self.weights.view(
                    -1,
                    self.in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_Y,
                    self.kernel_size_T,
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels,
                    -1,
                    self.kernel_size_Y,
                    self.kernel_size_Y,
                    self.kernel_size_T,
                )
                self.bias = self.B
        else:
            # Case: Group -> Group convolution (hidden layers)
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1, 1)

            # Rotate weights for p4 group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-3, -2])

                # Reflect weights for p4m group
                if self.reflection:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : (self.rt_group_size - 1)],
                            self.weights[:, k, :, (self.rt_group_size + 1) :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )

            if self.reflection:
                self.weights[:, self.rt_group_size :] = torch.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip([-3])

            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
                self.kernel_size_T_full,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        # Hermitian symmetry
        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_T :]

    def forward(self, x):
        self.get_weight()

        x = nn.functional.conv3d(input=x, weight=self.weights)

        if self.B is not None:
            x = x + self.bias

        return x


########################################
# G_MLP
########################################
class G_MLP(nn.Module):
    """
    Unified MLP/Conv structure for G-FNO.
    Equivalent to GMLP2d / GMLP3d but integrated with the FNO style.
    """

    def __init__(
        self,
        problem_dim,
        in_channels,
        out_channels,
        mid_channels,
        reflection=False,
        last_layer=False,
        first_layer=False,
        fun_act="gelu",
    ):
        super(G_MLP, self).__init__()
        self.problem_dim = problem_dim
        self.fun_act = fun_act

        if self.problem_dim == 2:
            self.mlp1 = GConv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                reflection=reflection,
                first_layer=first_layer,
            )
            self.mlp2 = GConv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                reflection=reflection,
                last_layer=last_layer,
            )

        elif self.problem_dim == 3:
            self.mlp1 = GConv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                kernel_size_T=1,
                reflection=reflection,
                first_layer=first_layer,
            )
            self.mlp2 = GConv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                kernel_size_T=1,
                reflection=reflection,
                last_layer=last_layer,
            )
        else:
            raise ValueError("G_FNO only supports 2D and 3D problems.")

    def forward(self, x):

        x = self.mlp1(x)
        x = activation(x, self.fun_act)
        x = self.mlp2(x)

        return x


########################################
# G_FourierLayer
########################################
class G_FourierLayer(nn.Module):
    """
    Unified Group Fourier Layer.
    Wraps GSpectralConv2d/GSpectralConv3d logic.
    """

    def __init__(
        self,
        problem_dim,
        in_channels,
        out_channels,
        modes,
        time_modes=None,
        reflection=False,
    ):
        super(G_FourierLayer, self).__init__()
        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.reflection = reflection

        if self.problem_dim == 2:
            # GSpectralConv2d logic
            self.conv = GConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * modes - 1,
                reflection=reflection,
                bias=False,
                spectral=True,
                Hermitian=True,
            )

        elif self.problem_dim == 3:
            # GSpectralConv3d logic
            assert time_modes is not None, "time_modes must be provided for 3D GFNO"
            self.time_modes = time_modes
            self.conv = GConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * modes - 1,
                kernel_size_T=2 * time_modes - 1,
                reflection=reflection,
                bias=False,
                spectral=True,
                Hermitian=True,
            )

        else:
            raise ValueError("G_FNO only supports 2D and 3D problems.")

        self.get_weight()

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        self.get_weight()

        if self.problem_dim == 2:
            freq0_y = (
                (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0)
                .nonzero()
                .item()
            )

            # RFFT2
            x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
            x_ft = x_ft[
                ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
            ]

            # Weighted multiplication
            out_ft = torch.zeros(
                batchsize,
                self.weights.shape[0],
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[
                ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
            ] = self.compl_mul2d(x_ft, self.weights)

            # IRFFT2
            x = torch.fft.irfft2(
                torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
            )

        elif self.problem_dim == 3:
            freq0_x = (
                (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0)
                .nonzero()
                .item()
            )
            freq0_y = (
                (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-3])) == 0)
                .nonzero()
                .item()
            )

            # RFFT3
            x_ft = torch.fft.fftshift(
                torch.fft.rfftn(x, dim=[-3, -2, -1]), dim=[-3, -2]
            )
            x_ft = x_ft[
                ...,
                (freq0_y - self.modes + 1) : (freq0_y + self.modes),
                (freq0_x - self.modes + 1) : (freq0_x + self.modes),
                : self.time_modes,
            ]

            # Weighted multiplication
            out_ft = torch.zeros(
                batchsize,
                self.weights.shape[0],
                x.size(-3),
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[
                ...,
                (freq0_y - self.modes + 1) : (freq0_y + self.modes),
                (freq0_x - self.modes + 1) : (freq0_x + self.modes),
                : self.time_modes,
            ] = self.compl_mul3d(x_ft, self.weights)

            # IRFFT3
            x = torch.fft.irfftn(
                torch.fft.ifftshift(out_ft, dim=[-3, -2]),
                s=(x.size(-3), x.size(-2), x.size(-1)),
            )

        return x


########################################
# G_FNO
########################################
class G_FNO(nn.Module):
    """
    Group equivariant FNO.
    """

    def __init__(
        self,
        problem_dim: int,
        in_dim: int,
        d_v: int,  # width (mid_channels)
        out_dim: int,
        L: int,
        modes: int,
        time_modes: int = None,
        fun_act: str = "gelu",
        reflection: bool = False,
        padding: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super(G_FNO, self).__init__()
        self.problem_dim = problem_dim
        self.in_dim = in_dim
        self.d_v = d_v
        self.out_dim = out_dim
        self.L = L
        self.modes = modes
        self.time_modes = time_modes
        self.fun_act = fun_act
        self.reflection = reflection
        self.padding = padding
        self.device = device
        self.group_size = 4 * (1 + reflection)

        # Lifting (P)
        self.p = G_MLP(
            problem_dim=self.problem_dim,
            in_channels=self.in_dim,
            out_channels=self.d_v,
            mid_channels=self.d_v * 4,
            reflection=reflection,
            first_layer=True,
            fun_act=self.fun_act,
        )
        # Linear Projection case
        # if self.problem_dim == 2:
        #     self.p = GConv2d(
        #         in_channels=self.in_dim,
        #         out_channels=self.d_v,
        #         kernel_size=1,
        #         reflection=reflection,
        #         first_layer=True,
        #     )
        # elif self.problem_dim == 3:
        #     self.p = GConv3d(
        #         in_channels=self.in_dim,
        #         out_channels=self.d_v,
        #         kernel_size=1,
        #         kernel_size_T=1,
        #         reflection=reflection,
        #         first_layer=True,
        #     )

        # Layers
        self.integrals = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.ws = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.L):
            # Integral Fourier Layer
            self.integrals.append(
                G_FourierLayer(
                    problem_dim=self.problem_dim,
                    in_channels=self.d_v,
                    out_channels=self.d_v,
                    modes=self.modes,
                    time_modes=self.time_modes,
                    reflection=reflection,
                )
            )

            # MLP
            self.mlps.append(
                G_MLP(
                    problem_dim=self.problem_dim,
                    in_channels=self.d_v,
                    out_channels=self.d_v,
                    mid_channels=self.d_v,
                    reflection=reflection,
                )
            )

            # Affine weights for skip connection
            if self.problem_dim == 2:
                self.ws.append(
                    GConv2d(
                        in_channels=self.d_v,
                        out_channels=self.d_v,
                        kernel_size=1,
                        reflection=reflection,
                    )
                )
            elif self.problem_dim == 3:
                self.ws.append(
                    GConv3d(
                        in_channels=self.d_v,
                        out_channels=self.d_v,
                        kernel_size=1,
                        kernel_size_T=1,
                        reflection=reflection,
                    )
                )

            # Original G-FNO uses GNorm (InstanceNorm3d).
            self.norms.append(
                nn.InstanceNorm3d(self.d_v)
                if self.problem_dim == 2
                else nn.InstanceNorm3d(self.d_v)
            )  # We will handle reshaping in forward

        # Projection (Q)
        self.q = G_MLP(
            problem_dim=self.problem_dim,
            in_channels=self.d_v,
            out_channels=self.out_dim,
            mid_channels=self.d_v * 4,
            reflection=reflection,
            last_layer=True,
            fun_act=self.fun_act,
        )
        # Linear Projection case
        # if self.problem_dim == 2:
        #     self.p = GConv2d(
        #         in_channels=self.d_v,
        #         out_channels=self.out_dim,
        #         kernel_size=1,
        #         reflection=reflection,
        #         last_layer=True,
        #     )
        # elif self.problem_dim == 3:
        #     self.p = GConv3d(
        #         in_channels=self.d_v,
        #         out_channels=self.out_dim,
        #         kernel_size=1,
        #         kernel_size_T=1,
        #         reflection=reflection,
        #         last_layer=True,
        #     )

        self.to(device)

    def g_norm(self, x, norm_layer):

        if self.problem_dim == 2:
            # x: (batch, channels*group_size, x, y)
            batch, c_g, nx, ny = x.shape
            c = c_g // self.group_size
            x = x.view(batch, c, self.group_size, nx, ny)
            x = norm_layer(x)
            x = x.view(batch, c_g, nx, ny)

        elif self.problem_dim == 3:
            # x: (batch, channels*group_size, x, y, t)
            pass

        return x

    def forward(self, x):
        # x shape: (batch, in_channels, *spatial_dims)

        #! no grid concatenation for the moment (it is not translation invariant)
        # Grid concatenation
        # if self.problem_dim == 2:
        #     grid = self.get_grid_2d(x.shape).to(x.device)

        # elif self.problem_dim == 3:
        #     grid = self.get_grid_3d(x.shape).to(x.device)

        # else:
        #     raise ValueError("Dimension not supported")
        # x = torch.cat((grid, x), dim=1)  # dim 1 is channels

        # Lifting P
        x = self.p(x)

        # Layers
        for i in range(self.L):

            # Integral
            x1 = self.integrals[i](x)

            # Note: For 3D, original code didn't use norm. We stick to that.
            if self.problem_dim == 2:
                x1 = self.g_norm(x1, self.norms[i])

            # MLP
            x1 = self.mlps[i](x1)

            # W
            x2 = self.ws[i](x)

            # Sum
            x = x1 + x2

            # Activation
            if i < self.L - 1:
                x = activation(x, self.fun_act)
            else:
                x = activation(
                    x, self.fun_act
                )  # At the moment I use the same activation for the last layer

        # Padding
        if self.padding > 0:
            if self.problem_dim == 2:
                x = x[..., : -self.padding, : -self.padding]
            elif self.problem_dim == 3:
                x = x[..., : -self.padding, : -self.padding, : -self.padding]

        # Projection Q
        x = self.q(x)

        return x

    @cache
    def get_grid_2d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)

    @cache
    def get_grid_3d(self, shape: torch.Size) -> Tensor:
        batchsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat(
            [batchsize, 1, 1, size_y, size_z]
        )
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat(
            [batchsize, 1, size_x, 1, size_z]
        )
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, 1, size_z).repeat(
            [batchsize, 1, size_x, size_y, 1]
        )
        return torch.cat((gridx, gridy, gridz), dim=1)

#########################################
# Grids
#########################################
class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()
    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)