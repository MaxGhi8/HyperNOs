"""Chebyshev transform and its inverse using PyTorch"""

from typing import Union

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

# torch.set_default_dtype(torch.float32)  # default tensor dtype

#########################################
# Functions for the change of basis
#########################################
def change_basis(x: Tensor, C: Tensor) -> Tensor:
    """
    Function to change the basis of the input x.
    x: torch.tensor
        x is a tensor of shape (modes, modes), coefficients to be transformed.
    C: torch.tensor
        C is a tensor of shape (modes, modes), matrix for the change of basis.
    """
    y = torch.matmul(torch.transpose(C, 0, 1), torch.matmul(x, C))
    return y


batched_change_basis = torch.vmap(
    torch.vmap(change_basis, in_dims=(-1, None), out_dims=-1),  # d_v dimension
    in_dims=(0, None),
    out_dims=0,
)  # batch dimension

patched_change_basis = torch.vmap(
    batched_change_basis, in_dims=(0, None), out_dims=0
)  # patch dimension


#########################################
# Grid generation
#########################################
@jaxtyped(typechecker=beartype)
def Chebyshev_grid_1d(n: int, a: float = -1.0, b: float = 1.0) -> Float[Tensor, "{n}"]:
    """
    Chebyshev grid 1D (second kind) from a to b with n points

    n:  int
        number of points

    a:  float
        initial value

    b:  float
        final value
    """
    grid = -torch.cos(torch.arange(n) * torch.pi / (n - 1))
    return ((b - a) / 2) * grid + (a + b) / 2


@jaxtyped(typechecker=beartype)
def Chebyshev_grid_2d(
    n: Union[int, list[int]], a: list[float] = [-1.0, -1.0], b: list[float] = [1.0, 1.0]
) -> Float[Tensor, "*n 2"]:
    """
    Chebyshev 2D grid (second kind)

    n:  list[int] or int
        number of points in x, y direction. If int is given, n = m

    a:  list
        contains the initial value for x and y (in this order)

    b:  list
        contains the final value for x and y (in this order)
    """
    n = n if isinstance(n, list) else [n] * 2
    # grid for x
    gridx = Chebyshev_grid_1d(n[0], a[0], b[0])
    gridx = gridx.view(n[0], 1, 1).repeat([1, n[1], 1])
    # grid for y
    gridy = Chebyshev_grid_1d(n[1], a[1], b[1])
    gridy = gridy.view(1, n[1], 1).repeat([n[0], 1, 1])
    return torch.cat((gridx, gridy), dim=-1)


#########################################
# Chebyshev Fast Transform (CFT)
#########################################
@jaxtyped(typechecker=beartype)
def values_to_coefficients(values: Float[Tensor, "... c"]) -> Float[Tensor, "... c"]:
    # input array has shape `(n1, n2, ..., c)`, where c stands for channels
    # transform values of the function on the Chebyshev grid to coefficients of Chebyshev series
    D = values.shape
    for N in D[:-1]:
        values = torch.cat((values, values[1:-1].flip(0)), dim=0)
        values = torch.real(torch.fft.rfft(values, dim=0)).div_(N - 1)
        values[0].div_(2)
        values[-1].div_(2)
        values = values * (
            (-1) ** torch.arange(N, dtype=torch.int, device=values.device)
        ).view([-1] + [1] * (len(D) - 1))
        values = torch.permute(
            values, tuple([i for i in range(1, len(D) - 1)]) + tuple((0, len(D) - 1))
        )

    return values


batched_values_to_coefficients = torch.vmap(
    values_to_coefficients
)  # batch_size is the first dimension and it is vmapped
patched_values_to_coefficients = torch.vmap(
    batched_values_to_coefficients
)  # patch_size is the first dimension and it is vmapped


#########################################
# Inverse Chebyshev Fast Transform (ICFT)
#########################################
@jaxtyped(typechecker=beartype)
def coefficients_to_values(
    coefficients: Float[Tensor, "... c"],
) -> Float[Tensor, "... c"]:
    D = coefficients.shape
    for N in D[:-1]:
        coefficients = coefficients * (
            (-1) ** torch.arange(N, dtype=torch.int, device=coefficients.device)
        ).view([-1] + [1] * (len(D) - 1))
        coefficients[0].mul_(2)
        coefficients[-1].mul_(2)
        coefficients = torch.cat((coefficients, coefficients[1:-1].flip(0)), dim=0)
        coefficients = torch.real(torch.fft.rfft(coefficients, dim=0)).div_(2)
        coefficients = torch.permute(
            coefficients,
            tuple([i for i in range(1, len(D) - 1)]) + tuple((0, len(D) - 1)),
        )
    return coefficients


batched_coefficients_to_values = torch.vmap(
    coefficients_to_values
)  # batch_size is the first dimension and it is vmapped
patched_coefficients_to_values = torch.vmap(
    batched_coefficients_to_values
)  # patch_size is the first dimension and it is vmapped


#########################################
# Differentiation in Chebyshev space
#########################################
def differentiate(coeff: Tensor, axis: int = 0) -> Tensor:
    # find derivative along 'axis'
    shape = coeff.shape
    n = shape[axis]
    A = torch.from_numpy(-np.eye(n - 1, k=+2) + np.eye(n - 1, k=0)).to(coeff.device)
    A[0, 0] = 2
    transposition = (axis,) + tuple([i for i in range(len(shape)) if i != axis])
    inv_transposition = (
        tuple([i for i in range(1, axis + 1)])
        + (0,)
        + tuple([i for i in range(axis + 1, len(shape))])
    )
    coeff = torch.permute(coeff, transposition)
    shape_ = coeff.shape
    w = ((torch.arange(1, shape_[0], device=coeff.device)).mul_(2)).view(-1, 1)
    coeff = torch.linalg.solve_triangular(
        A, coeff.reshape((shape_[0], -1))[1:] * w, upper=True
    )
    coeff = coeff.view((shape_[0] - 1,) + tuple(shape_[1:]))
    coeff = torch.permute(coeff, inv_transposition)
    
    # I add a zero at the end of the coeff expansion
    if axis == 0:
        coeff = torch.cat(
            (coeff, torch.zeros(1, *shape[1:], device=coeff.device)), axis=axis
        )
    elif axis == 1:
        coeff = torch.cat(
            (coeff, torch.zeros(shape[0], 1, *shape[2:], device=coeff.device)),
            axis=axis,
        )
    return coeff


batched_differentiate = torch.vmap(
    differentiate, in_dims=(0, None), out_dims=0
)  # batch_size is the first dimension

#### Implementation for 2D functions
# The vmaps are executed from the outside to the inside.
differentiate_2dx = torch.vmap(
    differentiate, in_dims=(1, None), out_dims=1
)  # y direction
batched_differentiate_2dx = torch.vmap(
    differentiate_2dx, in_dims=(0, None), out_dims=0
)  # batch dims
differentiate_2dy = torch.vmap(
    differentiate, in_dims=(0, None), out_dims=0
)  # x direction
batched_differentiate_2dy = torch.vmap(
    differentiate_2dy, in_dims=(0, None), out_dims=0
)  # batch dims


def batched_differentiate_2d(coeff: Tensor, axis: int = 0) -> Tensor:
    if axis == 0:
        return batched_differentiate_2dx(coeff, 0)
    elif axis == 1:
        return batched_differentiate_2dy(coeff, 0)
    else:
        raise ValueError("axis must be 0 or 1")


#########################################
# Integration in Chebyshev space
#########################################
@jaxtyped(typechecker=beartype)
def integrate(coeff: Float[Tensor, "... c"], axis: int) -> Float[Tensor, "... c"]:
    """find indefinite integral along `axis`"""
    n = coeff.shape[axis]
    sh = tuple([-1 if i == axis else 1 for i in range(len(coeff.shape))])
    w_0 = torch.hstack(
        [torch.tensor([1, 1 / 4]), (1 / torch.arange(3, n + 2)) / 2]
    ).view(sh)
    w_1 = torch.hstack([torch.tensor([0, 1 / 4]), -(1 / torch.arange(1, n)) / 2]).view(
        sh
    )
    # add a zero at the end of the axis
    tmp = []
    for i in range(len(coeff.shape) - 1, -1, -1):
        if i == axis:
            tmp.append(0)
            tmp.append(1)
        else:
            tmp.append(0)
            tmp.append(0)
    coeff = torch.nn.functional.pad(coeff, tuple(tmp))
    # compute d_k for k = 0, ..., n-1
    coeff = torch.roll(w_0 * coeff, 1, dims=axis) + torch.roll(
        w_1 * coeff, -1, dims=axis
    )

    # correct to have indefinite integral from -1 to x
    coeff[tuple([0] * (len(coeff.shape) - 1))] = 0
    # compute d_0
    w = (-1) ** torch.arange(n + 1, dtype=int)
    coeff = torch.moveaxis(coeff, axis, 0)
    coeff[0] = -(w + 0.0).matmul(coeff)
    coeff = torch.moveaxis(coeff, 0, axis)

    return coeff


batched_integrate = torch.vmap(
    integrate, in_dims=(0, None), out_dims=0
)  # batch_size is the first dimension
