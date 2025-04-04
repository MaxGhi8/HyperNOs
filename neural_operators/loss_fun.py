"""
This module contains the definition of the loss functions that can be used in the training of the Neural Operator.
"""

import torch
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped
from torch import Tensor


#########################################
# Loss function selector
#########################################
def loss_selector(loss_fn_str: str, problem_dim: int, beta: float = 1.0):
    match loss_fn_str.upper():
        case "L1":
            loss = LprelLoss(1, False)
        case "L2":
            loss = LprelLoss(2, False)
        case "l2":
            loss = lpLoss(2, False)
        case "H1":
            if problem_dim == 1:
                loss = H1relLoss_1D(beta, False, 1.0)
            elif problem_dim == 2:
                loss = H1relLoss(beta, False, 1.0)
        case "L1_SMOOTH":
            loss = torch.nn.SmoothL1Loss()  # L^1 smooth loss (Mishra)
        case "MSE":
            loss = torch.nn.MSELoss()  # L^2 smooth loss (Mishra)
        case _:
            raise ValueError("This value of p is not allowed")
    return loss


#########################################
# Smooth L1 relative loss
#########################################
class SmoothL1Loss_rel:
    def __init__(self, size_mean: bool = False):
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def Smooth_batched(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "n_samples"]:
        return torch.vmap(torch.nn.SmoothL1Loss(reduction="sum"))(x, y)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "*n_samples"]:

        diff_norms = self.Smooth_batched(x, y)
        y_norms = self.Smooth_batched(torch.zeros_like(y, device=y.device), y)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")

        if self.size_mean is True:
            return torch.mean(diff_norms / y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms / y_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms / y_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")


#########################################
# MSE relative loss
#########################################
class MSELoss_rel:
    def __init__(self, size_mean: bool = False):
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def MSE_batched(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "n_samples"]:
        return torch.vmap(torch.nn.MSELoss(reduction="sum"))(x, y)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "*n_samples"]:

        diff_norms = self.MSE_batched(x, y)
        y_norms = self.MSE_batched(torch.zeros_like(y, device=y.device), y)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")

        if self.size_mean is True:
            return torch.mean(diff_norms / y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms / y_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms / y_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")


#########################################
# l^p loss
#########################################
class lpLoss:
    """l^p loss for vectors"""

    def __init__(self, p: int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n"],
        y: Float[Tensor, "n_samples *n"],
    ) -> Float[Tensor, "*n_samples"]:

        diff_norms = torch.norm(x - y, p=self.p, dim=1)

        if self.size_mean is True:
            return torch.mean(diff_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")


#########################################
# L^p relative loss for N-D functions
#########################################
class LprelLoss:
    """
    Sum of relative errors in L^p norm

    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, out_dim)
          where *n indicates that the spatial dimensions can be arbitrary
    """

    def __init__(self, p: int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def rel(
        self,
        x: Float[Tensor, "n_samples *n {1}"],
        y: Float[Tensor, "n_samples *n {1}"],
    ) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=self.p, dim=1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")

        if self.size_mean is True:
            return torch.mean(diff_norms / y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms / y_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms / y_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "*n_samples"]:

        out_dim = x.size(-1)
        acc = 0
        for i in range(out_dim):
            acc += self.rel(x[..., [i]], y[..., [i]])
        return acc / out_dim


#########################################
# L^p relative loss for N-D functions and different output channels
#########################################
class LprelLoss_multiout:
    """
    I want to compute the relative error for each output channel separately, and return it separately.

    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, out_dim), with out_dim > 1
          where *n indicates that the spatial dimensions can be arbitrary
    """

    def __init__(self, p: int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "*n_samples out_dim"]:
        out_dim = x.size(-1)
        return torch.stack(
            [
                LprelLoss(p=self.p, size_mean=self.size_mean)(x[..., [i]], y[..., [i]])
                for i in range(out_dim)
            ],
            dim=-1,
        )


#########################################
#  H1 relative loss 1D
#########################################
class H1relLoss_1D:
    """
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform for 1D functions.
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = False, alpha: float = 1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)
    def rel(
        self,
        x: Complex[Tensor, "n_samples n_x out_dim"],
        y: Complex[Tensor, "n_samples n_x out_dim"],
    ) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")

        if self.size_mean is True:
            return torch.mean(diff_norms / y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms / y_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms / y_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples n_y out_dim"],
        y: Float[Tensor, "n_samples n_y out_dim"],
    ) -> Float[Tensor, "*n_samples"]:
        n_x, out_dim = x.size()[1:]

        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=n_x // 2, step=1),
                    torch.arange(start=-n_x // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(n_x, 1)
            .repeat(1, out_dim)
        )
        k_x = torch.abs(k_x).reshape(1, n_x, out_dim).to(x.device)

        x = torch.fft.fftn(x, dim=[1])
        y = torch.fft.fftn(y, dim=[1])

        weight = self.alpha * 1 + self.beta * (k_x**2)
        loss = self.rel(
            x * torch.sqrt(weight), y * torch.sqrt(weight)
        )  # Hadamard multiplication

        return loss


#########################################
# H1 relative loss for 1D functions and different output channels
#########################################
class H1relLoss_1D_multiout:
    """
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform, for 1D functions.

    I want to compute the relative error for each output channel separately, and return it separately.
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = False, alpha: float = 1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples n_x out_dim"],
        y: Float[Tensor, "n_samples n_x out_dim"],
    ) -> Float[Tensor, "*n_samples out_dim"]:
        out_dim = x.size(-1)
        return torch.stack(
            [
                H1relLoss_1D(self.beta, self.size_mean, self.alpha)(
                    x[..., [i]], y[..., [i]]
                )
                for i in range(out_dim)
            ],
            dim=-1,
        )


#########################################
#  H1 relative loss for 2D functions
#########################################
class H1relLoss:
    """
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = False, alpha: float = 1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)
    def rel(
        self,
        x: Complex[Tensor, "n_samples n_x n_y out_dim"],
        y: Complex[Tensor, "n_samples n_x n_y out_dim"],
    ) -> Float[Tensor, "*n_samples"]:
        num_examples = x.size(0)

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)

        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")

        if self.size_mean is True:
            return torch.mean(diff_norms / y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms / y_norms)  # sum along batchsize
        elif self.size_mean is None:
            return diff_norms / y_norms  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples n_x n_y out_dim"],
        y: Float[Tensor, "n_samples n_x n_y out_dim"],
    ) -> Float[Tensor, "*n_samples"]:
        n_x, n_y, out_dim = x.size()[1:]

        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=n_x // 2, step=1),
                    torch.arange(start=-n_x // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(n_x, 1, 1)
            .repeat(1, n_y, out_dim)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=n_y // 2, step=1),
                    torch.arange(start=-n_y // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(1, n_y, 1)
            .repeat(n_x, 1, out_dim)
        )
        k_x = torch.abs(k_x).reshape(1, n_x, n_y, out_dim).to(x.device)
        k_y = torch.abs(k_y).reshape(1, n_x, n_y, out_dim).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        weight = self.alpha * 1 + self.beta * (k_x**2 + k_y**2)
        loss = self.rel(x * torch.sqrt(weight), y * torch.sqrt(weight))

        return loss


#########################################
# H1 relative loss for 2D functions and different output channels
#########################################
class H1relLoss_multiout:
    """
    Relative H^1 = W^{1,2} norm, approximated with the Fourier transform, for 2D functions.

    I want to compute the relative error for each output channel separately, and return it separately.
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = False, alpha: float = 1.0):
        self.beta = beta
        self.size_mean = size_mean
        self.alpha = alpha

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples n_x n_y out_dim"],
        y: Float[Tensor, "n_samples n_x n_y out_dim"],
    ) -> Float[Tensor, "*n_samples out_dim"]:
        out_dim = x.size(-1)
        return torch.stack(
            [
                H1relLoss(self.beta, self.size_mean, self.alpha)(
                    x[..., [i]], y[..., [i]]
                )
                for i in range(out_dim)
            ],
            dim=-1,
        )
