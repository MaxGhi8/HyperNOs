"""
This module contains the definition of the loss functions that can be used in the training of the Neural Operator.
"""

import torch
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped
from torch import Tensor
import numpy as np


#########################################
# Loss function selector
#########################################
def loss_selector(loss_fn_str: str, problem_dim: int, beta: float = 1.0):
    match loss_fn_str.upper():
        case "L1":
            loss = LprelLoss(1, False)
        case "L2":
            loss = LprelLoss(2, False)
        case "L1_CHEB":
            loss = ChebyshevLprelLoss(1, False)
        case "L2_CHEB":
            loss = ChebyshevLprelLoss(2, False)
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
            raise ValueError("This loss function is not implemented, check the name passed to the loss_selector.")
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


class MSELoss:
    """Mean Squared Error loss for vectors"""

    def __init__(self, size_mean=False):
        self.size_mean = size_mean

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n"],
        y: Float[Tensor, "n_samples *n"],
    ) -> Float[Tensor, "*n_samples"]:
        # Calculate squared differences and mean across feature dimensions
        mse_per_sample = torch.mean((x - y) ** 2, dim=1)

        if self.size_mean is True:
            return torch.mean(mse_per_sample)  # Average MSE across samples
        elif self.size_mean is False:
            return torch.sum(mse_per_sample)  # Sum MSE across samples
        elif self.size_mean is None:
            return mse_per_sample  # MSE per sample, no reduction
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
# L^p relative loss for N-D functions on Chebyshev grids
#########################################
class ChebyshevLprelLoss:
    """
    Sum of relative errors in L^p norm for functions on Chebyshev grids
    Uses Chebyshev-Gauss-Lobatto quadrature for proper integration
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, out_dim)
          where *n indicates spatial dimensions with Chebyshev grid points
    """
    def __init__(self, p: int, size_mean=False):
        """
        Args:
            p: norm order
            size_mean: whether to take mean (True), sum (False), or no reduction (None)
        """
        self.p = p
        self.size_mean = size_mean
        self.device = None
        self.dtype = None
        
        # Cache for quadrature weights
        self._cached_weights = {}
        
    def _get_chebyshev_weights_1d(self, n: int) -> torch.Tensor:
        """
        Compute Chebyshev-Gauss-Lobatto quadrature weights for n points
        These are the weights for integrating over [-1, 1]
        """
        if n in self._cached_weights:
            return self._cached_weights[n]
            
        if n == 1:
            weights = torch.tensor([2.0])
        else:
            # Initialize weights array
            weights = torch.zeros(n, dtype=torch.float64)
            
            # Clenshaw-Curtis weights formula
            for j in range(n):
                weight = 1.0
                
                # Sum over even harmonics
                for k in range(1, n // 2 + 1):
                    if 2 * k <= n - 1:
                        angle = 2.0 * k * j * np.pi / (n - 1)
                        weight -= 2.0 * np.cos(angle) / (4.0 * k * k - 1.0)
                
                # Handle the Nyquist frequency for even n-1
                if (n - 1) % 2 == 0:
                    k = (n - 1) // 2
                    angle = k * j * np.pi / (n - 1)  # This is j*π when k = (n-1)/2
                    weight -= np.cos(angle) / (4.0 * k * k - 1.0)
                
                # Scale by interval length and normalization
                weights[j] = 2.0 * weight / (n - 1)
            
            # Correct the endpoints (they get half weight)
            weights[0] /= 2.0
            weights[n-1] /= 2.0
        
        # Convert to appropriate device and dtype when first tensor is seen
        if self.device is not None and self.dtype is not None:
            weights = weights.to(device=self.device, dtype=self.dtype)
            
        self._cached_weights[n] = weights
        return weights
    
    def _get_nd_weights(self, shape: tuple) -> torch.Tensor:
        """
        Compute N-dimensional quadrature weights as tensor product of 1D weights
        
        Args:
            shape: tuple of spatial dimensions (excluding batch and output dims)
        """
        # Get 1D weights for each dimension
        weights_1d = [self._get_chebyshev_weights_1d(n) for n in shape]
        
        # Compute tensor product
        if len(shape) == 1:
            return weights_1d[0]
        elif len(shape) == 2:
            w1, w2 = weights_1d[0], weights_1d[1]
            return torch.outer(w1, w2)
        elif len(shape) == 3:
            w1, w2, w3 = weights_1d[0], weights_1d[1], weights_1d[2]
            weights = torch.zeros(shape, device=self.device, dtype=self.dtype)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        weights[i, j, k] = w1[i] * w2[j] * w3[k]
            return weights
        else:
            # General N-D case using recursive tensor products
            weights = weights_1d[0]
            for w in weights_1d[1:]:
                # Expand dimensions and compute outer product
                new_shape = weights.shape + (1,) * len(w.shape)
                weights = weights.view(new_shape)
                w_shape = (1,) * len(weights.shape[:-1]) + w.shape
                w = w.view(w_shape)
                weights = weights * w
            return weights.squeeze()
    
    def _update_device_dtype(self, tensor: torch.Tensor):
        """Update cached device and dtype from input tensor"""
        if self.device != tensor.device or self.dtype != tensor.dtype:
            self.device = tensor.device
            self.dtype = tensor.dtype
            # Re-cache weights with correct device/dtype
            self._cached_weights = {}

    @jaxtyped(typechecker=beartype)
    def weighted_norm(
        self,
        x: Float[Tensor, "n_samples *n {1}"],
        weights: Float[Tensor, "*n"],
    ) -> Float[Tensor, "n_samples"]:
        """
        Compute weighted L^p norm using quadrature weights
        """
        num_examples = x.size(0)
        
        # Reshape for batch processing
        x_flat = x.reshape(num_examples, -1)  # (n_samples, spatial_points)
        weights_flat = weights.reshape(-1)    # (spatial_points,)
        
        # Compute weighted norm: ||f||_p = (\int |f|^p w dx)^(1/p)
        weighted_values = torch.abs(x_flat) ** self.p * weights_flat.unsqueeze(0)
        integrals = torch.sum(weighted_values, dim=1)  # Sum over spatial points
        norms = torch.pow(integrals, 1.0 / self.p)
        
        return norms

    @jaxtyped(typechecker=beartype)
    def rel(
        self,
        x: Float[Tensor, "n_samples *n {1}"],
        y: Float[Tensor, "n_samples *n {1}"],
    ) -> Float[Tensor, "*n_samples"]:
        """
        Compute relative error using Chebyshev quadrature
        """
        self._update_device_dtype(x)
        
        # Get spatial dimensions
        spatial_shape = x.shape[1:-1]  # Exclude batch and output dims
        
        # Get quadrature weights
        weights = self._get_nd_weights(spatial_shape)
        
        # Compute weighted norms
        diff = x - y
        diff_norms = self.weighted_norm(diff, weights)
        y_norms = self.weighted_norm(y, weights)
        
        # Check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero in denominator norm")
            
        rel_errors = diff_norms / y_norms
        
        if self.size_mean is True:
            return torch.mean(rel_errors)
        elif self.size_mean is False:
            return torch.sum(rel_errors)  # sum along batch dimension
        elif self.size_mean is None:
            return rel_errors  # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")

    @jaxtyped(typechecker=beartype) 
    def __call__(
        self,
        x: Float[Tensor, "n_samples *n out_dim"],
        y: Float[Tensor, "n_samples *n out_dim"],
    ) -> Float[Tensor, "*n_samples"]:
        """
        Compute relative error averaged over output dimensions
        """
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
