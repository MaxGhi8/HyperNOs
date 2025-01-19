import numpy as np
import torch
import torch.autograd as autograd


#########################################
# Derivative with torch.autograd
#########################################
class SpatialDerivativesAutograd:
    def __init__(self):
        pass

    def first_order_1d(self, u, x):
        """Compute first-order spatial derivatives in 1D using automatic differentiation."""
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x

    def second_order_1d(self, u, x):
        """Compute second-order spatial derivatives in 1D using automatic differentiation."""
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_x, u_xx

    def first_order_2d(self, u, x, y):
        """Compute first-order spatial derivatives in 2D using automatic differentiation."""
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = autograd.grad(u.sum(), y, create_graph=True)[0]
        return u_x, u_y

    def second_order_2d(self, u, x, y):
        """Compute second-order spatial derivatives in 2D using automatic differentiation."""
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = autograd.grad(u.sum(), y, create_graph=True)[0]
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]
        return u_x, u_y, u_xx, u_yy

    def get_grid_1d(self, shape: torch.Size) -> torch.Tensor:
        size_x = shape[1]
        # grid for x
        gridx = torch.tensor(
            np.linspace(0, 1, size_x), dtype=torch.float, requires_grad=True
        )
        gridx = gridx.reshape(size_x, 1)
        return gridx.unsqueeze(0)

    def get_grid_2d(self, shape: torch.Size) -> torch.Tensor:
        size_x, size_y = shape[1], shape[2]
        # grid for x
        gridx = torch.tensor(
            np.linspace(0, 1, size_x), dtype=torch.float, requires_grad=True
        )
        gridx = gridx.reshape(1, 1, size_x).repeat([1, size_y, 1])
        # grid for y
        gridy = torch.tensor(
            np.linspace(0, 1, size_y), dtype=torch.float, requires_grad=True
        )
        gridy = gridy.reshape(1, size_y, 1).repeat([1, 1, size_x])
        return gridx, gridy


class PoissonResidualAutograd:
    def __init__(self):
        pass

    def __call__(self, u, x, y, rhs):
        """Compute Poisson equation residual with rhs varying."""
        u = u.squeeze(-1)
        _, _, u_xx, u_yy = SpatialDerivativesAutograd().second_order_2d(u, x, y)
        laplacian_u = u_xx + u_yy
        residual = laplacian_u - rhs
        return residual


class DarcyResidualAutograd:
    def __init__(self, rhs):
        self.rhs = rhs

    def __call__(self, u, x, y, a):
        """Compute Darcy equation residual, with rhs fixed and diffusion coefficient varying."""
        u = u.squeeze(-1)
        u_x, u_y = SpatialDerivativesAutograd().first_order_2d(u, x, y)

        a_grad_u_x = a * u_x
        a_grad_u_y = a * u_y

        a_grad_u_x_x = autograd.grad(a_grad_u_x.sum(), x, create_graph=True)[0]
        a_grad_u_y_y = autograd.grad(a_grad_u_y.sum(), y, create_graph=True)[0]

        div_a_grad_u = a_grad_u_x_x + a_grad_u_y_y
        residual = div_a_grad_u + self.rhs
        return residual


class HelmholtzResidualAutograd:
    def __init__(self):
        pass

    def __call__(self, u, x, y, k):
        """Compute Helmholtz equation residual with k varying."""
        u = u.squeeze(-1)
        _, _, u_xx, u_yy = SpatialDerivativesAutograd().second_order_2d(u, x, y)
        laplacian_u = u_xx + u_yy

        residual = laplacian_u + (k**2) * u
        return residual


#########################################
# Derivative with finite differences
#########################################
import torch


class SpatialDerivativesFiniteDiff:
    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a  # domain width
        self.b = b  # domain height

    def first_order_1d(self, u):
        """
        Compute first-order spatial derivatives in 1D using finite differences.
        Input shape: (batch_size, x)
        Output shape: (batch_size, x)
        """
        dx = self.a / (u.shape[1] - 1)

        # Central difference for interior points
        u_x = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)

        # Handle boundary points
        u_x[:, 0] = (u[:, 1] - u[:, 0]) / dx
        u_x[:, -1] = (u[:, -1] - u[:, -2]) / dx

        return u_x

    def second_order_1d(self, u):
        """
        Compute second-order spatial derivatives in 1D using finite differences.
        Input shape: (batch_size, x)
        Output shape: (batch_size, x)
        """
        dx = self.a / (u.shape[1] - 1)

        # Central difference for interior points
        u_xx = (torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)) / (dx**2)

        # Handle boundary points
        u_xx[:, 0] = (u[:, 2] - 2 * u[:, 1] + u[:, 0]) / (dx**2)
        u_xx[:, -1] = (u[:, -1] - 2 * u[:, -2] + u[:, -3]) / (dx**2)

        return u_xx

    def first_order_2d(self, u):
        """
        Compute first-order spatial derivatives in 2D using finite differences.
        Input shape: (batch_size, x, y)
        Output shapes: (batch_size, x, y) for both u_x and u_y
        """
        dx = self.a / (u.shape[1] - 1)
        dy = self.b / (u.shape[2] - 1)

        # Central difference for interior points
        u_x = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
        u_y = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * dy)

        # Handle boundary points
        u_x[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dx
        u_x[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dx

        u_y[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dy
        u_y[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dy

        return u_x, u_y

    def second_order_2d(self, u):
        """
        Compute second-order spatial derivatives in 2D using finite differences.
        Input shape: (batch_size, x, y)
        Output shapes: (batch_size, x, y) for both u_xx and u_yy
        """
        dx = self.a / (u.shape[1] - 1)
        dy = self.b / (u.shape[2] - 1)

        # Central difference for interior points
        u_xx = (torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)) / (dx**2)
        u_yy = (torch.roll(u, -1, dims=2) - 2 * u + torch.roll(u, 1, dims=2)) / (dy**2)

        # Handle boundary points
        u_xx[:, 0, :] = (u[:, 2, :] - 2 * u[:, 1, :] + u[:, 0, :]) / (dx**2)
        u_xx[:, -1, :] = (u[:, -1, :] - 2 * u[:, -2, :] + u[:, -3, :]) / (dx**2)

        u_yy[:, :, 0] = (u[:, :, 2] - 2 * u[:, :, 1] + u[:, :, 0]) / (dy**2)
        u_yy[:, :, -1] = (u[:, :, -1] - 2 * u[:, :, -2] + u[:, :, -3]) / (dy**2)

        return u_xx, u_yy

    def mixed_derivative_2d(self, u):
        """
        Compute mixed second-order derivative in 2D using finite differences.
        Input shape: (batch_size, x, y)
        Output shape: (batch_size, x, y)
        """
        dx = self.a / (u.shape[1] - 1)
        dy = self.b / (u.shape[2] - 1)

        u_xy = (
            torch.roll(torch.roll(u, -1, dims=1), -1, dims=2)
            - torch.roll(torch.roll(u, -1, dims=1), 1, dims=2)
            - torch.roll(torch.roll(u, 1, dims=1), -1, dims=2)
            + torch.roll(torch.roll(u, 1, dims=1), 1, dims=2)
        ) / (4 * dx * dy)

        # For simplicity, we assume periodic boundaries or ignore boundaries for mixed derivatives

        return u_xy

    def get_grid_1d(self, shape: torch.Size) -> torch.Tensor:
        size_x = shape[1]

        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)

        return gridx.unsqueeze(0)

    def get_grid_2d(self, shape: torch.Size) -> torch.Tensor:
        size_x, size_y = shape[1], shape[2]

        # grid for x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([1, size_y, 1])
        # grid for y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1).repeat([1, 1, size_x])

        return gridx, gridy


class PoissonResidualFiniteDiff:
    def __init__(self):
        pass

    def __call__(self, u, rhs):
        """Compute Poisson equation residual with rhs varying."""
        u = u.squeeze(-1)
        u_xx, u_yy = SpatialDerivativesFiniteDiff().second_order_2d(u)
        laplacian_u = u_xx + u_yy
        residual = laplacian_u - rhs
        return residual


class DarcyResidualFiniteDiff:
    def __init__(self, rhs):
        self.rhs = rhs

    def __call__(self, u, a):
        """Compute Darcy equation residual, with rhs fixed and diffusion coefficient varying."""
        u = u.squeeze(-1)
        u_x, u_y = SpatialDerivativesFiniteDiff().first_order_2d(u)

        a_grad_u_x = a * u_x
        a_grad_u_y = a * u_y

        a_grad_u_x_x, _ = SpatialDerivativesFiniteDiff().first_order_2d(a_grad_u_x)
        _, a_grad_u_y_y = SpatialDerivativesFiniteDiff().first_order_2d(a_grad_u_y)

        div_a_grad_u = a_grad_u_x_x + a_grad_u_y_y
        residual = div_a_grad_u + self.rhs
        return residual


class HelmholtzResidualFiniteDiff:
    def __init__(self):
        pass

    def __call__(self, u, k):
        """Compute Helmholtz equation residual with k varying."""
        u = u.squeeze(-1)
        u_xx, u_yy = SpatialDerivativesFiniteDiff().second_order_2d(u)
        laplacian_u = u_xx + u_yy

        residual = laplacian_u + (k**2) * u
        return residual
