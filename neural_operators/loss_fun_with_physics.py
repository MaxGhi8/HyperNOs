import torch
import torch.autograd as autograd


def compute_spatial_derivatives_first_order(u, x, y):
    """Compute spatial derivatives using automatic differentiation, first order"""
    u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = autograd.grad(u.sum(), y, create_graph=True)[0]

    return u_x, u_y


def compute_spatial_derivatives_second_order(u, x, y):
    """Compute spatial derivatives using automatic differentiation, second order"""
    u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = autograd.grad(u.sum(), y, create_graph=True)[0]
    u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]

    return u_x, u_y, u_xx, u_yy


def poisson_residual(rhs, u, x, y):
    """
    Compute Poisson equation residual
    """
    _, _, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)
    laplacian_u = u_xx + u_yy
    residual = laplacian_u - rhs

    return residual


def darcy_residual(u, K, x, y, params):
    """
    Compute Darcy equation residual

    Args:
        u: Pressure field (output of the neural network)
        K: Permeability field (can be constant or a function of x, y)
        x, y: Spatial coordinates
        params: Dictionary containing physical parameters (optional)
    """
    # Compute spatial derivatives of u
    u_x, u_y = compute_spatial_derivatives_first_order(u, x, y)
    u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]

    # Compute ∇ · (K ∇u)
    # First, compute K * ∇u
    K_grad_u_x = K * u_x
    K_grad_u_y = K * u_y

    # Then, compute the divergence: ∂(K * ∂u/∂x)/∂x + ∂(K * ∂u/∂y)/∂y
    K_grad_u_x_x = autograd.grad(K_grad_u_x.sum(), x, create_graph=True)[0]
    K_grad_u_y_y = autograd.grad(K_grad_u_y.sum(), y, create_graph=True)[0]

    div_K_grad_u = K_grad_u_x_x + K_grad_u_y_y

    # Residual of the Darcy equation
    residual = -div_K_grad_u  # -∇ · (K ∇u) = 0

    return residual


def helmholtz_residual(u, x, y, params):
    """
    Compute Helmholtz equation residual

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        params: Dictionary containing physical parameters (k, source_term f(x, y))
    """
    # Compute second-order spatial derivatives of u
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)

    # Laplacian of u: ∇²u = u_xx + u_yy
    laplacian_u = u_xx + u_yy

    # Wave number k and source term f(x, y)
    k = params.get("k", 1.0)
    f = params.get("source_term", torch.zeros_like(u))

    # Residual of the Helmholtz equation: ∇²u + k²u - f = 0
    residual = laplacian_u + (k**2) * u - f

    return residual


def biharmonic_residual(u, x, y, params):
    """
    Compute Biharmonic equation residual

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        params: Dictionary containing physical parameters (source_term f(x, y))
    """
    # Compute second-order spatial derivatives of u
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)

    # Compute fourth-order derivatives
    u_xxx = autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    u_yyy = autograd.grad(u_yy.sum(), y, create_graph=True)[0]
    u_xxyy = autograd.grad(u_xx.sum(), y, create_graph=True)[0]

    # Biharmonic operator: ∇⁴u = u_xxxx + 2u_xxyy + u_yyyy
    biharmonic_u = u_xxx + 2 * u_xxyy + u_yyy

    # Source term f(x, y)
    f = params.get("source_term", torch.zeros_like(u))

    # Residual of the Biharmonic equation: ∇⁴u - f = 0
    residual = biharmonic_u - f

    return residual
