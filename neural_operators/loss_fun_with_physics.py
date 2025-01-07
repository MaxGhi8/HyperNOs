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


def advection_diffusion_residual(u, x, y, params):
    """
    Compute Advection-Diffusion equation residual

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        params: Dictionary containing physical parameters (D, v, source_term f(x, y))
    """
    # Compute first-order spatial derivatives of u
    u_x, u_y = compute_spatial_derivatives_first_order(u, x, y)

    # Diffusion coefficient D and velocity field v = [vx, vy]
    D = params.get("D", 1.0)
    vx = params.get("vx", 1.0)
    vy = params.get("vy", 1.0)

    # Compute second-order derivatives for diffusion term
    u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]

    # Diffusion term: ∇ · (D ∇u) = D (u_xx + u_yy)
    diffusion_term = D * (u_xx + u_yy)

    # Advection term: v · ∇u = vx * u_x + vy * u_y
    advection_term = vx * u_x + vy * u_y

    # Source term f(x, y)
    f = params.get("source_term", torch.zeros_like(u))

    # Residual of the Advection-Diffusion equation: ∇ · (D ∇u) + v · ∇u - f = 0
    residual = diffusion_term + advection_term - f

    return residual


def nonlinear_poisson_residual(u, x, y, params):
    """
    Compute Nonlinear Poisson equation residual

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        params: Dictionary containing physical parameters (nonlinear function g(u), source_term f(x, y))
    """
    # Compute second-order spatial derivatives of u
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)

    # Laplacian of u: ∇²u = u_xx + u_yy
    laplacian_u = u_xx + u_yy

    # Nonlinear function g(u) and source term f(x, y)
    g = params.get("g", lambda u: u**2)  # Default nonlinearity: g(u) = u²
    f = params.get("source_term", torch.zeros_like(u))

    # Residual of the Nonlinear Poisson equation: ∇²u + g(u) - f = 0
    residual = laplacian_u + g(u) - f

    return residual


def allen_cahn_residual(u, x, y, t, params):
    """
    Compute Allen-Cahn equation residual at a fixed time t

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        t: Time coordinate (fixed)
        params: Dictionary containing physical parameters (epsilon)
    """
    # Compute spatial derivatives
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)

    # Compute time derivative (fixed time, so ∂u/∂t is treated as a parameter)
    u_t = autograd.grad(u.sum(), t, create_graph=True)[0]

    # Parameter epsilon
    epsilon = params.get("epsilon", 0.1)

    # Residual of the Allen-Cahn equation: ∂u/∂t - ε∇²u + u³ - u = 0
    residual = u_t - epsilon * (u_xx + u_yy) + u**3 - u

    return residual


def navier_stokes_residual(u, v, p, x, y, t, params):
    """
    Compute Navier-Stokes equations residual at a fixed time t

    Args:
        u, v: Velocity components
        p: Pressure
        x, y: Spatial coordinates
        t: Time coordinate (fixed)
        params: Dictionary containing physical parameters (nu, etc.)
    """
    # Compute spatial derivatives for u and v
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)
    v_x, v_y, v_xx, v_yy = compute_spatial_derivatives_second_order(v, x, y)

    # Compute time derivatives (fixed time, so ∂u/∂t and ∂v/∂t are treated as parameters)
    u_t = autograd.grad(u.sum(), t, create_graph=True)[0]
    v_t = autograd.grad(v.sum(), t, create_graph=True)[0]

    # Compute pressure gradients
    p_x = autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = autograd.grad(p.sum(), y, create_graph=True)[0]

    # Kinematic viscosity
    nu = params.get("nu", 0.01)

    # Continuity equation: ∇ · u = 0
    continuity = u_x + v_y

    # Momentum equations
    momentum_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y


def reaction_diffusion_residual(u, x, y, t, params):
    """
    Compute Reaction-Diffusion equation residual at a fixed time t

    Args:
        u: Field (output of the neural network)
        x, y: Spatial coordinates
        t: Time coordinate (fixed)
        params: Dictionary containing physical parameters (D, reaction term f(u))
    """
    # Compute spatial derivatives
    u_x, u_y, u_xx, u_yy = compute_spatial_derivatives_second_order(u, x, y)

    # Compute time derivative (fixed time, so ∂u/∂t is treated as a parameter)
    u_t = autograd.grad(u.sum(), t, create_graph=True)[0]

    # Diffusion coefficient D and reaction term f(u)
    D = params.get("D", 1.0)
    f = params.get("f", lambda u: u * (1 - u))  # Default reaction term: logistic growth

    # Residual of the Reaction-Diffusion equation: ∂u/∂t - D∇²u - f(u) = 0
    residual = u_t - D * (u_xx + u_yy) - f(u)

    return residual
