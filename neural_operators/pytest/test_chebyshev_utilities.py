"""
This file contains the tests for the functions defined in chebyshev_utilities.py.
"""

import sys

import numpy as np
import torch

sys.path.append("..")
from BAMPNO import (
    Chebyshev_grid_1d,
    Chebyshev_grid_2d,
    batched_coefficients_to_values,
    batched_differentiate,
    batched_differentiate_2d,
    batched_integrate,
    batched_values_to_coefficients,
    coefficients_to_values,
    differentiate,
    integrate,
    patched_coefficients_to_values,
    patched_values_to_coefficients,
    values_to_coefficients,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)


#########################################
# Test Chebyshev Transfor
#########################################
def test_cft_1d():
    """
    Test the Chebyshev Fourier Transform (CFT) in 1D.
    """
    n = 50  # number of coefficients
    coeff = torch.randn((n,)) * 4  # random coefficients
    T = np.polynomial.chebyshev.Chebyshev(np.asarray(coeff))  # Chebyshev polynomial
    values = torch.tensor(
        T(np.asarray(Chebyshev_grid_1d(n + 10))).reshape(-1, 1)
    )  # values of the polynomial in the Chebyshev grid

    num_coeff = values_to_coefficients(values).reshape(
        -1,
    )

    assert torch.allclose(coeff, num_coeff[:n], atol=1e-10)


def test_cft_2d():
    """
    Test the Chebyshev Fourier Transform (CFT) in 2D.
    """
    n = 5  # number of coefficients
    c = torch.randn((n, n)) * 4  # random coefficients
    grid = Chebyshev_grid_2d(n, [-1.0, -1.0], [1.0, 1.0])
    X, Y = grid[:, :, 0], grid[:, :, 1]

    values = torch.tensor(np.polynomial.chebyshev.chebval2d(X, Y, np.asarray(c)))

    appro_coeffs = values_to_coefficients(values.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(c, appro_coeffs[:n, :n], atol=1e-10)


def test_patched_cft_2d():
    """
    Test the patched Chebyshev Fourier Transform (CFT) in 2D.
    """
    n_batch = 6
    n_patch = 4
    d_v = 4
    n = 20

    c = torch.randn((n_batch, n_patch, n, n, d_v))  # random coefficients
    grid = Chebyshev_grid_2d(n, [-1.0, -1.0], [1.0, 1.0])
    X, Y = grid[:, :, 0], grid[:, :, 1]

    values = torch.ones_like(c)
    for batch_idx in range(n_batch):
        for patch_idx in range(n_patch):
            for v in range(d_v):
                values[batch_idx, patch_idx, :, :, v] = torch.tensor(
                    np.polynomial.chebyshev.chebval2d(
                        X, Y, np.asarray(c[batch_idx, patch_idx, :, :, v])
                    )
                )

    appro_coeffs = patched_values_to_coefficients(values)

    assert torch.allclose(c, appro_coeffs, atol=1e-10)


#########################################
# Test Chebyshev Anti-transorm
#########################################
def test_icft_2d():
    """
    Test the Inverse Chebyshev Fourier Transform (ICFT) in 2D.
    """
    n = 50  # number of coefficients
    c = torch.randn((n, n))  # random coefficients
    grid = Chebyshev_grid_2d(n, [-1.0, -1.0], [1.0, 1.0])
    X, Y = grid[:, :, 0], grid[:, :, 1]

    values = torch.tensor(
        np.polynomial.chebyshev.chebval2d(X, Y, np.asarray(c))
    )  # Evaluated Chebyshev polynomial

    appro_values = coefficients_to_values(c.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(values, appro_values, atol=1e-10)


def test_patched_icft_2d():
    """
    Test the Inverse Chebyshev Fourier Transform (ICFT) in 2D using patched coefficients.
    """
    n_batch = 6
    n_patch = 4
    d_v = 4
    n = 20

    c = torch.randn((n_batch, n_patch, n, n, d_v))  # random coefficients
    grid = Chebyshev_grid_2d(n, [-1.0, -1.0], [1.0, 1.0])
    X, Y = grid[:, :, 0], grid[:, :, 1]

    values = torch.ones_like(c)
    for batch_idx in range(n_batch):
        for patch_idx in range(n_patch):
            for v in range(d_v):
                values[batch_idx, patch_idx, :, :, v] = torch.tensor(
                    np.polynomial.chebyshev.chebval2d(
                        X, Y, np.asarray(c[batch_idx, patch_idx, :, :, v])
                    )
                )

    appro_values = patched_coefficients_to_values(c)
    assert torch.allclose(values, appro_values, atol=1e-10)


def test_inverse():
    """
    Test if CFT is the inverse of ICFT and vice versa.
    """
    x = torch.randn((4, 3, 2, 2, 3)) * 4  # random values

    error1 = torch.norm(
        (x - values_to_coefficients(coefficients_to_values(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the values
    error2 = torch.norm(
        (x - coefficients_to_values(values_to_coefficients(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the coefficients
    assert error1 < 1e-12
    assert error2 < 1e-12

    error1 = torch.norm(
        (x - batched_values_to_coefficients(batched_coefficients_to_values(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the values
    error2 = torch.norm(
        (x - batched_coefficients_to_values(batched_values_to_coefficients(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the coefficients
    assert error1 < 1e-12
    assert error2 < 1e-12

    error1 = torch.norm(
        (x - patched_values_to_coefficients(patched_coefficients_to_values(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the values
    error2 = torch.norm(
        (x - patched_coefficients_to_values(patched_values_to_coefficients(x))).reshape(
            -1,
        ),
        float("inf"),
    )  # error in the coefficients
    assert error1 < 1e-12
    assert error2 < 1e-12


#########################################
# Test Chebyshev Differentiation
#########################################
def test_differentiation():
    """
    Test the Chebyshev differentiation.
    """
    n = 50
    k = 4
    for i in range(k):
        coeff = torch.randn((n,)) * 4  # random coefficients
        T = np.polynomial.chebyshev.Chebyshev(np.asarray(coeff))  # Chebyshev polynomial
        if i == 0:
            values = torch.tensor(
                T(np.asarray(Chebyshev_grid_1d(n))).reshape(-1, 1)
            )  # values of the polynomial in the Chebyshev grid
            coeff_diff = torch.tensor(
                T.deriv().coef.reshape(-1, 1)
            )  # values of the derivative of the polynomial in the Chebyshev grid
        else:
            values = torch.cat(
                (
                    values,
                    torch.tensor(T(np.asarray(Chebyshev_grid_1d(n))).reshape(-1, 1)),
                ),
                dim=1,
            )  # values of the polynomial in the Chebyshev grid
            coeff_diff = torch.cat(
                (coeff_diff, torch.tensor(T.deriv().coef).reshape(-1, 1)), dim=1
            )

    # Coefficients of the polynomial with our Chebyshev transform
    num_coeff = values_to_coefficients(values)
    coeff_diff_appro = differentiate(num_coeff, 0)

    # Test differentiation
    assert torch.allclose(coeff_diff_appro[:-1], coeff_diff, atol=1e-10)


#########################################
# Test Chebyshev Integration
#########################################
def test_integration():
    """
    Test the Chebyshev integration.
    """
    n = 50
    coeff = torch.randn((n,)) * 4  # random coefficients
    T = np.polynomial.chebyshev.Chebyshev(np.asarray(coeff))  # Chebyshev polynomial
    values = T(np.asarray(Chebyshev_grid_1d(n))).reshape(
        -1, 1
    )  # values of the polynomial in the Chebyshev grid

    # Coefficients of the polynomial with out Chebyshev transform
    num_coeff = values_to_coefficients(torch.tensor(values))

    # Test integration
    error = torch.norm(
        integrate(num_coeff, 0).reshape(
            -1,
        )[1 : (n + 1)]
        - torch.tensor(T.integ().coef)[1:],
        float("inf"),
    )

    assert error < 1e-10
