"""
This file contains the tests for the loss functions defined in Loss_fun.py.
"""

import random
import sys

import numpy as np
import pytest
import torch

sys.path.append("..")
from loss_fun import lpLoss, H1relLoss, LprelLoss, MSELoss_rel, SmoothL1Loss_rel, ChebyshevLprelLoss

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)


def test_lpLoss():
    # test 1: test L1(x, x)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    result = lpLoss(1)(x, x)
    assert result.item() == 0.0
    result = lpLoss(2)(x, x)
    assert result.item() == 0.0
    result = lpLoss(42, True)(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32)
    result = lpLoss(1)(x, x)
    assert result.item() == 0.0
    result = lpLoss(2)(x, x)
    assert result.item() == 0.0
    result = lpLoss(42, True)(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32)
    result = lpLoss(1)(x, x)
    assert result.item() == 0.0
    result = lpLoss(2)(x, x)
    assert result.item() == 0.0
    result = lpLoss(42, True)(x, x)
    assert result.item() == 0.0

    # test 4: hands on test
    x = torch.ones(1, 10)
    y = torch.zeros(1, 10)
    result = lpLoss(1)(x, y)
    assert result.item() == 10.0
    result = lpLoss(2)(x, y)
    assert np.abs(result.item() - np.sqrt(10.0)) < 1e-12
    result = lpLoss(2, True)(x, y)
    assert np.abs(result.item() - np.sqrt(10.0)) < 1e-12


def test_SmoothL1Loss_rel():
    # test 1: test L1(x, x)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = SmoothL1Loss_rel()(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32)
    result = SmoothL1Loss_rel()(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32)
    result = SmoothL1Loss_rel()(x, x)
    assert result.item() == 0.0

    # test 4: test on division by zero
    x = torch.randn(10, 3, 3)
    y = torch.zeros(10, 3, 3)
    with pytest.raises(ValueError, match="Division by zero"):
        SmoothL1Loss_rel()(x, y)


def test_MSELoss_rel():
    # test 1: test L1(x, x)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = MSELoss_rel()(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32)
    result = MSELoss_rel()(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32)
    result = MSELoss_rel()(x, x)
    assert result.item() == 0.0

    # test 4: test on division by zero
    x = torch.randn(10, 3, 3)
    y = torch.zeros(10, 3, 3)
    with pytest.raises(ValueError, match="Division by zero"):
        MSELoss_rel()(x, y)


def test_L1relLoss():
    # test 1: test L1(x, x)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = LprelLoss(1, False)(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32)
    result = LprelLoss(1, False)(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32)
    result = LprelLoss(1, False)(x, x)
    assert result.item() == 0.0

    # test 4: hands on test
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    y = torch.tensor([[[0.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    result = LprelLoss(1, False)(x, y)
    assert np.abs(result.item() - 1 / 9) < 1e-6  # single precision

    # test 5: test on division by zero
    x = torch.randn(10, 3, 3)
    y = torch.zeros(10, 3, 3)
    with pytest.raises(ValueError, match="Division by zero"):
        LprelLoss(1, False)(x, y)

    # test 6: test with x = 0 and y = random
    n = 100
    x = torch.zeros(n, 23, 23)
    y = torch.randn(n, 23, 23)
    assert LprelLoss(1, False)(x, y).item() == n

    # test 7: test y with norm 1 and x=y+1
    n = 30
    k = 64
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y + 1
    assert np.abs(LprelLoss(1, False)(x, y).item() - n) < 1e-12

    # test 8: test y with norm 1 and x=y-1
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y - 1
    assert np.abs(LprelLoss(1, False)(x, y).item() - n) < 1e-12

    # test 9: test y and x=y+random_const
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    const = random.normalvariate(0, 1)
    x = y + const
    assert np.abs(LprelLoss(1, False)(x, y).item() - np.abs(const) * n) < 1e-12


def test_L2relLoss():
    # test 1: test L2(x, x)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = LprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32)
    result = LprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32)
    result = LprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 4: hands on test
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    y = torch.tensor([[[0.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    result = LprelLoss(2, False)(x, y)
    assert np.abs(result.item() - 1 / (np.sqrt(29))) < 1e-6  # single precision

    # test 5: test on division by zero
    x = torch.randn(10, 3, 3)
    y = torch.zeros(10, 3, 3)
    with pytest.raises(ValueError, match="Division by zero"):
        LprelLoss(2, False)(x, y)

    # test 6: test with x = 0 and y = random
    n = 100
    x = torch.zeros(n, 23, 23)
    y = torch.randn(n, 23, 23)
    assert LprelLoss(2, False)(x, y).item() == n

    # test 7: test y with norm 1 and x=y+1
    n = 30
    k = 64
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y + 1
    assert np.abs(LprelLoss(2, False)(x, y).item() - n) < 1e-12

    # test 8: test y with norm 1 and x=y-1
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y - 1
    assert np.abs(LprelLoss(2, False)(x, y).item() - n) < 1e-12

    # test 9: test y with norm 1 and x=y+random_const
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    const = random.normalvariate(0, 1)
    x = y + const
    assert np.abs(LprelLoss(2, False)(x, y).item() - np.abs(const) * n) < 1e-12


def test_H1relLoss():
    # test 1: test H1(x, x)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    result = H1relLoss()(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32, 1)
    result = H1relLoss()(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32, 1)
    result = H1relLoss()(x, x)
    assert result.item() == 0.0

    # test 4: test on division by zero
    x = torch.randn(10, 3, 3, 1)
    y = torch.zeros(10, 3, 3, 1)
    with pytest.raises(ValueError, match="Division by zero"):
        H1relLoss()(x, y)

    # test 5: test y with norm 1 and x=y+1, in this case the loss is the same as the L2relLoss
    n = 30
    k = 42
    y = torch.ones(n, k, k, 1, dtype=torch.float64)
    x = y + 1
    assert np.abs(H1relLoss()(x, y).item() - n) < 1e-12

    # test 6: test y with norm 1 and x=y-1, in this case the loss is the same as the L2relLoss
    n = 30
    k = 42
    y = torch.ones(n, k, k, 1, dtype=torch.float64)
    x = y - 1
    assert np.abs(H1relLoss()(x, y).item() - n) < 1e-12

    # test 7: test y with norm 1 and y=x+random_const
    n = 30
    k = 42
    y = torch.ones(n, k, k, 1, dtype=torch.float64)
    const = random.normalvariate(0, 1)
    x = y + const
    assert np.abs(H1relLoss()(x, y).item() - np.abs(const) * n) < 1e-12

def test_L2relLoss_cheb():
    # test 1: test L2_cheb(x, x)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = ChebyshevLprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 2: as test 1 but with a single sample
    x = torch.rand(1, 32, 32)
    result = ChebyshevLprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 3: random test 1
    x = torch.rand(24, 32, 32)
    result = ChebyshevLprelLoss(2, False)(x, x)
    assert result.item() == 0.0

    # test 4: hands on test
    # x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    # y = torch.tensor([[[0.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).unsqueeze(-1)
    # result = ChebyshevLprelLoss(2, False)(x, y)
    # assert np.abs(result.item() - 1 / (np.sqrt(29))) < 1e-6  # single precision

    # test 5: test on division by zero
    x = torch.randn(10, 3, 3)
    y = torch.zeros(10, 3, 3)
    with pytest.raises(ValueError, match="Division by zero"):
        LprelLoss(2, False)(x, y)

    # test 6: test with x = 0 and y = random
    n = 100
    x = torch.zeros(n, 23, 23)
    y = torch.randn(n, 23, 23)
    assert LprelLoss(2, False)(x, y).item() == n

    # test 7: test y with norm 1 and x=y+1
    n = 30
    k = 64
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y + 1
    assert np.abs(LprelLoss(2, False)(x, y).item() - n) < 1e-12

    # test 8: test y with norm 1 and x=y-1
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    x = y - 1
    assert np.abs(LprelLoss(2, False)(x, y).item() - n) < 1e-12

    # test 9: test y with norm 1 and x=y+random_const
    n = 30
    k = 24
    y = torch.ones(n, k, k, dtype=torch.float64)
    const = random.normalvariate(0, 1)
    x = y + const
    assert np.abs(LprelLoss(2, False)(x, y).item() - np.abs(const) * n) < 1e-12
    

def test_chebyshev_quadrature_accuracy():
    """Test Chebyshev quadrature accuracy with known integrals"""
    
    def get_chebyshev_points(n: int) -> torch.Tensor:
        """Generate Chebyshev-Gauss-Lobatto points on [-1, 1]"""
        if n == 1:
            return torch.tensor([0.0])
        k = torch.arange(n, dtype=torch.float64)
        points = -torch.cos(np.pi * k / (n - 1))
        return points
    
    # Test polynomial exactness
    n = 17
    
    # Test constant function: \int_{-1}^{1} 1 dx = 2
    points = get_chebyshev_points(n)
    loss_fn = ChebyshevLprelLoss(p=2)
    weights = loss_fn._get_chebyshev_weights_1d(n)
    
    # Constant function
    f_const = torch.ones_like(points)
    integral_const = torch.sum(weights * f_const).item()
    assert abs(integral_const - 2.0) < 1e-14, f"Constant integration error: {abs(integral_const - 2.0)}"
    
    # Quadratic function: \int_{-1}^{1} x^2 dx = 2/3
    f_quad = points**2
    integral_quad = torch.sum(weights * f_quad).item()
    expected_quad = 2.0/3.0
    assert abs(integral_quad - expected_quad) < 1e-14, f"Quadratic integration error: {abs(integral_quad - expected_quad)}"
    
    # Quartic function: \int_{-1}^{1} x^4 dx = 2/5
    f_quart = points**4
    integral_quart = torch.sum(weights * f_quart).item()
    expected_quart = 2.0/5.0
    assert abs(integral_quart - expected_quart) < 1e-14, f"Quartic integration error: {abs(integral_quart - expected_quart)}"


def test_chebyshev_2d_quadrature():
    """Test 2D Chebyshev quadrature accuracy"""
    
    def get_chebyshev_points(n: int) -> torch.Tensor:
        if n == 1:
            return torch.tensor([0.0])
        k = torch.arange(n, dtype=torch.float64)
        points = -torch.cos(np.pi * k / (n - 1))
        return points
    
    nx, ny = 9, 9
    
    # Get 2D Chebyshev points
    x_points = get_chebyshev_points(nx)
    y_points = get_chebyshev_points(ny)
    X, Y = torch.meshgrid(x_points, y_points, indexing='ij')
    
    # Setup 2D loss function
    loss_fn_2d = ChebyshevLprelLoss(p=2)
    weights_2d = loss_fn_2d._get_nd_weights((nx, ny))
    
    # Test constant function: \int\int_{-1}^{1} 1 dx dy = 4
    f_const = torch.ones_like(X)
    integral_const = torch.sum(weights_2d * f_const).item()
    assert abs(integral_const - 4.0) < 1e-13, f"2D constant integration error: {abs(integral_const - 4.0)}"

    # Test separable quadratic: \int\int_{-1}^{1} x^2 y^2 dx dy = (2/3)^2
    f_sep_quad = X**2 * Y**2
    integral_sep_quad = torch.sum(weights_2d * f_sep_quad).item()
    expected_sep_quad = (2.0/3.0) * (2.0/3.0)
    assert abs(integral_sep_quad - expected_sep_quad) < 1e-13, f"2D separable quadratic error: {abs(integral_sep_quad - expected_sep_quad)}"
