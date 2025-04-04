"""
This file contains the tests for the loss functions defined in Loss_fun.py.
"""

import random
import sys

import numpy as np
import pytest
import torch

sys.path.append("..")
from loss_fun import lpLoss, H1relLoss, LprelLoss, MSELoss_rel, SmoothL1Loss_rel

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
