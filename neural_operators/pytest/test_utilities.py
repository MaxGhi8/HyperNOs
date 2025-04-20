"""Test function for testing the file utitlities.py"""

import sys

sys.path.append("..")
import torch
from utilities import UnitGaussianNormalizer, minmaxGlobalNormalizer, minmaxNormalizer


#########################################
# Test initial normalization
#########################################
def test_UnitGaussianNormalizer():
    # Tes for 1D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, d_v)
    normalizer = UnitGaussianNormalizer(x)
    assert normalizer.mean.size() == torch.Size([n, d_v])
    assert normalizer.std.size() == torch.Size([n, d_v])
    assert torch.all(normalizer.std >= 0)

    # Tes for 2D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, d_v)
    normalizer = UnitGaussianNormalizer(x)
    assert normalizer.mean.size() == torch.Size([n, n, d_v])
    assert normalizer.std.size() == torch.Size([n, n, d_v])
    assert torch.all(normalizer.std >= 0)

    # Tes for 3D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, n, d_v)
    normalizer = UnitGaussianNormalizer(x)
    assert normalizer.mean.size() == torch.Size([n, n, n, d_v])
    assert normalizer.std.size() == torch.Size([n, n, n, d_v])
    assert torch.all(normalizer.std >= 0)


def test_minmaxNormalizer():
    # Tes for 1D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, d_v)
    normalizer = minmaxNormalizer(x)
    assert normalizer.min.size() == torch.Size([n, d_v])
    assert normalizer.max.size() == torch.Size([n, d_v])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5

    # Tes for 2D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, d_v)
    normalizer = minmaxNormalizer(x)
    assert normalizer.min.size() == torch.Size([n, n, d_v])
    assert normalizer.max.size() == torch.Size([n, n, d_v])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5

    # Tes for 3D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, n, d_v)
    normalizer = minmaxNormalizer(x)
    assert normalizer.min.size() == torch.Size([n, n, n, d_v])
    assert normalizer.max.size() == torch.Size([n, n, n, d_v])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5


def test_minmaxGlobalNormalizer():
    # Tes for 1D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5

    # Tes for 2D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5

    # Tes for 3D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])

    x_normalized_min = normalizer.encode(x)
    x_normalized_max = normalizer.encode(x)
    for _ in range(x_normalized_min.dim()):
        x_normalized_min = torch.min(x_normalized_min, dim=0).values
        x_normalized_max = torch.max(x_normalized_max, dim=0).values

    assert abs(x_normalized_min - 0) < 1e-5
    assert abs(x_normalized_max - 1) < 1e-5
