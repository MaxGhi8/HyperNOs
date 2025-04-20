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

    # Tes for 2D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, d_v)
    normalizer = minmaxNormalizer(x)
    assert normalizer.min.size() == torch.Size([n, n, d_v])
    assert normalizer.max.size() == torch.Size([n, n, d_v])

    # Tes for 3D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, n, d_v)
    normalizer = minmaxNormalizer(x)
    assert normalizer.min.size() == torch.Size([n, n, n, d_v])
    assert normalizer.max.size() == torch.Size([n, n, n, d_v])


def test_minmaxGlobalNormalizer():
    # Tes for 1D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])

    # Tes for 2D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])

    # Tes for 3D input
    batch_size = 100
    n = 10
    d_v = 2
    x = torch.randn(batch_size, n, n, n, d_v)
    normalizer = minmaxGlobalNormalizer(x)
    assert normalizer.min.size() == torch.Size([])
    assert normalizer.max.size() == torch.Size([])
