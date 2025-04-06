import sys

sys.path.append("..")

import torch
from datasets import NO_load_data_model
from ResNet import (
    ResidualNetwork,
    centered_softmax,
    zero_mean_imposition,
)

torch.set_default_dtype(torch.float64)


def test_centered_softmax():
    x = torch.rand(32, 10)
    x = centered_softmax(x)
    assert x.shape == (32, 10)
    assert torch.allclose(x.sum(dim=1), torch.zeros(32), atol=1e-6)


def test_zero_mean_imposition():
    x = torch.rand(32, 10)
    x = zero_mean_imposition(x)
    assert x.shape == (32, 10)
    assert torch.allclose(x.sum(dim=1), torch.zeros(32), atol=1e-6)


def test_residualnetwork():
    in_channels = 7
    out_channels = 3
    hidden_channels = [5, 5, 5]
    n_blocks = 2
    activation_str = "relu"
    model = ResidualNetwork(
        in_channels, out_channels, hidden_channels, activation_str, n_blocks
    )
    batch_size = 32
    input = torch.rand(batch_size, in_channels)
    output = model.forward(input)
    # test if the model runs
    assert output.shape[-1] == out_channels
    assert output.shape[0] == batch_size

    # test zero mean
    model = ResidualNetwork(
        in_channels,
        out_channels,
        hidden_channels,
        activation_str,
        n_blocks,
        zero_mean=True,
    )
    output = model.forward(input)
    assert output.shape[-1] == out_channels
    assert output.shape[0] == batch_size
    assert torch.allclose(output.sum(dim=1), torch.zeros(batch_size), atol=1e-6)


def test_residual_normalization():
    batch_size = 100
    training_samples = 1500
    example = NO_load_data_model(
        which_example="afieti_homogeneous_neumann",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        in_dist=True,
    )

    in_channels = example.s_in
    out_channels = example.s_out
    hidden_channels = [5, 5, 5]
    n_blocks = 2
    activation_str = "relu"
    model = ResidualNetwork(
        in_channels,
        out_channels,
        hidden_channels,
        activation_str,
        n_blocks,
        zero_mean=True,
        example=example,
    )

    input = torch.rand(batch_size, in_channels)
    output = model.forward(input)
    assert output.shape[-1] == out_channels
    assert output.shape[0] == batch_size
    # Test zero mean
    assert torch.allclose(output.sum(dim=1), torch.zeros(batch_size), atol=1e-6)
