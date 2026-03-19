import sys

sys.path.append("..")

import pytest
import torch
from architectures import FNO
from datasets import NO_load_data_model
from utilities import initialize_hyperparameters
from wrappers import wrap_model

torch.set_default_dtype(torch.float32)


# Select some random params for testing fro 1D FNO
@pytest.mark.parametrize(
    "which_example, mode_hyperparams",
    [
        ("fhn", "default"),
        ("hh", "default"),
        ("hh", "best"),
        ("ord", "default"),
        ("burgers_zongyi", "default"),
    ],
)
def test_FNO_1d(which_example, mode_hyperparams):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", which_example, mode_hyperparams
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["filename"] if "filename" in config else None,
    )

    # Define the model builders
    model = FNO(
        config["problem_dim"],
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        (
            example.output_normalizer
            if ("internal_normalization" in config and config["internal_normalization"])
            else None
        ),
        config["retrain"],
    )

    model = wrap_model(model, which_example)

    dummy_input = torch.randn(32, 1000, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == config["out_dim"]


# Select some random params for testing fro 2D FNO
@pytest.mark.parametrize(
    "which_example, mode_hyperparams",
    [
        ("airfoil", "default"),
        ("allen", "best"),
        ("allen", "default"),
        ("cont_tran", "default"),
        # ("cont_tran", "best"),
        ("crosstruss", "best"),
        ("crosstruss", "default"),
        ("darcy", "best"),
        ("darcy", "default"),
        ("darcy_zongyi", "default"),
        ("disc_tran", "default"),
        # ("disc_tran", "best"),
    ],
)
def test_FNO_2d(which_example, mode_hyperparams):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", which_example, mode_hyperparams
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["filename"] if "filename" in config else None,
    )

    # Define the model builders
    model = FNO(
        config["problem_dim"],
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        (
            example.output_normalizer
            if ("internal_normalization" in config and config["internal_normalization"])
            else None
        ),
        config["retrain"],
    )

    model = wrap_model(model, which_example)

    dummy_input = torch.randn(32, 70, 70, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == config["out_dim"]


def test_FNO_3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model builders
    model = FNO(
        problem_dim=3,
        in_dim=1,
        d_v=5,
        out_dim=1,
        L=3,
        modes=2,
        fun_act="gelu",
        weights_norm="Kaiming",
        arc="Classic",
        RNN=False,
        FFTnorm=None,
        padding=1,
        device=device,
        retrain_fno=4,
    )

    dummy_input = torch.randn(10, 40, 40, 40, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == 1


def test_FNO_different_resolutions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model builders
    model = FNO(
        problem_dim=3,
        in_dim=1,
        d_v=5,
        out_dim=1,
        L=3,
        modes=2,
        fun_act="gelu",
        weights_norm="Kaiming",
        arc="Classic",
        RNN=False,
        FFTnorm=None,
        padding=1,
        device=device,
        retrain_fno=4,
    )

    dummy_input = torch.randn(10, 50, 60, 70, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == 1
    assert output.shape[-1] == 1
