import sys

sys.path.append("..")

import pytest
import torch
from FNO import FNO
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
        which_example, mode_hyperparams
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

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
        config["retrain"],
    )

    model = wrap_model(model, which_example)

    dummy_input = torch.randn(32, 70, 70, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == config["out_dim"]
