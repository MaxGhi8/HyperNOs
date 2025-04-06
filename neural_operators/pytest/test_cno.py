import sys

sys.path.append("..")

import pytest
import torch
from CNO import CNO
from utilities import initialize_hyperparameters
from wrappers import wrap_model


# Select some random params for testing fro 1D CNO
@pytest.mark.parametrize(
    "which_example, mode_hyperparams",
    [
        ("fhn", "default"),
        ("fhn", "best"),
        ("hh", "default"),
        ("burgers_zongyi", "default"),
    ],
)
def test_CNO_1d(which_example, mode_hyperparams):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the CNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "CNO", which_example, mode_hyperparams
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    model = CNO(
        problem_dim=config["problem_dim"],
        in_dim=config["in_dim"],
        out_dim=config["out_dim"],
        size=config["in_size"],
        N_layers=config["N_layers"],
        N_res=config["N_res"],
        N_res_neck=config["N_res_neck"],
        channel_multiplier=config["channel_multiplier"],
        kernel_size=config["kernel_size"],
        use_bn=config["bn"],
        device=device,
    )

    model = wrap_model(model, which_example)

    dummy_input = torch.randn(32, config["in_size"], config["in_dim"]).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == config["out_dim"]


# Select some random params for testing fro 2D CNO
@pytest.mark.parametrize(
    "which_example, mode_hyperparams",
    [
        ("airfoil", "default"),
        ("airfoil", "best"),
        ("allen", "default"),
        ("cont_tran", "default"),
        ("darcy", "default"),
        ("darcy", "best"),
        ("darcy_zongyi", "default"),
        ("disc_tran", "default"),
        ("disc_tran", "best"),
    ],
)
def test_CNO_2d(which_example, mode_hyperparams):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the CNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "CNO", which_example, mode_hyperparams
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    model = CNO(
        problem_dim=config["problem_dim"],
        in_dim=config["in_dim"],
        out_dim=config["out_dim"],
        size=config["in_size"],
        N_layers=config["N_layers"],
        N_res=config["N_res"],
        N_res_neck=config["N_res_neck"],
        channel_multiplier=config["channel_multiplier"],
        kernel_size=config["kernel_size"],
        use_bn=config["bn"],
        device=device,
    )

    model = wrap_model(model, which_example)

    dummy_input = torch.randn(
        32, config["in_size"], config["in_size"], config["in_dim"]
    ).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == config["out_dim"]
