import sys

sys.path.append("..")

import torch
from architectures import BAMPNO
from utilities import initialize_hyperparameters

torch.set_default_dtype(torch.float32)


def test_BAMPNO():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BC = {0: (1, 2, 3), 1: (1, 4), 2: (2, 3, 4)}
    internal_condition = {"horizontal": ((2, 1),), "vertical": ((1, 0),)}

    model = BAMPNO(
        problem_dim=2,
        n_patch=3,
        continuity_condition=internal_condition,
        n_pts=60,
        grid_filename="Darcy_Lshape_chebyshev_60pts.mat",
        in_dim=1,
        d_v=12,
        out_dim=1,
        L=2,
        modes=20,
        fun_act="gelu",
        weights_norm=None,
        zero_BC=BC,
        arc="Classic",
        RNN=False,
        same_params=False,
        FFTnorm=None,
        device=device,
        retrain_seed=4,
    )

    dummy_input = torch.randn(32, 3, 60, 60, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == 1


def test_BAMPNO_from_json():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "BAMPNO", "Lshape_domain", "default"
    )

    config = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    model = BAMPNO(
        problem_dim=config["problem_dim"],
        n_patch=config["n_patch"],
        continuity_condition=config["continuity_condition"],
        n_pts=config["n_pts"],
        grid_filename=config["grid_filename"],
        in_dim=config["in_dim"],
        d_v=config["d_v"],
        out_dim=config["out_dim"],
        L=config["L"],
        modes=config["modes"],
        fun_act=config["fun_act"],
        weights_norm=config["weights_norm"],
        zero_BC={int(k): v for k, v in config["zero_BC"].items()},
        arc=config["arc"],
        RNN=config["RNN"],
        same_params=config["same_params"],
        FFTnorm=config["FFTnorm"],
        device=device,
        retrain_seed=config["retrain_seed"],
    )

    dummy_input = torch.randn(32, 3, 60, 60, 1).to(device)
    output = model(dummy_input)
    assert output.shape[:-1] == dummy_input.shape[:-1]
    assert output.shape[-1] == 1
