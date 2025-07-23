"""
In this example I fix all the hyperparameters for the FNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from BAMPNO import BAMPNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters
from wrappers import wrap_model_builder


def train_bampno(
    which_example: str, which_domain: str, mode_hyperparams: str, loss_fn_str: str
):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "BAMPNO", which_example + "_" + which_domain, mode_hyperparams
    )

    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": default_hyper_params["FourierF"],
            "retrain": default_hyper_params["retrain_seed"],
        },
        batch_size=default_hyper_params["batch_size"],
        training_samples=default_hyper_params["training_samples"],
        filename=default_hyper_params["grid_filename"],
    )

    # Define the model builders
    model_builder = lambda config: BAMPNO(
        config["problem_dim"],
        config["n_patch"],
        config["continuity_condition"],
        config["n_pts"],
        config["grid_filename"],
        config["in_dim"],
        config["d_v"],
        config["out_dim"],
        config["L"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        (
            {int(k): v for k, v in config["zero_BC"].items()}
            if config["zero_BC"]
            else None
        ),
        config["arc"],
        config["RNN"],
        config["same_params"],
        config["FFTnorm"],
        device,
        example.output_normalizer if config["internal_normalization"] else None,
        config["retrain_seed"],
    )

    # Wrap the model builder
    # model_builder = wrap_model_builder(model_builder, which_example) # TODO

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain_seed"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["grid_filename"],
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = (
        f"BAMPNO/{which_domain}/loss_{loss_fn_str}_mode_{mode_hyperparams}"
    )

    # Create the right folder if it doesn't exist
    folder = f"../tests/{experiment_name}"
    if not os.path.isdir(folder):
        print("Generated new folder")
        os.makedirs(folder, exist_ok=True)

    # Save the norm information
    with open(folder + "/norm_info.txt", "w") as f:
        f.write("Norm used during the training:\n")
        f.write(f"{loss_fn_str}\n")

    # Call the library function to tune the hyperparameters
    train_fixed_model(
        default_hyper_params,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        get_plot_function(which_example, "input"),
        get_plot_function(which_example, "output"),
        full_validation=True,
    )


if __name__ == "__main__":
    train_bampno("bampno", "8_domain", "best", "L2_cheb_mp")
