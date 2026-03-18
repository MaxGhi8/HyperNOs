"""
In this example I fix all the hyperparameters for the G-FNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from architectures import G_FNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters
from wrappers import wrap_model_builder


def train_gfno(which_example: str, loss_fn_str: str):

    # Load the default hyperparameters for the G_FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "G_FNO", which_example, "default"
    )

    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    # Note: G_FNO signature: (problem_dim, in_dim, d_v, out_dim, L, modes, time_modes=None, reflection=False, ...)
    model_builder = lambda config: G_FNO(  # noqa: E731
        problem_dim=config["problem_dim"],
        in_dim=config["in_dim"],
        d_v=config["d_v"],
        out_dim=config["out_dim"],
        L=config["L"],
        modes=config["modes"],
        reflection=config["reflection"],
        padding=config["padding"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Wrap the model builder (channels last -> channels first)
    model_builder = wrap_model_builder(
        model_builder, which_example + "_neural_operator"
    )

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(  # noqa: E731
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = f"G_FNO/{which_example}/loss_{loss_fn_str}_mode_testing"

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
        output_folder=folder,
    )


if __name__ == "__main__":
    train_gfno("darcy", "L2")
