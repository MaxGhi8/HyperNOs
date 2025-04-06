"""
In this example I fix all the hyperparameters for the FNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from datasets import SinFrequency
from FNO import FNO
from loss_fun import loss_selector
from loss_fun_with_physics import PoissonResidualFiniteDiff
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters


def train_fno_with_phys(mode_hyperparams: str, loss_fn_str: str, alpha_phys: float):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", "poisson", mode_hyperparams
    )

    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    model_builder = lambda config: FNO(
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

    # Define the dataset builder
    dataset_builder = lambda config: SinFrequency(
        {
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
    loss_phys = PoissonResidualFiniteDiff(alpha=alpha_phys)

    experiment_name = f"FNO/poisson/loss_{loss_fn_str}_mode_{mode_hyperparams}"

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
        get_plot_function("poisson", "input"),
        get_plot_function("poisson", "output"),
        loss_phys=loss_phys,
    )


if __name__ == "__main__":
    train_fno_with_phys("default", "L2", 0.01)
