"""
In this example I fix all the hyperparameters for the CNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from CNO import CNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters
from wrappers import wrap_model_builder


def train_cno(which_example: str, mode_hyperparams: str, loss_fn_str: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the CNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "CNO", which_example, mode=mode_hyperparams
    )

    # Default value for the hyper-parameters in the search space
    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    model_builder = lambda config: CNO(
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
    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example)

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
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

    experiment_name = f"CNO/{which_example}/loss_{loss_fn_str}_mode_{mode_hyperparams}"

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
    )


if __name__ == "__main__":
    train_cno("shear_layer", "best", "L1")
