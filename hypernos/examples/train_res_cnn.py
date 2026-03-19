"""
In this example I fix all the hyperparameters for the CNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from architectures import CNN2DResidualNetwork as CNN_res
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters


def train_cnn(which_example: str, mode_hyperparams: str, loss_fn_str: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the CNN model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "CNN", which_example, mode=mode_hyperparams
    )

    # Default value for the hyper-parameters in the search space
    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the model builders
    model_builder = lambda config: CNN_res(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        activation_str=config["activation_str"],
        n_blocks=config["n_blocks"],
        include_grid=config["include_grid"],
        padding=config["padding"],
        normalization=config["normalization"],
        dropout_rate=config["dropout_rate"],
        device=device,
        example_input_normalizer=None,
        example_output_normalizer=None,
    )

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["filename"] if "filename" in config else None,
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = (
        f"CNN/Residual/{which_example}/loss_{loss_fn_str}_mode_{mode_hyperparams}"
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
        full_validation=False,
    )


if __name__ == "__main__":
    train_cnn("darcy", "default", "L2")
