"""
In this example I fix all the hyperparameters for the ResNet model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from architectures import GeometryConditionedLinearOperator
from datasets import NO_load_data_model
from loss_fun import lpLoss
from train import train_fixed_model
from utilities import initialize_hyperparameters


def train_iganet_transformer(which_example: str, filename: str, mode_hyperparams: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the IgaNet Transformer model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "IgaNet_transformer", which_example, mode_hyperparams
    )
    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example + "_transformer",
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=filename,
    )

    # Define the model builders
    example = NO_load_data_model(
        which_example=which_example + "_transformer",
        no_architecture={
            "FourierF": default_hyper_params["FourierF"],
            "retrain": default_hyper_params["retrain"],
        },
        batch_size=default_hyper_params["batch_size"],
        training_samples=default_hyper_params["training_samples"],
        filename=filename,
    )

    model_builder = lambda config: GeometryConditionedLinearOperator(
        n_dofs=config["n_dofs"],
        n_control_points=config["n_control_points"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers_geo=config["n_layers_geo"],
        dropout_rate=config["dropout_rate"],
        activation_str=config["activation_str"],
        zero_mean=config["zero_mean"],
        example_input_normalizer=(
            example.input_normalizer if config["internal_normalization"] else None
        ),
        example_output_normalizer=(
            example.output_normalizer if config["internal_normalization"] else None
        ),
        device=device,
    )

    # Define the loss function
    loss_fn = lpLoss(default_hyper_params["p"], True)
    loss_fn_str = "l2"

    experiment_name = (
        f"IgaNet_transformer/{which_example}/loss_{loss_fn_str}_mode_{mode_hyperparams}"
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
        full_validation=False,
    )


if __name__ == "__main__":
    train_iganet_transformer(
        "afieti_homogeneous_neumann",
        "dataset_homogeneous_Neumann_l_0_deg_2_crazygeom.mat",
        "default",
    )
