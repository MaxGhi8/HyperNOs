"""
In this example I fix all the hyperparameters for the FNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

import torch.nn as nn
from datasets import NO_load_data_model
from loss_fun import lpLoss
from ResNet.ResidualNetwork import ResidualNetwork
from ResNet.ResNet_utilities import ResNet_initialize_hyperparameters
from train import train_fixed_model


def train_resnet(which_example: str, mode_hyperparams: str, loss_fn_str: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the ResNet model
    hyperparams_train, hyperparams_arc = ResNet_initialize_hyperparameters(
        which_example, mode_hyperparams
    )
    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

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

    # Define the model builders
    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": default_hyper_params["FourierF"],
            "retrain": default_hyper_params["retrain"],
        },
        batch_size=default_hyper_params["batch_size"],
        training_samples=default_hyper_params["training_samples"],
    )

    model_builder = lambda config: ResidualNetwork(
        config["in_channels"],
        config["out_channels"],
        config["hidden_channels"],
        config["activation_str"],
        config["n_blocks"],
        device,
        zero_mean=config["zero_mean"],
        input_normalizer=lambda x: (
            example.input_normalizer.encode(x)
            if config["input_normalizer"]
            else nn.Identity()
        ),
        output_denormalizer=lambda x: (
            example.output_normalizer.decode(x)
            if config["output_denormalizer"]
            else nn.Identity()
        ),
    )
    # Wrap the model builder
    # model_builder = wrap_model_builder(model_builder, which_example) # TODO

    # Define the loss function
    loss_fn = lpLoss(default_hyper_params["p"], False)

    experiment_name = (
        f"ResNet/{which_example}/loss_{loss_fn_str}_mode_{mode_hyperparams}"
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
    )


if __name__ == "__main__":
    train_resnet("afieti_square_neumann", "default", "L2")
