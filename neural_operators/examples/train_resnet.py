"""
In this example I fix all the hyperparameters for the ResNet model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from datasets import NO_load_data_model
from loss_fun import lpLoss
from ResNet import ResidualNetwork
from train import train_fixed_model
from utilities import initialize_hyperparameters


def train_resnet(which_example: str, mode_hyperparams: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the ResNet model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "ResNet", which_example, mode_hyperparams
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
        layer_norm=config["layer_norm"],
        dropout_rate=config["dropout_rate"],
        zero_mean=config["zero_mean"],
        example=(example if config["internal_normalization"] else None),
    )
    # Wrap the model builder
    # model_builder = wrap_model_builder(model_builder, which_example) # TODO

    # Define the loss function
    loss_fn = lpLoss(default_hyper_params["p"], False)
    loss_fn_str = "l2"
    # loss_fn = LprelLoss(2, False)
    # loss_fn_str = "L2"

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
        full_validation=False,
    )


if __name__ == "__main__":
    train_resnet("afieti_homogeneous_neumann", "default")
