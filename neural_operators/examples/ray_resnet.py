"""
In this example I run Ray for the optimization of hyperparameters for the ResNet model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from architectures import ResidualNetwork
from datasets import NO_load_data_model
from loss_fun import lpLoss
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters


def ray_resnet(which_example: str, filename: str, mode_hyperparams: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the ResNet model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "ResNet", which_example, mode_hyperparams
    )
    # Default value for the hyper-parameters in the search space
    default_hyper_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-4),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "activation_str": tune.choice(
            ["tanh", "relu", "gelu", "leaky_relu", "sigmoid"]
        ),
        "n_blocks": tune.randint(1, 8),
        "dropout_rate": tune.quniform(0.0, 0.5, 1e-2),
    }
    # Add the hyperparameters hidden_channels
    config_space["hidden_channels"] = tune.sample_from(
        lambda spec: [
            tune.choice([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]).sample()
        ]
        * tune.randint(1, 8).sample()
    )

    # Set all the other parameters to fixed values
    fixed_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }
    parameters_to_tune = config_space.keys()
    for param in parameters_to_tune:
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
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
        which_example=which_example,
        no_architecture={
            "FourierF": default_hyper_params["FourierF"],
            "retrain": default_hyper_params["retrain"],
        },
        batch_size=default_hyper_params["batch_size"],
        training_samples=default_hyper_params["training_samples"],
        filename=filename,
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
        example_input_normalizer=(
            example.input_normalizer if config["internal_normalization"] else None
        ),
        example_output_normalizer=(
            example.output_normalizer if config["internal_normalization"] else None
        ),
    )
    # Wrap the model builder
    # model_builder = wrap_model_builder(model_builder, which_example) # TODO

    # Define the loss function
    loss_fn = lpLoss(default_hyper_params["p"], True)

    # Call the library function to tune the hyperparameters
    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        [default_hyper_params],
        runs_per_cpu=12.0,
        runs_per_gpu=1.0,
        grace_period=400,
        reduction_factor=4,
        max_epochs=2000,
        checkpoint_freq=1000,
    )


if __name__ == "__main__":
    ray_resnet(
        "afieti_homogeneous_neumann",
        "dataset_homogeneous_Neumann_l_3_deg_3.mat",
        "default",
    )
