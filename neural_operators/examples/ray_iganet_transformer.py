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
from ray import tune
from tune import tune_hyperparameters
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

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-4),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "activation_str": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "dropout_rate": tune.quniform(0.0, 0.05, 1e-3),
        "n_heads": tune.randint(1, 15),
        # Hidden_dim is determined by n_heads * head_dim, so n_head divides hidden_dim
        "head_dim": tune.randint(4, 16),
        "hidden_dim": tune.sample_from(
            lambda spec: spec.config.n_heads * spec.config.head_dim
        ),
        "n_heads_A": tune.randint(1, 15),
        "n_layers_geo": tune.randint(1, 5),
    }

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
        which_example=which_example + "_transformer",
        no_architecture={
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
        n_heads_A=config["n_heads_A"],
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

    # Call the library function to tune the hyperparameters
    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        [default_hyper_params],
        runs_per_cpu=12.0,
        runs_per_gpu=1.0,
        grace_period=125,
        reduction_factor=4,
        max_epochs=500,
        checkpoint_freq=1000,
    )


if __name__ == "__main__":
    train_iganet_transformer(
        "afieti_homogeneous_neumann",
        "dataset_homogeneous_Neumann_l_0_deg_2_crazygeom.mat",
        "default",
    )
