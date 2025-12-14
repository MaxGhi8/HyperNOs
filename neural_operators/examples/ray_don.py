"""
In this example I choose some parameters to tune and some to keep fixed for the FNO model.
"""

import sys

import torch

sys.path.append("..")

from architectures import DeepONet as DON
from datasets import NO_load_data_model
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters
from wrappers import wrap_model_builder


def ray_don(which_example: str, mode_hyperparams: str, loss_fn_str: str):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the DON model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "DON", which_example, mode=mode_hyperparams
    )

    # Define mappings for list-based hyperparameters
    branch_hidden_layer_options = [[32, 32, 32], [64, 64, 64], [128, 128, 128]]
    branch_channels_options = [[10, 20, 30], [20, 40, 60], [30, 60, 90]]
    trunk_hidden_layer_options = [[32, 32, 32], [64, 64, 64], [128, 128, 128]]

    # Define the hyperparameter search space
    config_space = {
        # Optimization hyperparameters
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        # DON hyperparameters
        "n_basis": tune.randint(10, 500),
        # Add branch network parameters (using indices for list-based params)
        "branch_channels_idx": tune.choice([0, 1, 2]),
        "branch_hidden_layer_idx": tune.choice([0, 1, 2]),
        "branch_act_fun": tune.choice(["tanh", "relu", "leaky_relu", "sigmoid"]),
        # "branch_n_blocks": tune.randint(2, 5), # TODO
        "branch_normalization": tune.choice(["none", "batch", "layer"]),
        "branch_dropout_rate": tune.quniform(0.0, 0.4, 0.01),
        # "branch_residual": tune.choice([0, 1]), # TODO
        # Add trunk network parameters (using indices for list-based params)
        "trunk_hidden_layer_idx": tune.choice([0, 1, 2]),
        "trunk_act_fun": tune.choice(["tanh", "relu", "leaky_relu", "sigmoid"]),
        "trunk_n_blocks": tune.randint(2, 5),
        "trunk_layer_norm": tune.choice([True, False]),
        "trunk_dropout_rate": tune.quniform(0.0, 0.4, 0.01),
        "trunk_residual": tune.choice([0, 1]),
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

    # Default value for the hyper-parameters in the search space
    default_hyper_params = [
        {
            **hyperparams_train,
            **hyperparams_arc,
            # Add default indices for list-based hyperparameters
            "trunk_hidden_layer_idx": 1,  # [64, 64, 64]
            "trunk_act_fun": hyperparams_arc["trunk_hyperparameters"]["act_fun"],
            "trunk_n_blocks": hyperparams_arc["trunk_hyperparameters"]["n_blocks"],
            "trunk_layer_norm": hyperparams_arc["trunk_hyperparameters"]["layer_norm"],
            "trunk_dropout_rate": hyperparams_arc["trunk_hyperparameters"][
                "dropout_rate"
            ],
            "trunk_residual": hyperparams_arc["trunk_hyperparameters"]["residual"],
            "branch_channels_idx": 0,  # [10, 20, 30]
            "branch_hidden_layer_idx": 1,  # [64, 64, 64]
            "branch_act_fun": hyperparams_arc["branch_hyperparameters"]["act_fun"],
            "branch_n_blocks": hyperparams_arc["branch_hyperparameters"]["n_blocks"],
            "branch_normalization": hyperparams_arc["branch_hyperparameters"][
                "normalization"
            ],
            "branch_dropout_rate": hyperparams_arc["branch_hyperparameters"][
                "dropout_rate"
            ],
        }
    ]

    # TODO: add internal normalization
    # example = NO_load_data_model(
    #     which_example=which_example + "_don",
    #     no_architecture={
    #         "FourierF": default_hyper_params[0]["FourierF"],
    #         "retrain": default_hyper_params[0]["retrain"],
    #     },
    #     batch_size=default_hyper_params[0]["batch_size"],
    #     training_samples=default_hyper_params[0]["training_samples"],
    #     filename=(
    #         default_hyper_params[0]["filename"]
    #         if "filename" in default_hyper_params[0]
    #         else None
    #     ),
    # )

    # Define the model builders
    model_builder = lambda config: DON(
        branch_hyperparameters={
            "n_inputs": config["branch_hyperparameters"]["n_inputs"],
            "channels_conv": branch_channels_options[config["branch_channels_idx"]],
            "hidden_layer": branch_hidden_layer_options[
                config["branch_hidden_layer_idx"]
            ],
            "kernel_size": config["branch_hyperparameters"]["kernel_size"],
            "stride": config["branch_hyperparameters"]["stride"],
            "padding": config["branch_hyperparameters"]["padding"],
            "include_grid": config["branch_hyperparameters"]["include_grid"],
            "act_fun": config["branch_act_fun"],
            "n_blocks": config["branch_n_blocks"],
            "normalization": config["branch_normalization"],
            "dropout_rate": config["branch_dropout_rate"],
            "residual": config["branch_hyperparameters"]["residual"],
        },
        trunk_hyperparameters={
            "n_inputs": config["trunk_hyperparameters"]["n_inputs"],
            "hidden_layer": trunk_hidden_layer_options[
                config["trunk_hidden_layer_idx"]
            ],
            "act_fun": config["trunk_act_fun"],
            "n_blocks": config["trunk_n_blocks"],
            "layer_norm": config["trunk_layer_norm"],
            "dropout_rate": config["trunk_dropout_rate"],
            "residual": config["trunk_residual"],
        },
        n_basis=config["n_basis"],
        n_output=config["n_output"],
        dim=config["problem_dim"],
        device=device,
    )

    # Get grid size from config, default to 64 for Darcy problem
    grid_size = default_hyper_params[0]["size_grid"]

    # Wrap the model builder with grid_size for DeepONet output reshaping
    model_builder = wrap_model_builder(
        model_builder, which_example + "_don", grid_size=grid_size
    )

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example + "_don",
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=(config["filename"] if "filename" in config else None),
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=config_space["problem_dim"],
        beta=config_space["beta"],
    )

    # Call the library function to tune the hyperparameters
    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params,
        runs_per_cpu=12.0,
        runs_per_gpu=1.0,
        grace_period=500,
        reduction_factor=4,
        max_epochs=1000,
        checkpoint_freq=1000,
    )


if __name__ == "__main__":
    ray_don("darcy", "default", "L2")
