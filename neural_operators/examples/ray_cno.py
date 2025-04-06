"""
In this example I choose some parameters to tune and some to keep fixed fot the CNO model.
"""

import sys

import torch

sys.path.append("..")

from CNO import CNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters
from wrappers import wrap_model_builder


def ray_cno(which_example: str, mode_hyperparams: str, loss_fn_str: str):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the CNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "CNO", which_example, mode=mode_hyperparams
    )

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "N_layers": tune.randint(1, 5),
        "channel_multiplier": tune.choice([8, 16, 24, 32, 40, 48]),
        "N_res_neck": tune.randint(1, 6),
        "N_res": tune.randint(1, 8),
    }
    # kernel size is different for different problem dimensions
    if hyperparams_arc["problem_dim"] == 1:
        config_space["kernel_size"] = tune.choice([11, 21, 31, 41, 51])
    if hyperparams_arc["problem_dim"] == 2:
        config_space["kernel_size"] = tune.choice([3, 5, 7])

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
        }
    ]

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
        runs_per_cpu=8.0,
        runs_per_gpu=0.5,
    )


if __name__ == "__main__":
    ray_cno("darcy", "default", "L1")
