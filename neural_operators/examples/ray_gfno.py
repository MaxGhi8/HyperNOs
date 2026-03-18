"""
This script runs Ray Tune hyperparameter optimization for the G-FNO model.
"""

import sys
from ray import tune
import torch

sys.path.append("..")
from architectures import G_FNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from tune import tune_hyperparameters
from wrappers import wrap_model_builder

def ray_gfno(which_example: str, loss_fn_str: str):

    # Base hyperparameters (fixed if not tuned)
    default_hyper_params = {
        "training_samples": 256,
        "learning_rate": 0.001,
        "epochs": 100,  # Lower epochs for tuning
        "batch_size": 32,
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 16,  # d_v
        "modes": 16,
        "n_layers": 2,  # L
        "input_dim": 1,
        "out_dim": 1,
        "reflection": False,
        "FourierF": 0,
        "retrain": 4,
        "problem_dim": 2,
        "padding": 0,
        "time_modes": 0
    }

    # Define the search space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "width": tune.choice([16, 24, 32]),
        "modes": tune.choice([16, 24]),
        "n_layers": tune.choice([2, 3, 4]),
    }

    # Set all the other parameters to fixed values
    fixed_params = {
        **default_hyper_params,
    }
    parameters_to_tune = config_space.keys()
    for param in parameters_to_tune:
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Define the model builder
    model_builder = lambda config: G_FNO( # noqa: E731
        problem_dim=config["problem_dim"],
        in_dim=config["input_dim"],
        d_v=config["width"],
        out_dim=config["out_dim"],
        L=config["n_layers"],
        modes=config["modes"],
        reflection=config["reflection"],
        padding=config["padding"],
        time_modes=config.get("time_modes", None),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example + "_neural_operator")

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model( # noqa: E731
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

    # Call the library function to tune the hyperparameters
    best_result = tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params=[default_hyper_params],
        num_samples=10,
        max_epochs=default_hyper_params["epochs"],
        checkpoint_freq=default_hyper_params["epochs"],
        grace_period=default_hyper_params["epochs"]//5,
        reduction_factor=4,
        runs_per_cpu=10.0,
        runs_per_gpu=1.0 if torch.cuda.is_available() else 0,
    )

    print("Best hyperparameters found were: ", best_result.config)


if __name__ == "__main__":
    ray_gfno("darcy", "L2")
