"""
In this example I choose some parameters to tune and some to keep fixed for the FNO model.
"""

import sys

import torch

sys.path.append("..")

from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import TFNO
from ray import tune
from tune import tune_hyperparameters


def ray_tfno(which_example: str, loss_fn_str: str):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the hyperparameter search space
    config_space = {
        "training_samples": tune.choice([1024]),
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "epochs": tune.choice([1000]),
        "batch_size": tune.choice([32]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_step": tune.choice([10]),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "beta": tune.choice([1]),
        "width": tune.choice([4, 8, 16, 32, 64, 96, 128, 160, 192]),
        "modes": tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]),  # modes1 = modes2
        "n_layers": tune.randint(1, 6),
        "input_dim": tune.choice([1]),
        "out_dim": tune.choice([1]),
        "rank": tune.choice([0.05]),
        "FourierF": tune.choice([0]),
        "retrain": tune.choice([4]),
        "problem_dim": tune.choice([2]),
    }

    # Define the model builders
    model_builder = lambda config: TFNO(
        n_modes=(config["modes"], config["modes"]),
        hidden_channels=config["width"],
        n_layers=config["n_layers"],
        in_channels=config["input_dim"] + 2,  # for the grid
        out_channels=config["out_dim"],
        factorization="tucker",
        implementation="factorized",
        rank=config["rank"],
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
        runs_per_cpu=8.0,
        runs_per_gpu=1.0,
    )


if __name__ == "__main__":
    ray_tfno("cont_tran", "L1")
