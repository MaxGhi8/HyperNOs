"""
In this example I choose some parameters to tune and some to keep fixed for the FNO model.
"""

import sys

import torch

sys.path.append("..")

from datasets import Darcy
from FNO import FNO
from loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters


def ray_model_problem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_space = {
        "FourierF": tune.choice([0]),
        "RNN": tune.choice([False]),
        "batch_size": tune.choice([32]),
        "beta": tune.choice([1]),
        "epochs": tune.choice([1000]),
        "fft_norm": tune.choice([None]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "in_dim": tune.choice([1]),
        "include_grid": tune.choice([1]),
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "modes": tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]),  # modes1 = modes2
        "n_layers": tune.randint(1, 6),
        "out_dim": tune.choice([1]),
        "padding": tune.randint(0, 16),
        "problem_dim": tune.choice([2]),
        "retrain": tune.choice([4]),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "scheduler_step": tune.choice([10]),
        "training_samples": tune.choice([256]),
        "val_samples": tune.choice([128]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "weights_norm": tune.choice(["Kaiming"]),
        "width": tune.choice([4, 8, 16, 32, 64, 128, 256]),
    }

    model_builder = lambda config: FNO(
        config["problem_dim"],
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        config["retrain"],
    )

    # Define the dataset builder
    dataset_builder = lambda config: Darcy(
        {
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
    )

    loss_fn = LprelLoss(2)

    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        num_samples=50,
        runs_per_cpu=0.0,
        runs_per_gpu=1.0,
    )


if __name__ == "__main__":
    ray_model_problem()
