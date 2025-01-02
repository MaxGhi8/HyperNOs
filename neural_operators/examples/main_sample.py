import torch
import sys

import os

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from datasets import Darcy
from FNO.FNO_arc import FNO_2D
from FNO.FNO_utilities import FNO_initialize_hyperparameters
from loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters


def main():

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = FNO_initialize_hyperparameters(
        "poisson", mode="default"
    )

    total_default_params = (
        hyperparams_arc["n_layers"]
        * hyperparams_arc["width"] ** 2
        * hyperparams_arc["modes"] ** hyperparams_arc["problem_dim"]
    )

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "width": tune.choice([4, 8, 16, 32, 64, 128, 256]),
        "n_layers": tune.randint(1, 6),
        "modes": tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]),  # modes1 = modes2
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "padding": tune.randint(0, 16),
    }
    # Default value for the hyper-parameters in the search space
    default_hyper_params = [
        {
            "learning_rate": hyperparams_train["learning_rate"],
            "weight_decay": hyperparams_train["weight_decay"],
            "scheduler_gamma": hyperparams_train["scheduler_gamma"],
            "width": hyperparams_arc["width"],
            "n_layers": hyperparams_arc["n_layers"],
            "modes": hyperparams_arc["modes"],
            "fun_act": hyperparams_arc["fun_act"],
            "fno_arc": hyperparams_arc["fno_arc"],
            "padding": hyperparams_arc["padding"],
        }
    ]

    # Set all the other parameters to fixed values
    fixed_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }

    parameters_to_tune = config_space.keys()
    for param in parameters_to_tune:
        fixed_params.pop(param, None)

    config_space.update(fixed_params)
    default_hyper_params[0].update(fixed_params)

    model_builder = lambda config: FNO_2D(
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        int(
            (total_default_params / (config["n_layers"] * config["width"] ** 2))
            ** (1 / hyperparams_arc["problem_dim"])
        ),
        int(
            (total_default_params / (config["n_layers"] * config["width"] ** 2))
            ** (1 / hyperparams_arc["problem_dim"])
        ),
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        config["retrain"],
    )

    dataset_builder = lambda config: Darcy(
        {
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        config["batch_size"],
        search_path="/",
    )

    loss_fn = LprelLoss(2, False)

    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params,
        runs_per_cpu=8.0,
        runs_per_gpu=1.0,
    )


if __name__ == "__main__":
    main()
