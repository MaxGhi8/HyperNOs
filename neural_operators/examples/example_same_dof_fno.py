""" In this example I choose some parameters to tune and some to keep fixed. 
    Moreover I set the modes for the FNO in order to have comparable number of parameters across the different models.
"""

import torch
import sys

sys.path.append("..")

from datasets import NO_load_data_model
from FNO.FNO_arc import FNO_2D
from FNO.FNO_utilities import FNO_initialize_hyperparameters
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters


def main(example_name, mode_hyperparams, loss_fn_str):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = FNO_initialize_hyperparameters(
        example_name, mode=mode_hyperparams
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
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "padding": tune.randint(0, 16),
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
        }
    ]

    # Define the model builders
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

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=example_name,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        in_dist=True,
        search_path="/",
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
    main("darcy", "default", "L1")
