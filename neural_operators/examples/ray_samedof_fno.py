"""
In this example I choose some parameters to tune and some to keep fixed.
Moreover I set the modes for the FNO in order to have comparable number of parameters across the different models.
"""

import sys

import torch

sys.path.append("..")

from datasets import NO_load_data_model
from FNO import FNO, compute_modes, count_params_fno
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters
from wrappers import wrap_model_builder


def ray_samedof_fno(
    which_example: str, mode_hyperparams: str, loss_fn_str: str, maximum: int
):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the baseline model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", which_example, mode=mode_hyperparams
    )

    fixed_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }
    # Approximate the total number of parameters (constant factor can be dropped)
    total_default_params = count_params_fno(fixed_params, accurate=False)
    # total_default_params = 500000

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "width": tune.choice([4, 8, 16, 32, 48, 64, 80, 96, 112, 128]),
        "n_layers": tune.randint(1, 6),
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "padding": tune.randint(0, 16),
    }

    # Set all the other parameters to fixed values
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
    model_builder = lambda config: FNO(
        config["problem_dim"],
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        compute_modes(total_default_params, maximum, config),
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        config["retrain"],
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
    ray_samedof_fno("poisson", "default", "L2", 33)
