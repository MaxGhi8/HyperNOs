"""
In this example I choose some parameters to tune and some to keep fixed for the FNO model.
"""

import sys

import torch

sys.path.append("..")

from architectures import FNO
from datasets import NO_load_data_model, concat_datasets
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters
from wrappers import wrap_model_builder


def ray_fno_multiple_resolutions(
    resolution: list[int],
    which_example: str,
    mode_hyperparams: str,
    loss_fn_str: str,
):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", which_example, mode=mode_hyperparams
    )

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "width": tune.choice([2, 4, 8, 16, 32, 64, 128, 256]),
        "n_layers": tune.randint(1, 6),
        "modes": tune.randint(1, 9),  # Satisfy the Shannon-Nyquist sampling theorem
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
    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example)

    dataset_builder = lambda config: concat_datasets(
        *(
            NO_load_data_model(
                which_example,
                no_architecture={
                    "FourierF": config["FourierF"],
                    "retrain": config["retrain"],
                },
                batch_size=config["batch_size"],
                training_samples=config["training_samples"],
                in_size=res,
            )
            for res in resolution
        )
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
        runs_per_cpu=6.0,
        runs_per_gpu=1.0,
    )


if __name__ == "__main__":
    ray_fno_multiple_resolutions([64, 32, 16], "poisson", "test", "L1")
