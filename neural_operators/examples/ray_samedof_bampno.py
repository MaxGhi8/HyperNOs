"""
In this example I choose some parameters to tune and some to keep fixed.
Moreover I set the modes for the FNO in order to have comparable number of parameters across the different models.
"""

import sys

import torch

sys.path.append("..")

from BAMPNO import BAMPNO, compute_modes, count_params_bampno
from datasets import NO_load_data_model
from loss_fun import loss_selector
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters


def ray_samedof_bampno(
    which_example: str,
    which_domain: str,
    mode_hyperparams: str,
    loss_fn_str: str,
    maximum: int,
):
    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the baseline model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "BAMPNO", which_example + "_" + which_domain, mode_hyperparams
    )

    fixed_params = {
        **hyperparams_train,
        **hyperparams_arc,
    }
    # Approximate the total number of parameters (constant factor can be dropped)
    # total_default_params = count_params_bampno(fixed_params, accurate=False)
    total_default_params = 20_000_000

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "d_v": tune.choice(
            [4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112]
        ),
        "L": tune.randint(1, 6),
        "modes": tune.choice([2, 4, 8, 12, 16]),  # modes1 = modes2
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "arc": tune.choice(["Classic", "Zongyi", "Residual"]),
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
    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": fixed_params["FourierF"],
            "retrain": fixed_params["retrain_seed"],
        },
        batch_size=fixed_params["batch_size"],
        training_samples=fixed_params["training_samples"],
        filename=fixed_params["grid_filename"],
    )

    # Define the model builders
    model_builder = lambda config: BAMPNO(
        config["problem_dim"],
        config["n_patch"],
        config["continuity_condition"],
        config["n_pts"],
        config["grid_filename"],
        config["in_dim"],
        config["d_v"],
        config["out_dim"],
        config["L"],
        compute_modes(total_default_params, maximum, config),
        config["fun_act"],
        config["weights_norm"],
        (
            {int(k): v for k, v in config["zero_BC"].items()}
            if config["zero_BC"]
            else None
        ),
        config["arc"],
        config["RNN"],
        config["same_params"],
        config["FFTnorm"],
        device,
        example.output_normalizer if config["internal_normalization"] else None,
        config["retrain_seed"],
    )
    # Wrap the model builder
    # model_builder = wrap_model_builder(model_builder, which_example)

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain_seed"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["grid_filename"],
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
        num_samples=200,
        grace_period=250,
        reduction_factor=4,
        max_epochs=1000,
        checkpoint_freq=1000,
        runs_per_cpu=16.0,
        runs_per_gpu=0.5,
    )


if __name__ == "__main__":
    ray_samedof_bampno("bampno", "8_domain", "default", "L2", 16)
