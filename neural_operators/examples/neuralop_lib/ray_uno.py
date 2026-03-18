"""
This script runs Ray Tune hyperparameter optimization for the UNO model.
"""

import sys
from ray import tune

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import UNO
from tune import tune_hyperparameters
from wrappers import wrap_model_builder

def ray_uno(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 32,
        "n_layers": 5,
        "input_dim": 1,
        "out_dim": 1,
        "problem_dim": 2,
        "FourierF": 0,
        "retrain": 4,
        "modes": 16,
    }

    # Define the hyperparameter space to tune
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "width": tune.choice([32, 64]),
    }

    # Set all the other parameters to fixed values
    fixed_params = {
        **default_hyper_params,
    }
    parameters_to_tune = config_space.keys()
    for param in parameters_to_tune:
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Define the model builders
    model_builder = lambda config: UNO(
        in_channels=config["input_dim"],
        out_channels=config["out_dim"],
        hidden_channels=config["width"],
        n_layers=config["n_layers"],
        uno_out_channels=[config["width"], config["width"]*2, config["width"]*2, config["width"]*2, config["width"]],
        uno_n_modes=[[config.get("modes", 16), config.get("modes", 16)]] * 5,
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]],
        channel_mlp_skip="linear",
    )
    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example + "_neural_operator")

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=default_hyper_params["training_samples"],
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
        runs_per_cpu=12.0,
        runs_per_gpu=1.0,
    )


    print("Best hyperparameters found were: ", best_result.config)


if __name__ == "__main__":
    ray_uno("poisson", "L1")
