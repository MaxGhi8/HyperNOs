"""
In this example I fix all the hyperparameters for the CODANO model and train it.
"""

import os
import sys

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import CODANO
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder

def train_codano(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 32,
        "modes": 16,
        "n_layers": 4,
        "input_dim": 1,
        "out_dim": 1,
        "FourierF": 0,
        "retrain": 4,
        "problem_dim": 2,
    }

    # Define the model builders
    model_builder = lambda config: CODANO(
        n_modes=[[config["modes"]] * config["problem_dim"]] * config["n_layers"],
        hidden_variable_codimension=config["width"],
        lifting_channels=config["width"],
        projection_channels=config["width"],
        n_layers=config["n_layers"],
        output_variable_codimension=config["out_dim"], 
        n_heads=[1] * config["n_layers"],
        per_layer_scaling_factors=[[1] * config["problem_dim"]] * config["n_layers"],
        attention_scaling_factors=[1] * config["n_layers"],
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
        training_samples=config["training_samples"],
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = f"CODANO/{which_example}/loss_{loss_fn_str}_mode_testing"

    # Create the right folder if it doesn't exist
    folder = f"../../tests/{experiment_name}"
    if not os.path.isdir(folder):
        print("Generated new folder")
        os.makedirs(folder, exist_ok=True)

    # Save the norm information
    with open(folder + "/norm_info.txt", "w") as f:
        f.write("Norm used during the training:\n")
        f.write(f"{loss_fn_str}\n")

    # Call the library function to tune the hyperparameters
    train_fixed_model(
        default_hyper_params,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        get_plot_function(which_example, "input"),
        get_plot_function(which_example, "output"),
        output_folder=folder,
    )


if __name__ == "__main__":
    train_codano("poisson", "L1")
