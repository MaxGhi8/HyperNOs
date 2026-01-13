"""
In this example I fix all the hyperparameters for the LocalNO model and train it.
"""

import os
import sys

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import LocalNO
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder

def train_local_no(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 64,
        "modes": 16,
        "n_layers": 4,
        "input_dim": 1,
        "out_dim": 1,
        "rank": 0.05,
        "FourierF": 0,
        "retrain": 4,
        "problem_dim": 2,
    }

    # Define the model builders
    model_builder = lambda config: LocalNO(
        n_modes=(config["modes"], config["modes"]),
        hidden_channels=config["width"],
        n_layers=config["n_layers"],
        in_channels=config["input_dim"],
        out_channels=config["out_dim"],
        default_in_shape=(64, 64),
        factorization="tucker",
        implementation="factorized",
        rank=config["rank"],
        disco_layers=False,
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

    experiment_name = f"LocalNO/{which_example}/loss_{loss_fn_str}_mode_testing"

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
    train_local_no("poisson", "L1")
