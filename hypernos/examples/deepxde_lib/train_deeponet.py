"""
In this example I fix all the hyperparameters for the DeepONet model and train it.
"""

import os
import sys
import torch
import deepxde as dde

# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder

def train_deeponet(which_example: str, loss_fn_str: str):

    # Fixed hyperparameters
    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 64, # Grid size
        "branch_depth": 2,
        "trunk_depth": 2,
        "network_width": 32,
        "retrain": 4,
        "problem_dim": 2,
        "out_dim": 1,
        "FourierF": 0,
    }

    # Define the model builder
    def model_builder(config):
        # Input size for branch: width * width * in_channels (assuming 1 channel currently)
        input_dim = config["width"] * config["width"] * (1 + 2 * config.get("FourierF", 0))
        
        # Output dim of branch/trunk networks
        p = config["network_width"] 
        
        # Branch net
        layer_sizes_branch = [input_dim] + [config["network_width"]] * config["branch_depth"] + [p]
        
        # Trunk net. Coordinate_dim is 2 for 2D grid
        layer_sizes_trunk = [2] + [config["network_width"]] * config["trunk_depth"] + [p]
        
        model = dde.nn.DeepONetCartesianProd(
            layer_sizes_branch,
            layer_sizes_trunk,
            "relu",
            "Glorot normal",
            num_outputs=config["out_dim"],
            multi_output_strategy="independent" if config["out_dim"] > 1 else None
        )

        model.out_dim = config["out_dim"]

        # Reset default tensor type to CPU because DeepXDE might have changed it during initialization
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float32)
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            
        return model

    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example + "_deepxde")

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

    experiment_name = f"DeepONet/{which_example}/loss_{loss_fn_str}_testing"

    # Create the right folder if it doesn't exist
    folder = f"../../tests/{experiment_name}"
    if not os.path.isdir(folder):
        print("Generated new folder")
        os.makedirs(folder, exist_ok=True)

    # Save the norm information
    with open(folder + "/norm_info.txt", "w") as f:
        f.write("Norm used during the training:\n")
        f.write(f"{loss_fn_str}\n")

    # Call the library function to train the fixed model
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
    train_deeponet("poisson", "L1")
