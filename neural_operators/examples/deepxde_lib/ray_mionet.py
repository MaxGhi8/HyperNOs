"""
This script runs Ray Tune hyperparameter optimization for the DeepXDE MIONet model.
"""

import os
# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import sys
import deepxde as dde
import torch
from ray import tune

# DeepXDE changes default tensor type to CUDA if available, which breaks datasets.py DataLoader generator
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from tune import tune_hyperparameters
from wrappers import wrap_model_builder

def ray_mionet(which_example: str, loss_fn_str: str):

    # Base hyperparameters
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

    # Define the search space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "network_width": tune.choice([32, 64]),
    }

    # Set all the other parameters to fixed values
    fixed_params = {
        **default_hyper_params,
    }
    parameters_to_tune = config_space.keys()
    for param in parameters_to_tune:
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Define the model builder
    def model_builder(config):
        # Input size for branch: width * width * in_channels
        input_dim = config["width"] * config["width"] * (1 + 2 * config.get("FourierF", 0))
        p = config["network_width"] 
        
        # Branch net 1
        layer_sizes_branch1 = [input_dim] + [config["network_width"]] * config["branch_depth"] + [p]
        # Branch net 2
        layer_sizes_branch2 = [input_dim] + [config["network_width"]] * config["branch_depth"] + [p]
        
        # Trunk net
        layer_sizes_trunk = [2] + [config["network_width"]] * config["trunk_depth"] + [p]
        
        model = dde.nn.MIONetCartesianProd(
            layer_sizes_branch1,
            layer_sizes_branch2,
            layer_sizes_trunk,
            "relu",
            "Glorot normal",
            num_outputs=config["out_dim"],
            multi_output_strategy="independent" if config["out_dim"] > 1 else None
        )
        
        model.out_dim = config["out_dim"]
        
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float32)
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            
        return model

    model_builder = wrap_model_builder(model_builder, which_example + "_mionet_deepxde")

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
        runs_per_cpu=10.0, 
        runs_per_gpu=1.0,
    )

    print("Best hyperparameters found were: ", best_result.config)


if __name__ == "__main__":
    ray_mionet("poisson", "L1")
