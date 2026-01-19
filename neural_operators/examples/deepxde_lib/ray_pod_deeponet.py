"""
This script runs Ray Tune hyperparameter optimization for the DeepXDE POD-DeepONet model.
"""

import os
# Ensure DeepXDE uses PyTorch backend
os.environ["DDE_BACKEND"] = "pytorch"

import sys
import deepxde as dde
import torch
import numpy as np
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

def ray_pod_deeponet(which_example: str, loss_fn_str: str):

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
        "network_width": 32, # Acts as p (number of modes)
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

    #####################################
    # Compute POD basis
    #####################################
    # Load Data to compute POD Basis
    print("\nLoading data to compute POD basis...")
    dataset_config = {
        "FourierF": default_hyper_params["FourierF"],
        "retrain": default_hyper_params["retrain"],
    }
    
    dataset = NO_load_data_model(
        which_example=which_example,
        no_architecture=dataset_config,
        batch_size=default_hyper_params["training_samples"],
        training_samples=default_hyper_params["training_samples"],
    )
    train_loader = dataset.train_loader

    # We group all the data into one single tensor
    y_train_list = []
    for _, y in train_loader:
        y_train_list.append(y)
    
    y_train = torch.cat(y_train_list, dim=0) # Shape: (Total_Samples, W, W, C)
    
    # Flatten spatial dims for POD: (N, W*W*C)
    N, W, _, C = y_train.shape
    y_train_flat = y_train.view(N, -1)
    data_matrix = y_train_flat.transpose(0, 1) # (W*W*C, N)
    
    print("Computing POD basis...")
    try:
        U, S, V = torch.linalg.svd(data_matrix, full_matrices=False)
    except Exception as e:
        print(f"SVD failed: {e}. Trying numpy.")
        U, S, Vh = np.linalg.svd(data_matrix.cpu().numpy(), full_matrices=False)
        U = torch.from_numpy(U).to(y_train.device)
    
    #####################################
    # Define the model builder
    #####################################
    def model_builder(config):
        # Input size for branch: width * width * in_channels
        input_dim = config["width"] * config["width"] * (1 + 2 * config.get("FourierF", 0))
        
        # Number of modes 'p' defined by network_width
        pod_modes = config["network_width"]
        pod_basis_slice = U[:, :pod_modes] # (Features, p)
        p = pod_modes
        
        # Branch net
        layer_sizes_branch = [input_dim] + [config["network_width"]] * config["branch_depth"] + [p]
        
        # Initialize PODDeepONet.
        model = dde.nn.PODDeepONet(
            pod_basis=pod_basis_slice.float(),
            layer_sizes_branch=layer_sizes_branch,
            activation="relu",
            kernel_initializer="Glorot normal",
            num_outputs=config["out_dim"],
            multi_output_strategy="independent" if config["out_dim"] > 1 else None,
            layer_sizes_trunk=None # POD-only mode
        )
        
        model.out_dim = config["out_dim"]
        
        # Manually register pod_basis as a buffer so it moves to device with the model
        if hasattr(model, "pod_basis"):
            basis = model.pod_basis
            del model.pod_basis
            model.register_buffer("pod_basis", basis)
        
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float32)
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            
        return model

    model_builder = wrap_model_builder(model_builder, which_example + "_deepxde")

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
    ray_pod_deeponet("poisson", "L1")
