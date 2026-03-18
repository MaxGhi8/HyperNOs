"""
This script runs Ray Tune hyperparameter optimization for the FNOGNO model.
"""

import sys
import torch
import torch.nn as nn
from ray import tune

sys.path.append("../../")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import FNOGNO
from neuralop.layers.gno_block import GNOBlock
from tune import tune_hyperparameters
from wrappers import wrap_model_builder

# Monkey patch GNOBlock to force use_torch_scatter_reduce=False
# This is to silence the "use_scatter is True but torch_scatter is not properly built" warning
# which is printed every forward pass and cannot be suppressed via warnings or configuration.
_original_gno_init = GNOBlock.__init__

def _patched_gno_init(self, *args, **kwargs):
    if 'use_torch_scatter_reduce' in kwargs:
        kwargs['use_torch_scatter_reduce'] = False
    _original_gno_init(self, *args, **kwargs)

GNOBlock.__init__ = _patched_gno_init

class FNOGNO_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.grid = None

    def forward(self, x):
        # x is (Batch, Channels, Height, Width) or (Batch, Height, Width, Channels)
        
        # Determine shape and ensure (B, C, H, W)
        if x.ndim == 4:
            if x.shape[1] == 1 or x.shape[1] == 2: # Assuming small channel dim at 1 means (B, C, H, W)
                 B, C, H, W = x.shape
                 x_f = x.permute(0, 2, 3, 1).reshape(B, -1, C) # (B, N, C)
            else: # (B, H, W, C)
                 B, H, W, C = x.shape
                 x_f = x.reshape(B, -1, C)
        
        N_points = H * W
        
        if self.grid is None or self.grid.shape[0] != 1:
            # Create grid (1, H*W, 2)
            grid_x = torch.linspace(0, 1, H, device=x.device)
            grid_y = torch.linspace(0, 1, W, device=x.device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
            self.grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)

        # Manual batching
        outputs = []
        for b in range(B):
            in_p = self.grid.squeeze(0).view(H, W, 2) 
            out_p = self.grid.squeeze(0) 
            f_b = x_f[b].view(H, W, -1) 
            
            # Forward pass for single sample
            out_b = self.model(in_p, out_p, f_b) 
            outputs.append(out_b.unsqueeze(0))
            
        out = torch.cat(outputs, dim=0) 
        
        # Reshape back to (B, H, W, out_C)
        out = out.reshape(B, H, W, -1)
        return out

def ray_fnogno(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 4, 
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 32,
        "modes": 16,
        "n_layers": 4,
        "input_dim": 1,
        "out_dim": 1,
        "problem_dim": 2,
        "FourierF": 0,
        "retrain": 4,
        "projection_channels": 32,
    }

    # Define the hyperparameter search space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "modes": tune.choice([8, 12, 16]),
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
    model_builder = lambda config: FNOGNO_Wrapper(FNOGNO(
        in_channels=config["input_dim"],
        out_channels=config["out_dim"],
        fno_n_modes=(config["modes"], config["modes"]),
        fno_hidden_channels=config["width"],
        fno_n_layers=config["n_layers"],
        gno_coord_dim=config["problem_dim"],
        projection_channels=config["projection_channels"],
        gno_use_open3d=False,
    ))
    
    # Bypass standard wrapper as we included our own
    model_builder = wrap_model_builder(model_builder, which_example)

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
    ray_fnogno("poisson", "L1")
