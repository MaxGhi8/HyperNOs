"""
In this example I fix all the hyperparameters for the FNOGNO model and train it.

WARNING: This script is experimental. FNOGNO typically requires geometry/mesh data.
We attempt to adapt 2D grid data using a wrapper, but this may be unstable or slow.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.append("..")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import FNOGNO
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder

from neuralop.layers.gno_block import GNOBlock

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
        # FNOGNO expects:
        # in_p: (Batch, N_points, coord_dim)
        # out_p: (Batch, N_points, coord_dim)
        # f: (Batch, N_points, in_channels)
        
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

        # Manual batching to avoid OOM and native_search issues
        outputs = []
        for b in range(B):
            in_p = self.grid.squeeze(0).view(H, W, 2) # (H, W, 2)
            out_p = self.grid.squeeze(0) # (N, 2)
            f_b = x_f[b].view(H, W, -1) # (H, W, C)
            
            # Forward pass for single sample
            out_b = self.model(in_p, out_p, f_b) # (N, out_C)
            outputs.append(out_b.unsqueeze(0))
            
        out = torch.cat(outputs, dim=0) # (B, N, out_C)
        
        # Reshape back to (B, H, W, out_C) - Channel Last to match Data
        out = out.reshape(B, H, W, -1)
        return out

def train_fnogno(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 4, # Reduced from 32 to avoid OOM with GNO on 64x64 grid
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 64,
        "modes": 16,
        "n_layers": 4,
        "input_dim": 1,
        "out_dim": 1,
        "problem_dim": 2,
        "FourierF": 0,
        "retrain": 4,
        "projection_channels": 64,
    }

    # Define the model builders
    # Custom wrapping for FNOGNO
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
        training_samples=config["training_samples"],
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = f"FNOGNO/{which_example}/loss_{loss_fn_str}"

    # Create the right folder if it doesn't exist
    folder = f"../tests/{experiment_name}"
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
    )


if __name__ == "__main__":
    train_fnogno("poisson", "L1")
