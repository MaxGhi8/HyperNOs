"""
In this example I fix all the hyperparameters for the GINO model and train it.

WARNING: This script is experimental. GINO typically requires geometry/mesh data.
We attempt to adapt 2D grid data using a wrapper, but this may be unstable or slow.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.append("..")
from datasets import NO_load_data_model
from loss_fun import loss_selector
from neuralop.models import GINO
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder

class GINO_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.grid = None
        self.latent_grid = None

    def forward(self, x):
        # x: (B, C, H, W)
        if x.ndim == 4:
            if x.shape[1] == 1 or x.shape[1] == 2:
                 B, C, H, W = x.shape
                 x_f = x.permute(0, 2, 3, 1).reshape(B, -1, C)
            else:
                 B, H, W, C = x.shape
                 x_f = x.reshape(B, -1, C)

        N_points = H * W
        
        if self.grid is None or self.grid.shape[0] != 1:
            # Create grid (1, H*W, 2)
            grid_x = torch.linspace(0, 1, H, device=x.device)
            grid_y = torch.linspace(0, 1, W, device=x.device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
            self.grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
            
            # Create latent grid (1, Hl, Wl, 2)
            H_l, W_l = H // 4, W // 4
            l_x = torch.linspace(0, 1, H_l, device=x.device)
            l_y = torch.linspace(0, 1, W_l, device=x.device)
            l_x, l_y = torch.meshgrid(l_x, l_y, indexing='ij')
            self.latent_grid = torch.stack((l_x, l_y), dim=-1).view(1, H_l, W_l, 2)

        # Manual batching
        outputs = []
        for b in range(B):
            # Input geom and output queries must be (1, N, 2) for GINO to squeeze them correctly to (N, 2)
            input_geom = self.grid 
            output_queries = self.grid.squeeze(0) # (N, 2) for GNO_out
            latent_queries = self.latent_grid 
            
            # Feature x must be (1, N, C) for batch_size=1
            f_b = x_f[b].unsqueeze(0) 
            
            out_b = self.model(
                input_geom=input_geom,
                latent_queries=latent_queries,
                output_queries=output_queries,
                x=f_b
            )
            # out_b is (1, N, C_out)
            outputs.append(out_b)
            
        out = torch.cat(outputs, dim=0) # (B, N, out_C)
        
        # Reshape back to (B, H, W, out_C) - Channel Last to match Data
        out = out.reshape(B, H, W, -1)
        return out

def train_gino(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 4, # Reduced from 32
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
    # Custom wrapping for GINO
    model_builder = lambda config: GINO_Wrapper(GINO(
        in_channels=config["input_dim"],
        out_channels=config["out_dim"],
        fno_n_modes=(config["modes"], config["modes"]),
        fno_hidden_channels=config["width"],
        fno_n_layers=config["n_layers"],
        gno_coord_dim=config["problem_dim"],
        projection_channels=config["projection_channels"],
        gno_use_open3d=False,
        gno_batched=True,
        gno_use_torch_scatter=False,
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

    experiment_name = f"GINO/{which_example}/loss_{loss_fn_str}"

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
    train_gino("poisson", "L1")
