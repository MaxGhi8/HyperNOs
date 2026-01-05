"""
In this example, we train a UQNO (Uncertainty Quantification Neural Operator).
UQNO wraps a base model (here TFNO) and trains a residual model to predict uncertainty intervals.

We use a custom QuantileLoss (Pinball loss) to train the residual model.
"""

import os
import sys
import torch
import torch.nn as nn
import warnings

sys.path.append("..")
from datasets import NO_load_data_model
from neuralop.models import TFNO, UQNO
from train import train_fixed_model
from utilities import get_plot_function
from wrappers import wrap_model_builder
from loss_fun import LprelLoss

class QuantileLoss:
    """
    Pinball loss for UQNO.
    UQNO returns (solution, uncertainty_interval).
    We interpret uncertainty_interval as the predicted spread around the solution.
    We aim to minimize the quantile loss between the error |y - y_pred| and the predicted interval.
    """
    def __init__(self, q=0.9):
        self.q = q

    def __call__(self, pred_tuple, target):
        # pred_tuple is (solution, quantile) from UQNO during training
        # BUT during validation, UQNO_Wrapper returns only solution (Tensor).
        # In that case, we fallback to L1 loss (or similar) since spread is missing.
        
        if not isinstance(pred_tuple, tuple):
             return torch.nn.functional.l1_loss(pred_tuple, target)

        pred_mean, pred_spread = pred_tuple
        
        # We model the spread of the absolute error
        abs_error = torch.abs(pred_mean - target)
        
        # Pinball loss between the predicted spread and the actual error magnitude
        # We want pred_spread to estimate the q-th quantile of the error distribution
        
        diff = abs_error - pred_spread
        # If diff > 0 (error > spread), punish by q
        # If diff < 0 (error < spread), punish by (1-q)
        loss = torch.max(diff * self.q, diff * (self.q - 1))
        
        return torch.mean(loss)

def train_uqno(which_example: str, loss_fn_str: str):
    
    # UQNO trains a residual model, typically on top of a decent base model.
    # Here we initialize a fresh TFNO as base (frozen) and train the residual.
    # In practice, you would load a pretrained base model.

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 50, # Fewer epochs as we iterate fast
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
        "projection_channels": 64,
        "problem_dim": 2,
    }

    # UQNO Wrapper to handle permutation for Tuple output
    class UQNO_Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Input: (Batch, ..., Channels) -> (Batch, Channels, ...)
            dims = list(range(x.ndim))
            dims.insert(1, dims.pop(-1))
            x = x.permute(*dims)

            # UQNO returns (solution, quantile)
            out = self.model(x)
            
            # Helper to permute back
            def permute_back(t):
                if t.ndim < 2: return t
                dims = list(range(t.ndim))
                dims.append(dims.pop(1))
                return t.permute(*dims)

            solution, quantile = out
            solution = permute_back(solution)
            quantile = permute_back(quantile)

            # CRITICAL: In Eval mode (validation), train.py expects a single Tensor output
            # (for loss calculation and concatenation). UQNO tuple breaks this.
            # We return only solution during eval. Custom QuantileLoss (used in training) handles tuple.
            if not self.training:
                return solution

            return (solution, quantile)

    # Define the model builders
    def build_uqno(config):
        # Base model (TFNO)
        base_model = TFNO(
            n_modes=(config["modes"], config["modes"]),
            hidden_channels=config["width"],
            in_channels=config["input_dim"],
            out_channels=config["out_dim"],
            n_layers=config["n_layers"],
            projection_channels=config["projection_channels"],
            factorization="tucker",
            rank=0.42,
        )
        
        # UQNO wraps base model
        model = UQNO(base_model=base_model)
        
        # Wrap with our custom wrapper
        return UQNO_Wrapper(model)

    # Use which_example WITHOUT _neural_operator suffix to avoid double wrapping/errors
    # The wrap_model_builder will return the model as is (which is our UQNO_Wrapper)
    model_builder = wrap_model_builder(build_uqno, which_example)

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

    # Custom Loss
    loss_fn = QuantileLoss(q=0.9)

    experiment_name = f"UQNO/{which_example}/quantile_0.9"

    # Create the right folder if it doesn't exist
    folder = f"../tests/{experiment_name}"
    if not os.path.isdir(folder):
        print("Generated new folder")
        os.makedirs(folder, exist_ok=True)

    with open(folder + "/norm_info.txt", "w") as f:
        f.write(f"Quantile Loss q=0.9\n")

    # Plotting wrapper to handle tuple output
    original_plotter = get_plot_function(which_example, "output")
    
    def uqno_plotter(example, data_plot, title, ep, writer, normalization=True, plotting=False):
        # Unwrap tuple if needed
        if isinstance(data_plot, tuple):
             # Plot the mean solution (first element)
             # We could also plot the uncertainty spread separately if needed
             data_plot = data_plot[0]
        
        if original_plotter:
            original_plotter(example, data_plot, title, ep, writer, normalization, plotting)

    # Call the library function
    train_fixed_model(
        default_hyper_params,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        get_plot_function(which_example, "input"),
        uqno_plotter,
    )


if __name__ == "__main__":
    train_uqno("poisson", "L1") # L1 argument ignored by custom loss logic
