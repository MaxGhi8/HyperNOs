"""
This script runs Ray Tune hyperparameter optimization for the UQNO model.
"""

import sys
import torch
import torch.nn as nn
from ray import tune

sys.path.append("../../")
from datasets import NO_load_data_model
from neuralop.models import TFNO, UQNO
from tune import tune_hyperparameters
from wrappers import wrap_model_builder

class QuantileLoss:
    """
    Pinball loss for UQNO.
    """
    def __init__(self, q=0.9):
        self.q = q

    def __call__(self, pred_tuple, target):
        if not isinstance(pred_tuple, tuple):
             return torch.nn.functional.l1_loss(pred_tuple, target)

        pred_mean, pred_spread = pred_tuple
        
        abs_error = torch.abs(pred_mean - target)
        diff = abs_error - pred_spread
        loss = torch.max(diff * self.q, diff * (self.q - 1))
        
        return torch.mean(loss)

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

        # In Eval mode (validation), return only solution
        if not self.training:
            return solution

        return (solution, quantile)

def ray_uqno(which_example: str, loss_fn_str: str):

    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 50,
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
        "projection_channels": 32,
        "problem_dim": 2,
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

    # Bypass standard wrapper as we included our own
    model_builder = wrap_model_builder(build_uqno, which_example)

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

    # Custom Loss
    loss_fn = QuantileLoss(q=0.9)

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
    ray_uqno("poisson", "L1")
