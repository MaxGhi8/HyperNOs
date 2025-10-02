"""
Example script to train the Parallel (patch-sharded) BAMPNO architecture.
Configuration mirrors train_bampno.py but swaps in ParallelBAMPNO which
splits the patch dimension across all available GPUs (when launched with torchrun).

Usage (single node multi-GPU):
  torchrun --standalone --nproc_per_node=NUM_GPUS neural_operators/examples/train_bampno_parallel.py

Usage (single GPU / CPU):
  python neural_operators/examples/train_bampno_parallel.py

Notes:
- Model auto-initializes torch.distributed if launched under torchrun.
- Outputs per-rank are restricted so that metadata files are only written by rank 0.
- For losses requiring global patch coupling beyond continuity, you may need to
  modify train_fixed_model to gather outputs. Continuity is already handled internally.
"""
import os
import sys

import torch

sys.path.append("..")

from architectures.BAMPNO.BAMPNO_parallel import ParallelBAMPNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters
from wrappers import wrap_model_builder


def train_bampno_parallel(
    which_example: str, which_domain: str, mode_hyperparams: str, loss_fn_str: str
):
    # Device hint (actual device per-rank decided inside ParallelBAMPNO)
    _device_hint = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load default hyperparameters
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "BAMPNO", which_example + "_" + which_domain, mode_hyperparams
    )
    default_hyper_params = {**hyperparams_train, **hyperparams_arc}

    # Base dataset / example (for output normalizer, etc.)
    example = NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": default_hyper_params["FourierF"],
            "retrain": default_hyper_params["retrain_seed"],
        },
        batch_size=default_hyper_params["batch_size"],
        training_samples=default_hyper_params["training_samples"],
        filename=default_hyper_params["grid_filename"],
    )

    # Model builder for ParallelBAMPNO
    model_builder = lambda config: ParallelBAMPNO(
        config["problem_dim"],
        config["n_patch"],
        config["continuity_condition"],
        config["n_pts"],
        config["grid_filename"],
        config["in_dim"],
        config["d_v"],
        config["out_dim"],
        config["L"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        (
            {int(k): v for k, v in config["zero_BC"].items()} if config["zero_BC"] else None
        ),
        config["arc"],
        config["RNN"],
        config["same_params"],
        config["FFTnorm"],
        example.output_normalizer if config["internal_normalization"] else None,
        config["retrain_seed"],
        enable_parallel=True,
    )

    # Dataset builder (same as standard example)
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain_seed"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["grid_filename"],
    )

    # Loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = (
        f"BAMPNO_parallel/{which_domain}/loss_{loss_fn_str}_mode_{mode_hyperparams}"
    )

    folder = f"../tests/{experiment_name}"
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder, exist_ok=True)
            print("Generated new folder")
        except Exception:
            pass

    # Rank-safe file writing (rank 0 only once model is built). For now, assume rank 0 (not initialized yet)
    with open(folder + "/norm_info.txt", "w") as f:
        f.write("Norm used during the training:\n")
        f.write(f"{loss_fn_str}\n")

    # Train
    train_fixed_model(
        default_hyper_params,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        get_plot_function(which_example, "input"),
        get_plot_function(which_example, "output"),
        full_validation=True,
    )


if __name__ == "__main__":
    # Mirror defaults of the original example
    train_bampno_parallel("bampno", "8_domain", "default", "L2_cheb_mp")
