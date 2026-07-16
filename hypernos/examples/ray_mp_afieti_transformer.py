"""Tune the in-package multi-patch AFIETI transformer with Ray/HyperOpt.

Unlike the standalone example in ``examples/mp_afieti_transformer``, this file
is intended to live inside the HyperNOs repository and therefore imports the
dataset and architecture directly from ``hypernos``.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from ray import tune

from hypernos.architectures import GeometryConditionedLinearOperator_mp_afieti
from hypernos.datasets import NO_load_data_model
from hypernos.loss_fun import lpLoss
from hypernos.tune import tune_hyperparameters
from hypernos.utilities import initialize_hyperparameters


# Sample width and head count as one categorical variable. This avoids invalid
# hidden_dim/n_heads combinations and works with HyperOpt without sample_from.
ATTENTION_LAYOUTS = (
    "16x1",
    "16x2",
    "20x1",
    "20x2",
    "20x4",
    "24x2",
    "24x3",
    "32x2",
    "32x4",
    "40x4",
    "40x5",
    "48x4",
    "48x6",
    "64x4",
    "64x8",
)


def decode_attention_layout(layout: str) -> tuple[int, int]:
    try:
        hidden_dim, n_heads = (int(value) for value in layout.split("x", 1))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid attention layout: {layout!r}") from error
    if hidden_dim <= 0 or n_heads <= 0 or hidden_dim % n_heads != 0:
        raise ValueError(
            f"Attention layout {layout!r} must have positive values and a "
            "hidden dimension divisible by the head count"
        )
    return hidden_dim, n_heads


def _default_attention_layout(config: dict[str, Any]) -> str:
    layout = f"{config['hidden_dim']}x{config['n_heads']}"
    decode_attention_layout(layout)
    return layout


def build_search_space(
    base_config: dict[str, Any], max_epochs: int
) -> dict[str, Any]:
    scheduler_steps = sorted(
        {
            int(base_config["scheduler_step"]),
            max(1, max_epochs // 4),
            max(1, max_epochs // 2),
            max_epochs,
        }
    )
    layouts = list(ATTENTION_LAYOUTS)
    default_layout = _default_attention_layout(base_config)
    if default_layout not in layouts:
        layouts.append(default_layout)

    search_domains = {
        "learning_rate": tune.loguniform(5.0e-5, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-7, 1.0e-3),
        "scheduler_step": tune.choice(scheduler_steps),
        "scheduler_gamma": tune.choice([0.90, 0.95, 0.98, 1.0]),
        "attention_layout": tune.choice(layouts),
        "n_layers_geo": tune.choice([1, 2, 3, 4]),
        "dropout_rate": tune.choice([0.0, 0.05, 0.10, 0.15]),
        "activation_str": tune.choice(["gelu", "relu"]),
    }

    fixed = dict(base_config)
    for name in search_domains:
        fixed.pop(name, None)
    for name in ("hidden_dim", "n_heads", "head_dim"):
        fixed.pop(name, None)

    # Keep batch_size fixed. HyperNOs validation divides an already
    # batch-averaged lpLoss by sample count, which biases cross-batch-size
    # comparisons.
    # zero_mean is fixed because the current SPD forward path does not apply
    # its post_processing member.
    return {**search_domains, **fixed}


def build_default_trial(base_config: dict[str, Any]) -> dict[str, Any]:
    default_trial = dict(base_config)
    for name in ("hidden_dim", "n_heads", "head_dim"):
        default_trial.pop(name, None)
    default_trial["attention_layout"] = _default_attention_layout(base_config)
    return default_trial


def _dataset_location(filename: str | Path) -> tuple[str, str]:
    """Adapt an explicit path to NO_load_data_model's recursive file finder."""

    candidate = Path(filename).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
        return resolved.name, str(resolved.parent)
    repository_root = Path(__file__).resolve().parents[2]
    return str(filename), str(repository_root)


def ray_mp_afieti_transformer(
    filename: str | Path = "yeti_dataset.csv",
    mode_hyperparams: str = "best",
    *,
    num_samples: int = 40,
    max_epochs: int = 300,
    grace_period: int = 50,
    reduction_factor: int = 4,
    cpus_per_trial: float = 2.0,
    gpus_per_trial: float | None = None,
):
    """Run constrained architecture/training optimization for ``mp_afieti``."""

    for name, value in {
        "num_samples": num_samples,
        "max_epochs": max_epochs,
        "grace_period": grace_period,
        "reduction_factor": reduction_factor,
        "cpus_per_trial": cpus_per_trial,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if grace_period > max_epochs:
        raise ValueError("grace_period cannot exceed max_epochs")

    training_config, architecture_config = initialize_hyperparameters(
        "IgaNet_transformer", "mp_afieti", mode_hyperparams
    )
    base_config = {**training_config, **architecture_config}
    base_config["epochs"] = max_epochs
    config_space = build_search_space(base_config, max_epochs)
    default_trial = build_default_trial(base_config)
    if set(default_trial) != set(config_space):
        raise RuntimeError("Default trial and Ray search-space keys do not agree")

    loader_filename, search_path = _dataset_location(filename)

    # Preserve only fitted statistics in the builder closure. Capturing the
    # complete example would unnecessarily serialize all dataset tensors into
    # Ray's object store.
    normalization_dataset = NO_load_data_model(
        which_example="mp_afieti",
        no_architecture={"retrain": base_config["retrain"]},
        batch_size=base_config["batch_size"],
        training_samples=base_config["training_samples"],
        filename=loader_filename,
        search_path=search_path,
    )
    input_normalizer = copy.deepcopy(normalization_dataset.input_normalizer)
    output_normalizer = copy.deepcopy(normalization_dataset.output_normalizer)
    del normalization_dataset

    def dataset_builder(config: dict[str, Any]):
        return NO_load_data_model(
            which_example="mp_afieti",
            no_architecture={"retrain": config["retrain"]},
            batch_size=config["batch_size"],
            training_samples=config["training_samples"],
            filename=loader_filename,
            search_path=search_path,
        )

    def model_builder(config: dict[str, Any]):
        hidden_dim, n_heads = decode_attention_layout(config["attention_layout"])
        trial_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return GeometryConditionedLinearOperator_mp_afieti(
            n_dofs=config["n_dofs"],
            n_control_points=config["n_control_points"],
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            # Keep this at one. The current SPD implementation averages these
            # projections, so larger values add redundant linear parameters.
            n_heads_A=config["n_heads_A"],
            n_layers_geo=config["n_layers_geo"],
            dropout_rate=config["dropout_rate"],
            activation_str=config["activation_str"],
            zero_mean=config["zero_mean"],
            example_input_normalizer=(
                copy.deepcopy(input_normalizer)
                if config["internal_normalization"]
                else None
            ),
            example_output_normalizer=(
                copy.deepcopy(output_normalizer)
                if config["internal_normalization"]
                else None
            ),
            device=trial_device,
        )

    if gpus_per_trial is None:
        gpus_per_trial = 1.0 if torch.cuda.is_available() else 0.0
    if gpus_per_trial < 0:
        raise ValueError("gpus_per_trial cannot be negative")

    best_result = tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        lpLoss(base_config["p"], True),
        default_hyper_params=[default_trial],
        num_samples=num_samples,
        max_epochs=max_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        runs_per_cpu=cpus_per_trial,
        runs_per_gpu=gpus_per_trial,
        checkpoint_freq=max_epochs + 1,
    )

    best_config = dict(best_result.config)
    hidden_dim, n_heads = decode_attention_layout(
        best_config.pop("attention_layout")
    )
    best_config.update(
        {
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "head_dim": hidden_dim // n_heads,
        }
    )
    print("Best expanded hyperparameters:", best_config)
    return best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("yeti_dataset.csv"))
    parser.add_argument("--mode", choices=("best", "default"), default="best")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=50)
    parser.add_argument("--reduction-factor", type=int, default=4)
    parser.add_argument("--cpus-per-trial", type=float, default=2.0)
    parser.add_argument("--gpus-per-trial", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    ray_mp_afieti_transformer(
        filename=arguments.data,
        mode_hyperparams=arguments.mode,
        num_samples=arguments.num_samples,
        max_epochs=arguments.max_epochs,
        grace_period=arguments.grace_period,
        reduction_factor=arguments.reduction_factor,
        cpus_per_trial=arguments.cpus_per_trial,
        gpus_per_trial=arguments.gpus_per_trial,
    )
