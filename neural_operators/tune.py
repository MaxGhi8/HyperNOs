import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from neural_operators.train import train_model


def tune_hyperparameters(
    config_space,
    model_builder,
    dataset,
    loss_fn,
    num_samples=200,
    grace_period=250,
    reduction_factor=2,
    device=torch.device("cpu"),
):
    def train_fn(config):
        model = model_builder(config)
        train_model(
            config,
            model,
            dataset.train_loader,
            dataset.val_loader,
            loss_fn,
            device,
        )

    scheduler = ASHAScheduler(
        metric="relative_loss",
        mode="min",
        max_t=config_space["epochs"],
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )

    search_alg = HyperOptSearch(metric="relative_loss", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_fn),
            resources={"cpu": 0, "gpu": 0.5},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        ),
        param_space=config_space,
    )

    results = tuner.fit()
    return results.get_best_result("relative_loss", "min")
