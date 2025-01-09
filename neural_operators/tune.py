import os
import tempfile
import torch
from ray import init, train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from train import train_epoch

from utilities import count_params


def tune_hyperparameters(
    config_space,
    model_builder,
    dataset_builder,
    loss_fn,
    default_hyper_params=[],
    num_samples: int = 200,
    grace_period: int = 250,
    reduction_factor: int = 4,
    max_epochs: int = 1000,
    checkpoint_freq: int = 500,
    runs_per_cpu: float = 0.0,
    runs_per_gpu: float = 1.0,
):
    # Check if the required keys are in the config_space
    required_keys = [
        "learning_rate",
        "weight_decay",
        "scheduler_step",
        "scheduler_gamma",
    ]
    missing_keys = [key for key in required_keys if key not in config_space]
    if missing_keys:
        print(
            f"Attention: the key {missing_keys} are missing in the config_space, so I'll using the default value."
        )

    # Define the training function
    def train_fn(config):
        dataset = dataset_builder(config)
        model = model_builder(config)
        train_model(
            model,
            dataset.train_loader,
            dataset.val_loader,
            loss_fn,
            max_epochs,
            model.device,
            config["learning_rate"],
            config["weight_decay"],
            config["scheduler_step"],
            config["scheduler_gamma"],
            checkpoint_freq=checkpoint_freq,
        )

    # Initialize Ray
    init(
        address="auto",
        runtime_env={
            "env_vars": {"PYTHONPATH": os.path.abspath("..")},
        },
    )

    # Define the scheduler
    scheduler = ASHAScheduler(
        metric="relative_loss",
        mode="min",
        time_attr="training_iteration",
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        stop_last_trials=True,
    )

    # Define the search algorithm
    search_alg = HyperOptSearch(
        metric="relative_loss",
        mode="min",
        points_to_evaluate=default_hyper_params,
        n_initial_points=20,
    )

    # Define the tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_fn),
            resources={"cpu": runs_per_cpu, "gpu": runs_per_gpu},
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

    # # Get the best trial
    # best_result = results.get_best_result("relative_loss", "min")
    # print("Best trial config: {}".format(best_result.config))
    # # print("Best trial test_relative_loss: {}".format(best_result.metrics["relative_loss"]))
    # print("Best trial directory: {}".format(best_result.path))

    return results.get_best_result("relative_loss", "min")


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    max_epochs,
    device,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    scheduler_step: int = 1,
    scheduler_gamma: float = 0.99,
    checkpoint_freq: int = 500,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    total_params, _ = count_params(model)
    print(f"Total Parameters: {total_params:,}")

    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])

    for ep in range(start_epoch, max_epochs):
        # Train the model for one epoch
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)

        # Validate the model for one epoch
        acc = validate_epoch(model, val_loader, loss_fn, device)

        if ep % checkpoint_freq == 0 or ep == max_epochs - 1:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report({"relative_loss": acc}, checkpoint=checkpoint)
        else:
            train.report({"relative_loss": acc})


def validate_epoch(
    model,
    val_loader,
    loss_fn,
    device: torch.device,
):
    with torch.no_grad():
        model.eval()
        loss = 0.0
        examples_count = 0

        for input_batch, output_batch in val_loader:
            input_batch = input_batch.to(device)
            examples_count += input_batch.size(0)
            output_batch = output_batch.to(device)
            output_pred_batch = model.forward(input_batch)
            loss += loss_fn(output_pred_batch, output_batch).item()

    return loss / examples_count
