import os
import tempfile

import torch
from ray import init, train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from train import train_epoch


def tune_hyperparameters(
    config_space,
    model_builder,
    dataset_builder,
    loss_fn,
    learning_rate=tune.quniform(1e-4, 1e-2, 1e-5),
    weight_decay=tune.quniform(1e-6, 1e-3, 1e-6),
    scheduler_step=tune.choice([10]),
    scheduler_gamma=tune.quniform(0.75, 0.99, 0.01),
    num_samples=200,
    grace_period=250,
    reduction_factor=2,
    max_epochs=1000,
    runs_per_cpu=0,
    runs_per_gpu=1,
):
    config_space["learning_rate"] = learning_rate
    config_space["weight_decay"] = weight_decay
    config_space["scheduler_step"] = scheduler_step
    config_space["scheduler_gamma"] = scheduler_gamma

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
        )

    init(address="auto")

    scheduler = ASHAScheduler(
        metric="relative_loss",
        mode="min",
        time_attr="training_iteration",
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        stop_last_trials=True,
    )

    search_alg = HyperOptSearch(
        metric="relative_loss",
        mode="min",
        points_to_evaluate=default_hyper_params,
    )

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
    return results.get_best_result("relative_loss", "min")


def train_model(config, model, train_loader, val_loader, loss_fn, max_epochs, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"]
    )

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

        if ep % 500 == 0 or ep == max_epochs - 1:
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
