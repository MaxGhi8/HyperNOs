import torch
from datasets import Darcy
from neural_operators.loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters

from CNO.CNO import CNO


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_space = {
        "FourierF": tune.choice([0]),
        "n_layers": tune.randint(1, 5),
        "n_res": tune.randint(1, 8),
        "n_res_neck": tune.randint(1, 6),
        "batch_size": tune.choice([32]),
        "beta": tune.choice([1]),
        "bn": tune.choice([True]),
        "channel_multiplier": tune.choice([8, 16, 24, 32, 40, 48, 56]),
        "epochs": tune.choice([1000]),
        "in_dim": tune.choice([1]),
        "in_size": tune.choice([64]),
        "kernel_size": tune.choice([3, 5, 7]),
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "out_dim": tune.choice([1]),
        "problem_dim": tune.choice([2]),
        "retrain": tune.choice([4]),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "scheduler_step": tune.choice([1]),
        "test_samples": tune.choice([128]),
        "training_samples": tune.choice([256]),
        "val_samples": tune.choice([128]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
    }

    model_builder = lambda config: CNO(
        config["in_dim"],
        config["out_dim"],
        config["in_size"],
        config["n_layers"],
        config["n_res"],
        config["n_res_neck"],
        config["channel_multiplier"],
        config["kernel_size"],
        config["bn"],
        "cpu",
    )

    dataset_builder = lambda config: Darcy(
        {
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        device,
        config["batch_size"],
        search_path="/",
    )

    loss_fn = LprelLoss(2, False)

    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        runs_per_cpu=8,
        runs_per_gpu=1,
    )


if __name__ == "__main__":
    main()
