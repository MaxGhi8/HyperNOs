import torch
from FNO.FNO_arc import FNO_2D
from FNO.FNO_benchmarks import SinFrequency
from Loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters


def main():
    config_space = {
        "FourierF": tune.choice([0]),
        "RNN": tune.choice([False]),
        "batch_size": tune.choice([32]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "d_a": tune.choice([1]),
        "d_u": tune.choice([1]),
        "epochs": tune.choice([1000]),
        "fft_norm": tune.choice([None]),
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "include_grid": tune.choice([1]),
        "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
        "modes": tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]),
        "n_layers": tune.randint(1, 6),
        "padding": tune.randint(0, 16),
        "problem_dim": tune.choice([2]),
        "retrain": tune.choice([4]),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
        "training_samples": tune.choice([1024]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
        "weights_norm": tune.choice(["Kaiming"]),
        "width": tune.choice([4, 8, 16, 32, 64, 128, 256]),
    }
    model_builder = lambda config: FNO_2D(
        config["d_a"],
        config["width"],
        config["d_u"],
        config["n_layers"],
        config["modes"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        torch.device("cpu"),
        config["retrain"],
    )
    dataset_builder = lambda config: SinFrequency(
        {
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        torch.device("cpu"),
        config["batch_size"],
        config["training_samples"],
        search_path="/",
    )
    loss_fn = LprelLoss(2, False)
    tune_hyperparameters(config_space, model_builder, dataset_builder, loss_fn)


if __name__ == "__main__":
    main()
