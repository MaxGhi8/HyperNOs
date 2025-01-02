import torch
from datasets import Airfoil, Darcy, concat_datasets
from FNO.FNO_arc import FNO_2D
from FNO.FNO_utilities import FNO_initialize_hyperparameters
from loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_space = {
        "FourierF": tune.choice([0]),
        "RNN": tune.choice([False]),
        "batch_size": tune.choice([32]),
        "fno_arc": tune.choice(["Classic", "Zongyi", "Residual"]),
        "in_dim": tune.choice([1]),
        "out_dim": tune.choice([1]),
        "fft_norm": tune.choice([None]),
        "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "include_grid": tune.choice([1]),
        "modes": tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]),
        "n_layers": tune.randint(1, 6),
        "padding": tune.randint(0, 16),
        "problem_dim": tune.choice([2]),
        "retrain": tune.choice([4]),
        "weights_norm": tune.choice(["Kaiming"]),
        "width": tune.choice([4, 8, 16, 32, 64, 128, 256]),
    }

    hyperparams_train, hyperparams_arc = FNO_initialize_hyperparameters(
        "poisson", mode="default"
    )

    default_hyper_params = [
        # {
        #     "learning_rate": hyperparams_train["learning_rate"],
        #     "weight_decay": hyperparams_train["weight_decay"],
        #     "scheduler_gamma": hyperparams_train["scheduler_gamma"],
        #     "width": hyperparams_arc["width"],
        #     "n_layers": hyperparams_arc["n_layers"],
        #     "modes": hyperparams_arc["modes"],
        #     "fun_act": hyperparams_arc["fun_act"],
        #     "fno_arc": hyperparams_arc["fno_arc"],
        #     "padding": hyperparams_arc["padding"],
        # }
    ]

    model_builder = lambda config: FNO_2D(
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        config["modes"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        config["retrain"],
    )

    dataset_builder = lambda config: concat_datasets(
        Airfoil(
            {
                "FourierF": config["FourierF"],
                "retrain": config["retrain"],
            },
            config["batch_size"],
            search_path="/",
        ),
        Darcy(
            {
                "FourierF": config["FourierF"],
                "retrain": config["retrain"],
            },
            config["batch_size"],
            search_path="/",
        ),
    )

    loss_fn = LprelLoss(2, False)

    tune_hyperparameters(
        config_space,
        model_builder,
        dataset_builder,
        loss_fn,
        default_hyper_params,
        runs_per_cpu=8,
        runs_per_gpu=0,
    )


if __name__ == "__main__":
    main()
