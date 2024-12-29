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
        "d_a": tune.choice([1]),
        "d_u": tune.choice([1]),
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
        "scheduler_step": tune.choice([10]),
        "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
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
        #     "arc": hyperparams_arc["arc"],
        #     "padding": hyperparams_arc["padding"],
        # }
    ]

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
