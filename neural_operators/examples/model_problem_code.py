# This is the model code for the hyperparameter tuning present in the article
# ``HyperNOs: Automated and Parallel Library for Neural Operators Research''
# IMPORTANT: before running this code please make sure to have installed all the necessary dependencies
# (see requirements.txt) and to be in this folder: HyperNOs/neural_operators/examples
# Also make sure to start ray with the command `ray start --head`.
import sys

import torch

sys.path.append("..")
from datasets import NO_load_data_model
from FNO import FNO
from loss_fun import LprelLoss
from ray import tune
from tune import tune_hyperparameters
from utilities import initialize_hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams_train, hyperparams_arc = initialize_hyperparameters(
    "FNO", "darcy", mode="default"
)

config_space = {
    "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
    "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
    "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
    "width": tune.choice([4, 8, 16, 32, 64, 96, 128]),
    "n_layers": tune.randint(1, 6),
    "modes": tune.choice([2, 4, 8, 12, 16, 20, 24]),
    "fun_act": tune.choice(["tanh", "relu", "gelu"]),
    "padding": tune.randint(0, 16),
}

fixed_params = {**hyperparams_train, **hyperparams_arc}
parameters_to_tune = config_space.keys()
for param in parameters_to_tune:
    fixed_params.pop(param, None)
config_space.update(fixed_params)

model_builder = lambda config: FNO(
    config["problem_dim"],
    config["in_dim"],
    config["width"],
    config["out_dim"],
    config["n_layers"],
    config["modes"],
    config["fun_act"],
    config["weights_norm"],
    config["fno_arc"],
    config["RNN"],
    config["fft_norm"],
    config["padding"],
    device,
    None,
    config["retrain"],
)

dataset_builder = lambda config: NO_load_data_model(
    which_example="darcy",
    no_architecture={"FourierF": config["FourierF"], "retrain": config["retrain"]},
    batch_size=config["batch_size"],
    training_samples=config["training_samples"],
)

loss_fn = LprelLoss(1)

tune_hyperparameters(
    config_space,
    model_builder,
    dataset_builder,
    loss_fn,
    num_samples=100,
    runs_per_cpu=8.0,
    runs_per_gpu=0.5,
)
