"""
This is the main file for hyperparameter search of the Neural Operator with the FNO architecture for different examples.

"which_example" can be one of the following options:
    poisson             : Poisson equation 
    wave_0_5            : Wave equation 
    cont_tran           : Smooth Transport 
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation # training_sample = 512
    shear_layer         : Navier-Stokes equations # training_sample = 512
    airfoil             : Compressible Euler equations 
    darcy               : Darcy equation

    burgers_zongyi      : Burgers equation
    darcy_zongyi        : Darcy equation
    navier_stokes_zongyi: Navier-Stokes equations

    fhn                 : FitzHugh-Nagumo equations in [0, 100]
    fhn_long            : FitzHugh-Nagumo equations in [0, 200]
    hh                  : Hodgkin-Huxley equation

    crosstruss          : Cross-shaped truss structure

"exp_norm" can be one of the following options:
    L1 : L^1 relative norm
    L2 : L^2 relative norm
    H1 : H^1 relative norm
    L1_smooth : L^1 smooth loss (Mishra)
    MSE : L^2 smooth loss (Mishra)
"""

import torch
import os
import sys

import tempfile
from ray import train, tune, init
from ray.train import Checkpoint

from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from train_fun import train_fun, test_fun
from Loss_fun import LprelLoss, H1relLoss_1D, H1relLoss
from utilities import count_params, plot_data

# FNO imports
from FNO.FNO_arc import FNO_1D, FNO_2D
from FNO.FNO_utilities import FNO_initialize_hyperparameters, FNO_load_data_model

# CNO imports
from CNO.CNO_1d import CNO1d
from CNO.CNO_2d import CNO2d
from CNO.CNO_utilities import CNO_load_data_model, CNO_initialize_hyperparameters

#########################################
# ray-tune parameters
#########################################
checkpoint_frequency = 500  # frequency to save the model
grace_period = 250  # minimum number of epochs to run before early stopping
reduce_factor = 2  # the factor to reduce the number of trials

#########################################
# Choose the example to run from CLI
#########################################
if len(sys.argv) < 5:
    raise ValueError("The user must choose the example and the model to run")
elif len(sys.argv) == 5:
    which_example = sys.argv[1].lower().strip()  # example to run
    arc = sys.argv[2].upper().strip()  # architecture to use
    exp_norm = sys.argv[3].upper().strip()  # loss function for training
    mode_str = sys.argv[4].lower().strip()  # mode of the training process
    in_dist = True  # default value
elif len(sys.argv) == 6:
    which_example = sys.argv[1].lower().strip()  # example to run
    arc = sys.argv[2].upper().strip()  # architecture to use
    exp_norm = sys.argv[3].upper().strip()  # loss function for training
    mode_str = sys.argv[4].lower().strip()  # mode of the training process
    in_dist = sys.argv[5]  # True or False (IN/OUT of distribution)
else:
    raise ValueError("The user must choose the example to run")

#########################################
# Hyperparameters
#########################################
match arc:
    case "FNO":
        hyperparams_train, hyperparams_arc = FNO_initialize_hyperparameters(
            which_example, mode=mode_str
        )
    case "CNO":
        hyperparams_train, hyperparams_arc = CNO_initialize_hyperparameters(
            which_example, mode=mode_str
        )
    case _:
        raise ValueError("This architecture is not allowed")

# loss function parameter
Norm_dict = {"L1": 0, "L2": 1, "H1": 2, "L1_SMOOTH": 3, "MSE": 4}
hyperparams_train["exp"] = Norm_dict[exp_norm]

#########################################
# Load the fixed hyperparameters
#########################################
epochs = hyperparams_train["epochs"]
p = hyperparams_train["exp"]
beta = hyperparams_train["beta"]
training_samples = hyperparams_train["training_samples"]
val_samples = hyperparams_train["val_samples"]
test_samples = hyperparams_train["test_samples"]
batch_size = hyperparams_train["batch_size"]
scheduler_step = hyperparams_train["scheduler_step"]

match arc:
    case "FNO":
        # fno fixed hyperparameters
        d_a = hyperparams_arc["d_a"]
        d_u = hyperparams_arc["d_u"]
        weights_norm = hyperparams_arc["weights_norm"]
        RNN = hyperparams_arc["RNN"]
        FFTnorm = hyperparams_arc["fft_norm"]
        retrain_fno = hyperparams_arc["retrain"]
        FourierF = hyperparams_arc["FourierF"]
        problem_dim = hyperparams_arc["problem_dim"]

    case "CNO":
        # cno architecture hyperparameters
        in_dim = hyperparams_arc["in_dim"]
        out_dim = hyperparams_arc["out_dim"]
        size = hyperparams_arc["in_size"]
        bn = hyperparams_arc["bn"]
        retrain = hyperparams_arc["retrain"]
        problem_dim = hyperparams_arc["problem_dim"]

    case _:
        raise ValueError("This architecture is not allowed")

# Loss function
match p:
    case 0:
        loss = LprelLoss(1, False)  # L^1 relative norm
    case 1:
        loss = LprelLoss(2, False)  # L^2 relative norm
    case 2:
        if problem_dim == 1:
            loss = H1relLoss_1D(beta, False, 1.0)
        elif problem_dim == 2:
            loss = H1relLoss(beta, False, 1.0)  # H^1 relative norm
    case 3:
        loss = torch.nn.SmoothL1Loss()  # L^1 smooth loss (Mishra)
    case 4:
        loss = torch.nn.MSELoss()  # L^2 smooth loss (Mishra)
    case _:
        raise ValueError("This value of p is not allowed")

#########################################
# load the model and data
#########################################
# count and print the total number of parameters
# par_tot = count_params(model)
# print("Total number of parameters is: ", par_tot)


def train_hyperparameter(config):
    # Hyperparameters to optimize for training process
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    scheduler_gamma = config["scheduler_gamma"]

    match arc:
        # FNO hyperparameters
        case "FNO":
            d_v = config["width"]
            L = config["n_layers"]
            modes = config["modes"]
            fun_act = config["fun_act"]
            fno_arc = config["arc"]
            padding = config["padding"]

        # CNO hyperparameters
        case "CNO":
            n_layers = config["N_layers"]
            chan_mul = config["channel_multiplier"]
            n_res_neck = config["N_res_neck"]
            n_res = config["N_res"]
            kernel_size = config["kernel_size"]

    # Device I can handle different devices for different ray trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Definition of the model
    match arc:
        case "FNO":
            example = FNO_load_data_model(
                which_example,
                hyperparams_arc,
                device,
                batch_size,
                training_samples,
                in_dist,
            )
        case "CNO":
            example = CNO_load_data_model(
                which_example,
                hyperparams_arc,
                device,
                batch_size,
                training_samples,
                in_dist,
            )

    # model = example.model
    match arc:
        case "FNO":
            if problem_dim == 1:
                model = FNO_1D(
                    d_a,
                    d_v,
                    d_u,
                    L,
                    modes,
                    fun_act,
                    weights_norm,
                    fno_arc,
                    RNN,
                    FFTnorm,
                    padding,
                    device,
                    retrain_fno,
                )
            elif problem_dim == 2:
                model = FNO_2D(
                    d_a,
                    d_v,
                    d_u,
                    L,
                    modes,
                    modes,
                    fun_act,
                    weights_norm,
                    fno_arc,
                    RNN,
                    FFTnorm,
                    padding,
                    device,
                    retrain_fno,
                )
        case "CNO":
            if problem_dim == 1:
                model = CNO1d(
                    in_dim,
                    out_dim,
                    size,
                    n_layers,
                    n_res,
                    n_res_neck,
                    chan_mul,
                    kernel_size,
                    bn,
                    device,
                )
            elif problem_dim == 2:
                model = CNO2d(
                    in_dim,
                    out_dim,
                    size,
                    n_layers,
                    n_res,
                    n_res_neck,
                    chan_mul,
                    kernel_size,
                    bn,
                    device,
                )

    # Definition of the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Definition of the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    # Load existing checkpoint through `get_checkpoint()` API.
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])

    # Load data
    train_loader = example.train_loader
    val_loader = example.val_loader

    ## Training process
    for ep in range(start_epoch, epochs):
        # Train the model for one epoch
        train_fun(
            model, train_loader, optimizer, scheduler, loss, p, device, which_example
        )

        # Test the model for one epoch
        acc = test_fun(
            model,
            val_loader,
            train_loader,
            loss,
            exp_norm,
            val_samples,
            training_samples,
            device,
            which_example,
            statistic=False,
        )

        if ep % checkpoint_frequency == 0 or ep == epochs - 1:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"relative_loss": acc}, checkpoint=checkpoint
                )  # report the accuracy to Tune
        else:
            train.report({"relative_loss": acc})

    print("Training completed")


def custom_trial_dirname_creator(trial):
    # Create a shorter name for each trial of ray tune
    return f"trial_{trial.trial_id}"


def main(num_samples, max_num_epochs=epochs):
    # Default hyperparameters from Mishra article to start the optimization search
    match arc:
        case "FNO":
            default_hyper_params = [
                {
                    "learning_rate": hyperparams_train["learning_rate"],
                    "weight_decay": hyperparams_train["weight_decay"],
                    "scheduler_gamma": hyperparams_train["scheduler_gamma"],
                    "width": hyperparams_arc["width"],
                    "n_layers": hyperparams_arc["n_layers"],
                    "modes": hyperparams_arc["modes"],
                    "fun_act": hyperparams_arc["fun_act"],
                    "arc": hyperparams_arc["arc"],
                    "padding": hyperparams_arc["padding"],
                }
            ]
        case "CNO":
            default_hyper_params = [
                {
                    "learning_rate": hyperparams_train["learning_rate"],
                    "weight_decay": hyperparams_train["weight_decay"],
                    "scheduler_gamma": hyperparams_train["scheduler_gamma"],
                    "N_layers": hyperparams_arc["N_layers"],
                    "channel_multiplier": hyperparams_arc["channel_multiplier"],
                    "N_res_neck": hyperparams_arc["N_res_neck"],
                    "N_res": hyperparams_arc["N_res"],
                    "kernel_size": hyperparams_arc["kernel_size"],
                }
            ]

    # Hyperparameter search space
    match arc:
        case "FNO":
            config = {
                "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
                "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
                "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
                "width": tune.choice([4, 8, 16, 32, 64, 128, 256]),
                "n_layers": tune.randint(1, 6),
                "modes": tune.choice(
                    [2, 4, 8, 12, 16, 20, 24, 28, 32]
                ),  # modes1 = modes2
                "fun_act": tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
                "arc": tune.choice(["Classic", "Zongyi", "Residual"]),
                "padding": tune.randint(0, 16),
            }
        case "CNO":
            config = {
                "learning_rate": tune.quniform(1e-4, 1e-2, 1e-5),
                "weight_decay": tune.quniform(1e-6, 1e-3, 1e-6),
                "scheduler_gamma": tune.quniform(0.75, 0.99, 0.01),
                "N_layers": tune.randint(1, 5),
                "channel_multiplier": tune.choice([8, 16, 24, 32, 40, 48, 56]),
                "N_res_neck": tune.randint(1, 6),
                "N_res": tune.randint(1, 8),
            }
            # kernel size is different for different problem dimensions
            if problem_dim == 1:
                config["kernel_size"] = tune.choice([11, 21, 31, 41, 51])
            if problem_dim == 2:
                config["kernel_size"] = tune.choice([3, 5, 7])

    # Automatically detect the available resources and use them
    init(
        address="auto"
    )  # run `ray start --head` in the terminal before running this script and at the end `ray stop`

    scheduler = ASHAScheduler(
        metric="relative_loss",
        mode="min",
        time_attr="training_iteration",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduce_factor,
        stop_last_trials=True,
    )

    optim_algo = HyperOptSearch(
        metric="relative_loss",
        mode="min",
        points_to_evaluate=default_hyper_params,
        n_initial_points=20,  # number of random points to evaluate before starting the hyperparameter search (default = 20)
        random_state_seed=None,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_hyperparameter),
            resources={"cpu": 0, "gpu": 0.5},  # allocate resources for each trials
        ),
        param_space=config,  # config is the hyperparameter space
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=optim_algo,
            num_samples=num_samples,
            trial_dirname_creator=custom_trial_dirname_creator,
        ),
    )
    # Run the hyperparameter search
    results = tuner.fit()

    # Get the best trial
    best_result = results.get_best_result("relative_loss", "min")
    print("Best trial config: {}".format(best_result.config))
    # print("Best trial test_relative_loss: {}".format(best_result.metrics["relative_loss"]))
    print("Best trial directory: {}".format(best_result.path))


if __name__ == "__main__":
    num_samples = 200  # number of trials
    main(num_samples)
