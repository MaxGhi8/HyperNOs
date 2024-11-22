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
# from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from train_fun import train_fun, test_fun
from Loss_fun import LprelLoss, H1relLoss_1D, H1relLoss
from FNO.FNO_arc import FNO_1D, FNO_2D
from utilities import count_params, plot_data, load_data_model
from FNO.FNO_utilities import FNO_initialize_hyperparameters

#########################################
# raytune parameters
#########################################
checkpoint_frequency = 500 # frequency to save the model
grace_period = 500 # minimum number of epochs to run before early stopping
reduce_factor = 2 # the factor to reduce the number of trials
mode_str = "default" # test base hyperparameters, can be "default" or "best"

#########################################
# Choose the example to run from CLI
#########################################
if len(sys.argv) == 1:
    raise ValueError("The user must choose the example to run")
elif len(sys.argv) == 2:
    which_example = sys.argv[1] # the example can be chosen by the user as second argument
    exp_norm = "L1"
    in_dist = True
elif len(sys.argv) == 3:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2] # the norm can be chosen by the user as third argument (see Norm_dict below)
    in_dist = True
elif len(sys.argv) == 4:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2]
    in_dist = sys.argv[3] # the in_dist can be chosen by the user as fourth argument
else:
    raise ValueError("The user must choose the example to run")

Norm_dict = {"L1":0, "L2":1, "H1":2, "L1_smooth":3, "MSE":4}

#########################################
# Hyperparameters
#########################################
training_properties, fno_architecture = FNO_initialize_hyperparameters(which_example, mode=mode_str)

# loss function parameter
training_properties["exp"] = Norm_dict[exp_norm]

# training hyperparameters 
epochs           = training_properties["epochs"]
p                = training_properties["exp"]
beta             = training_properties["beta"]
training_samples = training_properties["training_samples"]
val_samples      = training_properties["val_samples"]
test_samples     = training_properties["test_samples"]

# fno fixed hyperparameters
d_a          = fno_architecture["d_a"]
d_u          = fno_architecture["d_u"]
weights_norm = fno_architecture["weights_norm"]
RNN          = fno_architecture["RNN"]
FFTnorm      = fno_architecture["fft_norm"]
retrain_fno  = fno_architecture["retrain"]
FourierF     = fno_architecture["FourierF"]

# Loss function
match p:
    case 0:
        loss = LprelLoss(1, False) # L^1 relative norm
    case 1:
        loss = LprelLoss(2, False) # L^2 relative norm
    case 2:
        if fno_architecture["problem_dim"] == 1:
            loss = H1relLoss_1D(beta, False, 1.0)
        elif fno_architecture["problem_dim"] == 2:
            loss = H1relLoss(beta, False, 1.0) # H^1 relative norm
    case 3:
        loss = torch.nn.SmoothL1Loss() # L^1 smooth loss (Mishra)
    case 4:
        loss = torch.nn.MSELoss() # L^2 smooth loss (Mishra)
    case _:
        raise ValueError("This value of p is not allowed")

#########################################
# load the model and data
#########################################
# count and print the total number of parameters
# par_tot = count_params(model)
# print("Total number of parameters is: ", par_tot)

def train_hyperparameter(config):
    learning_rate   = config["learning_rate"]
    weight_decay    = config["weight_decay"]
    scheduler_step  = config["scheduler_step"]
    scheduler_gamma = config["scheduler_gamma"]
    batch_size      = config["batch_size"]
    d_v             = config["width"]
    L               = config["n_layers"]
    modes           = config["modes"]
    fun_act         = config["fun_act"]
    arc             = config["arc"]
    padding         = config["padding"]

    # Device I can handle different devices for different ray trials
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Definition of the model
    fno_architecture["problem_dim"] = 2 # default value
    example = load_data_model(which_example, fno_architecture, device, batch_size, training_samples, in_dist)

    # model = example.model
    if fno_architecture["problem_dim"] == 1:
        model = FNO_1D(d_a, d_v, d_u, L, modes, fun_act, weights_norm, arc, 
                        RNN, FFTnorm, padding, device, retrain_fno)    
    elif fno_architecture["problem_dim"] == 2:
        model = FNO_2D(d_a, d_v, d_u, L, modes, modes, fun_act, weights_norm, arc, 
                        RNN, FFTnorm, padding, device, retrain_fno)    

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
    test_loader = example.test_loader # for final testing

    # Definition of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                            lr = learning_rate, weight_decay = weight_decay)

    # Definition of the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                            step_size = scheduler_step, gamma = scheduler_gamma)
    
    ## Training process
    for ep in range(start_epoch, epochs):
        # Train the model for one epoch
        train_fun(model, train_loader, optimizer, scheduler, loss, p, device, which_example)
        
        # Test the model for one epoch
        acc = test_fun(model, val_loader, train_loader, loss, exp_norm, val_samples, 
                    training_samples, device, which_example, statistic=False)
        
        if ep % checkpoint_frequency == 0 or ep == epochs - 1:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report({"relative_loss": acc}, checkpoint=checkpoint) # report the accuracy to Tune
        else:
            train.report({"relative_loss": acc})
        
    print("Training completed")

def main(num_samples, max_num_epochs=epochs):
    # Default hyperparameters from Mishra article to start the optimization search
    default_mishra_params = [{
        "learning_rate"  : training_properties["learning_rate"],
        "weight_decay"   : training_properties["weight_decay"],
        "scheduler_step" : training_properties["scheduler_step"],
        "scheduler_gamma": training_properties["scheduler_gamma"],
        "batch_size"     : training_properties["batch_size"],
        "width"          : fno_architecture["width"],
        "n_layers"       : fno_architecture["n_layers"],
        "modes"          : fno_architecture["modes"],
        "fun_act"        : fno_architecture["fun_act"],
        "arc"            : fno_architecture["arc"],
        "padding"        : fno_architecture["padding"]
    }]

    config = {
        "learning_rate" : tune.quniform(1e-4, 1e-2, 1e-5),
        "weight_decay"  : tune.quniform(1e-6, 1e-3, 1e-6),
        "scheduler_step": tune.randint(1, 100),
        "scheduler_gamma": tune.quniform(0.75, 0.99, 0.001),
        "batch_size"    : tune.choice([20, 32, 48, 64]),
        "width"         : tune.choice([4, 8, 16, 32, 64, 128, 256]),
        "n_layers"      : tune.randint(1, 6),
        "modes"         : tune.choice([2, 4, 8, 12, 16, 20, 24, 28, 32]), # modes1 = modes2
        "fun_act"       : tune.choice(["tanh", "relu", "gelu", "leaky_relu"]),
        "arc"           : tune.choice(["Classic", "Zongyi", "Residual"]),
        "padding"       : tune.randint(0, 16),
    }

    # Automatically detect the available resources and use them
    init(address = "auto") # run `ray start --head` in the terminal before running this script and at the end `ray stop`

    scheduler = ASHAScheduler(
        metric = "relative_loss",
        mode = "min",
        time_attr = "training_iteration",
        max_t = max_num_epochs,
        grace_period = grace_period,
        reduction_factor = reduce_factor,
        stop_last_trials = True, 
    )

    optim_algo = HyperOptSearch(
        metric = "relative_loss",
        mode = "min",
        points_to_evaluate = default_mishra_params,
        n_initial_points = 20, # number of random points to evaluate before starting the hyperparameter search (default = 20)
        random_state_seed = None
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_hyperparameter),
            resources={"cpu": 0, "gpu": 0.5} # allocate resources for each trials
        ),
        param_space = config, # config is the hyperparameter space
        tune_config=tune.TuneConfig(
            scheduler = scheduler,
            search_alg = optim_algo,
            num_samples = num_samples,
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
    num_samples = 200 # number of trials
    main(num_samples)
