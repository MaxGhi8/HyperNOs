"""
This is the main file for makes test and plot the Neural Operator model.

"which_example" can be one of the following options:
    poisson             : Poisson equation 
    wave_0_5            : Wave equation 
    cont_tran           : Smooth Transport 
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation # training_sample = 512
    shear_layer         : Navier-Stokes equations # training_sample = 512
    airfoil             : Compressible Euler equations 
    darcy               : Darcy equation

    burgers_zongyi       : Burgers equation
    darcy_zongyi         : Darcy equation
    navier_stokes_zongyi : Navier-Stokes equations

    fhn                 : FitzHugh-Nagumo equations in [0, 100]
    fhn_long            : FitzHugh-Nagumo equations in [0, 200]
    hh                   : Hodgkin-Huxley equations

"exp_norm" can be one of the following options:
    L1 : L^1 relative norm
    L2 : L^2 relative norm
    H1 : H^1 relative norm
    L1_smooth : L^1 smooth loss (Mishra)
    MSE : L^2 smooth loss (Mishra)
"""

import argparse
import json
import os
import time
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from cli.utilities_plot import test_plot_samples
from jaxtyping import Float, jaxtyped
from loss_fun import (
    H1relLoss,
    H1relLoss_1D,
    H1relLoss_1D_multiout,
    H1relLoss_multiout,
    LprelLoss,
    LprelLoss_multiout,
)
from scipy.io import savemat
from torch import Tensor
from train_fun import test_fun, test_fun_tensors
from utilities import count_params

from datasets import NO_load_data_model

#########################################
# default values
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# torch.set_default_dtype(torch.float32) # default tensor dtype


#########################################
# Choose the example to run
#########################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a specific example with the desired model and training configuration."
    )

    parser.add_argument(
        "example",
        type=str,
        choices=[
            "poisson",
            "wave_0_5",
            "cont_tran",
            "disc_tran",
            "allen",
            "shear_layer",
            "airfoil",
            "darcy",
            "burgers_zongyi",
            "darcy_zongyi",
            "fhn",
            "fhn_long",
            "hh",
            "crosstruss",
        ],
        help="Select the example to run.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=["fno", "cno"],
        help="Select the architecture to use.",
    )
    parser.add_argument(
        "loss_function",
        type=str,
        choices=["l1", "l2", "h1", "l1_smooth"],
        help="Select the relative loss function to use during the training process.",
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["best", "default"],
        help="Select the hyper-params to use for define the architecture and the training, we have implemented the 'best' and 'default' options.",
    )
    parser.add_argument(
        "--in_dist",
        type=bool,
        default=True,
        help="For the datasets that are supported you can select if the test set is in-distribution or out-of-distribution.",
    )

    args = parser.parse_args()

    return {
        "example": args.example.lower(),
        "architecture": args.architecture.upper(),
        "loss_function": args.loss_function.upper(),
        "mode": args.mode.lower(),
        "in_dist": args.in_dist,
    }


config = parse_arguments()
which_example = config["example"]
arc = config["architecture"]
exp_norm = config["loss_function"]
mode_str = config["mode"]
in_dist = config["in_dist"]

Norm_dict = {"L1": 0, "L2": 1, "H1": 2, "L1_SMOOTH": 3, "MSE": 4}

# upload the model and the hyperparameters
model_folder = f"../{arc}/TrainedModels/"
description_test = "test_" + exp_norm
folder = (
    model_folder
    + which_example
    + "/exp_"
    + arc
    + "_"
    + description_test
    + "_"
    + mode_str
    + "_hyperparams"
)
name_model = (
    model_folder
    + which_example
    + "/model_"
    + arc
    + "_"
    + description_test
    + "_"
    + mode_str
    + "_hyperparams"
)

try:
    model = torch.load(name_model, weights_only=False)
except Exception:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed trhow the CLI."
    )

#########################################
# Hyperparameters
#########################################
# Load `hyper-params_train` from JSON
with open(folder + "/hyperparams_train.json", "r") as f:
    hyperparams_train = json.load(f)

# Load `hyper-params_arc` from JSON
with open(folder + "/hyperparams_arc.json", "r") as f:
    hyperparams_arc = json.load(f)

# Choose the Loss function
hyperparams_train["exp"] = Norm_dict[
    exp_norm
]  # 0 for L^1 relative norm, 1 for L^2 relative norm, 2 for H^1 relative norm

# Training hyperparameters
learning_rate = hyperparams_train["learning_rate"]
weight_decay = hyperparams_train["weight_decay"]
scheduler_step = hyperparams_train["scheduler_step"]
scheduler_gamma = hyperparams_train["scheduler_gamma"]
epochs = hyperparams_train["epochs"]
batch_size = hyperparams_train["batch_size"]
p = hyperparams_train["exp"]
beta = hyperparams_train["beta"]
training_samples = hyperparams_train["training_samples"]
test_samples = hyperparams_train["test_samples"]
val_samples = hyperparams_train["val_samples"]

match arc:
    case "FNO":
        # fno architecture hyperparameters
        in_dim = hyperparams_arc["in_dim"]
        d_v = hyperparams_arc["width"]
        d_u = hyperparams_arc["d_u"]
        L = hyperparams_arc["n_layers"]
        modes = hyperparams_arc["modes"]
        fun_act = hyperparams_arc["fun_act"]
        weights_norm = hyperparams_arc["weights_norm"]
        arc_fno = hyperparams_arc["arc"]
        RNN = hyperparams_arc["RNN"]
        FFTnorm = hyperparams_arc["fft_norm"]
        padding = hyperparams_arc["padding"]
        retrain = hyperparams_arc["retrain"]
        FourierF = hyperparams_arc["FourierF"]
        problem_dim = hyperparams_arc["problem_dim"]

    case "CNO":
        # cno architecture hyperparameters
        in_dim = hyperparams_arc["in_dim"]
        out_dim = hyperparams_arc["out_dim"]
        size = hyperparams_arc["in_size"]
        n_layers = hyperparams_arc["N_layers"]
        chan_mul = hyperparams_arc["channel_multiplier"]
        n_res_neck = hyperparams_arc["N_res_neck"]
        n_res = hyperparams_arc["N_res"]
        kernel_size = hyperparams_arc["kernel_size"]
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
# Data loader
#########################################
example = NO_load_data_model(
    which_example,
    hyperparams_arc,
    batch_size,
    training_samples,
    in_dist,
)

train_loader = example.train_loader
val_loader = example.val_loader
test_loader = example.test_loader  # for final testing
print(
    "Dimension of datasets are:",
    next(iter(train_loader))[0].shape,
    next(iter(val_loader))[0].shape,
    next(iter(test_loader))[0].shape,
)
print("")

# Count and print the total number of parameters
total_params, total_bytes = count_params(model)
total_mb = total_bytes / (1024**2)
print(f"Total Parameters: {total_params:,}")
print(f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)")

#########################################
# Compute mean error and print it
#########################################
(
    val_relative_l1,
    val_relative_l2,
    val_relative_semih1,
    val_relative_h1,
    train_loss,
) = test_fun(
    model,
    val_loader,
    train_loader,
    loss,
    exp_norm,
    val_samples,
    training_samples,
    device,
    which_example,
    statistic=True,
)

print("Train loss: ", train_loss)
print("Validation mean relative l1 norm: ", val_relative_l1)
print("Validation mean relative l2 norm: ", val_relative_l2)
print("Validation mean relative semi h1 norm: ", val_relative_semih1)
print("Validation mean relative h1 norm: ", val_relative_h1)
print("")

(
    test_relative_l1,
    test_relative_l2,
    test_relative_semih1,
    test_relative_h1,
    train_loss,
) = test_fun(
    model,
    test_loader,
    train_loader,
    loss,
    exp_norm,
    test_samples,
    training_samples,
    device,
    which_example,
    statistic=True,
)

print("Test mean relative l1 norm: ", test_relative_l1)
print("Test mean relative l2 norm: ", test_relative_l2)
print("Test mean relative semi h1 norm: ", test_relative_semih1)
print("Test mean relative h1 norm: ", test_relative_h1)
print("")


#########################################
# Compute median error and print it
#########################################
(
    input_tensor,
    output_tensor,
    prediction_tensor,
    test_rel_l1_tensor,
    test_rel_l2_tensor,
    test_rel_semih1_tensor,
    test_rel_h1_tensor,
) = test_fun_tensors(model, test_loader, loss, device, which_example)

# compute median error
test_median_rel_l1 = torch.median(test_rel_l1_tensor).item()
test_median_rel_l2 = torch.median(test_rel_l2_tensor).item()
test_median_rel_semih1 = torch.median(test_rel_semih1_tensor).item()
test_median_rel_h1 = torch.median(test_rel_h1_tensor).item()

print("Test median relative l1 norm: ", test_median_rel_l1)
print("Test median relative l2 norm: ", test_median_rel_l2)
print("Test median relative semi h1 norm: ", test_median_rel_semih1)
print("Test median relative h1 norm: ", test_median_rel_h1)
print("")

#########################################
# Compute mean error component per component
#########################################
if which_example in ["fhn", "fhn_long", "hh"]:
    test_rel_l1_componentwise = LprelLoss_multiout(1, True)(
        output_tensor, prediction_tensor
    )
    test_rel_l2_componentwise = LprelLoss_multiout(2, True)(
        output_tensor, prediction_tensor
    )
    test_rel_semih1_componentwise = H1relLoss_1D_multiout(0.0, True)(
        output_tensor, prediction_tensor
    )
    test_rel_h1_componentwise = H1relLoss_1D_multiout(1.0, True)(
        output_tensor, prediction_tensor
    )
    # print the mean error component-wise
    print("Test mean relative l1 norm componentwise: ", test_rel_l1_componentwise)
    print("Test mean relative l2 norm componentwise: ", test_rel_l2_componentwise)
    print(
        "Test mean relative semi h1 norm componentwise: ", test_rel_semih1_componentwise
    )
    print("Test mean relative h1 norm componentwise: ", test_rel_h1_componentwise)
    print("")

elif arc in ["crosstruss"]:
    test_rel_l1_componentwise = LprelLoss_multiout(1, True)(
        output_tensor, prediction_tensor
    )
    test_rel_l2_componentwise = LprelLoss_multiout(2, True)(
        output_tensor, prediction_tensor
    )
    test_rel_semih1_componentwise = H1relLoss_multiout(0.0, True)(
        output_tensor, prediction_tensor
    )
    test_rel_h1_componentwise = H1relLoss_multiout(1.0, True)(
        output_tensor, prediction_tensor
    )
    # print the mean error component-wise
    print("Test mean relative l1 norm componentwise: ", test_rel_l1_componentwise)
    print("Test mean relative l2 norm componentwise: ", test_rel_l2_componentwise)
    print(
        "Test mean relative semi h1 norm componentwise: ", test_rel_semih1_componentwise
    )
    print("Test mean relative h1 norm componentwise: ", test_rel_h1_componentwise)
    print("")


#########################################
# Time for evaluation
#########################################
# evaluation of all the test set
model.eval()
t_1 = time.time()
with torch.no_grad():
    _ = model(input_tensor)
t_2 = time.time()
print(f"Time for evaluation of {input_tensor.shape[0]} solutions is: ", t_2 - t_1)

# evaluation of one solution
ex = input_tensor[[0], ...]
t_1 = time.time()
with torch.no_grad():
    _ = model(ex)
t_2 = time.time()
print(f"Time for evaluation of one solution is: ", t_2 - t_1)
print("")


#########################################
# Prepare data for plotting
#########################################
# move tensors to cpu for plotting
input_tensor = input_tensor.to("cpu")
output_tensor = output_tensor.to("cpu")
prediction_tensor = prediction_tensor.to("cpu")

# de-normalize tensors to make it physical and the plot
match which_example:
    case "fhn" | "fhn_long":
        input_tensor = example.a_normalizer.decode(input_tensor)
        output_tensor[:, :, [0]] = example.v_normalizer.decode(output_tensor[:, :, [0]])
        output_tensor[:, :, [1]] = example.w_normalizer.decode(output_tensor[:, :, [1]])
        prediction_tensor[:, :, [0]] = example.v_normalizer.decode(
            prediction_tensor[:, :, [0]]
        )
        prediction_tensor[:, :, [1]] = example.w_normalizer.decode(
            prediction_tensor[:, :, [1]]
        )

    case "hh":
        input_tensor = example.a_normalizer.decode(input_tensor)
        output_tensor[:, :, [0]] = example.v_normalizer.decode(output_tensor[:, :, [0]])
        output_tensor[:, :, [1]] = example.m_normalizer.decode(output_tensor[:, :, [1]])
        output_tensor[:, :, [2]] = example.h_normalizer.decode(output_tensor[:, :, [2]])
        output_tensor[:, :, [3]] = example.n_normalizer.decode(output_tensor[:, :, [3]])
        prediction_tensor[:, :, [0]] = example.v_normalizer.decode(
            prediction_tensor[:, :, [0]]
        )
        prediction_tensor[:, :, [1]] = example.m_normalizer.decode(
            prediction_tensor[:, :, [1]]
        )
        prediction_tensor[:, :, [2]] = example.h_normalizer.decode(
            prediction_tensor[:, :, [2]]
        )
        prediction_tensor[:, :, [3]] = example.n_normalizer.decode(
            prediction_tensor[:, :, [3]]
        )

    case "crosstruss":
        # denormalize
        output_tensor[:, :, :, 0] = (example.max_x - example.min_x) * output_tensor[
            :, :, :, 0
        ] + example.min_x
        output_tensor[:, :, :, 1] = (example.max_y - example.min_y) * output_tensor[
            :, :, :, 1
        ] + example.min_y
        prediction_tensor[:, :, :, 0] = (
            example.max_x - example.min_x
        ) * prediction_tensor[:, :, :, 0] + example.min_x
        prediction_tensor[:, :, :, 1] = (
            example.max_y - example.min_y
        ) * prediction_tensor[:, :, :, 1] + example.min_y
        # multiplication for domain
        output_tensor[:, :, :, [0]] = (
            output_tensor[:, :, :, [0]] * input_tensor[:, :, :]
        )
        output_tensor[:, :, :, [1]] = (
            output_tensor[:, :, :, [1]] * input_tensor[:, :, :]
        )
        prediction_tensor[:, :, :, [0]] = (
            prediction_tensor[:, :, :, [0]] * input_tensor[:, :, :]
        )
        prediction_tensor[:, :, :, [1]] = (
            prediction_tensor[:, :, :, [1]] * input_tensor[:, :, :]
        )
    case (
        "poisson"
        | "wave_0_5"
        | "cont_tran"
        | "disc_tran"
        | "allen"
        | "shear_layer"
        | "airfoil"
        | "darcy"
    ):
        output_tensor = (
            example.max_model - example.min_model
        ) * output_tensor + example.min_model
        prediction_tensor = (
            example.max_model - example.min_model
        ) * prediction_tensor + example.min_model

    case "darcy_zongyi":
        input_tensor = example.a_normalizer.decode(input_tensor)
        output_tensor = example.u_normalizer.decode(output_tensor)
        prediction_tensor = example.u_normalizer.decode(prediction_tensor)

    case "burgers_zongyi" | "navier_stokes_zongyi":
        pass
        # TODO burgers and navier

    case _:
        raise ValueError("The example chosen is not allowed")


#########################################
# Example 1: Plot the histogram
#########################################
@jaxtyped(typechecker=beartype)
def plot_histogram(error: Float[Tensor, "n_samples"], str_norm: str):
    plt.figure(figsize=(8, 6))
    plt.hist(error.to("cpu"), bins=100)
    plt.xlabel("Relative error")
    plt.ylabel("Number of samples")
    plt.title(f"Histogram of the relative error in norm {str_norm}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


# call the functions to plot histograms for errors
plot_histogram(test_rel_l1_tensor, "L1")
plot_histogram(test_rel_l2_tensor, "L2")
plot_histogram(test_rel_semih1_tensor, "semi H1")
plot_histogram(test_rel_h1_tensor, "H1")

#########################################
# Example 2: Plot the worst and best samples
#########################################
# call the function to plot data
test_plot_samples(
    input_tensor,
    output_tensor,
    prediction_tensor,
    test_rel_l1_tensor,
    "worst",
    which_example,
    ntest=100,
    str_norm=exp_norm,
    n_idx=5,
)


def save_tensor(
    input_tensor, output_tensor, prediction_tensor, which_example: str, norm_str: str
):
    # Prepare data for saving
    flag = True
    match which_example:
        case "fhn" | "fhn_long":
            data_to_save = {
                "input": input_tensor.numpy().squeeze(),
                "V_exact": output_tensor[:, :, 0].numpy().squeeze(),
                "w_exact": output_tensor[:, :, 1].numpy().squeeze(),
                "V_pred": prediction_tensor[:, :, 0].numpy().squeeze(),
                "w_pred": prediction_tensor[:, :, 1].numpy().squeeze(),
            }

        case "hh":
            data_to_save = {
                "input": input_tensor.numpy().squeeze(),
                "V_exact": output_tensor[:, :, 0].numpy().squeeze(),
                "m_exact": output_tensor[:, :, 1].numpy().squeeze(),
                "h_exact": output_tensor[:, :, 2].numpy().squeeze(),
                "n_exact": output_tensor[:, :, 3].numpy().squeeze(),
                "V_pred": prediction_tensor[:, :, 0].numpy().squeeze(),
                "m_pred": prediction_tensor[:, :, 1].numpy().squeeze(),
                "h_pred": prediction_tensor[:, :, 2].numpy().squeeze(),
                "n_pred": prediction_tensor[:, :, 3].numpy().squeeze(),
            }

        case _:
            flag = False

    if flag:
        directory = f"../data/{which_example}/"
        os.makedirs(
            directory, exist_ok=True
        )  # Create the directory if it doesn't exist
        str_file = f"{directory}{which_example}_train{norm_str}_n_{output_tensor.shape[0]}_points_{output_tensor.shape[1]}_tf_{'200' if 'long' in which_example else '100'}.mat"
        savemat(str_file, data_to_save)
        print(f"Data saved in {str_file}")
    else:
        raise ValueError("The example chosen is not allowed")


# call the function to save tensors
save_tensor(input_tensor, output_tensor, prediction_tensor, which_example, exp_norm)
