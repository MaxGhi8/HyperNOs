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

"loss_fn_str" can be one of the following options:
    L1 : L^1 relative norm
    L2 : L^2 relative norm
    H1 : H^1 relative norm
    L1_smooth : L^1 smooth loss (Mishra)
    MSE : L^2 smooth loss (Mishra)
"""

import argparse
import json
import os
import sys
import time

sys.path.append("..")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from beartype import beartype
from cli.utilities_recover_model import test_fun, test_fun_tensors, test_plot_samples
from datasets import NO_load_data_model
from jaxtyping import Float, jaxtyped
from loss_fun import (
    H1relLoss_1D_multiout,
    H1relLoss_multiout,
    LprelLoss_multiout,
    loss_selector,
)
from scipy.io import savemat
from torch import Tensor
from utilities import count_params

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
        "loss_fn_str",
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
        "loss_fn_str": args.loss_fn_str.upper(),
        "mode": args.mode.lower(),
        "in_dist": args.in_dist,
    }


config = parse_arguments()
which_example = config["example"]
arc = config["architecture"]
loss_fn_str = config["loss_fn_str"]
mode_str = config["mode"]
in_dist = config["in_dist"]

Norm_dict = {"L1": 0, "L2": 1, "H1": 2, "L1_SMOOTH": 3, "MSE": 4}

# upload the model and the hyperparameters
folder = f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_{mode_str}/"
files = os.listdir(folder)
name_model = folder + [file for file in files if file.startswith("model_")][0]

try:
    model = torch.load(name_model, weights_only=False)
except Exception:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed trhow the CLI."
    )

#########################################
# Hyperparameters
#########################################
# Load `hyperparameters` from JSON
with open(folder + "/chosed_hyperparams.json", "r") as f:
    hyperparams = json.load(f)

# Choose the Loss function
hyperparams["loss_fn_str"] = loss_fn_str

# Training hyperparameters
batch_size = hyperparams["batch_size"]
beta = hyperparams["beta"]
FourierF = hyperparams["FourierF"]
problem_dim = hyperparams["problem_dim"]
retrain = hyperparams["retrain"]
val_samples = hyperparams["val_samples"]
test_samples = hyperparams["test_samples"]
training_samples = hyperparams["training_samples"]

# Loss function
loss = loss_selector(loss_fn_str=loss_fn_str, problem_dim=problem_dim, beta=beta)

#########################################
# Data loader
#########################################
example = NO_load_data_model(
    which_example,
    {
        "FourierF": hyperparams["FourierF"],
        "retrain": hyperparams["retrain"],
    },
    batch_size,
    training_samples,
)

train_loader = example.train_loader
val_loader = example.val_loader
test_loader = example.test_loader
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
    config["problem_dim"],
    loss_fn_str,
    val_samples,
    training_samples,
    device,
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
    config["problem_dim"],
    loss_fn_str,
    test_samples,
    training_samples,
    device,
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
    error_np = error.to("cpu").numpy()

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="deep")

    plt.figure(figsize=(10, 6))
    sns.histplot(error_np, bins=100, kde=True, color="skyblue", edgecolor="black")

    # Add labels and title
    plt.xlabel("Relative Error", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(
        f"Histogram of the Relative Error in Norm {str_norm}", fontsize=14, pad=20
    )

    # Improve grid and layout
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Resets the style to default
    plt.style.use("default")


@jaxtyped(typechecker=beartype)
def plot_overlapped_histograms(
    error1: Float[Tensor, "n_samples"],
    error2: Float[Tensor, "n_samples"],
    str_norm: str,
    label1: str = "Error 1",
    label2: str = "Error 2",
):
    # Convert tensors to numpy arrays
    error1_np = error1.to("cpu").numpy()
    error2_np = error2.to("cpu").numpy()

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="deep")

    plt.figure(figsize=(10, 6))

    # Plot the first histogram
    sns.histplot(
        error1_np,
        bins=100,
        kde=True,
        color="skyblue",
        edgecolor="black",
        label=label1,
        alpha=0.6,
    )

    # Plot the second histogram
    sns.histplot(
        error2_np,
        bins=100,
        kde=True,
        color="salmon",
        edgecolor="black",
        label=label2,
        alpha=0.6,
    )

    # Add labels and title
    plt.xlabel("Relative Error", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(
        f"Histogram of the Relative Error in Norm {str_norm}", fontsize=14, pad=20
    )

    # Add a legend
    plt.legend()

    # Improve grid and layout
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Resets the style to default
    plt.style.use("default")


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
    str_norm=loss_fn_str,
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
save_tensor(input_tensor, output_tensor, prediction_tensor, which_example, loss_fn_str)
