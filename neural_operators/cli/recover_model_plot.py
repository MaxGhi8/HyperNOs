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
from cli.utilities_recover_model import get_tensors, test_fun, test_plot_samples
from datasets import NO_load_data_model
from jaxtyping import Float, jaxtyped
from loss_fun import (
    H1relLoss,
    H1relLoss_1D,
    H1relLoss_1D_multiout,
    H1relLoss_multiout,
    LprelLoss,
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
# device = torch.device("cpu")
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
            "hh",
            "ord",
            "crosstruss",
            "afieti_homogeneous_neumann",
            "diffusion_reaction",
            "fhn_1d",
        ],
        help="Select the example to run.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=["FNO", "CNO", "ResNet"],
        help="Select the architecture to use.",
    )
    parser.add_argument(
        "loss_fn_str",
        type=str,
        choices=["L1", "L2", "H1", "L1_smooth", "l2"],
        help="Select the relative loss function to use during the training process.",
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=[
            "best",
            "default",
            "best_samedofs",
            "best_linear",
            "best_50M",  # for the cont_tran FNO tests
            "best_150M",  # for the cont_tran FNO tests
            "best_500k",  # for the cont_tran FNO tests
        ],
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
        "architecture": args.architecture,
        "loss_fn_str": args.loss_fn_str,
        "mode": args.mode,
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
folder_best = f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_best/"
files_best = os.listdir(folder_best)
name_model_best = (
    folder_best + [file for file in files_best if file.startswith("model_")][0]
)

folder_samedofs = (
    f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_best_samedofs/"
)
files_samedofs = os.listdir(folder_samedofs)
name_model_samedofs = (
    folder_samedofs + [file for file in files_samedofs if file.startswith("model_")][0]
)

try:
    model_best = torch.load(name_model_best, weights_only=False, map_location=device)
    model_best.eval()

    model_samedofs = torch.load(
        name_model_samedofs, weights_only=False, map_location=device
    )
    model_samedofs.eval()
    # torch.save(model.state_dict(), name_model_best + "_state_dict")
except Exception:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed trhow the CLI."
    )

#########################################
# Hyperparameters
#########################################
# Load `hyperparameters` from JSON
with open(folder_best + "/chosed_hyperparams.json", "r") as f:
    hyperparams = json.load(f)

# Choose the Loss function
hyperparams["loss_fn_str"] = loss_fn_str

# Training hyperparameters
batch_size = hyperparams["batch_size"]
try:
    beta = hyperparams["beta"]
except KeyError:
    beta = 1.0  # default value
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
    "Dimension of datasets inputs are:",
    next(iter(train_loader))[0].shape,
    next(iter(val_loader))[0].shape,
    next(iter(test_loader))[0].shape,
)
print(
    "Dimension of datasets outputs are:",
    next(iter(train_loader))[1].shape,
    next(iter(val_loader))[1].shape,
    next(iter(test_loader))[1].shape,
)
print("")

# Count and print the total number of parameters
total_params, total_bytes = count_params(model_best)
total_mb = total_bytes / (1024**2)
print(f"Total Parameters: {total_params:,}")
print(f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)")

total_params, total_bytes = count_params(model_samedofs)
total_mb = total_bytes / (1024**2)
print(f"Total Parameters: {total_params:,}")
print(f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)")

#########################################
# Get the tensors
#########################################
(
    input_tensor,
    output_tensor,
    prediction_tensor,
) = get_tensors(model_best, test_loader, device)
(
    train_input_tensor,
    train_output_tensor,
    train_prediction_tensor,
) = get_tensors(model_best, train_loader, device)

(
    input_tensor_samedofs,
    output_tensor_samedofs,
    prediction_tensor_samedofs,
) = get_tensors(model_samedofs, test_loader, device)
(
    train_input_tensor_samedofs,
    train_output_tensor_samedofs,
    train_prediction_tensor_samedofs,
) = get_tensors(model_samedofs, train_loader, device)

#########################################
# Compute mean error and print it
#########################################
# Error tensors
train_relative_l2_tensor = LprelLoss(2, None)(
    train_output_tensor, train_prediction_tensor
)
test_relative_l2_tensor = LprelLoss(2, None)(output_tensor, prediction_tensor)

train_relative_l2_tensor_samedofs = LprelLoss(2, None)(
    train_output_tensor_samedofs, train_prediction_tensor_samedofs
)
test_relative_l2_tensor_samedofs = LprelLoss(2, None)(
    output_tensor_samedofs, prediction_tensor_samedofs
)

#########################################
# Time for evaluation
#########################################
# evaluation of all the test set
# t_1 = time.time()
# with torch.no_grad():
#     _ = model(input_tensor)
# t_2 = time.time()
# print(f"Time for evaluation of {input_tensor.shape[0]} solutions is: ", t_2 - t_1)

# # evaluation of one solution
# ex = input_tensor[[0], ...]
# t_1 = time.time()
# with torch.no_grad():
#     _ = model(ex)
# t_2 = time.time()
# print(f"Time for evaluation of one solution is: ", t_2 - t_1)
# print("")


#########################################
# Example 1: Plot the histogram
#########################################
@jaxtyped(typechecker=beartype)
def plot_histogram(
    errors: list[Float[Tensor, "n_samples"]], str_norm: str, legends: list[str] = None
):

    if legends != None:
        assert len(legends) == len(
            errors
        ), "Legend is not consistent with input errros, have different length."

        error_np = {}
        for legend, error in zip(legends, errors):
            error_np[legend] = error.to("cpu").numpy()

    else:
        error_np = [error.to("cpu").numpy() for error in errors]

    # Set seaborn style for better aesthetics
    sns.set(style="white", palette="deep")

    plt.figure(figsize=(8, 6), layout="constrained")
    plt.xscale("log")
    sns.histplot(
        error_np,
        bins=100,
        # kde=True,
        color="skyblue",
        edgecolor="black",
        multiple="stack",
        legend=True if legends else False,
    )

    # Add labels and title
    plt.xlabel("Relative Error", fontsize=18)
    plt.ylabel("Number of Samples", fontsize=18)
    plt.xticks([0.001, 0.01, 0.1, 1], fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(
    #     f"Histogram of the Relative Error in Norm {str_norm}", fontsize=20, pad=20
    # )

    # Improve grid and layout
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")

    # Show the plot
    plt.savefig(
        f"./{which_example}_histograms_{str_norm}_{mode_str}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # Resets the style to default
    plt.style.use("default")


# call the functions to plot histograms for errors
# plot_histogram(
#     [train_relative_l1_tensor, test_relative_l1_tensor],
#     "L1",
#     ["Train error", "Test error"],
# )
plot_histogram(
    (
        [train_relative_l2_tensor, test_relative_l2_tensor]
        if mode_str == "best"
        else [train_relative_l2_tensor_samedofs, test_relative_l2_tensor_samedofs]
    ),
    "L2",
    ["Train error", "Test error"],
)
# plot_histogram(
#     [train_relative_semih1_tensor, test_relative_semih1_tensor],
#     "Semi H1",
#     ["Train error", "Test error"],
# )
# plot_histogram(
#     [train_relative_h1_tensor, test_relative_h1_tensor],
#     "H1",
#     ["Train error", "Test error"],
# )


#########################################
# Example 1_bis: boxplot
#########################################
@jaxtyped(typechecker=beartype)
def plot_boxplot(
    errors: list[Float[Tensor, "n_samples"]], str_norm: str, legends: list[str] = None
):

    if legends != None:
        assert len(legends) == len(
            errors
        ), "Legend is not consistent with input errros, have different length."

        error_np = {}
        for legend, error in zip(legends, errors):
            error_np[legend] = error.to("cpu").numpy()

    else:
        error_np = [error.to("cpu").numpy() for error in errors]

    # Set seaborn style for better aesthetics
    sns.set(style="white", palette="deep")

    plt.figure(figsize=(12, 6), layout="constrained")
    # flierprops = dict(
    #     marker="o",
    #     markerfacecolor="red",
    #     markersize=4,
    #     linestyle="none",
    #     markeredgecolor="black",
    # )
    # sns.boxplot(error_np, log_scale=True, flierprops=flierprops)
    sns.stripplot(
        error_np,
        orient="v",
        log_scale=True,
        jitter=0.4,
        edgecolor="black",
        size=1.5,
        linewidth=0.15,
    )
    sns.boxplot(error_np, fliersize=0, whis=1.5)

    # Add labels and title
    plt.ylabel("Relative Error", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks([0.001, 0.01, 0.1, 1], fontsize=18)
    plt.ylim(0.001, 1)
    # plt.title(
    #     f"boxplot of the Relative Error in Norm {str_norm}", fontsize=20, pad=20
    # )

    # Improve grid and layout
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")

    # Show the plot
    plt.savefig(
        f"./{which_example}_boxplot_{str_norm}_swarmplot.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # Resets the style to default
    plt.style.use("default")


# call the functions to plot swarm-plots for errors
# plot_boxplot(
#     [train_relative_l1_tensor, test_relative_l1_tensor],
#     "L1",
#     ["Train error", "Test error"],
# )
plot_boxplot(
    [
        train_relative_l2_tensor,
        test_relative_l2_tensor,
        train_relative_l2_tensor_samedofs,
        test_relative_l2_tensor_samedofs,
    ],
    "L2",
    [
        "Train error unconstr.",
        "Test error unconstr.",
        "Train error constr.",
        "Test error constr.",
    ],
)
# plot_boxplot(
#     [train_relative_semih1_tensor, test_relative_semih1_tensor],
#     "Semi H1",
#     ["Train error", "Test error"],
# )
# plot_boxplot(
#     [train_relative_h1_tensor, test_relative_h1_tensor],
#     "H1",
#     ["Train error", "Test error"],
# )


#########################################
# Example 4: save input, output and prediction tensor
#########################################
def save_tensor(
    input_tensor, output_tensor, prediction_tensor, which_example: str, norm_str: str
):
    # Prepare data for saving
    flag = True
    print(flag, which_example)
    match which_example:

        case (
            "poisson"
            | "wave_0_5"
            | "allen"
            | "shear_layer"
            | "darcy"
            | "cont_tran"
            | "disc_tran"
            | "airfoil"
        ):
            data_to_save = {
                "input": input_tensor.numpy().squeeze(),
                "output": output_tensor.numpy().squeeze(),
                "prediction": prediction_tensor.numpy().squeeze(),
            }

        case "fhn":
            tf = 100

            data_to_save = {
                "input": input_tensor.cpu().numpy().squeeze(),
                "V_exact": output_tensor[:, :, 0].cpu().numpy().squeeze(),
                "w_exact": output_tensor[:, :, 1].cpu().numpy().squeeze(),
                "V_pred": prediction_tensor[:, :, 0].cpu().numpy().squeeze(),
                "w_pred": prediction_tensor[:, :, 1].cpu().numpy().squeeze(),
            }

        case "hh":
            tf = 100
            data_to_save = {
                "input": input_tensor.cpu().numpy().squeeze(),
                "V_exact": output_tensor[:, :, 0].cpu().numpy().squeeze(),
                "m_exact": output_tensor[:, :, 1].cpu().numpy().squeeze(),
                "h_exact": output_tensor[:, :, 2].cpu().numpy().squeeze(),
                "n_exact": output_tensor[:, :, 3].cpu().numpy().squeeze(),
                "V_pred": prediction_tensor[:, :, 0].cpu().numpy().squeeze(),
                "m_pred": prediction_tensor[:, :, 1].cpu().numpy().squeeze(),
                "h_pred": prediction_tensor[:, :, 2].cpu().numpy().squeeze(),
                "n_pred": prediction_tensor[:, :, 3].cpu().numpy().squeeze(),
            }

        case "ord":
            tf = 500
            data_to_save = {
                "input": input_tensor[:, :, :].cpu().numpy().squeeze(),
            }
            for idx, field in enumerate(example.fields_to_concat):
                key_exact = field + "_exact"
                key_pred = field + "_pred"
                data_to_save[key_exact] = (
                    output_tensor[:, :, idx].cpu().numpy().squeeze()
                )
                data_to_save[key_pred] = (
                    prediction_tensor[:, :, idx].cpu().numpy().squeeze()
                )

        case _:
            flag = False

    if flag:
        if which_example in ["hh", "fhn", "ord"]:
            directory = f"../../data/{which_example}/"
            os.makedirs(directory, exist_ok=True)
        elif which_example in [
            "poisson",
            "wave_0_5",
            "allen",
            "shear_layer",
            "darcy",
            "cont_tran",
            "disc_tran",
            "airfoil",
        ]:
            directory = f"../../data/mishra/outputs_for_website/"
            os.makedirs(directory, exist_ok=True)

        if which_example == "ord":
            str_file = f"{directory}{which_example}_train{norm_str}_n_{output_tensor.shape[0]}_points_{output_tensor[:, :, 0].shape[1]}_tf_{tf}_{mode_str}.mat"
        elif which_example in ["hh", "fhn"]:
            str_file = f"{directory}{which_example}_train{norm_str}_n_{output_tensor.shape[0]}_points_{output_tensor[:, :, ].shape[1]}_tf_{tf}_{mode_str}.mat"
        elif which_example in [
            "poisson",
            "wave_0_5",
            "allen",
            "shear_layer",
            "darcy",
            "cont_tran",
            "disc_tran",
            "airfoil",
        ]:
            str_file = f"{directory}{which_example}_{arc}_train{norm_str}_{mode_str.replace('_', '')}.mat"

        savemat(str_file, data_to_save)
        print(f"Data saved in {str_file}")

    else:
        # raise ValueError("The example chosen is not allowed")
        pass


# call the function to save tensors
if mode_str == "best":
    save_tensor(
        input_tensor, output_tensor, prediction_tensor, which_example, loss_fn_str
    )
else:
    save_tensor(
        input_tensor_samedofs,
        output_tensor_samedofs,
        prediction_tensor_samedofs,
        which_example,
        loss_fn_str,
    )
