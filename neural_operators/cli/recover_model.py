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
from CNO import CNO
from datasets import NO_load_data_model
from FNO import FNO
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
from ResNet import ResidualNetwork
from scipy.io import savemat
from torch import Tensor
from utilities import count_params, initialize_hyperparameters
from wrappers import wrap_model

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
            "hh",
            "ord",
            "crosstruss",
            "afieti_homogeneous_neumann",
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
folder = f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_{mode_str}/"
files = os.listdir(folder)
name_model = folder + [file for file in files if file.startswith("model_")][0]

try:
    try:  # For models saved with torch.save(model.state_dict())
        # Load the default hyperparameters for the FNO model
        hyperparams_train, hyperparams_arc = initialize_hyperparameters(
            arc, which_example, mode_str
        )

        default_hyper_params = {
            **hyperparams_train,
            **hyperparams_arc,
        }

        example = NO_load_data_model(
            which_example=which_example,
            no_architecture={
                "FourierF": default_hyper_params["FourierF"],
                "retrain": default_hyper_params["retrain"],
            },
            batch_size=default_hyper_params["batch_size"],
            training_samples=default_hyper_params["training_samples"],
        )

        match arc:
            case "FNO":
                model = FNO(
                    default_hyper_params["problem_dim"],
                    default_hyper_params["in_dim"],
                    default_hyper_params["width"],
                    default_hyper_params["out_dim"],
                    default_hyper_params["n_layers"],
                    default_hyper_params["modes"],
                    default_hyper_params["fun_act"],
                    default_hyper_params["weights_norm"],
                    default_hyper_params["fno_arc"],
                    default_hyper_params["RNN"],
                    default_hyper_params["fft_norm"],
                    default_hyper_params["padding"],
                    device,
                    default_hyper_params["retrain"],
                )
                model = wrap_model(model, which_example)

            case "CNO":
                model = CNO(
                    problem_dim=default_hyper_params["problem_dim"],
                    in_dim=default_hyper_params["in_dim"],
                    out_dim=default_hyper_params["out_dim"],
                    size=default_hyper_params["in_size"],
                    N_layers=default_hyper_params["N_layers"],
                    N_res=default_hyper_params["N_res"],
                    N_res_neck=default_hyper_params["N_res_neck"],
                    channel_multiplier=default_hyper_params["channel_multiplier"],
                    kernel_size=default_hyper_params["kernel_size"],
                    use_bn=default_hyper_params["bn"],
                    device=device,
                )
                model = wrap_model(model, which_example)

            case "ResNet":
                model = ResidualNetwork(
                    default_hyper_params["in_channels"],
                    default_hyper_params["out_channels"],
                    default_hyper_params["hidden_channels"],
                    default_hyper_params["activation_str"],
                    default_hyper_params["n_blocks"],
                    device,
                    layer_norm=default_hyper_params["layer_norm"],
                    dropout_rate=default_hyper_params["dropout_rate"],
                    zero_mean=default_hyper_params["zero_mean"],
                    example=(
                        example
                        if default_hyper_params["internal_normalization"]
                        else None
                    ),
                )

        checkpoint = torch.load(name_model, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    except Exception:
        # save with torch.save(model)
        model = torch.load(name_model, weights_only=False, map_location=device)

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
total_params, total_bytes = count_params(model)
total_mb = total_bytes / (1024**2)
print(f"Total Parameters: {total_params:,}")
print(f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)")

#########################################
# Compute train error and print it
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
    problem_dim,
    loss_fn_str,
    device,
    statistic=True,
)

print("Train loss: ", train_loss)
print("Validation mean relative l1 norm: ", val_relative_l1)
print("Validation mean relative l2 norm: ", val_relative_l2)
print("Validation mean relative semi h1 norm: ", val_relative_semih1)
print("Validation mean relative h1 norm: ", val_relative_h1)
print("")

#########################################
# Get the tensors
#########################################
(
    input_tensor,
    output_tensor,
    prediction_tensor,
) = get_tensors(model, test_loader, device)
(
    train_input_tensor,
    train_output_tensor,
    train_prediction_tensor,
) = get_tensors(model, train_loader, device)

#########################################
# Compute mean error and print it
#########################################
# Error tensors
train_relative_l1_tensor = LprelLoss(1, None)(
    train_output_tensor, train_prediction_tensor
)
test_relative_l1_tensor = LprelLoss(1, None)(output_tensor, prediction_tensor)

train_relative_l2_tensor = LprelLoss(2, None)(
    train_output_tensor, train_prediction_tensor
)
test_relative_l2_tensor = LprelLoss(2, None)(output_tensor, prediction_tensor)

if problem_dim == 1:
    train_relative_semih1_tensor = H1relLoss_1D(1.0, None, 0.0)(
        train_output_tensor, train_prediction_tensor
    )
    train_relative_h1_tensor = H1relLoss_1D(1.0, None)(
        train_output_tensor, train_prediction_tensor
    )

    test_relative_semih1_tensor = H1relLoss_1D(1.0, None, 0.0)(
        output_tensor, prediction_tensor
    )
    test_relative_h1_tensor = H1relLoss_1D(1.0, None)(output_tensor, prediction_tensor)

elif problem_dim == 2:
    train_relative_semih1_tensor = H1relLoss(1.0, None, 0.0)(
        train_output_tensor, train_prediction_tensor
    )
    train_relative_h1_tensor = H1relLoss(1.0, None)(
        train_output_tensor, train_prediction_tensor
    )

    test_relative_semih1_tensor = H1relLoss(1.0, None, 0.0)(
        output_tensor, prediction_tensor
    )
    test_relative_h1_tensor = H1relLoss(1.0, None)(output_tensor, prediction_tensor)

# Error mean
test_mean_l1 = test_relative_l1_tensor.mean().item()
test_mean_l2 = test_relative_l2_tensor.mean().item()
test_mean_semih1 = test_relative_semih1_tensor.mean().item()
test_mean_h1 = test_relative_h1_tensor.mean().item()

# Error median
test_median_l1 = torch.median(test_relative_l1_tensor).item()
test_median_l2 = torch.median(test_relative_l2_tensor).item()
test_median_semih1 = torch.median(test_relative_semih1_tensor).item()
test_median_h1 = torch.median(test_relative_h1_tensor).item()

print("Test mean relative l1 norm: ", test_mean_l1)
print("Test mean relative l2 norm: ", test_mean_l2)
print("Test mean relative semi h1 norm: ", test_mean_semih1)
print("Test mean relative h1 norm: ", test_mean_h1)
print("")

print("Test median relative l1 norm: ", test_median_l1)
print("Test median relative l2 norm: ", test_median_l2)
print("Test median relative semi h1 norm: ", test_median_semih1)
print("Test median relative h1 norm: ", test_median_h1)
print("")

#########################################
# Compute mean error component per component
#########################################
if which_example in ["fhn", "hh", "ord"]:
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

elif which_example in ["crosstruss"]:
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
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")
    # plt.title(
    #     f"Histogram of the Relative Error in Norm {str_norm}", fontsize=20, pad=20
    # )

    # Show the plot
    plt.savefig(
        f"./{which_example}_histograms_{str_norm}.png", dpi=300, bbox_inches="tight"
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
    [train_relative_l2_tensor, test_relative_l2_tensor],
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
    plt.figure(figsize=(8, 6), layout="constrained")

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
    plt.yticks(fontsize=18)
    plt.grid(True, which="both", ls="-", alpha=0.1, color="black")
    # plt.title(
    #     f"boxplot of the Relative Error in Norm {str_norm}", fontsize=20, pad=20
    # )

    # Show the plot
    plt.savefig(
        f"./{which_example}_boxplot_{str_norm}.png", dpi=300, bbox_inches="tight"
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
    [train_relative_l2_tensor, test_relative_l2_tensor],
    "L2",
    ["Train error", "Test error"],
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
# Example 2: Plot the worst and best samples
#########################################
# move tensors to cpu for plotting
input_tensor = input_tensor.to("cpu")
output_tensor = output_tensor.to("cpu")
prediction_tensor = prediction_tensor.to("cpu")

# Denormalize the tensors for plotting
stats_to_save = None  # default value
match which_example:
    case "fhn":
        input_tensor = example.a_normalizer.decode(input_tensor)
        output_tensor[:, :, [0]] = example.v_normalizer.decode(output_tensor[:, :, [0]])
        output_tensor[:, :, [1]] = example.w_normalizer.decode(output_tensor[:, :, [1]])
        prediction_tensor[:, :, [0]] = example.v_normalizer.decode(
            prediction_tensor[:, :, [0]]
        )
        prediction_tensor[:, :, [1]] = example.w_normalizer.decode(
            prediction_tensor[:, :, [1]]
        )

        stats_to_save = {
            "mean_input": example.a_normalizer.mean,
            "std_input": example.a_normalizer.std,
            "mean_V": example.v_normalizer.mean,
            "std_V": example.v_normalizer.std,
            "mean_w": example.w_normalizer.mean,
            "std_w": example.w_normalizer.std,
        }

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

        stats_to_save = {
            "mean_input": example.a_normalizer.mean,
            "std_input": example.a_normalizer.std,
            "mean_V": example.v_normalizer.mean,
            "std_V": example.v_normalizer.std,
            "mean_m": example.m_normalizer.mean,
            "std_m": example.m_normalizer.std,
            "mean_h": example.h_normalizer.mean,
            "std_h": example.h_normalizer.std,
            "mean_n": example.n_normalizer.mean,
            "std_n": example.n_normalizer.std,
        }

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

    case "poisson" | "wave_0_5" | "allen" | "shear_layer" | "darcy":
        # denormalize input
        input_tensor = (
            input_tensor * (example.max_data - example.min_data) + example.min_data
        )
        # denormalize outputs
        output_tensor = (
            example.max_model - example.min_model
        ) * output_tensor + example.min_model

        prediction_tensor = (
            example.max_model - example.min_model
        ) * prediction_tensor + example.min_model

    case "cont_tran" | "disc_tran" | "airfoil":
        # data does not need to be normalized
        pass

    case "darcy_zongyi":
        input_tensor = example.a_normalizer.decode(input_tensor)
        output_tensor = example.u_normalizer.decode(output_tensor)
        prediction_tensor = example.u_normalizer.decode(prediction_tensor)

    case "burgers_zongyi" | "navier_stokes_zongyi":
        pass
        # TODO burgers and navier

    case "ord":
        pass
        input_tensor = example.dict_normalizers["I_app_dataset"].decode(input_tensor)

        for i in range(output_tensor.size(-1)):
            output_tensor[:, :, [i]] = example.dict_normalizers[
                example.fields_to_concat[i]
            ].decode(output_tensor[:, :, [i]])

        for i in range(prediction_tensor.size(-1)):
            prediction_tensor[:, :, [i]] = example.dict_normalizers[
                example.fields_to_concat[i]
            ].decode(prediction_tensor[:, :, [i]])

        stats_to_save = {
            "mean_input": example.dict_normalizers["I_app_dataset"].mean,
            "std_input": example.dict_normalizers["I_app_dataset"].std,
        }
        for i in range(output_tensor.size(-1)):
            stats_to_save[f"mean_{example.fields_to_concat[i]}"] = (
                example.dict_normalizers[example.fields_to_concat[i]].mean
            )
            stats_to_save[f"std_{example.fields_to_concat[i]}"] = (
                example.dict_normalizers[example.fields_to_concat[i]].std
            )

    case _:
        raise ValueError("The example chosen is not allowed")

if stats_to_save:
    directory = f"../../data/{which_example}/"
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    str_file = f"{directory}{which_example}_stats_n_points_{output_tensor.shape[1]}.mat"
    savemat(str_file, stats_to_save)
    print(f"Data saved in {str_file}")

# call the function to plot data
test_plot_samples(
    input_tensor,
    output_tensor,
    prediction_tensor,
    test_relative_l1_tensor,
    "worst",
    which_example,
    ntest=100,
    str_norm=loss_fn_str,
    n_idx=5,
)


#########################################
# Example 3: Plot the weight matrix
#########################################
def print_matrix(model, name_param):
    for name, par in model.named_parameters():
        if name == name_param:
            print(name, par.shape)
            fig = plt.imshow(torch.abs(par[:, :, 0, 0]).detach().cpu().numpy())
            plt.colorbar(fig)

            plt.savefig("output_figure.png", dpi=300, bbox_inches="tight")

            # plt.show()


# print_matrix(model, "integrals.0.weights1")


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
                "input": input_tensor.numpy().squeeze(),
                "V_exact": output_tensor[:, :, 0].numpy().squeeze(),
                "w_exact": output_tensor[:, :, 1].numpy().squeeze(),
                "V_pred": prediction_tensor[:, :, 0].numpy().squeeze(),
                "w_pred": prediction_tensor[:, :, 1].numpy().squeeze(),
            }

        case "hh":
            tf = 100
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

        case "ord":
            tf = 500
            data_to_save = {
                "input": input_tensor[:, :, :].numpy().squeeze(),
            }
            for idx, field in enumerate(example.fields_to_concat):
                key_exact = field + "_exact"
                key_pred = field + "_pred"
                data_to_save[key_exact] = output_tensor[:, :, idx].numpy().squeeze()
                data_to_save[key_pred] = prediction_tensor[:, :, idx].numpy().squeeze()

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
# save_tensor(input_tensor, output_tensor, prediction_tensor, which_example, loss_fn_str)
