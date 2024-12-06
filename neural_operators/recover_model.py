"""
This is the main file for makes test and plot the Neural Operator with the trained FNO

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

import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
import numpy as np
import time
import sys
import json
import matplotlib.pyplot as plt
import os
from scipy.io import savemat

from Loss_fun import LprelLoss, H1relLoss_1D, H1relLoss
from train_fun import test_fun, test_fun_tensors
from utilities import count_params
from FNO.FNO_utilities import FNO_load_data_model

#########################################
# default values
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# torch.set_default_dtype(torch.float32) # default tensor dtype

mode_str = "best"  # test base hyperparameters, can be "default" or "best"

#########################################
# Choose the example to run
#########################################
# choose example from CLI
if len(sys.argv) == 1:
    raise ValueError("The user must choose the example to run")
elif len(sys.argv) == 2:
    which_example = sys.argv[
        1
    ]  # the example can be chosen by the user as second argument
    exp_norm = "L1"
    in_dist = True
elif len(sys.argv) == 3:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2]
    in_dist = True
elif len(sys.argv) == 4:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2]
    in_dist = sys.argv[3]
else:
    raise ValueError("The user must choose the example to run")

Norm_dict = {"L1": 0, "L2": 1, "H1": 2, "L1_smooth": 3, "MSE": 4}

# upload the model and the hyperparameters
arc = "FNO"
model_folder = f"./{arc}/TrainedModels/"
description_test = "test_" + exp_norm
folder = (
    model_folder
    + which_example
    + "/exp_FNO_"
    + description_test
    + "_"
    + mode_str
    + "_hyperparams"
)
name_model = (
    model_folder
    + which_example
    + "/model_FNO_"
    + description_test
    + "_"
    + mode_str
    + "_hyperparams"
)

try:
    model = torch.load(name_model, weights_only=False)
except:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed trhow the CLI."
    )

#########################################
# Hyperparameters
#########################################
# Load `training_properties` from JSON
with open(folder + "/hyperparams_train.json", "r") as f:
    training_properties = json.load(f)

# Load `fno_architecture` from JSON
with open(folder + "/hyperparams_arc.json", "r") as f:
    fno_architecture = json.load(f)

# Choose the Loss function
training_properties["exp"] = Norm_dict[
    exp_norm
]  # 0 for L^1 relative norm, 1 for L^2 relative norm, 2 for H^1 relative norm

# Training hyperparameters
learning_rate = training_properties["learning_rate"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
p = training_properties["exp"]
beta = training_properties["beta"]
training_samples = training_properties["training_samples"]
test_samples = training_properties["test_samples"]
val_samples = training_properties["val_samples"]

# FNO architecture hyperparameters
d_a = fno_architecture["d_a"]
d_v = fno_architecture["width"]
d_u = fno_architecture["d_u"]
L = fno_architecture["n_layers"]
modes = fno_architecture["modes"]
fun_act = fno_architecture["fun_act"]
weights_norm = fno_architecture["weights_norm"]
arc = fno_architecture["arc"]
RNN = fno_architecture["RNN"]
FFTnorm = fno_architecture["fft_norm"]
padding = fno_architecture["padding"]
retrain_fno = fno_architecture["retrain"]
FourierF = fno_architecture["FourierF"]
problem_dim = fno_architecture["problem_dim"]

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
example = FNO_load_data_model(
    which_example, fno_architecture, device, batch_size, training_samples, in_dist
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

# Count and print the total number of parameters
par_tot = count_params(model)
print("Total number of parameters is: ", par_tot)

#########################################
# Compute error and print error
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
print("Validation relative l1 norm: ", val_relative_l1)
print("Validation relative l2 norm: ", val_relative_l2)
print("Validation relative semi h1 norm: ", val_relative_semih1)
print("Validation relative h1 norm: ", val_relative_h1)

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

print("Test relative l1 norm: ", test_relative_l1)
print("Test relative l2 norm: ", test_relative_l2)
print("Test relative semi h1 norm: ", test_relative_semih1)
print("Test relative h1 norm: ", test_relative_h1)

# Compute all the relative errors and outputs
(
    input_tensor,
    output_tensor,
    prediction_tensor,
    test_rel_l1_tensor,
    test_rel_l2_tensor,
    test_rel_semih1_tensor,
    test_rel_h1_tensor,
) = test_fun_tensors(model, test_loader, loss, device, which_example)

# evaluate the model and compute time
model.eval()
t_1 = time.time()
with torch.no_grad():
    _ = model(input_tensor)
t_2 = time.time()
print(f"Time for evaluation of {input_tensor.shape[0]} solutions is: ", t_2 - t_1)


ex = input_tensor[[0], ...]
t_1 = time.time()
with torch.no_grad():
    _ = model(ex)
t_2 = time.time()
print(f"Time for evaluation of one solution is: ", t_2 - t_1)

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


#########################################
# Example 2: Plot the worst and best samples
#########################################
# @jaxtyped(typechecker=beartype)
def test_plot_samples(
    input_tensor: Float[Tensor, "n_samples *n d_a"],
    output_tensor: Float[Tensor, "n_samples *n d_v"],
    prediction_tensor: Float[Tensor, "n_samples *n d_v"],
    error: Float[Tensor, "n_samples"],
    mode: str,
    which_example: str,
    ntest: int,
    str_norm: str,
    n_idx: int = 5,
):
    """
    Function to plot the worst and best samples in the test set.
    mode: str
        can be "best", "worst" or "random", to plot the best, worst or random samples respectively.
    """
    error, indices = torch.sort(
        error, descending=True
    )  # Sort the error in descending order
    if mode == "worst":
        idx = indices[:n_idx].to("cpu")
        error = error[:n_idx].to("cpu")
    elif mode == "best":
        idx = indices[-n_idx:].to("cpu")
        error = error[-n_idx:].to("cpu")
    elif mode == "random":
        idx = torch.tensor(np.random.randint(0, ntest, size=(n_idx,)))
        error = error[idx].to("cpu")
    else:
        raise ValueError("The mode must be 'best', 'worst' or 'random'")

    print(f"The {mode} samples are: {idx} with error {error} in norms {str_norm}")

    if which_example == "fhn" or which_example == "fhn_long":
        n_points = input_tensor.shape[1]
        x_grid = torch.linspace(0, 100, n_points).to("cpu")

        fig, axs = plt.subplots(3, len(idx), figsize=(12, 4))

        for i in range(3):
            for j in range(idx.shape[0]):
                if i == 0:  # input
                    axs[i, j].plot(
                        x_grid,
                        input_tensor[idx[j], :, :].squeeze(),
                        label="Input (I_app)",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$I_app$(t)")
                elif i == 1:  # v approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 0].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 0].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$v$(t)")
                else:  # w approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 1].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 1].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$w$(t)")

                axs[i, j].set_xlabel("t")
                axs[0, j].set_ylim([0, 2])
                axs[1, j].set_ylim([-0.5, 1.5])
                axs[2, j].set_ylim([0, 2.5])
                axs[i, j].grid()
                axs[i, j].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, len(idx), figsize=(12, 4))

        for i in range(2):
            for j in range(idx.shape[0]):
                if i == 0:  # input
                    axs[i, j].plot(
                        x_grid,
                        input_tensor[idx[j], :, :].squeeze(),
                        label="Input (I_app)",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$I_app$(t)")
                    axs[i, j].set_xlabel("t")
                else:  # phase space
                    axs[i, j].plot(
                        output_tensor[idx[j], :, 0].squeeze(),
                        output_tensor[idx[j], :, 1].squeeze(),
                        label="sol",
                    )
                    axs[i, j].plot(
                        prediction_tensor[idx[j], :, 0].squeeze(),
                        prediction_tensor[idx[j], :, 1].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$w$(t)")
                    axs[i, j].set_xlabel("$v$(t)")

                axs[i, j].grid()
                axs[i, j].legend(loc="upper right")
                axs[0, j].set_ylim([0, 2])
                axs[1, j].set_xlim([-0.5, 1.5])
                axs[1, j].set_ylim([0, 2.5])

        plt.tight_layout()
        plt.show()

    if which_example == "hh":
        n_points = input_tensor.shape[1]
        x_grid = torch.linspace(0, 100, n_points).to("cpu")

        fig, axs = plt.subplots(5, len(idx), figsize=(12, 8))

        for i in range(5):
            for j in range(idx.shape[0]):

                if i == 0:  # input
                    axs[i, j].plot(
                        x_grid,
                        input_tensor[idx[j], :, :].squeeze(),
                        label="Input (I_app)",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$I_app$(t)")

                elif i == 1:  # v approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 0].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 0].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$v$(t)")

                elif i == 2:  # m approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 1].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 1].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$m$(t)")

                elif i == 3:  # h approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 2].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 2].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$h$(t)")

                elif i == 4:  # n approximation
                    axs[i, j].plot(
                        x_grid, output_tensor[idx[j], :, 3].squeeze(), label="sol"
                    )
                    axs[i, j].plot(
                        x_grid,
                        prediction_tensor[idx[j], :, 3].squeeze(),
                        "r",
                        label="FNO",
                    )
                    if j == 0:
                        axs[i, j].set_ylabel("$n$(t)")

                axs[i, j].set_xlabel("t")
                # axs[0, j].set_ylim([0, 2])
                # axs[1, j].set_ylim([-0.5, 1.5])
                # axs[2, j].set_ylim([0, 2.5])
                axs[i, j].grid()
                axs[i, j].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    if which_example == "crosstruss":

        fig, axs = plt.subplots(7, len(idx), figsize=(16, 16))
        for i in range(7):
            for j in range(idx.shape[0]):
                if i == 0:  # input
                    im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    colorbar.set_ticks([0, 1])
                    if j == 0:
                        axs[i, j].set_ylabel("domain input")

                elif i == 1:  # output x
                    im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    colorbar.set_ticks([-0.005, 0.005])
                    if j == 0:
                        axs[i, j].set_ylabel("displacement x")

                elif i == 2:  # predicted x
                    im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    colorbar.set_ticks([-0.005, 0.005])
                    if j == 0:
                        axs[i, j].set_ylabel("pred disp x")

                elif i == 3:  # error x
                    error = torch.abs(
                        output_tensor[idx[j], :, :, 0]
                        - prediction_tensor[idx[j], :, :, 0]
                    )
                    im = axs[i, j].imshow(error.squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    if j == 0:
                        axs[i, j].set_ylabel("error x")

                elif i == 4:  # output y
                    im = axs[i, j].imshow(output_tensor[idx[j], :, :, 1].squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    colorbar.set_ticks([0.0, 0.1])
                    if j == 0:
                        axs[i, j].set_ylabel("displacement y")

                elif i == 5:  # predicted y
                    im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 1].squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    colorbar.set_ticks([0.0, 0.1])
                    if j == 0:
                        axs[i, j].set_ylabel("pred disp y")

                elif i == 6:  # error y
                    error = torch.abs(
                        output_tensor[idx[j], :, :, 1]
                        - prediction_tensor[idx[j], :, :, 1]
                    )
                    im = axs[i, j].imshow(error.squeeze())
                    colorbar = fig.colorbar(im, ax=axs[i, j])
                    if j == 0:
                        axs[i, j].set_ylabel("error y")

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                # axs[i, j].set_xlabel('x')

        plt.tight_layout()
        plt.show()
        plt.savefig("figure.png")


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


# call the functions to plot histograms for errors
plot_histogram(test_rel_l1_tensor, "L1")
plot_histogram(test_rel_l2_tensor, "L2")
plot_histogram(test_rel_semih1_tensor, "semi H1")
plot_histogram(test_rel_h1_tensor, "H1")

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

# call the function to save tensors
save_tensor(input_tensor, output_tensor, prediction_tensor, which_example, exp_norm)
