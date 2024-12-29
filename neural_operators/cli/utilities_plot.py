"""
Plotting of input/output and prediction for the trained model.

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
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


#########################################
# Plotting poisson example
#########################################
def plot_poisson(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 12))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Source term f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Poisson equation: $-\Delta u = f$ with Dirichlet BCs")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting wave_0_5 example
#########################################
def plot_wave_0_5(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Initial condition f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Wave equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting cont_tran example
#########################################
def plot_cont_tran(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Continuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Transport equation with continuous quantities")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting disc_tran example
#########################################
def plot_disc_tran(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Transport equation with discontinuous quantities")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting allen example
#########################################
def plot_allen(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Allen-Cahn equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_shear_layer(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Discontinuous initial data f")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u at final time T")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u at final time T")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Navier-Stokes equations")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_darcy(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Diffusion coefficient a")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Darcy flow equation")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting shear_layer example
#########################################
def plot_airfoil(input_tensor, output_tensor, prediction_tensor, idx):
    fig, axs = plt.subplots(4, len(idx), figsize=(16, 10))

    for i in range(4):
        for j in range(idx.shape[0]):
            if i == 0:  # input
                im = axs[i, j].imshow(input_tensor[idx[j], :, :].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Initial condition")

            elif i == 1:  # output x
                im = axs[i, j].imshow(output_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Exact solution u")

            elif i == 2:  # predicted x
                im = axs[i, j].imshow(prediction_tensor[idx[j], :, :, 0].squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Approximated solution u")

            elif i == 3:  # error x
                error = torch.abs(
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
                )
                im = axs[i, j].imshow(error.squeeze())
                fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("Error")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Compressible Euler flow past airfoils")  # title
    plt.tight_layout()
    plt.show()
    # plt.savefig("figure.png")


#########################################
# Plotting fhn example
#########################################
def plot_fhn(input_tensor, output_tensor, prediction_tensor, idx):
    n_points = input_tensor.shape[1]
    x_grid = torch.linspace(0, 100, n_points).to("cpu")

    # Plot the input, output and prediction for the selected samples
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

    plt.suptitle("FitzHugh-Nagumo equations")  # title
    plt.tight_layout()
    plt.show()

    # Plot of the phase space for the selected samples
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

    plt.suptitle("Phase space of the FitzHugh-Nagumo equations")  # title
    plt.tight_layout()
    plt.show()


#########################################
# Plotting hh example
#########################################
def plot_hh(input_tensor, output_tensor, prediction_tensor, idx):
    n_points = input_tensor.shape[1]
    x_grid = torch.linspace(0, 100, n_points).to("cpu")

    # Plot the input, output and prediction for the selected samples
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

    plt.suptitle("Hodgkin-Huxley equations")  # title
    plt.tight_layout()
    plt.show()


#########################################
# Plotting crosstruss example
#########################################
def plot_crosstruss(input_tensor, output_tensor, prediction_tensor, idx):
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
                    output_tensor[idx[j], :, :, 0] - prediction_tensor[idx[j], :, :, 0]
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
                    output_tensor[idx[j], :, :, 1] - prediction_tensor[idx[j], :, :, 1]
                )
                im = axs[i, j].imshow(error.squeeze())
                colorbar = fig.colorbar(im, ax=axs[i, j])
                if j == 0:
                    axs[i, j].set_ylabel("error y")

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel('x')

    plt.suptitle("Cross-truss structure")  # title
    plt.tight_layout()
    plt.show()
    plt.savefig("figure.png")


@jaxtyped(typechecker=beartype)
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

    match which_example:
        case "poisson":
            plot_poisson(input_tensor, output_tensor, prediction_tensor, idx)
        case "wave_0_5":
            plot_wave_0_5(input_tensor, output_tensor, prediction_tensor, idx)
        case "cont_tran":
            plot_cont_tran(input_tensor, output_tensor, prediction_tensor, idx)
        case "disc_tran":
            plot_disc_tran(input_tensor, output_tensor, prediction_tensor, idx)
        case "allen":
            plot_allen(input_tensor, output_tensor, prediction_tensor, idx)
        case "shear_layer":
            plot_shear_layer(input_tensor, output_tensor, prediction_tensor, idx)
        case "airfoil":
            plot_airfoil(input_tensor, output_tensor, prediction_tensor, idx)
        case "darcy":
            plot_darcy(input_tensor, output_tensor, prediction_tensor, idx)
        case "crosstruss":
            plot_crosstruss(input_tensor, output_tensor, prediction_tensor, idx)
        case "fhn":
            plot_fhn(input_tensor, output_tensor, prediction_tensor, idx)
        case "fhn_long":
            plot_fhn(input_tensor, output_tensor, prediction_tensor, idx)
        case "hh":
            plot_hh(input_tensor, output_tensor, prediction_tensor, idx)
        case _:
            raise ValueError(f"Unsupported example type: {which_example}.")
