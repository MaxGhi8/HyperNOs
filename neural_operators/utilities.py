"""
In this file there are some utilities functions that are used in the main file.
"""

import operator
import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from tensorboardX import SummaryWriter
from torch import Tensor


#########################################
# Function to find a file in a directory
#########################################
def find_file(file_name, search_path):
    # Set the directory to start the search, for example 'C:\' on Windows or '/' on Unix-based systems.
    # Walk through all directories and files in the search_path.
    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            print(
                f"File {file_name} found in {root}"
            )  # print the path where the file was found
            return os.path.join(root, file_name)  # full path
    raise FileNotFoundError(f"File {file_name} not found in {search_path}")


#########################################
# Function to count the number of parameters
#########################################
def count_params(model):
    """Count the number of parameters in a model."""

    par_tot = 0
    bytes_tot = 0
    for par in model.parameters():
        # print(par.shape)
        tmp = reduce(
            operator.mul, list(par.shape + (2,) if par.is_complex() else par.shape)
        )
        par_tot += tmp
        bytes_tot += tmp * par.data.element_size()

    return par_tot, bytes_tot


#########################################
# initial normalization
#########################################
class UnitGaussianNormalizer(object):
    """
    Initial normalization is the point-wise gaussian normalization over the tensor x
    dimension: (n_samples)*(nx)*(ny)
    """

    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x, 0).to(x.device)
        self.std = torch.std(x, 0).to(x.device)
        self.eps = torch.tensor(eps).to(x.device)

    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[Tensor, "n_samples *n"]) -> Float[Tensor, "n_samples *n"]:
        x = (x - self.mean) / (self.std + self.eps)
        return x

    @jaxtyped(typechecker=beartype)
    def decode(self, x: Float[Tensor, "n_samples *n"]) -> Float[Tensor, "n_samples *n"]:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        eps = self.eps.to(x.device)
        x = x * (std + eps) + mean
        return x


#########################################
# Fourier features
#########################################
class FourierFeatures(nn.Module):
    """
    Class to compute the Fourier features.
    """

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.B = scale * torch.randn((self.mapping_size, 2)).to(device)

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nx, ny, 2)
        if self.scale != 0:
            x_proj = torch.matmul((2.0 * np.pi * x), self.B.T)
            inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return inp
        else:
            return x


#########################################
# Function to plot a generic 1d data
#########################################
def plot_data_generic_1d(
    data_plot: Tensor,
    t_final: float,
    title: str,
    y_label: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    n_idx = data_plot.size(0)

    n_points = data_plot.shape[1]
    x_grid = torch.linspace(0, t_final, n_points).to("cpu")

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 4))
    fig.suptitle(title)
    ax[0].set(ylabel=y_label)
    for i in range(n_idx):
        ax[i].set(xlabel="x")
        if "error" in title.lower():
            ax[i].semilogy(
                x_grid,
                data_plot[i, :].squeeze(),
            )
        else:
            ax[i].plot(
                x_grid,
                data_plot[i, :].squeeze(),
            )
        ax[i].grid()

    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)


#########################################
# Function to plot the phase space of 1D data
#########################################
def plot_data_phield_space(
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    n_idx = data_plot.size(0)

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 4))
    fig.suptitle(title)
    for i in range(n_idx):
        ax[i].set(xlabel="V(t)", ylabel="w(t)")
        ax[i].plot(
            data_plot[i, :, 0].squeeze(),
            data_plot[i, :, 1].squeeze(),
        )
        ax[i].grid()

    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)


#########################################
# Function to plot the data of the FHN example
#########################################
def plot_data_fhn_input(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    # Denormalize the data
    if normalization:
        data_plot = example.a_normalizer.decode(data_plot)
    # Plot the data
    plot_data_generic_1d(data_plot, 100, title, "I(t)", ep, writer, plotting)


def plot_data_fhn(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot[:, :, [0]] = example.v_normalizer.decode(data_plot[:, :, [0]])
        data_plot[:, :, [1]] = example.w_normalizer.decode(data_plot[:, :, [1]])

        # phase-phield space (not for the error)
        plot_data_phield_space(data_plot, title + " phase space", ep, writer, plotting)

    # Plot the data
    plot_data_generic_1d(
        data_plot[..., 0],
        100,
        title + " V(t)",
        "V(t)",
        ep,
        writer,
        plotting,
    )
    plot_data_generic_1d(
        data_plot[..., 1],
        100,
        title + " w(t)",
        "w(t)",
        ep,
        writer,
        plotting,
    )


#########################################
# Function to plot the data of the HH example
#########################################
def plot_data_hh_input(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    # Denormalize the data
    if normalization:
        data_plot = example.a_normalizer.decode(data_plot)
    # Plot the data
    plot_data_generic_1d(data_plot, 100, title, "I(t)", ep, writer, plotting)


def plot_data_hh(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot[:, :, [0]] = example.v_normalizer.decode(data_plot[:, :, [0]])
        data_plot[:, :, [1]] = example.w_normalizer.decode(data_plot[:, :, [1]])

        # phase-phield space (not for the error)
        plot_data_phield_space(data_plot, title + " phase space", ep, writer, plotting)

    # Plot the data
    plot_data_generic_1d(
        data_plot[..., 0],
        100,
        title + " V(t)",
        "V(t)",
        ep,
        writer,
        plotting,
    )
    plot_data_generic_1d(
        data_plot[..., 1],
        100,
        title + " w(t)",
        "w(t)",
        ep,
        writer,
        plotting,
    )


#########################################
# Function to plot a generic 2d data
#########################################
def plot_data_generic_2d(
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    # select the data to plot
    n_idx = data_plot.size(0)

    # plot
    fig, ax = plt.subplots(1, n_idx, figsize=(18, 4))
    fig.suptitle(title)
    ax[0].set(ylabel="y")
    for i in range(n_idx):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set(xlabel="x")
        im = ax[i].imshow(data_plot[i])
        fig.colorbar(im, ax=ax[i])
    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)


#########################################
# Function to plot cross-truss example
#########################################
def plot_data_crosstruss_input(
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    n_idx = data_plot.size(0)

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 5))
    fig.suptitle(title)
    ax[0].set(ylabel="Geometry domain")
    for i in range(n_idx):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set(xlabel="x")
        im = ax[i].imshow(data_plot[i].squeeze())
        fig.colorbar(im, ax=ax[i])

    if plotting:
        plt.show()
    writer.add_figure(title, fig, ep)


def plot_data_crosstruss(
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    n_idx = data_plot.size(0)

    fig, ax = plt.subplots(2, n_idx, figsize=(18, 10))
    fig.suptitle(title)
    ax[0, 0].set(ylabel="Displacement x")
    ax[1, 0].set(ylabel="Displacement y")
    for i in range(2):
        for j in range(n_idx):
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])
            ax[i, j].set(xlabel="x")
            im = ax[i, j].imshow(data_plot[j, ..., i].squeeze())
            fig.colorbar(im, ax=ax[i, j])

    if plotting:
        plt.show()
    writer.add_figure(title, fig, ep)


def plot_data(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    which_example: str,
    plotting: bool = False,
):
    """
    Function to makes the plots of the data.

    data_plot: torch.tensor
        data_plot is a tensor of shape (n_samples, n_patch, *n).

    title: str
        title is the title of the plot.

    ep: int
        ep is the epoch number.

    writer: tensorboardX.SummaryWriter
        writer is the tensorboard writer.

    which_example: str
        which_example is the name of the example.

    problem_dim: int
        problem_dim is the dimension of the problem.

    plotting: bool (default=False)
        plotting is a boolean to decide if the plot is shown or not.
    """
    ## 1D problem
    match which_example:

        case "fhn_long":
            pass
            # TODO

        case "fhn":
            if "input" in title.lower():
                data_plot = example.a_normalizer.decode(data_plot)
                plot_data_fhn_input(
                    example,
                    data_plot,
                    title,
                    ep,
                    writer,
                    True,
                    plotting,
                )
            else:
                # Denormalize the data
                if "error" not in title.lower():
                    data_plot[:, :, [0]] = example.v_normalizer.decode(
                        data_plot[:, :, [0]]
                    )
                    data_plot[:, :, [1]] = example.w_normalizer.decode(
                        data_plot[:, :, [1]]
                    )
                    # Plot the phase space (not for the error)
                    plot_data_phield_space(
                        data_plot, title + " phase space", ep, writer, plotting
                    )

                # Plot the data
                plot_data_generic_1d(
                    data_plot[..., 0],
                    100,
                    title + " V(t)",
                    "V(t)",
                    ep,
                    writer,
                    plotting,
                )
                plot_data_generic_1d(
                    data_plot[..., 1],
                    100,
                    title + " w(t)",
                    "w(t)",
                    ep,
                    writer,
                    plotting,
                )

        case "hh":
            if "input" in title.lower():
                # Denormalize the data
                data_plot = example.a_normalizer.decode(data_plot)
                # Plot the data
                plot_data_generic_1d(
                    data_plot, 100, title, "I(t)", ep, writer, plotting
                )
            else:
                # Denormalize the data
                if "error" not in title.lower():
                    data_plot[:, :, [0]] = example.v_normalizer.decode(
                        data_plot[:, :, [0]]
                    )
                    data_plot[:, :, [1]] = example.m_normalizer.decode(
                        data_plot[:, :, [1]]
                    )
                    data_plot[:, :, [2]] = example.h_normalizer.decode(
                        data_plot[:, :, [2]]
                    )
                    data_plot[:, :, [3]] = example.n_normalizer.decode(
                        data_plot[:, :, [3]]
                    )

                # Plot the data
                plot_data_generic_1d(
                    data_plot[..., 0],
                    100,
                    title + " V(t)",
                    "V(t)",
                    ep,
                    writer,
                    plotting,
                )
                plot_data_generic_1d(
                    data_plot[..., 1],
                    100,
                    title + " m(t)",
                    "m(t)",
                    ep,
                    writer,
                    plotting,
                )
                plot_data_generic_1d(
                    data_plot[..., 2],
                    100,
                    title + " h(t)",
                    "h(t)",
                    ep,
                    writer,
                    plotting,
                )
                plot_data_generic_1d(
                    data_plot[..., 3],
                    100,
                    title + " n(t)",
                    "n(t)",
                    ep,
                    writer,
                    plotting,
                )

    ## 2D problem
    if which_example == "crosstruss":
        if "input" in title.lower():
            plot_data_crosstruss_input(data_plot, title, ep, writer, plotting)
        else:
            if "error" not in title.lower():
                data_plot[:, :, :, 0] = (example.max_x - example.min_x) * data_plot[
                    :, :, :, 0
                ] + example.min_x
                data_plot[:, :, :, 1] = (example.max_y - example.min_y) * data_plot[
                    :, :, :, 1
                ] + example.min_y
            plot_data_crosstruss(data_plot, title, ep, writer, plotting)

    elif which_example in [
        "poisson",
        "wave_0_5",
        "cont_tran",
        "disc_tran",
        "allen",
        "shear_layer",
        "airfoil",
        "darcy",
    ]:
        # Denormalize the data
        if "input" in title.lower():
            data_plot = (
                example.max_data - example.min_data
            ) * data_plot + example.min_data
        elif "error" not in title.lower():
            data_plot = (
                example.max_model - example.min_model
            ) * data_plot + example.min_model

        # Plot the data
        plot_data_generic_2d(data_plot, title, ep, writer, plotting)

    elif which_example in [
        "burgers_zongyi",
        "darcy_zongyi",
        "navier_stokes_zongyi",
    ]:
        pass
        # TODO
