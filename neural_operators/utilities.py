"""
In this file there are some utilities functions that are used in the main file.
"""

import os
from functools import reduce
import operator
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from jaxtyping import jaxtyped, Float
from beartype import beartype


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
# Function to plot the data
#########################################
def plot_data_crosstruss(
    data_plot: Tensor,
    idx: list,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
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


def plot_data_generic_2d(
    data_plot: Tensor,
    idx: list,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    # select the data to plot
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
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


def plot_data_generic_1d(
    data_plot: Tensor,
    idx: list,
    t_final: float,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
        n_idx = data_plot.size(0)

    n_points = data_plot.shape[1]
    x_grid = torch.linspace(0, t_final, n_points).to("cpu")

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 4))
    fig.suptitle(title)
    for i in range(n_idx):
        ax[i].set(xlabel="x")
        ax[i].plot(
            x_grid,
            data_plot[i, :, 0].squeeze(),
        )

    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)


def plot_data_phield_space(
    data_plot: Tensor,
    idx: list,
    title: str,
    ep: int,
    writer: SummaryWriter,
    plotting: bool = False,
):
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
        n_idx = data_plot.size(0)

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 4))
    fig.suptitle(title)
    for i in range(n_idx):
        ax[i].set(xlabel="x")
        ax[i].plot(
            data_plot[i, :, 0].squeeze(),
            data_plot[i, :, 1].squeeze(),
        )

    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)


def plot_data(
    data_plot: Tensor,
    idx: list,
    title: str,
    ep: int,
    writer: SummaryWriter,
    which_example: str,
    problem_dim: int,
    plotting: bool = False,
):
    """
    Function to makes the plots of the data.

    data_plot: torch.tensor
        data_plot is a tensor of shape (n_samples, n_patch, *n).

    idx: list
        idx is a list of indices to plot.

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
    if problem_dim == 1:
        match which_example:
            case "fhn" | "fhn_long":
                if "input" in title:
                    plot_data_generic_1d(
                        data_plot, idx, 100, title, ep, writer, plotting
                    )
                else:
                    plot_data_generic_1d(
                        data_plot[..., 0],
                        idx,
                        100,
                        title + " V(t)",
                        ep,
                        writer,
                        plotting,
                    )
                    plot_data_generic_1d(
                        data_plot[..., 1],
                        idx,
                        100,
                        title + " w(t)",
                        ep,
                        writer,
                        plotting,
                    )
                    plot_data_phield_space(
                        data_plot, idx, title + " phase space", ep, writer, plotting
                    )

            case "hh":
                if "input" in title:
                    plot_data_generic_1d(
                        data_plot, idx, 100, title, ep, writer, plotting
                    )
                else:
                    plot_data_generic_1d(
                        data_plot[..., 0],
                        idx,
                        100,
                        title + " V(t)",
                        ep,
                        writer,
                        plotting,
                    )
                    plot_data_generic_1d(
                        data_plot[..., 1],
                        idx,
                        100,
                        title + " m(t)",
                        ep,
                        writer,
                        plotting,
                    )
                    plot_data_generic_1d(
                        data_plot[..., 2],
                        idx,
                        100,
                        title + " h(t)",
                        ep,
                        writer,
                        plotting,
                    )
                    plot_data_generic_1d(
                        data_plot[..., 3],
                        idx,
                        100,
                        title + " n(t)",
                        ep,
                        writer,
                        plotting,
                    )

    ## 2D problem
    elif problem_dim == 2:
        if which_example == "crosstruss":
            plot_data_crosstruss(data_plot, idx, title, ep, writer, plotting)

        else:
            plot_data_generic_2d(data_plot, idx, title, ep, writer, plotting)


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
