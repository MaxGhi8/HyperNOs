"""
In this file there are some utilities functions that are used in the main file.
"""

import json
import operator
import os
import pathlib
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
def find_file(file_path, search_path):
    # Extract the filename and path components
    file_name = os.path.basename(file_path)
    path_components = pathlib.Path(file_path).parts[:-1]  # All except filename

    # Walk through all directories and files in the search_path
    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            # Get the full path of the found file
            full_path = os.path.join(root, file_name)

            # Check if the path structure matches what we're looking for
            relative_path = os.path.relpath(full_path, search_path)
            relative_parts = pathlib.Path(relative_path).parts

            # If no specific path structure was requested, or if the path matches our pattern
            if not path_components or all(
                pc in relative_parts for pc in path_components
            ):
                print(f"📂 File {file_name} found in {root}")
                return full_path

    raise FileNotFoundError(f"File {file_path} not found in {search_path}")


##########################################
# Function to initialize the hyperparameters from JSON
##########################################
def initialize_hyperparameters(arc: str, which_example: str, mode: str):
    """
    Function to initialize the hyperparameters of an architecture loading a JSON.

    Parameters
    ----------
    arc: str
        The architecture to load the hyperparameters for.

    which_example: str
        The name of the example to load the hyperparameters for.

    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_path = find_file(
        f"{arc}/configurations/{mode}_{which_example}.json",
        os.path.dirname("../HyperNOs"),
    )

    # Load the configuration from the JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract the training properties and FNO architecture from the loaded configuration
    training_properties = config["training_properties"]
    no_architecture = config[f"{arc.lower()}_architecture"]

    if arc == "FNO":
        no_architecture["weights_norm"] = (
            "Xavier" if no_architecture["fun_act"] == "gelu" else "Kaiming"
        )

    return training_properties, no_architecture


#########################################
# Function to count the number of parameters
#########################################
def count_params(model):
    """Count the number of parameters in a model."""

    par_tot = 0
    bytes_tot = 0
    for par in model.parameters():
        tmp = reduce(
            operator.mul, list(par.shape + (2,) if par.is_complex() else par.shape)
        )
        par_tot += tmp
        bytes_tot += tmp * par.data.element_size()

    return par_tot, bytes_tot


def count_weight_params(model):
    """Count the number of parameters in a model, excluding biases."""
    par_tot = 0
    bytes_tot = 0

    for name, par in model.named_parameters():
        if "weight" in name:
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
        x = (x - self.mean.to(x.device)) / (
            self.std.to(x.device) + self.eps.to(x.device)
        )
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
class FourierFeatures1D(nn.Module):
    """
    Class to compute the Fourier features for 1D inputs.
    """

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size  # Number of Fourier features
        self.scale = scale  # Scaling factor for the random projection
        self.B = scale * torch.randn((self.mapping_size, 1)).to(
            device
        )  # Random projection matrix for 1D

    def forward(self, x):
        """
        Forward pass for 1D Fourier features.

        Args:
            x (torch.Tensor): Input coordinates of shape (num_points, 1) or (num_points,).

        Returns:
            torch.Tensor: Fourier features of shape (num_points, 2 * mapping_size).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Ensure x is 2D: (num_points, 1)

        if self.scale != 0:
            # Project input coordinates using the random matrix B
            x_proj = torch.matmul(
                (2.0 * np.pi * x), self.B.T
            )  # Shape: (num_points, mapping_size)
            # Concatenate sine and cosine of the projected values
            inp = torch.cat(
                [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
            )  # Shape: (num_points, 2 * mapping_size)
            return inp
        else:
            # If scale is 0, return the original input
            return x


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
        data_plot[:, :, [1]] = example.m_normalizer.decode(data_plot[:, :, [1]])
        data_plot[:, :, [2]] = example.h_normalizer.decode(data_plot[:, :, [2]])
        data_plot[:, :, [3]] = example.n_normalizer.decode(data_plot[:, :, [3]])

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
# Function to plot the data of the OHara-Rudy example
#########################################
def plot_data_ord_input(
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
        data_plot = example.dict_normalizers["I_app_dataset"].decode(data_plot)
    # Plot the data
    plot_data_generic_1d(
        data_plot, 2000, title, "I_app_dataset(t)", ep, writer, plotting
    )


def plot_data_ord(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        for i in range(data_plot.size(-1)):
            data_plot[:, :, [i]] = example.dict_normalizers[
                example.fields_to_concat[i]
            ].decode(data_plot[:, :, [i]])

        # phase-phield space (not for the error)
        plot_data_phield_space(data_plot, title + " phase space", ep, writer, plotting)

    # Plot the data
    for i in range(data_plot.size(-1)):
        plot_data_generic_1d(
            data_plot[..., i],
            2000,
            title + f" {example.fields_to_concat[i]}(t)",
            f"{example.fields_to_concat[i]}(t)",
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
# Function to plot the data for all the Mishra's example
#########################################
def plot_data_mishra_input(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot = (example.max_data - example.min_data) * data_plot + example.min_data

    plot_data_generic_2d(data_plot, title, ep, writer, plotting)


def plot_data_mishra(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot = (
            example.max_model - example.min_model
        ) * data_plot + example.min_model

    plot_data_generic_2d(data_plot, title, ep, writer, plotting)


#########################################
# Function to plot cross-truss example
#########################################
def plot_data_crosstruss_input(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        pass  # no normalization needed for the input of the crosstruss example

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
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot[:, :, :, 0] = (example.max_x - example.min_x) * data_plot[
            :, :, :, 0
        ] + example.min_x
        data_plot[:, :, :, 1] = (example.max_y - example.min_y) * data_plot[
            :, :, :, 1
        ] + example.min_y

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


#########################################
# Function to plot stiff matrix example
#########################################
def plot_data_stiffness_matrix(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    normalization: bool = True,
    plotting: bool = False,
):
    if normalization:
        data_plot[:, :, :, 0] = (example.max_x - example.min_x) * data_plot[
            :, :, :, 0
        ] + example.min_x

    n_idx = data_plot.size(0)

    fig, ax = plt.subplots(1, n_idx, figsize=(18, 10))
    fig.suptitle(title)
    ax[0].set(ylabel="Stiffness matrix")
    for j in range(n_idx):
        ax[j].set_yticklabels([])
        ax[j].set_xticklabels([])
        ax[j].set(xlabel="x")
        im = ax[j].imshow(data_plot[j, ..., 0].squeeze())
        fig.colorbar(im, ax=ax[j])

    if plotting:
        plt.show()
    writer.add_figure(title, fig, ep)


def get_plot_function(
    which_example: str,
    title: str,
):
    ## 1D problem
    match which_example:
        case "fhn":
            if "input" in title.lower():
                return plot_data_fhn_input
            return plot_data_fhn

        case "hh":
            if "input" in title.lower():
                return plot_data_hh_input
            return plot_data_hh

        case "ord":
            if "input" in title.lower():
                return plot_data_ord_input
            return plot_data_ord

        case "crosstruss":
            if "input" in title.lower():
                return plot_data_crosstruss_input
            return plot_data_crosstruss

        case "stiffness_matrix":
            if "input" in title.lower():
                return plot_data_crosstruss_input
            return plot_data_stiffness_matrix

    if which_example in [
        "poisson",
        "wave_0_5",
        "cont_tran",
        "disc_tran",
        "allen",
        "shear_layer",
        "airfoil",
        "darcy",
    ]:
        if "input" in title.lower():
            return plot_data_mishra_input
        return plot_data_mishra

    elif which_example in [
        "burgers_zongyi",
        "darcy_zongyi",
        "navier_stokes_zongyi",
    ]:
        return None  # TODO

    return None


def plot_data(
    example,
    data_plot: Tensor,
    title: str,
    ep: int,
    writer: SummaryWriter,
    which_example: str,
    plotting: bool = False,
):
    plot_func = get_plot_function(which_example, title)
    if plot_func is not None:
        plot_func(
            example,
            data_plot,
            title,
            ep,
            writer,
            True,
            plotting,
        )
    else:
        print(f"Plot function for {which_example} not found.")
