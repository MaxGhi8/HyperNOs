"""
In this file there are some utilities functions that are used in the main file.
"""

import math

import torch
import torch.nn as nn


def count_params_bampno(config, accurate=True):
    """
    function to approximate the number of parameters for the BAMPNO model and classical architecture
    """
    latent = 128
    bias = 0 if config["zero_BC"] is not None else 1
    P_Q = (
        config["in_dim"] + 2 * config["d_v"] + config["out_dim"] + 2 * bias
    ) * latent + config["d_v"] * config["out_dim"] * bias

    hidden = (
        config["L"]
        * (config["d_v"] ** 2)
        * (config["modes"] ** config["problem_dim"])
        * config["n_patch"]
    )

    if accurate:
        return hidden + P_Q + config["L"] * (config["d_v"] ** 2 + config["d_v"] * bias)
    else:
        return hidden


def compute_modes(total_param, maximum, config):
    modes = min(
        max(
            int(
                (total_param / (config["L"] * (config["d_v"] ** 2) * config["n_patch"]))
                ** (1 / config["problem_dim"])
            ),
            1,
        ),
        maximum,
    )

    return modes
