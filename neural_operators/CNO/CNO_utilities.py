"""
In this file there are some utilities functions that are used in the main file.
"""

import json
import sys

sys.path.append("..")
from utilities import find_file


def count_params_cno(config):
    """
    Function to approximate the number of parameters in the CNO model.
    """
    latent = 64
    P_Q = (
        config["kernel_size"] ** config["problem_dim"]
        * latent
        * (
            config["in_dim"]
            + config["out_dim"]
            + (3 / 2) * config["channel_multiplier"]
        )
    )
    pow4 = 4 ** (config["N_layers"] - 1)
    sq = (pow4 - 1) / (4 - 1)
    hidden = (
        config["kernel_size"] ** config["problem_dim"]
        * config["channel_multiplier"] ** 2
        * (
            pow4 * 2 * config["N_res_neck"]
            + 2 * config["N_res"] * (1 / 4 + sq)
            + (31 / 6) * pow4
            - 11 / 12
        )
    )
    return hidden + P_Q


def CNO_initialize_hyperparameters(which_example: str, mode: str):
    """
    Function to initialize the hyperparameters in according to the best
    results obtained in the paper of Mishra on CNOs, by loading them from external JSON files.

    which_example: str
        The name of the example to load the hyperparameters for.
    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_directory = "../CNO/configurations/"
    config_path = find_file(f"{mode}_{which_example}.json", config_directory)

    # Load the configuration from the JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract the training properties and FNO architecture from the loaded configuration
    training_properties = config["training_properties"]
    cno_architecture = config["cno_architecture"]

    return training_properties, cno_architecture
