"""
In this file there are some utilities functions that are used in the main file.
"""

import json
import sys

sys.path.append("..")

from utilities import find_file


def count_params_fno(config, accurate=True):
    """
    function to approximate the number of parameters for the FNO model and classical architecture
    """
    latent = 128
    P_Q = (
        config["in_dim"] + 2 * config["width"] + config["out_dim"] + 2
    ) * latent + config["width"] * config["out_dim"]

    hidden = (
        config["n_layers"]
        * (config["width"] ** 2)
        * config["modes"] ** config["problem_dim"]
        * 2 ** config["problem_dim"]
    )

    if accurate:
        return (
            hidden + P_Q + config["n_layers"] * (config["width"] ** 2 + config["width"])
        )
    else:
        return hidden


def compute_modes(total_param, maximum, config):
    modes = min(
        max(
            int(
                (
                    total_param
                    / (
                        2 ** config["problem_dim"]
                        * config["n_layers"]
                        * config["width"] ** 2
                    )
                )
                ** (1 / config["problem_dim"])
            ),
            1,
        ),
        maximum,
    )

    return modes


def FNO_initialize_hyperparameters(which_example: str, mode: str):
    """
    Function to initialize the hyperparameters in according to the best
    results obtained in the paper of Mishra on CNOs, by loading them from external JSON files.

    which_example: str
        The name of the example to load the hyperparameters for.
    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_directory = "../FNO/configurations/"
    config_path = find_file(f"{mode}_{which_example}.json", config_directory)

    # Load the configuration from the JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract the training properties and FNO architecture from the loaded configuration
    training_properties = config["training_properties"]
    fno_architecture = config["fno_architecture"]
    fno_architecture["weights_norm"] = (
        "Xavier" if fno_architecture["fun_act"] == "gelu" else "Kaiming"
    )

    return training_properties, fno_architecture
