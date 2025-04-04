"""
In this file there are some utilities functions that are used in the main file.
"""

import json
import sys

sys.path.append("..")

from utilities import find_file


def count_params_resnet(config, accurate=True):
    """
    function to approximate the number of parameters for the ResNet model and classical architecture
    """
    pass


def compute_modes(total_param, maximum, config):
    pass


def ResNet_initialize_hyperparameters(which_example: str, mode: str):
    """
    which_example: str
        The name of the example to load the hyperparameters for.
    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_directory = "../ResNet/configurations/"
    config_path = find_file(f"{mode}_{which_example}.json", config_directory)

    # Load the configuration from the JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract the training properties and architecture structure from the loaded configuration
    training_properties = config["training_properties"]
    fno_architecture = config["resnet_architecture"]

    return training_properties, fno_architecture
