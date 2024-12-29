"""
In this file there are some utilities functions that are used in the main file.
"""

import json
import sys
sys.path.append("..")

from utilities import find_file


#########################################
# function to load the hyperparameters
#########################################
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
