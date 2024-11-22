"""
In this file there are some utilities functions that are used in the main file.
"""
import os
import json

from utilities import find_file
from CNO_benchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer, Darcy
# from CNO_benchmarks import Darcy_Zongyi, Burgers_Zongyi # todo
# from CNO_benchmarks import FitzHughNagumo, HodgkinHuxley # todo
# from CNO_benchmarks import CrossTruss # todo

#########################################
# function to load the data and model
#########################################
def load_data_model(which_example:str, fno_architecture, device, batch_size, training_samples, in_dist, search_path:str='/'):
    """
    Function to load the data and the model.
    """
    match which_example:
        case "shear_layer":
            example = ShearLayer(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "poisson":
            example = SinFrequency(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "wave_0_5":
            example = WaveEquation(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "allen":
            example = AllenCahn(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "cont_tran":
            example = ContTranslation(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "disc_tran":
            example = DiscContTranslation(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "airfoil":
            example = Airfoil(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)
        case "darcy":
            example = Darcy(fno_architecture, device, batch_size, training_samples, in_dist=in_dist, search_path=search_path)

        # todo
        # case "burgers_zongyi":
        #     example = Burgers_Zongyi(fno_architecture, batch_size, search_path=search_path)
        #     fno_architecture["problem_dim"] = 1
        # case "darcy_zongyi":
        #     example = Darcy_Zongyi(fno_architecture, batch_size, search_path=search_path)
        # case "navier_stokes_zongyi":
        #     pass # TODO

        # todo
        # case "fhn":
        #     time = "_tf_100"
        #     example = FitzHughNagumo(time, fno_architecture, batch_size, search_path=search_path)
        #     fno_architecture["problem_dim"] = 1
        # case "fhn_long":
        #     time = "_tf_200"
        #     example = FitzHughNagumo(time, fno_architecture, batch_size, search_path=search_path)
        #     fno_architecture["problem_dim"] = 1
        # case "hh":
        #     example = HodgkinHuxley(fno_architecture, batch_size, search_path=search_path)
        #     fno_architecture["problem_dim"] = 1

        # case "crosstruss":
        #     example = CrossTruss(fno_architecture, batch_size, search_path=search_path)

        case _:
            raise ValueError("the variable which_example is typed wrong")

    return example

#########################################
# function to load the hyperparameters
#########################################
def CNO_initialize_hyperparameters(which_example:str, mode:str):
    """
    Function to initialize the hyperparameters in according to the best 
    results obtained in the paper of Mishra on CNOs, by loading them from external JSON files.

    which_example: str
        The name of the example to load the hyperparameters for.
    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_directory = "./configurations/"
    config_path = find_file(f"{mode}_{which_example}.json", config_directory)

    # Load the configuration from the JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract the training properties and FNO architecture from the loaded configuration
    training_properties = config["training_properties"]
    cno_architecture = config["cno_architecture"]
    
    return training_properties, cno_architecture