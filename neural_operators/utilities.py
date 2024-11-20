"""
In this file there are some utilities functions that are used in the main file.
"""
import os
import json
from functools import reduce
import operator
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn

from FNObenchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer, Darcy
from FNObenchmarks import Darcy_Zongyi, Burgers_Zongyi
from FNObenchmarks import FitzHughNagumo, HodgkinHuxley
from FNObenchmarks import CrossTruss

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

        case "burgers_zongyi":
            example = Burgers_Zongyi(fno_architecture, batch_size, search_path=search_path)
            fno_architecture["problem_dim"] = 1
        case "darcy_zongyi":
            example = Darcy_Zongyi(fno_architecture, batch_size, search_path=search_path)
        case "navier_stokes_zongyi":
            pass # TODO

        case "fhn":
            time = "_tf_100"
            example = FitzHughNagumo(time, fno_architecture, batch_size, search_path=search_path)
            fno_architecture["problem_dim"] = 1
        case "fhn_long":
            time = "_tf_200"
            example = FitzHughNagumo(time, fno_architecture, batch_size, search_path=search_path)
            fno_architecture["problem_dim"] = 1
        case "hh":
            example = HodgkinHuxley(fno_architecture, batch_size, search_path=search_path)
            fno_architecture["problem_dim"] = 1

        case "crosstruss":
            example = CrossTruss(fno_architecture, batch_size, search_path=search_path)

        case _:
            raise ValueError("the variable which_example is typed wrong")

    return example


#########################################
# function to count the number of parameters
#########################################
def count_params(model):
    """ Count the number of parameters in a model. """
    par_tot = 0
    for par in model.parameters():
        # print(par.shape)
        par_tot += reduce(operator.mul, list(par.shape + (2,) if par.is_complex() else par.shape))
    return par_tot

#########################################
# function to plot the data
#########################################
def plot_data(data_plot:Tensor, idx:list, title:str, ep:int, writer, plotting:bool = True):
    """ 
    Function to makes the plots of the data.
    
    data_plot: torch.tensor
        data_plot is a tensor of shape (n_samples, n_patch, *n).
    """  
    # select the data to plot
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
        n_idx = data_plot.size(0)
    # plot
    fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    fig.suptitle(title)
    ax[0].set(ylabel = 'y')
    for i in range(n_idx):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set(xlabel = 'x')
        im = ax[i].imshow(data_plot[i])
        fig.colorbar(im, ax = ax[i])
    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)

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
            x_proj = torch.matmul((2. * np.pi * x), self.B.T)
            inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return inp
        else:
            return x

#########################################
# function to load the hyperparameters
#########################################
def FNO_initialize_hyperparameters(which_example:str, mode:str):
    """
    Function to initialize the hyperparameters in according to the best 
    results obtained in the paper of Mishra on CNOs, by loading them from external JSON files.

    which_example: str
        The name of the example to load the hyperparameters for.
    mode: str
        The mode to use to load the hyperparameters (this can be either 'best' or 'default').
    """
    # Here I use relative path
    config_directory = "./configurations"
    config_path = os.path.join(config_directory, f"{mode}_{which_example}.json")
    print("The default hyperparameters are loaded from:", config_path)

    # Check if the configuration file exists
    if not os.path.exists(config_path):
        raise ValueError(f"The configuration file for '{which_example}' does not exist.")
    
    # Load the configuration from the JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract the training properties and FNO architecture from the loaded configuration
    training_properties = config["training_properties"]
    fno_architecture = config["fno_architecture"]
    fno_architecture["weights_norm"] = "Xavier" if fno_architecture["fun_act"] == 'gelu' else "Kaiming"
    
    return training_properties, fno_architecture