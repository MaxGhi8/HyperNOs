"""
This is the main file for training the Neural Operator with the FNO architecture for different examples.

"which_example" can be one of the following options:
    poisson             : Poisson equation 
    wave_0_5            : Wave equation 
    cont_tran           : Smooth Transport 
    disc_tran           : Discontinuous Transport
    allen               : Allen-Cahn equation # training_sample = 512
    shear_layer         : Navier-Stokes equations # training_sample = 512
    airfoil             : Compressible Euler equations 
    darcy               : Darcy equation

    burgers_zongyi      : Burgers equation
    darcy_zongyi        : Darcy equation
    navier_stokes_zongyi: Navier-Stokes equations

    fhn                 : FitzHugh-Nagumo equations in [0, 100]
    hh                  : Hodgkin-Huxley equations

    crosstruss          : Cross-shaped truss structure

"arc" can be one of the following options:
    FNO : Fourier Neural Operator
    CNO : Convolutional Neural Operator

"loss_fn_str" can be one of the following options:
    L1 : L^1 relative norm
    L2 : L^2 relative norm
    H1 : H^1 relative norm
    L1_smooth : L^1 smooth loss (Mishra)
    MSE : L^2 smooth loss (Mishra)
"""

import argparse
import sys

sys.path.append("..")

import torch
from examples.train_cno import train_cno
from examples.train_fno import train_fno

#########################################
# default values
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# torch.set_default_dtype(torch.float32) # default tensor dtype


#########################################
# Choose the example to run
#########################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a specific example with the desired model and training configuration."
    )

    parser.add_argument(
        "example",
        type=str,
        choices=[
            "poisson",
            "wave_0_5",
            "cont_tran",
            "disc_tran",
            "allen",
            "shear_layer",
            "airfoil",
            "darcy",
            "burgers_zongyi",
            "darcy_zongyi",
            "fhn",
            "hh",
            "crosstruss",
        ],
        help="Select the example to run.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=["fno", "cno"],
        help="Select the architecture to use.",
    )
    parser.add_argument(
        "loss_fn_str",
        type=str,
        choices=["l1", "l2", "h1", "l1_smooth"],
        help="Select the relative loss function to use during the training process.",
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["best", "default"],
        help="Select the hyper-params to use for define the architecture and the training, we have implemented the 'best' and 'default' options.",
    )

    args = parser.parse_args()

    return {
        "example": args.example.lower(),
        "architecture": args.architecture.upper(),
        "loss_fn_str": args.loss_fn_str.upper(),
        "mode": args.mode.lower(),
    }


config = parse_arguments()
which_example = config["example"]
arc = config["architecture"]
loss_fn_str = config["loss_fn_str"]
mode_str = config["mode"]

# Run the chosed example
match arc:
    case "FNO":
        train_fno(which_example, mode_str, loss_fn_str)
    case "CNO":
        train_cno(which_example, mode_str, loss_fn_str)
    case _:
        raise ValueError("This architecture is not allowed")
