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
    fhn_long            : FitzHugh-Nagumo equations in [0, 200]
    hh                  : Hodgkin-Huxley equations

    crosstruss          : Cross-shaped truss structure

"exp_norm" can be one of the following options:
    L1 : L^1 relative norm
    L2 : L^2 relative norm
    H1 : H^1 relative norm
    L1_smooth : L^1 smooth loss (Mishra)
    MSE : L^2 smooth loss (Mishra)
"""

import torch
from tensorboardX import SummaryWriter
import os
import json
from tqdm import tqdm
import sys 

from Loss_fun import LprelLoss, H1relLoss_1D, H1relLoss
from train_fun import train_fun, test_fun
from FNO.FNO_arc import FNO_1D, FNO_2D
from utilities import count_params, plot_data
from FNO.FNO_utilities import FNO_initialize_hyperparameters, load_data_model

#########################################
# default values
#########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
# torch.set_default_dtype(torch.float32) # default tensor dtype
mode_str = "best" # test base hyperparameters, can be "default" or "best"

#########################################
# Choose the example to run
#########################################
if len(sys.argv) == 1:
    raise ValueError("The user must choose the example to run")
elif len(sys.argv) == 2:
    which_example = sys.argv[1] # the example can be chosen by the user as second argument
    exp_norm = "L1"
    in_dist = True
elif len(sys.argv) == 3:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2]
    in_dist = True
elif len(sys.argv) == 4:
    which_example = sys.argv[1]
    exp_norm = sys.argv[2]
    in_dist = sys.argv[3]
else:
    raise ValueError("The user must choose the example to run")

Norm_dict = {"L1":0, "L2":1, "H1":2, "L1_smooth":3, "MSE":4}

#########################################
# parameters for save the model
#########################################
arc = "FNO"
model_folder = f"./{arc}/TrainedModels/"
description_test = "test_" + exp_norm
folder = model_folder + which_example + "/exp_FNO_" + description_test + "_" + mode_str + "_hyperparams"
name_model = model_folder + which_example + "/model_FNO_" + description_test + "_" + mode_str + "_hyperparams"

writer = SummaryWriter(log_dir = folder) # tensorboard

if not os.path.isdir(folder):
    # create the right folder if it doesn't exist
    print("Generated new folder")
    os.mkdir(folder)

#########################################
# Hyperparameters
#########################################
training_properties, fno_architecture = FNO_initialize_hyperparameters(which_example, mode = mode_str)

# choose the Loss function
training_properties["exp"] = Norm_dict[exp_norm] # 0 for L^1 relative norm, 1 for L^2 relative norm, 2 for H^1 relative norm

# Training hyperparameters 
learning_rate    = training_properties["learning_rate"]
weight_decay     = training_properties["weight_decay"]
scheduler_step   = training_properties["scheduler_step"]
scheduler_gamma  = training_properties["scheduler_gamma"]
epochs           = training_properties["epochs"]
batch_size       = training_properties["batch_size"]
p                = training_properties["exp"]
beta             = training_properties["beta"]
training_samples = training_properties["training_samples"]
test_samples     = training_properties["test_samples"]
val_samples      = training_properties["val_samples"]

# fno architecture hyperparameters
d_a          = fno_architecture["d_a"]
d_v          = fno_architecture["width"]
d_u          = fno_architecture["d_u"]
L            = fno_architecture["n_layers"]
modes        = fno_architecture["modes"]
fun_act      = fno_architecture["fun_act"]
weights_norm = fno_architecture["weights_norm"]
arc          = fno_architecture["arc"]
RNN          = fno_architecture["RNN"]
FFTnorm      = fno_architecture["fft_norm"]
padding      = fno_architecture["padding"]
retrain_fno  = fno_architecture["retrain"]
FourierF     = fno_architecture["FourierF"]


#########################################
# Parameters for plots and tensorboard
#########################################
ep_step  = 50
n_idx    = 4 # number of random test that we plot
plotting = False 

#########################################
# Data and model loader, depending on the example chosen
#########################################
fno_architecture["problem_dim"] = 2 # default value
example = load_data_model(which_example, fno_architecture, device, batch_size, training_samples, in_dist)
train_loader = example.train_loader
val_loader = example.val_loader 
test_loader = example.test_loader # for final testing

#########################################
# save hyper-parameter in txt files
#########################################
# Write this information about the norm in a txt file
with open(folder + '/norm_info.txt', 'w') as f:
    f.write("Norm used during the training:\n")
    f.write(f"{training_properties["exp"]}\n")

# Save `training_properties` as JSON
with open(folder + '/training_properties.json', 'w') as f:
    json.dump(training_properties, f, indent=4)

# Save `fno_architecture` as JSON
with open(folder + '/net_architecture.json', 'w') as f:
    json.dump(fno_architecture, f, indent=4)

#########################################
# load the model
#########################################
if fno_architecture["problem_dim"] == 1:
    model = FNO_1D(d_a, d_v, d_u, L, modes, fun_act, weights_norm, arc, 
                    RNN, FFTnorm, padding, device, retrain_fno)    
elif fno_architecture["problem_dim"] == 2:
    model = FNO_2D(d_a, d_v, d_u, L, modes, modes, fun_act, weights_norm, arc, 
                    RNN, FFTnorm, padding, device, retrain_fno)    

# count and print the total number of parameters
par_tot = count_params(model)
print("Total number of parameters is: ", par_tot)
writer.add_text("Parameters", 'total number of parameters is' + str(par_tot), 0)

#########################################
# Training
#########################################
# optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(),
                        lr = learning_rate, weight_decay = weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                        step_size = scheduler_step, gamma = scheduler_gamma)

# Loss function
match p:
    case 0:
        loss = LprelLoss(1, False) # L^1 relative norm
    case 1:
        loss = LprelLoss(2, False) # L^2 relative norm
    case 2:
        if fno_architecture["problem_dim"] == 1:
            loss = H1relLoss_1D(beta, False, 1.0)
        elif fno_architecture["problem_dim"] == 2:
            loss = H1relLoss(beta, False, 1.0) # H^1 relative norm
    case 3:
        loss = torch.nn.SmoothL1Loss() # L^1 smooth loss (Mishra)
    case 4:
        loss = torch.nn.MSELoss() # L^2 smooth loss (Mishra)
    case _:
        raise ValueError("This value of p is not allowed")
    

#### Training process
for epoch in range(epochs):
    with tqdm(desc = f"Epoch {epoch}", bar_format = "{desc}: [{elapsed_s:.2f}{postfix}]") as tepoch:

        # train the model for one epoch
        if epoch == 0: # extract the test data
            esempio_test, soluzione_test = train_fun(model, train_loader, optimizer, scheduler, loss, p, device, which_example, tepoch, n_idx)
        else:
            train_fun(model, train_loader, optimizer, scheduler, loss, p, device, which_example, tepoch, n_idx)

        # test the model for one epoch
        test_relative_l1, test_relative_l2, test_relative_semih1, test_relative_h1, train_loss = test_fun(model, val_loader, train_loader, loss, exp_norm, 
                                                                                                          val_samples, training_samples, device, which_example, tepoch, statistic=True)

        # save the results on tensorboard
        writer.add_scalars(f"FNO_{fno_architecture["problem_dim"]}D_" + which_example, {"Train loss " + exp_norm: train_loss,
                                                "Test rel. L^1 error": test_relative_l1,
                                                "Test rel. L^2 error": test_relative_l2,
                                                "Test rel. semi-H^1 error": test_relative_semih1,
                                                "Test rel. H^1 error": test_relative_h1}, epoch)
                    
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training loss " + exp_norm + ': ' + str(train_loss) + "\n")
            file.write("Test relative L^1 error: " + str(test_relative_l1) + "\n")
            file.write("Test relative L^2 error: " + str(test_relative_l2) + "\n")
            file.write("Test relative semi-H^1 error: " + str(test_relative_semih1) + "\n")
            file.write("Test relative H^1 error: " + str(test_relative_h1) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(par_tot) + "\n")
            
        # plot data during the training and save on tensorboard
        if epoch == 0:
            # plot the input data
            plot_data(esempio_test, [], "Coefficients a(x)", epoch, writer, plotting)

            # plot the exact solution
            plot_data(soluzione_test, [], "Exact solution", epoch, writer, plotting)
                
        # Approximate solution with FNO
        if epoch % ep_step == 0:
            with torch.no_grad(): # no grad for efficiency
                out_test = model(esempio_test.to(device))
                out_test = out_test.cpu()
            
            # plot the approximate solution 
            plot_data(out_test, [], "Approximate solution with FNO", epoch, writer, plotting)

            # Module of the difference
            diff = torch.abs(out_test - soluzione_test)
            plot_data(diff, [], "Module of the difference", epoch, writer, plotting)

writer.flush() # for saving final data
writer.close() # close the tensorboard writer

torch.save(model, name_model)  
