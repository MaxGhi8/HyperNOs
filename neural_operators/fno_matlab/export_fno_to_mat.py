"""
Export FNO PyTorch model parameters to MATLAB .mat file

This script loads a trained FNO model and exports all its parameters
(weights and biases) to a MATLAB .mat file for use in MATLAB environments.
"""

import argparse
import json
import os
import sys

sys.path.append("..")

import numpy as np
import torch
from architectures import BAMPNO, CNO, FNO, FNO_lin, ResidualNetwork
from datasets import NO_load_data_model

# from architectures import FNO_lin
from loss_fun import loss_selector
from scipy.io import savemat
from utilities import count_params, initialize_hyperparameters
from wrappers import wrap_model

# torch.set_default_dtype(torch.float64)  # default tensor dtype

#########################################
# default values
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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
            "eig",
            "coeff_rhs",
            "coeff_rhs_1d",
            "fhn",
            "hh",
            "ord",
            "crosstruss",
            "afieti_homogeneous_neumann",
            "afieti_fno",
            "bampno_8_domain",
            "bampno_S_domain",
            "bampno_continuation",
        ],
        help="Select the example to run.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=["FNO", "CNO", "ResNet", "FNO_lin", "BAMPNO"],
        help="Select the architecture to use.",
    )
    parser.add_argument(
        "loss_fn_str",
        type=str,
        choices=["L1", "L2", "H1", "L1_smooth", "l2", "L2_cheb_mp", "H1_cheb_mp"],
        help="Select the relative loss function to use during the training process.",
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=[
            "best",
            "default",
            "best_samedofs",
            "best_linear",
            "best_50M",  # for the cont_tran FNO tests
            "best_150M",  # for the cont_tran FNO tests
            "best_500k",  # for the cont_tran FNO tests
        ],
        help="Select the hyper-params to use for define the architecture and the training, we have implemented the 'best' and 'default' options.",
    )
    parser.add_argument(
        "--in_dist",
        type=bool,
        default=True,
        help="For the datasets that are supported you can select if the test set is in-distribution or out-of-distribution.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="trained_params.mat",
        help="Specify the output file name for the MATLAB .mat file.",
    )

    args = parser.parse_args()

    return {
        "example": args.example.lower(),
        "architecture": args.architecture,
        "loss_fn_str": args.loss_fn_str,
        "mode": args.mode,
        "in_dist": args.in_dist,
        "output_file": args.output_file,
    }


# Parse CLI arguments
config = parse_arguments()
which_example = config["example"]
arc = config["architecture"]
loss_fn_str = config["loss_fn_str"]
mode_str = config["mode"]
in_dist = config["in_dist"]
output_file = config["output_file"]


#########################################
# Upload the model
#########################################
folder = f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_{mode_str}/"
files = os.listdir(folder)
name_model = folder + [file for file in files if file.startswith("model_")][0]
print("Model name: ", name_model)

try:
    try:  # For models saved with torch.save(model.state_dict())
        # Load the default hyperparameters for the model
        hyperparams_train, hyperparams_arc = initialize_hyperparameters(
            arc, which_example, mode_str
        )

        default_hyper_params = {
            **hyperparams_train,
            **hyperparams_arc,
        }

        example = NO_load_data_model(
            which_example=which_example,
            no_architecture={
                "FourierF": default_hyper_params["FourierF"],
                "retrain": default_hyper_params["retrain"],
            },
            batch_size=default_hyper_params["batch_size"],
            training_samples=default_hyper_params["training_samples"],
            filename=(
                default_hyper_params["filename"]
                if "filename" in default_hyper_params
                else None
            ),
        )

        match arc:
            case "FNO":
                model = FNO(
                    default_hyper_params["problem_dim"],
                    default_hyper_params["in_dim"],
                    default_hyper_params["width"],
                    default_hyper_params["out_dim"],
                    default_hyper_params["n_layers"],
                    default_hyper_params["modes"],
                    default_hyper_params["fun_act"],
                    default_hyper_params["weights_norm"],
                    default_hyper_params["fno_arc"],
                    default_hyper_params["RNN"],
                    default_hyper_params["fft_norm"],
                    default_hyper_params["padding"],
                    device,
                    (
                        example.output_normalizer
                        if (
                            "internal_normalization" in default_hyper_params
                            and default_hyper_params["internal_normalization"]
                        )
                        else None
                    ),
                    default_hyper_params["retrain"],
                )
                model = wrap_model(model, which_example)

            case "CNO":
                model = CNO(
                    problem_dim=default_hyper_params["problem_dim"],
                    in_dim=default_hyper_params["in_dim"],
                    out_dim=default_hyper_params["out_dim"],
                    size=default_hyper_params["in_size"],
                    N_layers=default_hyper_params["N_layers"],
                    N_res=default_hyper_params["N_res"],
                    N_res_neck=default_hyper_params["N_res_neck"],
                    channel_multiplier=default_hyper_params["channel_multiplier"],
                    kernel_size=default_hyper_params["kernel_size"],
                    use_bn=default_hyper_params["bn"],
                    device=device,
                )
                model = wrap_model(model, which_example)

            case "ResNet":
                model = ResidualNetwork(
                    default_hyper_params["in_channels"],
                    default_hyper_params["out_channels"],
                    default_hyper_params["hidden_channels"],
                    default_hyper_params["activation_str"],
                    default_hyper_params["n_blocks"],
                    device,
                    layer_norm=default_hyper_params["layer_norm"],
                    dropout_rate=default_hyper_params["dropout_rate"],
                    zero_mean=default_hyper_params["zero_mean"],
                    example_input_normalizer=(
                        example.input_normalizer
                        if default_hyper_params["internal_normalization"]
                        else None
                    ),
                    example_output_normalizer=(
                        example.output_normalizer
                        if default_hyper_params["internal_normalization"]
                        else None
                    ),
                )

            case "BAMPNO":
                model = BAMPNO(
                    default_hyper_params["problem_dim"],
                    default_hyper_params["n_patch"],
                    default_hyper_params["continuity_condition"],
                    default_hyper_params["n_pts"],
                    default_hyper_params["grid_filename"],
                    default_hyper_params["in_dim"],
                    default_hyper_params["d_v"],
                    default_hyper_params["out_dim"],
                    default_hyper_params["L"],
                    default_hyper_params["modes"],
                    default_hyper_params["fun_act"],
                    default_hyper_params["weights_norm"],
                    (
                        {int(k): v for k, v in default_hyper_params["zero_BC"].items()}
                        if default_hyper_params["zero_BC"]
                        else None
                    ),
                    default_hyper_params["arc"],
                    default_hyper_params["RNN"],
                    default_hyper_params["same_params"],
                    default_hyper_params["FFTnorm"],
                    device,
                    (
                        example.output_normalizer
                        if default_hyper_params["internal_normalization"]
                        else None
                    ),
                    default_hyper_params["retrain_seed"],
                )

            case "FNO_lin":
                model = FNO_lin(
                    default_hyper_params["problem_dim"],
                    default_hyper_params["in_dim"],
                    default_hyper_params["width"],
                    default_hyper_params["out_dim"],
                    default_hyper_params["n_layers"],
                    default_hyper_params["modes"],
                    default_hyper_params["fun_act"],
                    default_hyper_params["weights_norm"],
                    default_hyper_params["fno_arc"],
                    default_hyper_params["RNN"],
                    default_hyper_params["fft_norm"],
                    default_hyper_params["padding"],
                    device,
                    (
                        example.output_normalizer
                        if (
                            "internal_normalization" in default_hyper_params
                            and default_hyper_params["internal_normalization"]
                        )
                        else None
                    ),
                    default_hyper_params["retrain"],
                )

        checkpoint = torch.load(name_model, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    except Exception:
        print("Model not found, trying to load the model with torch.load()")
        # save with torch.save(model)
        # model = torch.load(name_model, weights_only=False, map_location=device)

except Exception:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed through the CLI."
    )

#########################################
# Upload hyperparameters
#########################################
# Load `hyperparameters` from JSON
with open(folder + "/chosed_hyperparams.json", "r") as f:
    hyperparams = json.load(f)

# Choose the Loss function
hyperparams["loss_fn_str"] = loss_fn_str

# Training hyperparameters
batch_size = hyperparams["batch_size"]
try:
    beta = hyperparams["beta"]
except KeyError:
    beta = 1.0  # default value
FourierF = hyperparams["FourierF"]
problem_dim = hyperparams["problem_dim"]
try:
    retrain = hyperparams["retrain"]
except KeyError:
    retrain = hyperparams["retrain_seed"]
val_samples = hyperparams["val_samples"]
test_samples = hyperparams["test_samples"]
training_samples = hyperparams["training_samples"]

try:
    filename = hyperparams["filename"]
except KeyError:
    try:
        filename = hyperparams["grid_filename"]
    except KeyError:
        filename = None

# Loss function
loss = loss_selector(loss_fn_str=loss_fn_str, problem_dim=problem_dim, beta=beta)

#########################################
# Data loader
#########################################
example = NO_load_data_model(
    which_example=which_example,
    no_architecture={
        "FourierF": hyperparams["FourierF"],
        "retrain": retrain,
    },
    batch_size=hyperparams["batch_size"],
    training_samples=hyperparams["training_samples"],
    filename=(
        default_hyper_params["filename"] if "filename" in default_hyper_params else None
    ),
)

# train_loader = example.train_loader
# val_loader = example.val_loader
# test_loader = example.test_loader


#########################################
## Export model parameters to .mat
#########################################
# Count and print the total number of parameters
total_params, total_bytes = count_params(model)
total_mb = total_bytes / (1024**2)
print(f"Total Parameters: {total_params:,}")
print(f"Total Model Size: {total_bytes:,} bytes ({total_mb:.2f} MB)")

# Dictionary to store all parameters
params_dict = {}

# Get model state dict
state_dict = model.state_dict()

print(f"\nExporting {len(state_dict)} parameters...")

# Convert each parameter to numpy and store
for name, param in state_dict.items():
    # Convert to numpy
    param_np = param.cpu().detach().numpy()

    # Replace dots with underscores for MATLAB compatibility
    matlab_name = name.replace(".", "_")

    # MATLAB doesn't support complex numbers in the same way
    # If the parameter is complex, split into real and imaginary parts
    if np.iscomplexobj(param_np):
        params_dict[f"{matlab_name}_real"] = param_np.real
        params_dict[f"{matlab_name}_imag"] = param_np.imag
        print(
            f"  {name}: shape={param_np.shape}, dtype={param_np.dtype} (split into real/imag)"
        )
    else:
        params_dict[matlab_name] = param_np
        print(f"  {name}: shape={param_np.shape}, dtype={param_np.dtype}")

# Add metadata
params_dict["model_type"] = "FNO"
params_dict["problem_dim"] = model.problem_dim
params_dict["in_dim"] = model.in_dim
params_dict["d_v"] = model.d_v
params_dict["out_dim"] = model.out_dim
params_dict["L"] = model.L
params_dict["modes"] = model.modes
params_dict["fun_act"] = model.fun_act
params_dict["arc"] = model.arc
params_dict["RNN"] = model.RNN
params_dict["padding"] = model.padding

# Add normalization parameters if available
if (
    hasattr(example, "input_normalizer")
    and example.input_normalizer is not None
    and params_dict.get("internal_normalization", False)
):
    try:  # for UnitGaussianNormalizer
        params_dict["input_normalizer_mean"] = (
            example.input_normalizer.mean.cpu().detach().numpy()
        )
        params_dict["input_normalizer_std"] = (
            example.input_normalizer.std.cpu().detach().numpy()
        )
        params_dict["input_normalizer_eps"] = (
            example.input_normalizer.eps.cpu().detach().numpy()
        )
        print(
            f"\nAdded input normalizer (Gaussian point-wise scaling): mean shape={example.input_normalizer.mean.shape}, std shape={example.input_normalizer.std.shape}"
        )
        params_dict["has_input_normalizer_gaussian"] = True

    except:
        # for minmaxGlobalNormalizer
        params_dict["input_normalizer_min"] = (
            example.input_normalizer.min.cpu().detach().numpy()
        )
        params_dict["input_normalizer_max"] = (
            example.input_normalizer.max.cpu().detach().numpy()
        )
        print(
            f"\nAdded input normalizer (min-max scaling): min shape={example.input_normalizer.min.shape}, max shape={example.input_normalizer.max.shape}"
        )
        params_dict["has_input_normalizer_minmax"] = True

else:
    params_dict["has_input_normalizer"] = False
    print("\nNo input normalizer found")

if (
    hasattr(example, "output_normalizer")
    and example.output_normalizer is not None
    and params_dict.get("internal_normalization", False)
):
    try:  # for UnitGaussianNormalizer
        params_dict["output_normalizer_mean"] = (
            example.output_normalizer.mean.cpu().detach().numpy()
        )
        params_dict["output_normalizer_std"] = (
            example.output_normalizer.std.cpu().detach().numpy()
        )
        params_dict["output_normalizer_eps"] = (
            example.output_normalizer.eps.cpu().detach().numpy()
        )
        print(
            f"Added output normalizer (Gaussian point-wise scaling): mean shape={example.output_normalizer.mean.shape}, std shape={example.output_normalizer.std.shape}"
        )
        params_dict["has_output_normalizer_gaussian"] = True

    except:  # for minmaxGlobalNormalizer
        params_dict["output_normalizer_min"] = (
            example.output_normalizer.min.cpu().detach().numpy()
        )
        params_dict["output_normalizer_max"] = (
            example.output_normalizer.max.cpu().detach().numpy()
        )
        print(
            f"Added output normalizer (min-max scaling): min shape={example.output_normalizer.min.shape}, max shape={example.output_normalizer.max.shape}"
        )
        params_dict["has_output_normalizer_minmax"] = True

else:
    params_dict["has_output_normalizer"] = False
    print("No output normalizer found")

#########################################
# Export test data batch for verification
#########################################
print("\n" + "=" * 60)
print("Exporting test data batch for verification...")
print("=" * 60)

# Get one batch from test loader
try:
    # Get test loader
    test_loader = example.test_loader

    # Get first batch
    test_batch = next(iter(test_loader))
    X_test_batch, y_test_batch = test_batch

    # Move to CPU and convert to numpy
    X_test_np = X_test_batch.cpu().detach().numpy()
    y_test_np = y_test_batch.cpu().detach().numpy()

    # Run inference on the batch with PyTorch
    model.eval()
    with torch.no_grad():
        y_pred_pytorch = model(X_test_batch.to(device))
        y_pred_np = y_pred_pytorch.cpu().detach().numpy()

    print(f"\nTest batch shapes:")
    print(f"  Input (X_test):  {X_test_np.shape}")
    print(f"  Target (y_test): {y_test_np.shape}")
    print(f"  PyTorch output (y_pred_pytorch): {y_pred_np.shape}")

    # Add to params_dict
    params_dict["test_X_batch"] = X_test_np
    params_dict["test_y_batch"] = y_test_np
    params_dict["test_y_pred_pytorch"] = y_pred_np
    params_dict["has_test_batch"] = True

    print(f"\n Test batch exported successfully!")
    print(f"  PyTorch output stats:")
    print(f"    Mean: {y_pred_np.mean():.6f}")
    print(f"    Std:  {y_pred_np.std():.6f}")
    print(f"    Min:  {y_pred_np.min():.6f}")
    print(f"    Max:  {y_pred_np.max():.6f}")

except Exception as e:
    print(f"\n Warning: Could not export test batch: {e}")
    print("  Continuing without test data...")
    params_dict["has_test_batch"] = False

# Save to .mat file in the model folder
output_path = os.path.join(folder, output_file)
savemat(output_path, params_dict, oned_as="column")

print(f"\nSuccessfully exported model parameters to: {output_path}")
print(
    f"Total parameters exported: {len([k for k in params_dict.keys() if not k.startswith('__')])}"
)
