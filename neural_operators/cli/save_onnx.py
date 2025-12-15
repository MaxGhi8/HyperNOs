"""
Given a model, save it to ONNX format.
It is used to save the model in ONNX format for deployment or inference.
"""

import argparse
import os
import sys

import torch

sys.path.append("..")

from architectures import (
    CNO,
    FNO_ONNX,
    GeometryConditionedLinearOperator,
    ResidualNetwork,
)
from datasets import NO_load_data_model
from utilities import initialize_hyperparameters
from wrappers import wrap_model

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
            "ord",
            "crosstruss",
            "afieti_homogeneous_neumann",
            "afieti_fno",
        ],
        help="Select the example to run.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=["FNO", "CNO", "ResNet", "IgaNet_transformer"],
        help="Select the architecture to use.",
    )
    parser.add_argument(
        "loss_fn_str",
        type=str,
        choices=["L1", "L2", "H1", "L1_smooth", "l2"],
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

    args = parser.parse_args()

    return {
        "example": args.example.lower(),
        "architecture": args.architecture,
        "loss_fn_str": args.loss_fn_str,
        "mode": args.mode,
        "in_dist": args.in_dist,
    }


config = parse_arguments()
which_example = config["example"]
arc = config["architecture"]
loss_fn_str = config["loss_fn_str"]
mode_str = config["mode"]
in_dist = config["in_dist"]

# upload the model and the hyperparameters
folder = f"../tests/{arc}/{which_example}/loss_{loss_fn_str}_mode_{mode_str}/"
files = os.listdir(folder)
name_model = folder + [file for file in files if file.startswith("model_")][0]
print(f"🤖 Uploading model's name: {name_model}")

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
            which_example=which_example + "_transformer" * ("transformer" in arc),
            no_architecture={
                "FourierF": (
                    default_hyper_params["FourierF"]
                    if "FourierF" in default_hyper_params
                    else None
                ),
                "retrain": default_hyper_params["retrain"],
            },
            batch_size=default_hyper_params["batch_size"],
            training_samples=default_hyper_params["training_samples"],
            filename="dataset_homogeneous_Neumann_l_0_deg_2_crazygeom.mat",
        )

        match arc:
            case "FNO":
                print("Loading FNO ONNX model.")
                model = FNO_ONNX(
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
                            "internal_normalization" in config
                            and config["internal_normalization"]
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

            case "IgaNet_transformer":
                model = GeometryConditionedLinearOperator(
                    n_dofs=default_hyper_params["n_dofs"],
                    n_control_points=default_hyper_params["n_control_points"],
                    hidden_dim=default_hyper_params["hidden_dim"],
                    n_heads=default_hyper_params["n_heads"],
                    n_layers_geo=default_hyper_params["n_layers_geo"],
                    dropout_rate=default_hyper_params["dropout_rate"],
                    activation_str=default_hyper_params["activation_str"],
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
                    device=device,
                )

        checkpoint = torch.load(name_model, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    except Exception:
        # save with torch.save(model)
        model = torch.load(name_model, weights_only=False, map_location=device)

except Exception:
    raise ValueError(
        "The model is not found, please check the hyperparameters passed trhow the CLI."
    )

model.eval()
model.to("cpu")

# Create dummy input
batch_input = next(iter(example.train_loader))[0]

if isinstance(batch_input, list) or isinstance(batch_input, tuple):
    first, second = batch_input
    first = first.to("cpu")
    second = second.to("cpu")
    dummy_input = (first, second)
else:
    dummy_input = batch_input.to("cpu")

# Export the model to ONNX
torch.onnx.export(
    model,
    (dummy_input,),  # pass a single positional argument
    f"{name_model[:-4]}.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    # dynamo=True,
)

print(f"\n 🤖 Model saved to {name_model[:-4]}.onnx")
