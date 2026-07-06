"""
Given a model, save it to ONNX format.
It is used to save the model in ONNX format for deployment or inference.
"""

import argparse
import csv
import os
from pathlib import Path

import torch

from hypernos.architectures import (
    CNO,
    FNO_ONNX,
    GeometryConditionedLinearOperator,
    GeometryConditionedLinearOperator_mp_afieti,
    GeometryConditionedLinearOperatorExport_mp_afieti,
    ResidualNetwork,
)
from hypernos.datasets import NO_load_data_model
from hypernos.utilities import initialize_hyperparameters
from hypernos.wrappers import wrap_model

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
            "mp_afieti",
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

        if which_example == "mp_afieti":
            dataset_which_example = which_example
            dataset_filename = "yeti_dataset.csv"
        else:
            dataset_which_example = which_example + "_transformer" * (
                "transformer" in arc
            )
            dataset_filename = "dataset_homogeneous_Neumann_l_0_deg_2_crazygeom.mat"

        example = NO_load_data_model(
            which_example=dataset_which_example,
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
            filename=dataset_filename,
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
                iganet_transformer_cls = (
                    GeometryConditionedLinearOperator_mp_afieti
                    if which_example == "mp_afieti"
                    else GeometryConditionedLinearOperator
                )
                model = iganet_transformer_cls(
                    n_dofs=default_hyper_params["n_dofs"],
                    n_control_points=default_hyper_params["n_control_points"],
                    hidden_dim=default_hyper_params["hidden_dim"],
                    n_heads=default_hyper_params["n_heads"],
                    n_heads_A=default_hyper_params["n_heads_A"],
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

if isinstance(model, GeometryConditionedLinearOperator_mp_afieti):
    # Export Q, K_scaled and epsilon alongside u so external inference code
    # (e.g. a C++ application) can apply the operator without materializing
    # the dense (n_dofs, n_dofs) matrix.
    model = GeometryConditionedLinearOperatorExport_mp_afieti(model)

model.eval()
model.to("cpu")

# print the parameters of the model
# print("Model parameters:")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}")

print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))


def _tensor_to_flat_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).tolist()


def _save_tensor_csv(base_path: str, tensor_name: str, tensor: torch.Tensor) -> tuple[str, str]:
    csv_path = Path(f"{base_path}_{tensor_name}.csv")
    shape = list(tensor.shape)
    flat_values = _tensor_to_flat_list(tensor)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(flat_values)

    shape_text = "x".join(str(dimension) for dimension in shape)
    return str(csv_path), shape_text


def _save_sample_io(base_path: str, sample_input, sample_output, output_names=None) -> None:
    manifest_rows = []

    if isinstance(sample_input, (list, tuple)):
        for index, tensor in enumerate(sample_input):
            csv_path, shape_text = _save_tensor_csv(base_path, f"input_{index}", tensor)
            manifest_rows.append(
                ["input", f"input_{index}", csv_path, shape_text, str(tensor.dtype), tensor.numel()]
            )
    else:
        csv_path, shape_text = _save_tensor_csv(base_path, "input", sample_input)
        manifest_rows.append(
            ["input", "input", csv_path, shape_text, str(sample_input.dtype), sample_input.numel()]
        )

    if isinstance(sample_output, (list, tuple)):
        if output_names is None or len(output_names) != len(sample_output):
            output_names = [f"output_{index}" for index in range(len(sample_output))]
        for name, tensor in zip(output_names, sample_output):
            csv_path, shape_text = _save_tensor_csv(base_path, name, tensor)
            manifest_rows.append(
                [
                    "output",
                    name,
                    csv_path,
                    shape_text,
                    str(tensor.dtype),
                    tensor.numel(),
                ]
            )
    else:
        csv_path, shape_text = _save_tensor_csv(base_path, "output", sample_output)
        manifest_rows.append(
            ["output", "output", csv_path, shape_text, str(sample_output.dtype), sample_output.numel()]
        )

    manifest_path = Path(f"{base_path}_sample_io_manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "name", "filename", "shape", "dtype", "numel"])
        writer.writerows(manifest_rows)

    print(f"Saved sample input/output CSV files and manifest to {manifest_path}")


def _get_model_epsilon(model):
    current_model = model

    for _ in range(5):
        if hasattr(current_model, "epsilon"):
            epsilon_value = current_model.epsilon
            if callable(epsilon_value):
                epsilon_value = epsilon_value()
            return float(epsilon_value.detach().cpu().item())

        if hasattr(current_model, "model"):
            current_model = current_model.model
            continue

        if hasattr(current_model, "module"):
            current_model = current_model.module
            continue

        break

    return None


def _save_epsilon(base_path: str, epsilon_value: float) -> None:
    epsilon_path = Path(f"{base_path}_epsilon.txt")
    with epsilon_path.open("w", encoding="utf-8") as file:
        file.write(f"{epsilon_value:.18e}\n")

    print(f"Saved epsilon to {epsilon_path}")


# Create dummy input
batch_input = next(iter(example.train_loader))[0]

if isinstance(batch_input, list) or isinstance(batch_input, tuple):
    first, second = batch_input
    first = first[[0], :].to("cpu")  # TODO: works only for IgaNet_transformer
    second = second[[0], :, :].to("cpu")  # TODO: works only for IgaNet_transformer
    dummy_input = (first, second)
else:
    dummy_input = batch_input.to("cpu")

with torch.no_grad():
    dummy_output = model(dummy_input)

output_names = (
    ["u", "Q", "K_scaled", "epsilon"]
    if isinstance(model, GeometryConditionedLinearOperatorExport_mp_afieti)
    else ["output"]
)

_save_sample_io(name_model[:-4], dummy_input, dummy_output, output_names=output_names)

epsilon_value = _get_model_epsilon(model)
if epsilon_value is not None:
    print(f"Epsilon: {epsilon_value:.18e}")
    _save_epsilon(name_model[:-4], epsilon_value)
else:
    print("Epsilon: not available on this model")

# Export the model to ONNX
torch.onnx.export(
    model,
    (dummy_input,),  # pass a single positional argument
    f"{name_model[:-4]}.onnx",
    input_names=["input"],
    output_names=output_names,
    export_params=True,
    # dynamo=True,
)

print(f"\n 🤖 Model saved to {name_model[:-4]}.onnx")
