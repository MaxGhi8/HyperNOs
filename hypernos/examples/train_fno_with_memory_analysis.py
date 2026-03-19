"""
In this example I fix all the hyperparameters for the FNO model and train it.
"""

import os
import sys

import torch

sys.path.append("..")

from architectures import FNO
from datasets import NO_load_data_model
from loss_fun import loss_selector
from train import train_fixed_model
from utilities import get_plot_function, initialize_hyperparameters
from wrappers import wrap_model_builder


def compute_model_memory(model, print_details=True):
    """
    Compute memory usage of model parameters and buffers.
    
    Returns:
        dict: Memory statistics in MB
    """
    param_size = 0
    param_count = 0
    buffer_size = 0
    buffer_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    memory_stats = {
        'param_count': param_count,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': total_size / 1024**2,
    }
    
    if print_details:
        print(f"\n{'='*60}")
        print("MODEL MEMORY BREAKDOWN")
        print(f"{'='*60}")
        print(f"Parameters: {param_count:,} ({memory_stats['param_size_mb']:.2f} MB)")
        print(f"Buffers: {buffer_count:,} ({memory_stats['buffer_size_mb']:.2f} MB)")
        print(f"Total Model Size: {memory_stats['total_size_mb']:.2f} MB")
        print(f"{'='*60}\n")
    
    return memory_stats


def track_gpu_memory(device, stage=""):
    """
    Track current and peak GPU memory usage.
    
    Returns:
        dict: Memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {}
    
    torch.cuda.synchronize()
    
    stats = {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
        'max_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    
    if stage:
        print(f"\n--- GPU Memory at {stage} ---")
        print(f"Allocated: {stats['allocated_mb']:.2f} MB")
        print(f"Reserved: {stats['reserved_mb']:.2f} MB")
        print(f"Peak Allocated: {stats['max_allocated_mb']:.2f} MB")
        print(f"Peak Reserved: {stats['max_reserved_mb']:.2f} MB")
    
    return stats


def analyze_memory_usage(model, example_batch, device, loss_fn):
    """
    Perform detailed memory analysis during forward and backward passes.
    
    Args:
        model: The neural network model
        example_batch: A sample batch from the dataset
        device: torch device
        loss_fn: Loss function
    
    Returns:
        dict: Comprehensive memory analysis
    """
    print(f"\n{'='*60}")
    print("DETAILED MEMORY ANALYSIS")
    print(f"{'='*60}")
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # 1. Model parameters only
    model_memory = compute_model_memory(model, print_details=True)
    mem_after_model = track_gpu_memory(device, "After Model Load")
    
    # 2. After loading data to GPU
    x, y = example_batch
    x = x.to(device)
    y = y.to(device)
    
    input_size_mb = (x.numel() * x.element_size() + y.numel() * y.element_size()) / 1024**2
    print(f"\nInput Batch Size: {input_size_mb:.2f} MB")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {y.shape}")
    
    mem_after_data = track_gpu_memory(device, "After Data Load")
    
    # 3. After forward pass
    torch.cuda.reset_peak_memory_stats(device)
    output = model(x)
    mem_after_forward = track_gpu_memory(device, "After Forward Pass")
    
    # 4. After backward pass
    torch.cuda.reset_peak_memory_stats(device)
    loss = loss_fn(output, y)
    if isinstance(loss, torch.Tensor):
        loss = loss.mean() if loss.dim() > 0 else loss
    loss.backward()
    mem_after_backward = track_gpu_memory(device, "After Backward Pass")
    
    # Memory breakdown analysis
    print(f"\n{'='*60}")
    print("MEMORY BREAKDOWN ANALYSIS")
    print(f"{'='*60}")
    
    forward_activation_mb = mem_after_forward['max_allocated_mb'] - mem_after_data['allocated_mb']
    backward_gradient_mb = mem_after_backward['max_allocated_mb'] - mem_after_forward['allocated_mb']
    
    print(f"\nModel Parameters: {model_memory['total_size_mb']:.2f} MB")
    print(f"Input/Output Data: {input_size_mb:.2f} MB")
    print(f"Forward Activations (estimated): {forward_activation_mb:.2f} MB")
    print(f"Backward Gradients (estimated): {backward_gradient_mb:.2f} MB")
    print(f"\nPeak Memory Usage: {mem_after_backward['max_allocated_mb']:.2f} MB")
    
    # Theoretical estimates
    print(f"\n{'='*60}")
    print("THEORETICAL ESTIMATES")
    print(f"{'='*60}")
    
    # Parameter gradients = same size as parameters
    gradient_memory_mb = model_memory['total_size_mb']
    
    # Optimizer state (Adam uses 2x params for momentum and velocity)
    optimizer_memory_mb = 2 * model_memory['total_size_mb']
    
    print(f"Parameter Gradients: {gradient_memory_mb:.2f} MB")
    print(f"Optimizer State (Adam): {optimizer_memory_mb:.2f} MB")
    print(f"Total Training Memory (estimated): {gradient_memory_mb + optimizer_memory_mb:.2f} MB")
    
    # Activation memory analysis
    total_activations = 0
    print(f"\nActivation Memory by Layer:")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            # Rough estimate: assume activation size ~ weight size for fully connected layers
            # For conv layers, depends on spatial dimensions
            if len(name.split('.')) <= 2:  # Only print main layers
                print(f"  {name}: parameters = {module.weight.numel():,}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    
    ratio_activation_to_params = forward_activation_mb / model_memory['total_size_mb']
    ratio_peak_to_params = mem_after_backward['max_allocated_mb'] / model_memory['total_size_mb']
    
    print(f"\nActivation/Parameters Ratio: {ratio_activation_to_params:.2f}x")
    print(f"Peak Memory/Parameters Ratio: {ratio_peak_to_params:.2f}x")
    
    if ratio_activation_to_params > 3:
        print("\n⚠️  Forward activations dominate memory usage!")
        print("   Consider: gradient checkpointing, smaller batch size, or model sharding")
    
    if ratio_peak_to_params > 10:
        print("\n⚠️  Peak memory is significantly higher than parameters!")
        print("   Backpropagation is likely the bottleneck due to stored activations")
    
    # Clean up
    del x, y, output, loss
    torch.cuda.empty_cache()
    
    return {
        'model_memory': model_memory,
        'input_size_mb': input_size_mb,
        'forward_activation_mb': forward_activation_mb,
        'backward_gradient_mb': backward_gradient_mb,
        'peak_memory_mb': mem_after_backward['max_allocated_mb'],
        'gradient_memory_mb': gradient_memory_mb,
        'optimizer_memory_mb': optimizer_memory_mb,
    }


def train_fno(which_example: str, mode_hyperparams: str, loss_fn_str: str):

    # Select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default hyperparameters for the FNO model
    hyperparams_train, hyperparams_arc = initialize_hyperparameters(
        "FNO", which_example, mode_hyperparams
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

    # Define the model builders
    model_builder = lambda config: FNO(
        config["problem_dim"],
        config["in_dim"],
        config["width"],
        config["out_dim"],
        config["n_layers"],
        config["modes"],
        config["fun_act"],
        config["weights_norm"],
        config["fno_arc"],
        config["RNN"],
        config["fft_norm"],
        config["padding"],
        device,
        (
            example.output_normalizer
            if ("internal_normalization" in config and config["internal_normalization"])
            else None
        ),
        config["retrain"],
    )
    # Wrap the model builder
    model_builder = wrap_model_builder(model_builder, which_example)

    # Define the dataset builder
    dataset_builder = lambda config: NO_load_data_model(
        which_example=which_example,
        no_architecture={
            "FourierF": config["FourierF"],
            "retrain": config["retrain"],
        },
        batch_size=config["batch_size"],
        training_samples=config["training_samples"],
        filename=config["filename"] if "filename" in config else None,
    )

    # Define the loss function
    loss_fn = loss_selector(
        loss_fn_str=loss_fn_str,
        problem_dim=default_hyper_params["problem_dim"],
        beta=default_hyper_params["beta"],
    )

    experiment_name = f"FNO/{which_example}/loss_{loss_fn_str}_mode_{mode_hyperparams}"

    # Create the right folder if it doesn't exist
    folder = f"../tests/{experiment_name}"
    if not os.path.isdir(folder):
        print("Generated new folder")
        os.makedirs(folder, exist_ok=True)

    # MEMORY ANALYSIS: Build model and analyze memory
    model = model_builder(default_hyper_params).to(device)
    
    # Get a sample batch for memory analysis
    train_loader = example.train_loader
    example_batch = next(iter(train_loader))
    
    # Perform comprehensive memory analysis
    memory_analysis = analyze_memory_usage(model, example_batch, device, loss_fn)
    
    # Save memory analysis to file
    with open(folder + "/memory_analysis.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("MEMORY ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: FNO\n")
        f.write(f"Example: {which_example}\n")
        f.write(f"Loss: {loss_fn_str}\n")
        f.write(f"Batch Size: {default_hyper_params['batch_size']}\n\n")
        
        f.write("Model Parameters\n")
        f.write("-"*60 + "\n")
        f.write(f"Parameter Count: {memory_analysis['model_memory']['param_count']:,}\n")
        f.write(f"Model Size: {memory_analysis['model_memory']['total_size_mb']:.2f} MB\n\n")
        
        f.write("Memory Breakdown\n")
        f.write("-"*60 + "\n")
        f.write(f"Input/Output Batch: {memory_analysis['input_size_mb']:.2f} MB\n")
        f.write(f"Forward Activations: {memory_analysis['forward_activation_mb']:.2f} MB\n")
        f.write(f"Backward Gradients: {memory_analysis['backward_gradient_mb']:.2f} MB\n")
        f.write(f"Parameter Gradients: {memory_analysis['gradient_memory_mb']:.2f} MB\n")
        f.write(f"Optimizer State (Adam): {memory_analysis['optimizer_memory_mb']:.2f} MB\n\n")
        
        f.write("Peak Memory\n")
        f.write("-"*60 + "\n")
        f.write(f"Peak GPU Memory: {memory_analysis['peak_memory_mb']:.2f} MB\n")
        f.write(f"Peak/Parameters Ratio: {memory_analysis['peak_memory_mb'] / memory_analysis['model_memory']['total_size_mb']:.2f}x\n\n")
        
        f.write("Analysis\n")
        f.write("-"*60 + "\n")
        ratio = memory_analysis['forward_activation_mb'] / memory_analysis['model_memory']['total_size_mb']
        f.write(f"Activation/Parameters Ratio: {ratio:.2f}x\n")
        if ratio > 3:
            f.write("⚠️  Forward activations dominate memory usage.\n")
            f.write("   Peak memory is dominated by backpropagation (storing forward activations).\n")
        if memory_analysis['peak_memory_mb'] / memory_analysis['model_memory']['total_size_mb'] > 10:
            f.write("⚠️  Peak memory significantly exceeds parameter count.\n")
            f.write("   This indicates that activation storage for backprop is the main bottleneck.\n")

    # Save the norm information
    with open(folder + "/norm_info.txt", "w") as f:
        f.write("Norm used during the training:\n")
        f.write(f"{loss_fn_str}\n")
    
    # Clean up before training
    del model
    torch.cuda.empty_cache()

    # Call the library function to tune the hyperparameters
    train_fixed_model(
        default_hyper_params,
        model_builder,
        dataset_builder,
        loss_fn,
        experiment_name,
        get_plot_function(which_example, "input"),
        get_plot_function(which_example, "output"),
        full_validation=True,
    )


if __name__ == "__main__":
    train_fno("ord", "best_samedofs", "L2")