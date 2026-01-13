# Examples

This directory provides example scripts demonstrating how to train and configure various neural operators.

## Subdirectories

The examples are organized into two main subfolders containing library-specific implementations:

- **`deepxde_lib`**: Contains examples using the DeepXDE library.
- **`neuralop_lib`**: Contains examples using the NeuralOperator library.

## Available Models

The following models are available across the directory and its subfolders:

**Root Directory:**

- **CNO** (Convolutional Neural Operator)
- **FNO** (Fourier Neural Operator)
- **ResNet** (Residual Neural Network)

**DeepXDE Library (`deepxde_lib`):**

- **DeepONet** (Deep Operator Network)
- **MIONet** (Multiple Input Operator Network)
- **POD-DeepONet** (POD-reduced DeepONet)
- **POD-MIONet** (POD-reduced MIONet)

**NeuralOperator Library (`neuralop_lib`):**

- **FNOGNO** (Fourier Neural Operator with Graph Neural Operator)
- **GINO** (Graph-Informed Neural Operator)
- **SFNO** (Spherical Fourier Neural Operator)
- **CODANO** (Codomain Attention Neural Operator)
- **TFNO** (Tensorized Fourier Neural Operator)
- **UNO** (U-Net Neural Operator)
- **UQNO** (Uncertainty Quantification Neural Operator)
- **RNO** (Recurrent Neural Operator)
- **LocalNO** (Local Neural Operator)
- **OTNO** (Optimal Transport Neural Operator)

## Script Types

- **Training Scripts** (`train_*.py`): Standalone scripts for training specific models.
- **Ray Tune Scripts** (`ray_*.py`): Scripts configured for hyperparameter optimization using Ray Tune.

These examples cover different use cases, including multiple datasets, varying resolutions, and physics-constrained learning.
