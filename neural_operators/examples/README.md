## Codebase Structure: Custom vs Official

This repository contains both **custom implementations** of Neural Operators and wrappers around **official libraries**.

### 1. Custom Implementations (Root Examples)

Scripts directly in this folder (e.g., `train_fno.py`, `train_cno.py`) use our **custom architectures** located in `../architectures/`.

### 2. Official Library Wrappers

We also provide subfolders demonstrating how to use official libraries within our pipeline:

- **`neuralop_lib/`**: Examples using the official `neuraloperator` library (e.g., `TFNO`, `CODANO`, `UNO`, `RNO`, `LocalNO`, `OTNO`).
- **`deepxde_lib/`**: Examples using `deepxde` (e.g., `DeepONet`, `MIONet`, `POD-DeepONet`, `POD-MIONet`).

This dual approach allows us to benchmark our custom models against community standards and easily integrate new findings.

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
