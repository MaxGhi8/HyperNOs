# HyperNOs: Automated and Parallel Neural Operators Research

HyperNOs is a Python framework for automated, distributed, and parallel hyperparameter optimization (HPO) of Neural Operators. It leverages **PyTorch** for model development and **Ray Tune** for scalable hyperparameter search across various physics-informed benchmarks.

## Project Overview

- **Core Technology:** Python 3.12, PyTorch, Ray Tune, H5PY, JAXTyping.
- **Architectures:** Supports multiple Neural Operator variants, including:
  - **FNO** (Fourier Neural Operator)
  - **CNO** (Convolutional Neural Operator)
  - **BAMPNO** (Basis-Augmented Multi-Resolution Parallel Neural Operator)
  - **DeepONet** (DON), ResNet, RationalNN, etc.
- **Benchmarks:** Covers various PDEs and physical systems (Poisson, Wave, Allen-Cahn, Navier-Stokes, Darcy flow, etc.).
- **Research Context:** Associated with the paper "HyperNOs: Automated and Parallel Library for Neural Operators Research" (arXiv:2503.18087).

## Building and Running

### Environment Setup
1. **Python Version:** 3.12.7 is recommended.
2. **Installation:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Acquisition:**
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```

### Running Experiments
The `neural_operators/cli/` directory contains the primary entry points.

- **Single Run Training:**
  ```bash
  python neural_operators/cli/main_singlerun.py <example> <architecture> <loss_fn> <mode>
  ```
  Example: `python neural_operators/cli/main_singlerun.py poisson fno l2 best`

- **Hyperparameter Search (Ray Tune):**
  ```bash
  python neural_operators/cli/main_raytune.py <example> <architecture> <loss_fn> <mode>
  ```

- **Positional Arguments:**
  - `example`: `poisson`, `wave_0_5`, `cont_tran`, `disc_tran`, `allen`, `shear_layer`, `airfoil`, `darcy`, etc.
  - `architecture`: `fno`, `cno`.
  - `loss_fn`: `l1`, `l2`, `h1`, `l1_smooth`.
  - `mode`: `best`, `default` (determines which JSON config to load).

### Testing
Execute unit tests using `pytest`:
```bash
pytest neural_operators/pytest/
```

## Development Conventions

- **Type Safety:** Uses `jaxtyping` and `beartype` for robust runtime type and shape checking of tensors.
- **Code Style:** Strictly adheres to `black` (line length 88) and `isort`.
- **Configuration Management:** Hyperparameters are decentralized into JSON files within `neural_operators/architectures/<ARC>/configurations/`.
- **Modularity:** 
  - `architectures/`: Contains model definitions.
  - `datasets.py`: Unified data loading and normalization logic.
  - `loss_fun.py` / `loss_fun_with_physics.py`: Implementation of relative norms and physics-informed losses.
- **Logging:** Training progress is logged via **TensorBoard**.

## Key Directories
- `data/`: Dataset storage (generated after `download_data.sh`).
- `neural_operators/architectures/`: Implementations of various NO models.
- `neural_operators/cli/`: Command-line interface for experiments and plotting.
- `neural_operators/examples/`: High-level training routines.
- `neural_operators/wrappers/`: Model wrappers for specific use cases (e.g., Airfoil, StiffnessMatrix).
- `neural_operators/pytest/`: Comprehensive test suite.
