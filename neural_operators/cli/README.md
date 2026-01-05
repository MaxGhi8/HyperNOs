# Command Line Interface (CLI) Utilities

This directory contains command-line scripts for running experiments on trained models and visualizing results.

## Key Scripts

- **Utilities**:

  - `recover_model.py`: Tools for loading and inspecting saved models, with some functions already implemented, like inference, error analysis, and visualization.
  - `save_onnx.py`: Export models to ONNX format.

- **Visualization & Analysis**:

  - `plot_loss.py`: Plot training and validation loss curves.
  - `plot_wall_time_loss.py`: Plot training loss curves of all the runs of a hyperparameter tuning experiment.
  - `plot_error_distribution.py`: Visualize error distributions.
  - `convergence_plot.py` and `convergence_plot_comparison.py`: Generate convergence plots.

- **Experiment Runners**:

  - `main_singlerun.py`: Execute a single training run, analogously to the `train_*.py` files in the `neural_operators/examples` directory.
  - `main_raytune.py`: Orchestrate hyperparameter tuning using Ray Tune, analogously to the `ray_*.py` files in the `neural_operators/examples` directory.
