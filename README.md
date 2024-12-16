# HyperNOs
## HyperNOs Documentation

### Introduction
HyperNOs is a Python project focused on the implementation of completely automatic, distributed, parallel neural operators hyperparameter optimization. The project aims to provide a framework for training neural operator models using Pytorch and Ray Tune for hyperparameter tuning.

### Installation
To set up the HyperNOs project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/MaxGhi8/HyperNOs.git
   cd HyperNOs
   ```
2. Install the required dependencies. It is recommended to create a virtual environment before installing the dependencies; I personally use `pyenv` and Python version `3.12.7` for this purpose:
    ```bash
    pyenv install 3.12.7
    pyenv virtualenv 3.12.7 hypernos
    pyenv activate hypernos
    ```
    Then, install the dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
    ```

<!-- TODO  -->
3. Download the dataset using the `download_data.py` script:
   ```bash
   python download_data.py 
   ```
<!-- TODO  -->

### Usage
The main functionalities of the project are implemented in the `neural_operators` directory. Here are the key scripts and their purposes:

- `main_singlerun.py`: Script for load the selected dataset, architecture and hyperparameters, and train the model.
- `main_raytune.py`: Main script for running hyper-parameters optimization using Ray Tune.

----------------------------------------------------------------------------------------------------------------------------

### Examples `main_singlerun.py`

To run the script, execute the following command in your terminal:
```bash
python main_singlerun.py <example> <architecture> <loss_function> <mode> [--in_dist]
```

**Positional Arguments**
1. **`example`**  
   Specifies the example/problem to train on. Choices include:
   - **`poisson`**: Poisson equation  
   - **`wave_0_5`**: Wave equation  
   - **`cont_tran`**: Continuous transport  
   - **`disc_tran`**: Discontinuous transport  
   - **`allen`**: Allen-Cahn equation (512 training samples)  
   - **`shear_layer`**: Navier-Stokes equations (512 training samples)  
   - **`airfoil`**: Compressible Euler equations  
   - **`darcy`**: Darcy equation  
   - **`burgers_zongyi`**: Burgers equation  
   - **`darcy_zongyi`**: Darcy equation  
   - **`fhn`**: FitzHugh-Nagumo equations over [0, 100]  
   - **`fhn_long`**: FitzHugh-Nagumo equations over [0, 200]  
   - **`hh`**: Hodgkin-Huxley equations  
   - **`crosstruss`**: Cross-shaped truss structure  

2. **`architecture`**  
   Specifies the architecture to use:
   - **`fno`**: Fourier Neural Operator  
   - **`cno`**: Convolution Neural Operator  

3. **`loss_function`**  
   Specifies the loss function for training. Choices include:
   - **`l1`**: $L^1$ relative norm  
   - **`l2`**: $L^2$ relative norm  
   - **`h1`**: $H^1$ relative norm  
   - **`l1_smooth`**: $L^1$ smooth loss 

4. **`mode`**  
   Specifies the mode for defining hyperparameters and architecture settings:
   - **`best`**: Optimized hyperparameters for better performance  
   - **`default`**: General-purpose default settings  

**Optional Arguments**
- **`--in_dist`**  
   (Boolean, default: `True`) Specifies whether the test set is in-distribution or out-of-distribution for supported datasets.

#### Saved Files
- Model: 
  ```./{architecture}/TrainedModels/{example}/model_{architecture}_test_{loss_function}_{mode}_hyperparams```
- Training parameters:
```./{architecture}/TrainedModels/{example}/{experiment_name}/hyperparams_train.json```  
- Architecture parameters:
```./{architecture}/TrainedModels/{example}/{experiment_name}/hyperparams_arc.txt```
- Simplified log:  
  ```./{architecture}/TrainedModels/{example}/{experiment_name}/errors.txt```

#### **TensorBoard Integration**
- Logs training metrics (e.g., loss, error rates) for real-time visualization.
- Summaries stored in `./{architecture}/TrainedModels/{example}/`.
- This feature requires TensorBoard to be installed. To start TensorBoard, run the following command:
  ```bash
  tensorboard --logdir ./{architecture}/TrainedModels/{example}/{experiment_name}
  ```


#### **Examples of Usage**
1. Training the Poisson (2D) example with FNO architecture using $L^2$ loss and loading the optimized hyper-parameters found:
```bash
python main.py poisson fno l2 best
```

2. Training a Hodgkin-Huxley (1D) example with CNO architecture using $L^1$ loss and default hyper-parameters:
```bash
python main.py hh cno l1 default
```

----------------------------------------------------------------------------------------------------------------------------