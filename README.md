# HyperNOs
## HyperNOs Documentation

### Introduction
HyperNOs is a Python project focused on the implementation of completely automatic, distributed and parallel neural operators hyperparameter optimization. The project aims to provide a framework for training neural operator models using Pytorch and Ray Tune for hyperparameter tuning.

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
TODO
3. Download the dataset using the `download_data.py` script:
   ```bash
   python download_data.py 
   ```
<!-- TODO  -->

### Usage
The main functionalities of the project are implemented in the `neural_operators` directory. Here are the key scripts and their purposes:

- `main_raytune.py`: Main script of the project, it runs hyper-parameters optimization using Ray Tune.
- `main_singlerun.py`: Script for load the selected dataset, architecture and hyperparameters, and train the model.
- `recover_model.py`: Script for recovering the trained model and then use it to make predictions or statistics.

----------------------------------------------------------------------------------------------------------------------------

### `main_raytune.py` file documentation

Handles user inputs to configure the script. Users can select:
- **Example**: Defines the differential equation problem to solve.
- **Architecture**: Model type.
- **Loss Function**: Training loss.
- **Mode**: Specifies whether to use `best` or `default` hyperparameters.
- **in_dist**: Optional flag for in-distribution or out-of-distribution test sets.

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
- In the folder ```~/ray_results/train_hyperparameters_{date}``` are stored the checkpoints for all the models for each trial.

#### **TensorBoard Integration**
- Logs training metrics for real-time visualization, in particular the loss function and the statistic about the time needed for training each model. 
- This feature requires TensorBoard to be installed. To start the visualization you have to run Tensorboard inside the folder where the logs are stored:
  ```bash
  tensorboard --logdir /tmp/ray/session_{date-time}_{id}/artifacts/{date-time}/train_hyperparameter_{date-time}/driver_artifacts
  ```


#### **Examples of Usage**
1. Training the Poisson (2D) example with FNO architecture using $L^2$ loss and loading the optimized hyper-parameters found:
```bash
python main_raytune.py poisson fno l2 best
```

2. Training a Hodgkin-Huxley (1D) example with CNO architecture using $L^1$ loss and default hyper-parameters:
```bash
python main_raytune.py hh cno l1 default
```

----------------------------------------------------------------------------------------------------------------------------

### `main_singlerun.py` file documentation

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
python main_singlerun.py poisson fno l2 best
```

2. Training a Hodgkin-Huxley (1D) example with CNO architecture using $L^1$ loss and default hyper-parameters:
```bash
python main_singlerun.py hh cno l1 default
```

----------------------------------------------------------------------------------------------------------------------------

### `recover_model.py` file documentation

To run the script, execute the following command in your terminal:
```bash
python recover_model.py <example> <architecture> <loss_function> <mode> [--in_dist]
```
The script recovers the trained model and uses it to make predictions or statistics. The basic functionalities are:
- **Recover Model**: Load the trained model.
- **Train Error**: Compute the mean error in $L^1$, $L^2$ and $H^1$ relative error on the train set and print it, this is useful to check with ```./{architecture}/TrainedModels/{example}/{experiment_name}/errors.txt``` file if the model is upload correctly and all is fine.
- **Test Error**: Compute the mean error in $L^1$, $L^2$ and $H^1$ relative error on the test set and print it, this is useful to check with ```./{architecture}/TrainedModels/{example}/{experiment_name}/errors.txt``` file if the model is upload correctly and all is fine.
- **Validation Error** Compute the mean and median error in $L^1$, $L^2$ and $H^1$ relative error on the validation set and print it, the validation error was not used before in the scripts so this is the final result of the accuracy of the model. For the dataset supported the validation set can be in distribution or out of distribution, this is specified with the flag `--in_dist`.
- **Predictions**: Make predictions on the test set and save the results in the folder  TODO
- **Execution time**: print the time needed for evaluate the model on the entire validation set and the time needed for a single prediction.
- **Plots**: Make the plots of the frequency of the error in the validation set and of some random examples.

**Arguments**

Handles user inputs to configure the script, the arguments are the same as the `main_singlerun.py` script and are needed to select the model to recover. 
- **Example**: Defines the differential equation problem to solve.
- **Architecture**: Model type.
- **Loss Function**: Training loss.
- **Mode**: Specifies whether to use `best` or `default` hyperparameters.
- **in_dist**: Optional flag for in-distribution or out-of-distribution test sets.

For detailed information about the arguments, refer to the `main_singlerun.py` section. 


#### Saved Files
<!-- TODO -->
TODO



#### **Examples of Usage**
1. Training the Poisson (2D) example with FNO architecture using $L^2$ loss and loading the optimized hyper-parameters found:
```bash
python main_singlerun.py poisson fno l2 best
```

2. Training a Hodgkin-Huxley (1D) example with CNO architecture using $L^1$ loss and default hyper-parameters:
```bash
python main_singlerun.py hh cno l1 default
```

----------------------------------------------------------------------------------------------------------------------------