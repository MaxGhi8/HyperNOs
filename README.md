# HyperNOs
## HyperNOs Documentation

### Introduction
HyperNOs is a Python project focused on the implementation of completely automatic, distributed and parallel neural operators hyperparameter optimization. The project aims to provide a framework for training neural operator models using Pytorch and Ray Tune for hyperparameter tuning.

### Installation
To set up the HyperNOs project, follow these steps:
1. Clone the repository:
   ```bash
   git clone --depth=1 https://github.com/MaxGhi8/HyperNOs.git
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
   pip install .
    ```

3. Download the dataset using the `download_data.sh` script:
   ```bash
   ./download_data.sh
   ```

4. If you want to download our trained model for each datasets you can clone our dedicated repository:
   ```bash
    git clone --depth=1 git@github.com:MaxGhi8/tests.git
   ```

