# HyperNOs Documentation

### Introduction
HyperNOs is a Python project focused on the implementation of completely automatic, distributed and parallel neural operators hyperparameter optimization. The project aims to provide a framework for training neural operator models using Pytorch and Ray Tune for hyperparameter tuning.

### Installation
To set up the HyperNOs project, follow these steps:
1. Clone the repository:
   ```bash
   git clone --depth=1 https://github.com/MaxGhi8/HyperNOs.git
   cd HyperNOs
   ```
2. Install the required dependencies. It is recommended to create a virtual environment before installing the dependencies; I personally use [`pyenv`](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) (but others, like [`uv`](https://docs.astral.sh/uv/), are fine) and Python version `3.12.7` for this purpose:
    ```bash
    pyenv install 3.12.7
    pyenv virtualenv 3.12.7 hypernos
    pyenv activate hypernos
    ```
    Then, install the dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
    ```
   ⚠️ **Attention:** for pytorch can be need more attention in the installation, if you do not want the default installation please follow the [documentation](https://pytorch.org/get-started/locally/).

3. Download the dataset using the `download_data.sh` script:
   ```bash
   ./download_data.sh
   ```
   ⚠️ **Attention:** only **for Windows** I recommend to install [WSL](https://ubuntu.com/desktop/wsl). Then open the WSL terminal and navigate where you have installed the HyperNOs library
   ```bash
   cd /mnt/c/Users/<your_user>/<your_path_to_HyperNOs>
   ```
   and then try to run the program with `./download_data.sh` if you get an error like `/bin/bash^M: bad interpreter. No such file or directory` this can be due to `CR` and `LF` in Windows. In this case try to run the following line and then rerun the program.
   ```bash
   sed -i -e 's/\r$//' download_data.sh
   ./download_data.sh
   ```

4. If you want to download our trained model this have to be done in two steps. First of all clone the following github repository:
   ```bash
    git clone --depth=1 https://github.com/MaxGhi8/tests
   ```
   The previous repository contains the Tensorboard support for every model, the information about the training and the architecture's hyperparameters chosen. Then you can download running the following script and select the model that you want to download:
   ```bash
   ./download_trained_model.sh
   ```
   ⚠️ **Attention:** as before, for **Windows**, if you are on WSL and get the error `/bin/bash^M: bad interpreter. No such file or directory` try to run `sed -i -e 's/\r$//' download_trained_model.sh` and then rerun the script `./download_trained_model.sh`.

