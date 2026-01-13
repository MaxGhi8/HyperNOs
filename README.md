# HyperNOs Documentation

## Introduction

HyperNOs is a Python project focused on the implementation of completely automatic, distributed and parallel neural operators hyperparameter optimization. The project aims to provide a framework for training neural operator models using Pytorch and Ray Tune for hyperparameter tuning.

## Supported Libraries

HyperNOs allows users to easily integrate and use models from popular neural-operator libraries or custom models. Is also very flexible and can be used with many different datasets.

I already implemented some examples of usage with the following popular libraries:

- **[NeuralOperator](https://github.com/neuraloperator/neuraloperator)**: Implement neural operator architectures like FNO, SFNO, TFNO, UNO, UQNO, GINO, FNOGNO, LocalNO, RNO, CODANO, OTNO.
- **[DeepXDE](https://github.com/lululxvi/deepxde)**: Implement operator learning models like DeepONet, MIONet, POD-DeepONet, POD-MIONet.

You can find examples of how to use these models in the `neural_operators/examples` directory with two dedicated subdirectories: `deepxde_lib` and `neuralop_lib`. There are implemented examples both for training a given architecture and for hyperparameter optimization routines.

## Visualization website

The project also includes a visualization [website: (https://hypernos.streamlit.app)](https://hypernos.streamlit.app) that allows users to visualize the results obtained with HyperNOs library.

## Installation

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

   If you want to use the `neuraloperator` library, since we use the cutting-edge features of NeuralOperator, please install the library directly from GitHub and not with pip, you can found the instructions [here](https://github.com/neuraloperator/neuraloperator).

   > [!WARNING]
   > For PyTorch, more attention may be needed during installation. We describe the default installation; however, we highly recommend following the [official documentation](https://pytorch.org/get-started/locally/) to install the correct version for your system. You can check your CUDA driver version by running `nvidia-smi` in your terminal to ensure compatibility.

3. Download the dataset using the `download_data.sh` script:

   ```bash
   ./download_data.sh
   ```

   > [!WARNING]
   > Only **for Windows** I recommend to install [WSL](https://ubuntu.com/desktop/wsl). Then open the WSL terminal and navigate where you have installed the HyperNOs library
   >
   > ```bash
   > cd /mnt/c/Users/<your_user>/<your_path_to_HyperNOs>
   > ```
   >
   > and then try to run the program with `./download_data.sh` if you get an error like `/bin/bash^M: bad interpreter. No such file or directory` this can be due to `CR` and `LF` in Windows. In this case try to run the following line and then rerun the program.
   >
   > ```bash
   > sed -i -e 's/\r$//' download_data.sh
   > ./download_data.sh
   > ```

4. If you want to download our trained model this have to be done in two steps. First of all clone the following github repository:
   ```bash
    git clone --depth=1 https://github.com/MaxGhi8/tests
   ```
   The previous repository contains the Tensorboard support for every model, the information about the training and the architecture's hyperparameters chosen. Then you can download running the following script and select the model that you want to download:
   ```bash
   ./download_trained_model.sh
   ```
   > [!WARNING]
   > As before, for **Windows**, if you are on WSL and get the error `/bin/bash^M: bad interpreter. No such file or directory` try to run `sed -i -e 's/\r$//' download_trained_model.sh` and then rerun the script `./download_trained_model.sh`.

## Citation

If you use our library please consider citing our paper:

```bibtex
@misc{ghiotto2025hypernosautomatedparallellibrary,
      title={HyperNOs: Automated and Parallel Library for Neural Operators Research},
      author={Massimiliano Ghiotto},
      year={2025},
      eprint={2503.18087},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.18087},
}
```
