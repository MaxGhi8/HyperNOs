"""
This file contains the necessary functions to load the data for the Fourier Neural Operator benchmarks.
"""

import os
import random

import h5py
import numpy as np
import scipy
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utilities import (
    FourierFeatures,
    FourierFeatures1D,
    UnitGaussianNormalizer,
    find_file,
)


#########################################
# function to load the data and model
#########################################
def NO_load_data_model(
    which_example: str,
    no_architecture,
    batch_size: int,
    training_samples: int,
    s: int = None,  # Make s optional
    in_dist: bool = True,
    search_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
):
    """
    Function to load the data and the model.
    If s is specified, it modifies the parameter "s" of the functions.
    Otherwise, the functions maintain their default value for "s".
    """
    # Define a dictionary to map which_example to the corresponding class and its arguments
    example_map = {
        "shear_layer": ShearLayer,
        "poisson": SinFrequency,
        "wave_0_5": WaveEquation,
        "allen": AllenCahn,
        "cont_tran": ContTranslation,
        "disc_tran": DiscContTranslation,
        "airfoil": Airfoil,
        "darcy": Darcy,
        ###
        "burgers_zongyi": Burgers_Zongyi,
        "darcy_zongyi": Darcy_Zongyi,
        # "navier_stokes_zongyi" #todo
        ###
        "fhn": FitzHughNagumo,
        "hh": HodgkinHuxley,
        "ord": OHaraRudy,
        ###
        "afieti_homogeneous_neumann": AFIETI,
        ###
        "crosstruss": CrossTruss,
        "stiffness_matrix": StiffnessMatrix,
    }

    # Define additional parameters for specific cases
    additional_params = {
        "fhn": ["_tf_100"],
        # "fhn_long": [ "time": "_tf_200" ],
        "afieti_homogeneous_neumann": ["dataset_homogeneous_Neumann.mat"],
    }

    # Check if the example is valid
    if which_example not in example_map:
        raise ValueError("The variable which_example is typed wrong")

    # Get the class for the example
    example_class = example_map[which_example]

    # Prepare the arguments for the class
    args = [no_architecture, batch_size, training_samples]
    kwargs = {
        "in_dist": in_dist,
        "search_path": search_path,
    }

    # Add additional kwargs for specific cases
    if which_example in additional_params:
        args = [*additional_params[which_example]] + args

    # Add s to kwargs if it is specified
    if s is not None:
        if which_example in ["fhn", "hh", "ord"]:
            points = 10080 if which_example == "ord" else 5040
            stride = points // s
            if abs(stride - points / s) > 1e-3:
                raise ValueError(
                    f"Invalid size, the s must divide the original grid size (in this example {points})"
                )
            kwargs["s"] = stride

        elif which_example == "crosstruss":
            points = 210
            stride = points // s
            if abs(stride - points / s) > 1e-3:
                raise ValueError(
                    f"Invalid size, the s must divide the original grid size (in this example {points})"
                )
            kwargs["s"] = stride

        elif which_example == "stiffness_matrix":
            points = 100
            stride = points // s
            if abs(stride - points / s) > 1e-3:
                raise ValueError(
                    f"Invalid size, the s must divide the original grid size (in this example {points})"
                )
            kwargs["s"] = stride

        else:
            kwargs["s"] = s

    # Create the example instance
    example = example_class(*args, **kwargs)

    return example


def concat_datasets(*datasets):

    def flatten(iterables):
        class MyIterable:
            def __iter__(self):
                for iterable in iterables:
                    for batch in iterable:
                        yield batch

        return MyIterable()

    class ConcatenatedDataset:
        def __init__(self):
            self.train_loader = flatten(
                list(map(lambda dataset: dataset.train_loader, datasets))
            )
            self.val_loader = flatten(
                list(map(lambda dataset: dataset.val_loader, datasets))
            )
            self.test_loader = flatten(
                list(map(lambda dataset: dataset.test_loader, datasets))
            )

    return ConcatenatedDataset()


#########################################
# Some functions needed for loading the Navier-Stokes data
#########################################
def samples_fft(u):
    return scipy.fft.fft2(u, norm="forward", workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm="forward", workers=-1).real


def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1 / N_old)
    sel = np.logical_and(freqs >= -N / 2, freqs <= N / 2 - 1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:, :, sel, :][:, :, :, sel]
    u_down = samples_ifft(u_hat_down)
    return u_down


# ------------------------------------------------------------------------------
# Navier-Stokes data (from Mishra CNO article)
#   From 0 to 750 : training samples (750)
#   From 1024 - 128 - 128 to 1024 - 128 : validation samples (128)
#   From 1024 - 128 to 1024 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)


class ShearLayerDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=750,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        self.in_dist = in_dist

        if in_dist:
            if self.s <= 64:
                self.file_data = find_file(
                    "NavierStokes_64x64_IN.h5", search_path
                )  # In-distribution file 64x64
            else:
                self.file_data = find_file(
                    "NavierStokes_128x128_IN.h5", search_path
                )  # In-distribution file 128x128
        else:
            self.file_data = find_file(
                "NavierStokes_128x128_OUT.h5", search_path
            )  # Out-of_-distribution file 128x128

        self.reader = h5py.File(self.file_data, "r")
        self.N_max = 1024

        self.n_val = 128
        self.n_test = 128
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model = -2.0383243560791016

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max - self.n_test

        # Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.s <= 64 and self.in_dist:
            inputs = (
                torch.from_numpy(
                    self.reader["Sample_" + str(index + self.start)]["input"][:]
                )
                .type(torch.float32)
                .reshape(1, self.s, self.s)
            )
            labels = (
                torch.from_numpy(
                    self.reader["Sample_" + str(index + self.start)]["output"][:]
                )
                .type(torch.float32)
                .reshape(1, self.s, self.s)
            )

            # Down-sample the data
            stride = 64 // self.s
            if abs(stride - 64 / self.s) > 1e-3:
                raise ValueError(
                    "Invalid size, the in_size must divide the original grid size (in this example 64)"
                )
            inputs = inputs[:, ::stride, ::stride]
            labels = labels[:, ::stride, ::stride]

        else:
            inputs = self.reader["Sample_" + str(index + self.start)]["input"][
                :
            ].reshape(1, 1, self.s, self.s)
            labels = self.reader["Sample_" + str(index + self.start)]["output"][
                :
            ].reshape(1, 1, self.s, self.s)

            if self.s < 128:
                inputs = downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)

            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class ShearLayer:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 128

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = ShearLayerDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            self.s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            ShearLayerDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                self.s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            ShearLayerDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                self.s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    @property
    def min_data(self):
        return self.train_set.min_data

    @property
    def max_data(self):
        return self.train_set.max_data

    @property
    def min_model(self):
        return self.train_set.min_model

    @property
    def max_model(self):
        return self.train_set.max_model


# ------------------------------------------------------------------------------
# Poisson data (from Mishra CNO article)
#   From 0 to 1024 : training samples (1024)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)


class SinFrequencyDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        if in_dist:
            self.file_data = find_file("PoissonData_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("PoissonData_64x64_OUT.h5", search_path)

        # Load normalization constants from the TRAINING set:
        self.reader = h5py.File(self.file_data, "r")
        self.min_data = self.reader["min_inp"][()]
        self.max_data = self.reader["max_inp"][()]
        self.min_model = self.reader["min_out"][()]
        self.max_model = self.reader["max_out"][()]
        self.s = s  # Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0

        # Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class SinFrequency:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = SinFrequencyDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            SinFrequencyDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            SinFrequencyDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    @property
    def min_data(self):
        return self.train_set.min_data

    @property
    def max_data(self):
        return self.train_set.max_data

    @property
    def min_model(self):
        return self.train_set.min_model

    @property
    def max_model(self):
        return self.train_set.max_model


# ------------------------------------------------------------------------------
# Wave data (from Mishra CNO article)
#   From 0 to 512 : training samples (512)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)


class WaveEquationDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=512,
        t=5,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        if in_dist:
            self.file_data = find_file("WaveData_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("WaveData_64x64_OUT.h5", search_path)

        self.reader = h5py.File(self.file_data, "r")

        # Load normalization constants:
        self.min_data = self.reader["min_u0"][()]
        self.max_data = self.reader["max_u0"][()]
        self.min_model = self.reader["min_u"][()]
        self.max_model = self.reader["max_u"][()]

        self.t = t

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0

        self.s = s
        assert self.s <= 64

        # If the reader changed:
        self.reader = h5py.File(self.file_data, "r")

        # Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start) + "_t_" + str(self.t)][
                    "input"
                ][:]
            )
            .type(torch.float32)
            .reshape(1, self.s, self.s)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start) + "_t_" + str(self.t)][
                    "output"
                ][:]
            )
            .type(torch.float32)
            .reshape(1, self.s, self.s)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class WaveEquation:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = WaveEquationDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            5,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            WaveEquationDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                5,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            WaveEquationDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                5,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    @property
    def min_data(self):
        return self.train_set.min_data

    @property
    def max_data(self):
        return self.train_set.max_data

    @property
    def min_model(self):
        return self.train_set.min_model

    @property
    def max_model(self):
        return self.train_set.max_model


# ------------------------------------------------------------------------------
# Allen-Cahn data (from Mishra CNO article)
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)


class AllenCahnDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=256,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        if in_dist:
            self.file_data = find_file("AllenCahn_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("AllenCahn_64x64_OUT.h5", search_path)
        self.reader = h5py.File(self.file_data, "r")

        # Load normalization constants:
        self.min_data = self.reader["min_u0"][()]
        self.max_data = self.reader["max_u0"][()]
        self.min_model = self.reader["min_u"][()]
        self.max_model = self.reader["max_u"][()]

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 256
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 256 + 128
            else:
                self.length = 128
                self.start = 0

        # Default:
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class AllenCahn:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = AllenCahnDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            AllenCahnDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            AllenCahnDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    @property
    def min_data(self):
        return self.train_set.min_data

    @property
    def max_data(self):
        return self.train_set.max_data

    @property
    def min_model(self):
        return self.train_set.min_model

    @property
    def max_model(self):
        return self.train_set.max_model


# ------------------------------------------------------------------------------
# Smooth Transport data (from Mishra CNO article)
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)


class ContTranslationDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        # The data is already normalized
        if in_dist:
            self.file_data = find_file("ContTranslation_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("ContTranslation_64x64_OUT.h5", search_path)

        self.reader = h5py.File(self.file_data, "r")

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512 + 256
            else:
                self.length = 256
                self.start = 0

        # Default:
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class ContTranslation:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = ContTranslationDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            ContTranslationDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            ContTranslationDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    # When data is already normalized I set the min and max to 0 and 1 (have no effects)
    @property
    def min_data(self):
        return 0.0

    @property
    def max_data(self):
        return 1.0

    @property
    def min_model(self):
        return 0.0

    @property
    def max_model(self):
        return 1.0


# ------------------------------------------------------------------------------
# Discontinuous Transport data (from Mishra CNO article)
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)


class DiscContTranslationDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        # The data is already normalized
        if in_dist:
            self.file_data = find_file("DiscTranslation_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("DiscTranslation_64x64_OUT.h5", search_path)

        if which == "training":
            self.length = training_samples
            self.start = 0

        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512 + 256
            else:
                self.length = 256
                self.start = 0

        self.reader = h5py.File(self.file_data, "r")

        # Default:
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class DiscContTranslation:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = DiscContTranslationDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            DiscContTranslationDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            DiscContTranslationDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    # When data is already normalized I set the min and max to 0 and 1 (have no effects)
    @property
    def min_data(self):
        return 0.0

    @property
    def max_data(self):
        return 1.0

    @property
    def min_model(self):
        return 0.0

    @property
    def max_model(self):
        return 1.0


# ------------------------------------------------------------------------------
# Compressible Euler data (from Mishra CNO article)
#   From 0 to 750 : training samples (750)
#   From 750 to 750 + 128 : validation samples (128)
#   From 750 + 128 to 750 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)


class AirfoilDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=512,
        s=128,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        # We DO NOT normalize the data in this case
        if in_dist:
            self.file_data = find_file("Airfoil_128x128_IN.h5", search_path)
        else:
            self.file_data = find_file("Airfoil_128x128_OUT.h5", search_path)

        if which == "training":
            self.length = training_samples
            self.start = 0

        elif which == "validation":
            self.length = 128
            self.start = 750
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 750 + 128
            else:
                self.length = 128
                self.start = 0

        self.reader = h5py.File(self.file_data, "r")

        # Default:
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 128, 128)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 128, 128)
        )
        # post process the output to fit the domain
        labels[inputs == 1] = 1

        # Down-sample the data
        stride = 128 // self.s
        if abs(stride - 128 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 128)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class Airfoil:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=512,
        s=128,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 128

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = AirfoilDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            AirfoilDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            AirfoilDataset(
                "test",
                self.N_Fourier_F,
                training_samples,
                s,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    # When data is already normalized I set the min and max to 0 and 1 (have no effects)
    @property
    def min_data(self):
        return 0.0

    @property
    def max_data(self):
        return 1.0

    @property
    def min_model(self):
        return 0.0

    @property
    def max_model(self):
        return 1.0


# ------------------------------------------------------------------------------
# Darcy Flow data (from Mishra CNO article)
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)


class DarcyDataset(Dataset):
    def __init__(
        self,
        which="training",
        nf=0,
        training_samples=256,
        s=64,
        insample=True,
        search_path="/",
    ):
        self.s = s
        if insample:
            self.file_data = find_file("Darcy_64x64_IN.h5", search_path)
        else:
            self.file_data = find_file("Darcy_64x64_OUT.h5", search_path)

        self.reader = h5py.File(self.file_data, "r")

        self.min_data = self.reader["min_inp"][()]
        self.max_data = self.reader["max_inp"][()]
        self.min_model = self.reader["min_out"][()]
        self.max_model = self.reader["max_out"][()]

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "testing":
            if insample:
                self.length = 128
                self.start = training_samples + 128
            else:
                self.length = 128
                self.start = 0

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = (
            torch.from_numpy(
                self.reader["sample_" + str(index + self.start)]["input"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )
        labels = (
            torch.from_numpy(
                self.reader["sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(1, 64, 64)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        # Down-sample the data
        stride = 64 // self.s
        if abs(stride - 64 / self.s) > 1e-3:
            raise ValueError(
                "Invalid size, the in_size must divide the original grid size (in this example 64)"
            )
        inputs = inputs[:, ::stride, ::stride]
        labels = labels[:, ::stride, ::stride]

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs.permute(1, 2, 0), labels.permute(1, 2, 0)

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid


class Darcy:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=256,
        s=64,
        in_dist=True,
        search_path="/",
    ):
        self.s = s
        assert self.s <= 64

        self.N_Fourier_F = network_properties["FourierF"]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = DarcyDataset(
            "training",
            self.N_Fourier_F,
            training_samples,
            s=self.s,
            search_path=search_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            DarcyDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s=self.s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            DarcyDataset(
                "testing",
                self.N_Fourier_F,
                training_samples,
                s=self.s,
                insample=in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

    @property
    def min_data(self):
        return self.train_set.min_data

    @property
    def max_data(self):
        return self.train_set.max_data

    @property
    def min_model(self):
        return self.train_set.min_model

    @property
    def max_model(self):
        return self.train_set.max_model


# ------------------------------------------------------------------------------
# Darcy's Equation data (from Zongyi Li FNO article)
#   Training samples (1000)
#   Testing samples  (200)
#   Validation samples (200)


@jaxtyped(typechecker=beartype)
def MatReader_darcy(
    file_path: str,
) -> tuple[Float[Tensor, "n_samples n_x n_y"], Float[Tensor, "n_samples n_x n_y"]]:
    """
    Function to read .mat files for the darcy_zongyi problem.
    I prefer to makes a separate function to use jaxtyped decorator to check dimensions.

    Parameters
    ----------
    file_path : string
        path to the .mat file

    Returns
    -------
    a : tensor
        point-wise evaluation of the coefficient tensor a(x) in the Darcy equation
    u : tensor
        point-wise evaluation of the solution approximation of the solution u(x) of the Darcy equation

    Note: The grid is uniform and with 421 points in each direction.

    """
    data = scipy.io.loadmat(file_path)
    a = data["coeff"]
    a = torch.from_numpy(a).float()
    u = data["sol"]
    u = torch.from_numpy(u).float()
    a, u = a.to("cpu"), u.to("cpu")
    return a, u


class Darcy_Zongyi:
    def __init__(
        self,
        network_properties,
        batch_size,
        ntrain=1000,
        ntest=200,
        s=5,
        in_dist=True,
        search_path="/",
    ):
        assert in_dist, "Out-of-distribution testing samples are not available"

        # s = 5 --> (421)//s + 1 = 85 points per direction

        self.TrainDataPath = find_file("piececonst_r421_N1024_smooth1.mat", search_path)
        a_train, u_train = MatReader_darcy(self.TrainDataPath)
        self.TestDataPath = find_file("piececonst_r421_N1024_smooth2.mat", search_path)
        a_test, u_test = MatReader_darcy(self.TestDataPath)

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Training data
        idx_train = torch.randperm(1024, device="cpu", generator=g)[:ntrain]
        a_train, u_train = a_train[idx_train, ::s, ::s], u_train[idx_train, ::s, ::s]
        # Compute mean and std (for gaussian point-wise normalization)
        a_normalizer = UnitGaussianNormalizer(a_train)
        u_normalizer = UnitGaussianNormalizer(u_train)
        # Normalize
        a_train = a_normalizer.encode(a_train).unsqueeze(-1)
        u_train = u_normalizer.encode(u_train).unsqueeze(-1)

        # Validation data
        idx_test_tot = torch.randperm(1024, device="cpu", generator=g)
        idx_val = idx_test_tot[:ntest]
        a_val, u_val = a_test[idx_val, ::s, ::s], u_test[idx_val, ::s, ::s]
        a_val = a_normalizer.encode(a_val).unsqueeze(-1)
        u_val = u_normalizer.encode(u_val).unsqueeze(-1)

        # Test data
        idx_test = idx_test_tot[ntest : 2 * ntest]
        a_test, u_test = a_test[idx_test, ::s, ::s], u_test[idx_test, ::s, ::s]
        a_test = a_normalizer.encode(a_test).unsqueeze(-1)
        u_test = u_normalizer.encode(u_test).unsqueeze(-1)

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(a_test.shape[1])
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_x, n_y, 2*ff)

            a_train = torch.cat(
                (a_train, ff_grid.repeat(a_train.shape[0], 1, 1, 1)), -1
            )
            a_val = torch.cat((a_val, ff_grid.repeat(a_val.shape[0], 1, 1, 1)), -1)
            a_test = torch.cat((a_test, ff_grid.repeat(a_test.shape[0], 1, 1, 1)), -1)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = a_test.shape[1]

    def get_grid(self, res):
        x = torch.linspace(0, 1, res)
        y = torch.linspace(0, 1, res)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


# ------------------------------------------------------------------------------
# 1D Burger's Equation data (from Zongyi Li FNO article)
#   Training samples (1000)
#   Testing samples  (200)
#   Validation samples (200)


@jaxtyped(typechecker=beartype)
def MatReader_burgers(
    file_path: str,
) -> tuple[Float[Tensor, "n_samples n_x"], Float[Tensor, "n_samples n_x"]]:
    """
    Function to read .mat files for the burgers_zongyi problem
    """
    data = scipy.io.loadmat(file_path)
    a = data["a"]
    a = torch.from_numpy(a).float()
    u = data["u"]
    u = torch.from_numpy(u).float()
    a, u = a.to("cpu"), u.to("cpu")
    return a, u


class Burgers_Zongyi:
    def __init__(
        self,
        network_properties,
        batch_size,
        ntrain=1000,
        ntest=200,
        s=8,
        in_dist=True,
        search_path="/",
    ):
        assert in_dist, "Out-of-distribution testing samples are not available"
        # s = 8 --> (8192)//s + 1 = 1025 points

        self.DataPath = find_file("burgers_data_R10.mat", search_path)
        a, u = MatReader_burgers(self.DataPath)

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Training data
        idx = torch.randperm(2048, device="cpu", generator=g)
        idx_train = idx[:ntrain]
        a_train, u_train = a[idx_train, ::s], u[idx_train, ::s]
        # Compute mean and std (for gaussian point-wise normalization)
        a_normalizer = UnitGaussianNormalizer(a_train)
        u_normalizer = UnitGaussianNormalizer(u_train)
        # Normalize
        a_train = a_normalizer.encode(a_train).unsqueeze(-1)
        u_train = u_normalizer.encode(u_train).unsqueeze(-1)

        # Validation data
        idx_val = idx[ntrain : ntrain + ntest]
        a_val, u_val = a[idx_val, ::s], u[idx_val, ::s]
        a_val = a_normalizer.encode(a_val).unsqueeze(-1)
        u_val = u_normalizer.encode(u_val).unsqueeze(-1)

        # Test data
        idx_test = idx[ntrain + ntest : ntrain + 2 * ntest]
        a_test, u_test = a[idx_test, ::s], u[idx_test, ::s]
        a_test = a_normalizer.encode(a_test).unsqueeze(-1)
        u_test = u_normalizer.encode(u_test).unsqueeze(-1)

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(a_test.shape[1])
            FF = FourierFeatures1D(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_y, 2*ff)

            a_train = torch.cat((a_train, ff_grid.repeat(a_train.shape[0], 1, 1)), -1)
            a_val = torch.cat((a_val, ff_grid.repeat(a_val.shape[0], 1, 1)), -1)
            a_test = torch.cat((a_test, ff_grid.repeat(a_test.shape[0], 1, 1)), -1)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = a_test.shape[1]

    def get_grid(self, res):
        return torch.linspace(0, 1, res).unsqueeze(-1)


# ------------------------------------------------------------------------------
# Navier Stokes's Equation data (from Zongyi Li FNO article)


# ------------------------------------------------------------------------------
# 0D FitzHugh-Nagumo model data
#   Training samples (3000)
#   Testing samples  (375)
#   Validation samples (375)


@jaxtyped(typechecker=beartype)
def MatReader_fhn(
    file_path: str,
) -> tuple[
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
]:
    """
    Function to read .mat files for the FitzHugh-Nagumo problem
    """
    data = scipy.io.loadmat(file_path)
    a = data["I_app_dataset"]
    a = torch.from_numpy(a).transpose(0, 1).float()
    v = data["V_dataset"]
    v = torch.from_numpy(v).transpose(0, 1).float()
    w = data["w_dataset"]
    w = torch.from_numpy(w).transpose(0, 1).float()
    a, v, w = a.to("cpu"), v.to("cpu"), w.to("cpu")
    return a, v, w


class FitzHughNagumo:
    def __init__(
        self,
        time: str,
        network_properties,
        batch_size,
        training_samples=3000,
        s=4,
        in_dist=True,
        search_path="/",
    ):
        assert training_samples <= 3000, "Training samples must be less than 3000"
        assert in_dist, "Out-of-distribution testing samples are not available"
        # s = 4 --> (5040)//s  = 1260 points

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Training data
        self.TrainDataPath = find_file(
            f"Training_dataset_FHN_n_3000_points_5040{time}.mat", search_path
        )
        a, v, w = MatReader_fhn(self.TrainDataPath)
        a_train, v_train, w_train = (
            a[:training_samples, ::s].unsqueeze(-1),
            v[:training_samples, ::s].unsqueeze(-1),
            w[:training_samples, ::s].unsqueeze(-1),
        )
        # Compute mean and std (for gaussian point-wise normalization)
        self.a_normalizer = UnitGaussianNormalizer(a_train)
        self.v_normalizer = UnitGaussianNormalizer(v_train)
        self.w_normalizer = UnitGaussianNormalizer(w_train)
        # Normalize
        a_train = self.a_normalizer.encode(a_train)
        u_train = torch.concatenate(
            (self.v_normalizer.encode(v_train), self.w_normalizer.encode(w_train)),
            dim=2,
        )

        # Validation data
        self.ValDataPath = find_file(
            f"Validation_dataset_FHN_n_375_points_5040{time}.mat", search_path
        )
        a, v, w = MatReader_fhn(self.ValDataPath)
        a_val, v_val, w_val = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            w[:, ::s].unsqueeze(-1),
        )
        a_val = self.a_normalizer.encode(a_val)
        u_val = torch.concatenate(
            (self.v_normalizer.encode(v_val), self.w_normalizer.encode(w_val)), dim=2
        )

        # Test data
        self.TestDataPath = find_file(
            f"Test_dataset_FHN_n_375_points_5040{time}.mat", search_path
        )
        a, v, w = MatReader_fhn(self.TestDataPath)
        a_test, v_test, w_test = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            w[:, ::s].unsqueeze(-1),
        )
        a_test = self.a_normalizer.encode(a_test)
        u_test = torch.concatenate(
            (self.v_normalizer.encode(v_test), self.w_normalizer.encode(w_test)), dim=2
        )

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(a_test.shape[1])
            FF = FourierFeatures1D(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_y, 2*ff)

            a_train = torch.cat((a_train, ff_grid.repeat(a_train.shape[0], 1, 1)), -1)
            a_val = torch.cat((a_val, ff_grid.repeat(a_val.shape[0], 1, 1)), -1)
            a_test = torch.cat((a_test, ff_grid.repeat(a_test.shape[0], 1, 1)), -1)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = a_test.shape[1]

    def get_grid(self, res):
        return torch.linspace(0, 1, res).unsqueeze(-1)


# ------------------------------------------------------------------------------
# 0D Hodgkin-Huxley model data
#   Training samples (3000)
#   Testing samples  (375)
#   Validation samples (375)


@jaxtyped(typechecker=beartype)
def MatReader_hh(
    file_path: str,
) -> tuple[
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
    Float[Tensor, "n_samples n_x"],
]:
    """
    Function to read .mat files for the Hodgkin-Huxley problem
    """
    data = scipy.io.loadmat(file_path)

    a = data["I_app_dataset"]
    a = torch.from_numpy(a).transpose(0, 1).float()

    v = data["V_dataset"]
    v = torch.from_numpy(v).transpose(0, 1).float()

    m = data["m_dataset"]
    m = torch.from_numpy(m).transpose(0, 1).float()

    h = data["h_dataset"]
    h = torch.from_numpy(h).transpose(0, 1).float()

    n = data["n_dataset"]
    n = torch.from_numpy(n).transpose(0, 1).float()

    return a.to("cpu"), v.to("cpu"), m.to("cpu"), h.to("cpu"), n.to("cpu")


class HodgkinHuxley:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=3000,
        s=4,
        in_dist=True,
        search_path="/",
    ):
        assert training_samples <= 3000, "Training samples must be less than 3000"
        assert in_dist, "Out-of-distribution testing samples are not available"

        # s = 4 --> (5040)//s  = 1260 points

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Training data
        self.TrainDataPath = find_file(
            "Training_dataset_HH_n_3000_points_5040_tf_100.mat", search_path
        )
        a, v, m, h, n = MatReader_hh(self.TrainDataPath)
        a_train, v_train, m_train, h_train, n_train = (
            a[:training_samples, ::s].unsqueeze(-1),
            v[:training_samples, ::s].unsqueeze(-1),
            m[:training_samples, ::s].unsqueeze(-1),
            h[:training_samples, ::s].unsqueeze(-1),
            n[:training_samples, ::s].unsqueeze(-1),
        )
        # Compute mean and std (for gaussian point-wise normalization)
        self.a_normalizer = UnitGaussianNormalizer(a_train)
        self.v_normalizer = UnitGaussianNormalizer(v_train)
        self.m_normalizer = UnitGaussianNormalizer(m_train)
        self.h_normalizer = UnitGaussianNormalizer(h_train)
        self.n_normalizer = UnitGaussianNormalizer(n_train)
        # Normalize
        a_train = self.a_normalizer.encode(a_train)
        u_train = torch.concatenate(
            (
                self.v_normalizer.encode(v_train),
                self.m_normalizer.encode(m_train),
                self.h_normalizer.encode(h_train),
                self.n_normalizer.encode(n_train),
            ),
            dim=2,
        )

        # Validation data
        self.ValDataPath = find_file(
            "Validation_dataset_HH_n_375_points_5040_tf_100.mat", search_path
        )
        a, v, m, h, n = MatReader_hh(self.ValDataPath)
        a_val, v_val, m_val, h_val, n_val = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            m[:, ::s].unsqueeze(-1),
            h[:, ::s].unsqueeze(-1),
            n[:, ::s].unsqueeze(-1),
        )
        a_val = self.a_normalizer.encode(a_val)
        u_val = torch.concatenate(
            (
                self.v_normalizer.encode(v_val),
                self.m_normalizer.encode(m_val),
                self.h_normalizer.encode(h_val),
                self.n_normalizer.encode(n_val),
            ),
            dim=2,
        )

        # Test data
        self.TestDataPath = find_file(
            "Test_dataset_HH_n_375_points_5040_tf_100.mat", search_path
        )
        a, v, m, h, n = MatReader_hh(self.TestDataPath)
        a_test, v_test, m_test, h_test, n_test = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            m[:, ::s].unsqueeze(-1),
            h[:, ::s].unsqueeze(-1),
            n[:, ::s].unsqueeze(-1),
        )
        a_test = self.a_normalizer.encode(a_test)
        u_test = torch.concatenate(
            (
                self.v_normalizer.encode(v_test),
                self.m_normalizer.encode(m_test),
                self.h_normalizer.encode(h_test),
                self.n_normalizer.encode(n_test),
            ),
            dim=2,
        )

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(a_test.shape[1])
            FF = FourierFeatures1D(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_y, 2*ff)

            a_train = torch.cat((a_train, ff_grid.repeat(a_train.shape[0], 1, 1)), -1)
            a_val = torch.cat((a_val, ff_grid.repeat(a_val.shape[0], 1, 1)), -1)
            a_test = torch.cat((a_test, ff_grid.repeat(a_test.shape[0], 1, 1)), -1)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = a_test.shape[1]

    def get_grid(self, res):
        return torch.linspace(0, 1, res).unsqueeze(-1)


# ------------------------------------------------------------------------------
# 0D O'Hara-Rudy model data
#   Training samples (3000)
#   Testing samples  (375)
#   Validation samples (375)


def MatReader_ord(fields: list[str], file_path: str) -> dict[str, torch.Tensor]:
    """
    Function to read .mat files for the new dataset.
    Returns a dictionary of tensors for each dataset field.
    """
    data = scipy.io.loadmat(file_path)
    tensors = {}

    # Convert each field to a tensor and store in the dictionary
    for field in fields:
        if field in data:
            tensor = torch.from_numpy(data[field]).transpose(0, 1).float()
            tensors[field] = tensor.to("cpu")
        else:
            raise KeyError(f"Field {field} not found in the dataset.")

    return tensors


class OHaraRudy:
    def __init__(
        self,
        network_properties,
        batch_size,
        training_samples=3000,
        s=1,
        in_dist=True,
        search_path="/",
    ):
        assert training_samples <= 3000, "Training samples must be less than 3000"
        assert in_dist, "Out-of-distribution testing samples are not available"
        assert s == 1, "I want to test with all points"

        # List of all dataset fields
        self.fields = [
            "CaMK_trap_dataset",
            "Ca_i_dataset",
            "Ca_jsr_dataset",
            "Ca_nsr_dataset",
            "Ca_ss_dataset",
            "I_app_dataset",
            "J_rel_CaMK_dataset",
            "J_rel_NP_dataset",
            "K_i_dataset",
            "K_ss_dataset",
            "Na_i_dataset",
            "Na_ss_dataset",
            "V_dataset",
            "a_CaMK_dataset",
            "a_dataset",
            "d_dataset",
            "f_CaMK_fast_dataset",
            "f_Ca_CaMK_fast_dataset",
            "f_Ca_fast_dataset",
            "f_Ca_slow_dataset",
            "f_fast_dataset",
            "f_slow_dataset",
            "h_CaMK_slow_dataset",
            "h_L_CaMK_dataset",
            "h_L_dataset",
            "h_fast_dataset",
            "h_slow_dataset",
            "i_CaMK_fast_dataset",
            "i_CaMK_slow_dataset",
            "i_fast_dataset",
            "i_slow_dataset",
            "j_CaMK_dataset",
            "j_Ca_dataset",
            "j_dataset",
            "m_L_dataset",
            "m_dataset",
            "n_dataset",
            "x_k1_dataset",
            "x_r_fast_dataset",
            "x_r_slow_dataset",
            "x_s1_dataset",
            "x_s2_dataset",
        ]

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        #### Training data
        self.TrainDataPath = find_file(
            "Training_dataset_ORd_n_3000_points_5040_tf_500.mat", search_path
        )
        dict_train = MatReader_ord(self.fields, self.TrainDataPath)

        # Compute mean and std (for gaussian point-wise normalization)
        self.dict_normalizers = {}
        for field in self.fields:
            # extraction
            dict_train[field] = dict_train[field][:training_samples, ::s].unsqueeze(-1)
            # normalizer
            self.dict_normalizers[field] = UnitGaussianNormalizer(dict_train[field])
            # normalization
            dict_train[field] = self.dict_normalizers[field].encode(dict_train[field])

        # Concatenation
        a_train = dict_train["I_app_dataset"]
        self.fields_to_concat = [
            field for field in self.fields if field != "I_app_dataset"
        ]
        u_train = torch.cat(
            [dict_train[field] for field in self.fields_to_concat], dim=2
        )

        #### Validation data
        self.ValDataPath = find_file(
            "Validation_dataset_ORd_n_375_points_5040_tf_500.mat", search_path
        )
        dict_val = MatReader_ord(self.fields, self.ValDataPath)

        # Compute mean and std (for gaussian point-wise normalization)
        for field in self.fields:
            # extraction
            dict_val[field] = dict_val[field][:training_samples, ::s].unsqueeze(-1)
            # normalization
            dict_val[field] = self.dict_normalizers[field].encode(dict_val[field])

        # Concatenation
        a_val = dict_val["I_app_dataset"]
        u_val = torch.cat([dict_val[field] for field in self.fields_to_concat], dim=2)

        #### Validation data
        self.TestDataPath = find_file(
            "Test_dataset_ORd_n_375_points_5040_tf_500.mat", search_path
        )
        dict_test = MatReader_ord(self.fields, self.TestDataPath)

        # Compute mean and std (for gaussian point-wise normalization)
        for field in self.fields:
            # extraction
            dict_test[field] = dict_test[field][:training_samples, ::s].unsqueeze(-1)
            # normalization
            dict_test[field] = self.dict_normalizers[field].encode(dict_test[field])

        # Concatenation
        a_test = dict_test["I_app_dataset"]
        u_test = torch.cat([dict_test[field] for field in self.fields_to_concat], dim=2)

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(a_test.shape[1])
            FF = FourierFeatures1D(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_y, 2*ff)

            a_train = torch.cat((a_train, ff_grid.repeat(a_train.shape[0], 1, 1)), -1)
            a_val = torch.cat((a_val, ff_grid.repeat(a_val.shape[0], 1, 1)), -1)
            a_test = torch.cat((a_test, ff_grid.repeat(a_test.shape[0], 1, 1)), -1)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.s = a_test.shape[1]

    def get_grid(self, res):
        return torch.linspace(0, 1, res).unsqueeze(-1)


# ------------------------------------------------------------------------------
# AF-IETI data
# Training samples (16000)
# Testing samples (2000)
# Validation samples (2000)


class AFIETI:
    def __init__(
        self,
        filename: str,
        network_properties: dict,
        batch_size: int,
        training_samples: int,
        s=1,
        in_dist=True,
        search_path="/",
    ):
        assert training_samples <= 16000, "Training samples must be less than 3000"
        assert in_dist, "Out-of-distribution testing samples are not available"
        assert s == 1, "Sampling rate must be 1, no subsampling allowed in this example"

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        self.TrainDataPath = find_file(filename, search_path)
        reader = h5py.File(self.TrainDataPath, "r")
        input = torch.from_numpy(reader["input"][:]).type(torch.float32)
        output = torch.from_numpy(reader["output"][:]).type(torch.float32)

        # Training data
        input_train, output_train = (
            input[:training_samples, ::s],
            output[:training_samples, ::s],
        )

        # Compute mean and std (for gaussian point-wise normalization)
        self.input_normalizer = UnitGaussianNormalizer(input_train)
        self.output_normalizer = UnitGaussianNormalizer(output_train)

        # Normalize
        # input_train = self.input_normalizer.encode(input_train)
        # output_train = self.output_normalizer.encode(output_train)

        # Validation data
        input_val, output_val = (
            input[training_samples : training_samples + 2000, ::s],
            output[training_samples : training_samples + 2000, ::s],
        )
        # input_val = self.input_normalizer.encode(input_val)
        # output_val = self.output_normalizer.encode(output_val)

        # Test data
        input_test, output_test = (
            input[training_samples + 2000 :, ::s],
            output[training_samples + 2000 :, ::s],
        )
        # input_test = self.input_normalizer.encode(input_test)
        # output_test = self.output_normalizer.encode(output_test)

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(input_test.shape[1])
            FF = FourierFeatures1D(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_y, 2*ff)

            input_train = torch.cat(
                (input_train, ff_grid.repeat(input_train.shape[0], 1, 1)), -1
            )
            input_val = torch.cat(
                (input_val, ff_grid.repeat(input_val.shape[0], 1, 1)), -1
            )
            input_test = torch.cat(
                (input_test, ff_grid.repeat(input_test.shape[0], 1, 1)), -1
            )

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(input_train, output_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(input_val, output_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(input_test, output_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s_in = input_test.shape[1]
        self.s_out = output_test.shape[1]

    def get_grid(self, res):
        return torch.linspace(0, 1, res).unsqueeze(-1)


# ------------------------------------------------------------------------------
# Cross truss data
# Training samples (1000)
# Testing samples (250)
# Validation samples (250)


class CrossTruss(Dataset):
    def __init__(
        self,
        network_properties,
        batch_size,
        ntrain=1000,
        ntest=250,
        s=2,
        in_dist=True,
        search_path="/",
    ):
        assert in_dist, "Out-of-distribution testing samples are not available"
        # s = 2 --> (211)//s + 1 = 105 points

        # Read the data
        self.file_data = find_file("elasticity_dataset_211x211_n1500.h5", search_path)
        self.reader = h5py.File(self.file_data, "r")
        inputs = torch.from_numpy(self.reader["domain"][:]).type(torch.float32)
        outputs = torch.from_numpy(self.reader["displacements"][:]).type(torch.float32)
        # Train set
        inputs_train = inputs[:ntrain, ::s, ::s].unsqueeze(-1)
        outputs_train = outputs[:ntrain, ::s, ::s]
        # Validation set
        inputs_val = inputs[ntrain : ntrain + ntest, ::s, ::s].unsqueeze(-1)
        outputs_val = outputs[ntrain : ntrain + ntest, ::s, ::s]
        # Test set
        inputs_test = inputs[ntrain + ntest : ntrain + 2 * ntest, ::s, ::s].unsqueeze(
            -1
        )
        outputs_test = outputs[ntrain + ntest : ntrain + 2 * ntest, ::s, ::s]

        # Normalize the outputs (min-max normalization), the inputs are already normalized in {0,1}
        self.min_x = torch.min(outputs_train[:, :, :, 0])
        self.max_x = torch.max(outputs_train[:, :, :, 0])
        outputs_train[:, :, :, 0] = (outputs_train[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )
        outputs_val[:, :, :, 0] = (outputs_val[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )
        outputs_test[:, :, :, 0] = (outputs_test[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )

        self.min_y = torch.min(outputs_train[:, :, :, 1])
        self.max_y = torch.max(outputs_train[:, :, :, 1])
        outputs_train[:, :, :, 1] = (outputs_train[:, :, :, 1] - self.min_y) / (
            self.max_y - self.min_y
        )
        outputs_val[:, :, :, 1] = (outputs_val[:, :, :, 1] - self.min_y) / (
            self.max_y - self.min_y
        )
        outputs_test[:, :, :, 1] = (outputs_test[:, :, :, 1] - self.min_y) / (
            self.max_y - self.min_y
        )

        # post processing the outputs to fit to the domain
        for i in range(outputs_train.shape[-1]):
            outputs_train[:, :, :, [i]] *= inputs_train

        for i in range(outputs_val.shape[-1]):
            outputs_val[:, :, :, [i]] *= inputs_val

        for i in range(outputs_test.shape[-1]):
            outputs_test[:, :, :, [i]] *= inputs_test

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(inputs_test.shape[1])
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_x, n_y, 2*ff)

            inputs_train = torch.cat(
                (inputs_train, ff_grid.repeat(inputs_train.shape[0], 1, 1, 1)), -1
            )
            inputs_val = torch.cat(
                (inputs_val, ff_grid.repeat(inputs_val.shape[0], 1, 1, 1)), -1
            )
            inputs_test = torch.cat(
                (inputs_test, ff_grid.repeat(inputs_test.shape[0], 1, 1, 1)), -1
            )

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(inputs_train, outputs_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(inputs_val, outputs_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(inputs_test, outputs_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = inputs_test.shape[1]

    def get_grid(self, res):
        x = torch.linspace(0, 1, res)
        y = torch.linspace(0, 1, res)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


# ------------------------------------------------------------------------------
# Stiffness matrix data
# Training samples (1500)
# Testing samples (250)
# Validation samples (250)


class StiffnessMatrix(Dataset):
    def __init__(
        self,
        network_properties,
        batch_size,
        ntrain=1500,
        ntest=250,
        s=1,
        in_dist=True,
        search_path="/",
    ):
        assert in_dist, "Out-of-distribution testing samples are not available"
        # s = 1 --> we have 100x100 points in input and 8x8 points in output

        # Read the data
        self.file_data = find_file(
            "domain_100x100_stiffness_matrix_8x8_n2000.h5", search_path
        )
        self.reader = h5py.File(self.file_data, "r")
        inputs = torch.from_numpy(self.reader["domain"][:]).type(torch.float32)
        outputs = torch.from_numpy(self.reader["stiffness_matrix"][:]).type(
            torch.float32
        )
        # Train set
        inputs_train = inputs[:ntrain, ::s, ::s].unsqueeze(-1)
        outputs_train = outputs[:ntrain, ::s, ::s].unsqueeze(-1)
        # Validation set
        inputs_val = inputs[ntrain : ntrain + ntest, ::s, ::s].unsqueeze(-1)
        outputs_val = outputs[ntrain : ntrain + ntest, ::s, ::s].unsqueeze(-1)
        # Test set
        inputs_test = inputs[ntrain + ntest : ntrain + 2 * ntest, ::s, ::s].unsqueeze(
            -1
        )
        outputs_test = outputs[ntrain + ntest : ntrain + 2 * ntest, ::s, ::s].unsqueeze(
            -1
        )

        # Normalize the outputs (min-max normalization), the inputs are already normalized in {0,1}
        self.min_x = torch.min(outputs_train[:, :, :, 0])
        self.max_x = torch.max(outputs_train[:, :, :, 0])
        outputs_train[:, :, :, 0] = (outputs_train[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )
        outputs_val[:, :, :, 0] = (outputs_val[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )
        outputs_test[:, :, :, 0] = (outputs_test[:, :, :, 0] - self.min_x) / (
            self.max_x - self.min_x
        )

        self.N_Fourier_F = network_properties["FourierF"]
        if self.N_Fourier_F > 0:
            grid = self.get_grid(inputs_test.shape[1])
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid).unsqueeze(0)  # (1, n_x, n_y, 2*ff)

            inputs_train = torch.cat(
                (inputs_train, ff_grid.repeat(inputs_train.shape[0], 1, 1, 1)), -1
            )
            inputs_val = torch.cat(
                (inputs_val, ff_grid.repeat(inputs_val.shape[0], 1, 1, 1)), -1
            )
            inputs_test = torch.cat(
                (inputs_test, ff_grid.repeat(inputs_test.shape[0], 1, 1, 1)), -1
            )

        g = torch.Generator()

        retrain = network_properties["retrain"]
        if retrain > 0:
            os.environ["PYTHONHASHSEED"] = str(retrain)
            random.seed(retrain)
            np.random.seed(retrain)
            torch.manual_seed(retrain)
            torch.cuda.manual_seed(retrain)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            g.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(inputs_train, outputs_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(inputs_val, outputs_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )
        self.test_loader = DataLoader(
            TensorDataset(inputs_test, outputs_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=g,
        )

        self.s = inputs_test.shape[1]

    def get_grid(self, res):
        x = torch.linspace(0, 1, res)
        y = torch.linspace(0, 1, res)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid
