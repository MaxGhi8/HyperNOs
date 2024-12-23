"""
This file contains the necessary functions to load the data for the Fourier Neural Operator benchmarks.
"""

import h5py
import numpy as np
import scipy
import torch
import sys
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import Tensor
from jaxtyping import jaxtyped, Float
from beartype import beartype

sys.path.append("../")
from utilities import find_file, FourierFeatures, UnitGaussianNormalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            if self.s == 64:
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

        if self.s == 64 and self.in_dist:
            inputs = (
                torch.from_numpy(
                    self.reader["Sample_" + str(index + self.start)]["input"][:]
                )
                .type(torch.float32)
                .reshape(self.s, self.s, 1)
            )
            labels = (
                torch.from_numpy(
                    self.reader["Sample_" + str(index + self.start)]["output"][:]
                )
                .type(torch.float32)
                .reshape(self.s, self.s, 1)
            )

        else:

            inputs = self.reader["Sample_" + str(index + self.start)]["input"][
                :
            ].reshape(1, 1, self.s, self.s)
            labels = self.reader["Sample_" + str(index + self.start)]["output"][
                :
            ].reshape(1, 1, self.s, self.s)

            if self.s < 128:
                inputs = downsample(inputs, self.s).reshape(self.s, self.s, 1)
                labels = downsample(labels, self.s).reshape(self.s, self.s, 1)
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

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class ShearLayer:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples,
        size=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        if "s" in network_properties:
            s = size
        else:
            s = 64  # Default value

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            ShearLayerDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            ShearLayerDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            ShearLayerDataset(
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

        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = find_file("PoissonData_64x64_IN.h5", search_path)
        self.reader = h5py.File(file_data_train, "r")
        self.min_data = self.reader["min_inp"][()]
        self.max_data = self.reader["max_inp"][()]
        self.min_model = self.reader["min_out"][()]
        self.max_model = self.reader["max_out"][()]

        if in_dist:
            self.file_data = file_data_train
        else:
            self.file_data = find_file("PoissonData_64x64_OUT.h5", search_path)

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

        # If the reader changed.
        self.reader = h5py.File(self.file_data, "r")

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
            .reshape(self.s, self.s, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(self.s, self.s, 1)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class SinFrequency:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            SinFrequencyDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Wave data
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

        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = find_file("WaveData_64x64_IN.h5", search_path)

        self.reader = h5py.File(file_data_train, "r")
        self.min_data = self.reader["min_u0"][()]
        self.max_data = self.reader["max_u0"][()]
        self.min_model = self.reader["min_u"][()]
        self.max_model = self.reader["max_u"][()]

        # Default file:
        if in_dist:
            self.file_data = file_data_train
        else:
            self.file_data = find_file("WaveData_64x64_OUT.h5", search_path)

        # What time? DEFAULT : t = 5
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
        if s != 64:
            self.file_data = "data/WaveData_24modes_s" + str(s) + ".h5"
            self.start = 0

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
            .reshape(self.s, self.s, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start) + "_t_" + str(self.t)][
                    "output"
                ][:]
            )
            .type(torch.float32)
            .reshape(self.s, self.s, 1)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        grid = torch.zeros((self.s, self.s, 2))

        for i in range(self.s):
            for j in range(self.s):
                grid[i, j][0] = i / (self.s - 1)
                grid[i, j][1] = j / (self.s - 1)

        return grid


class WaveEquation:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers accoirding to your preference
        num_workers = 0

        self.train_set = (
            WaveEquationDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                5,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Allen-Cahn data
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

        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = find_file("AllenCahn_64x64_IN.h5", search_path)
        self.reader = h5py.File(file_data_train, "r")
        self.min_data = self.reader["min_u0"][()]
        self.max_data = self.reader["max_u0"][()]
        self.min_model = self.reader["min_u"][()]
        self.max_model = self.reader["max_u"][()]

        # Default file:
        if in_dist:
            self.file_data = file_data_train
        else:
            self.file_data = find_file("AllenCahn_64x64_OUT.h5", search_path)

        self.reader = h5py.File(self.file_data, "r")

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
            .reshape(64, 64, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(64, 64, 1)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        # print(inputs.shape)
        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class AllenCahn:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=1024,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            AllenCahnDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Smooth Transport data
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

        # The data is already normalized
        # Default file:
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
            .reshape(64, 64, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(64, 64, 1)
        )

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class ContTranslation:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            ContTranslationDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Discontinuous Transport data
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
            .reshape(64, 64, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(64, 64, 1)
        )

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class DiscContTranslation:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers accoirding to your preference
        num_workers = 0

        self.train_set = (
            DiscContTranslationDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Compressible Euler data
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

        # We DO NOT normalize the data in this case
        if in_dist:
            self.file_data = find_file("Airfoil_128x128_IN.h5", search_path)
        else:
            self.file_data = find_file("Airfoil_128x128_OUT.h5", search_path)

        # in_dist = False

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
            .reshape(128, 128, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["Sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(128, 128, 1)
        )

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 128)
        y = torch.linspace(0, 1, 128)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class Airfoil:
    def __init__(
        self,
        network_properties,
        device,
        batch_size,
        training_samples=512,
        s=128,
        in_dist=True,
        search_path="/",
    ):
        # Must have parameters: ------------------------------------------------

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            AirfoilDataset(
                "training",
                self.N_Fourier_F,
                training_samples,
                s,
                search_path=search_path,
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
# Darcy Flow data
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
        insample=True,
        search_path="/",
    ):

        # Note: Normalization constants for both ID and OOD should be used from the training set!
        # Load normalization constants from the TRAINING set:
        file_data_train = find_file("Darcy_64x64_IN.h5", search_path)

        self.reader = h5py.File(file_data_train, "r")
        self.min_data = self.reader["min_inp"][()]
        self.max_data = self.reader["max_inp"][()]
        self.min_model = self.reader["min_out"][()]
        self.max_model = self.reader["max_out"][()]

        if insample:
            self.file_data = file_data_train
        else:
            self.file_data = find_file("Darcy_64x64_OUT.h5", find_file)

        self.reader = h5py.File(self.file_data, "r")

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
            .reshape(64, 64, 1)
        )
        labels = (
            torch.from_numpy(
                self.reader["sample_" + str(index + self.start)]["output"][:]
            )
            .type(torch.float32)
            .reshape(64, 64, 1)
        )

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

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
        device,
        batch_size,
        training_samples=512,
        s=64,
        in_dist=True,
        search_path="/",
    ):

        # Must have parameters: ------------------------------------------------

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size <= 128
        else:
            raise ValueError("You must specify the computational grid size.")

        # Seed
        self.N_Fourier_F = network_properties["FourierF"]
        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_set = (
            DarcyDataset(
                "training", self.N_Fourier_F, training_samples, search_path=search_path
            ),
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            DarcyDataset(
                "validation",
                self.N_Fourier_F,
                training_samples,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            DarcyDataset(
                "testing",
                self.N_Fourier_F,
                training_samples,
                in_dist,
                search_path=search_path,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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
        search_path="/",
    ):
        # s = 5 --> (421)//s + 1 = 85 points per direction

        self.TrainDataPath = find_file("piececonst_r421_N1024_smooth1.mat", search_path)
        a_train, u_train = MatReader_darcy(self.TrainDataPath)
        self.TestDataPath = find_file("piececonst_r421_N1024_smooth2.mat", search_path)
        a_test, u_test = MatReader_darcy(self.TestDataPath)

        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Training data
        g = torch.Generator().manual_seed(1)
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

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


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
        search_path="/",
    ):
        # s = 8 --> (8192)//s + 1 = 1025 points

        self.DataPath = find_file("burgers_data_R10.mat", search_path)
        a, u = MatReader_burgers(self.DataPath)

        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Training data
        g = torch.Generator().manual_seed(1)
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

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


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
        ntrain=3000,
        ntest=375,
        s=4,
        search_path="/",
    ):
        # s = 4 --> (5040)//s  = 1260 points

        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Training data
        self.TrainDataPath = find_file(
            f"Training_dataset_FHN_n_3000_points_5040{time}.mat", search_path
        )
        a, v, w = MatReader_fhn(self.TrainDataPath)
        a_train, v_train, w_train = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            w[:, ::s].unsqueeze(-1),
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

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


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
        ntrain=3000,
        ntest=375,
        s=4,
        search_path="/",
    ):
        # s = 4 --> (5040)//s  = 1260 points

        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Training data
        self.TrainDataPath = find_file(
            "Training_dataset_HH_n_3000_points_5040_tf_100.mat", search_path
        )
        a, v, m, h, n = MatReader_hh(self.TrainDataPath)
        a_train, v_train, m_train, h_train, n_train = (
            a[:, ::s].unsqueeze(-1),
            v[:, ::s].unsqueeze(-1),
            m[:, ::s].unsqueeze(-1),
            h[:, ::s].unsqueeze(-1),
            n[:, ::s].unsqueeze(-1),
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

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(a_train, u_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(a_val, u_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(a_test, u_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


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
        search_path="/",
    ):
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

        retrain = network_properties["retrain"]
        if retrain > 0:
            torch.manual_seed(retrain)

        # Change number of workers according to your preference
        num_workers = 0

        self.train_loader = DataLoader(
            TensorDataset(inputs_train, outputs_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(inputs_val, outputs_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(inputs_test, outputs_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
