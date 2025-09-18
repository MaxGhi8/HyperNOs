import random
import sys

import pytest
import torch

sys.path.append("..")
from datasets import (
    AFIETI,
    Airfoil,
    AllenCahn,
    Burgers_Zongyi,
    ContTranslation,
    CrossTruss,
    Darcy,
    Darcy_Zongyi,
    DiscContTranslation,
    FitzHughNagumo,
    HodgkinHuxley,
    NO_load_data_model,
    OHaraRudy,
    ShearLayer,
    SinFrequency,
    WaveEquation,
)

random.seed(42)  # Set a seed for reproducibility

#### Test cases for valid examples 2D
num_test_cases = 10  # Number of test cases
random_batch_sizes = [random.randint(1, 50) for _ in range(num_test_cases)]
random_retrain = [random.choice([1, -1]) for _ in range(num_test_cases)]
random_fourierf = [random.randint(0, 10) for _ in range(num_test_cases)]


@pytest.mark.parametrize(
    "example_name, expected_class, batch_size, retrain, fourierf, output_size",
    [
        (
            "airfoil",
            Airfoil,
            random_batch_sizes[0],
            random_retrain[0],
            random_fourierf[0],
            1,
        ),
        (
            "allen",
            AllenCahn,
            random_batch_sizes[1],
            random_retrain[1],
            random_fourierf[1],
            1,
        ),
        (
            "cont_tran",
            ContTranslation,
            random_batch_sizes[2],
            random_retrain[2],
            random_fourierf[2],
            1,
        ),
        (
            "crosstruss",
            CrossTruss,
            random_batch_sizes[3],
            random_retrain[3],
            random_fourierf[3],
            2,
        ),
        (
            "darcy",
            Darcy,
            random_batch_sizes[4],
            random_retrain[4],
            random_fourierf[4],
            1,
        ),
        (
            "darcy_zongyi",
            Darcy_Zongyi,
            random_batch_sizes[5],
            random_retrain[5],
            random_fourierf[5],
            1,
        ),
        (
            "disc_tran",
            DiscContTranslation,
            random_batch_sizes[6],
            random_retrain[6],
            random_fourierf[6],
            1,
        ),
        (
            "shear_layer",
            ShearLayer,
            random_batch_sizes[7],
            random_retrain[7],
            random_fourierf[7],
            1,
        ),
        (
            "poisson",
            SinFrequency,
            random_batch_sizes[8],
            random_retrain[8],
            random_fourierf[8],
            1,
        ),
        (
            "wave_0_5",
            WaveEquation,
            random_batch_sizes[9],
            random_retrain[9],
            random_fourierf[9],
            1,
        ),
    ],
)
def test_valid_example_2d(
    example_name, expected_class, batch_size, retrain, fourierf, output_size
):
    # Test valid examples without in_size
    example = NO_load_data_model(
        which_example=example_name,
        no_architecture={
            "FourierF": fourierf,
            "retrain": retrain,
        },
        batch_size=batch_size,
        training_samples=60,
    )

    assert isinstance(example, expected_class)

    # train set
    assert next(iter(example.train_loader))[0].shape == (  # input
        batch_size,
        example.s,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.train_loader))[1].shape == (  # output
        batch_size,
        example.s,
        example.s,
        output_size,
    )
    # test set
    assert next(iter(example.test_loader))[0].shape == (  # input
        batch_size,
        example.s,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.test_loader))[1].shape == (  # output
        batch_size,
        example.s,
        example.s,
        output_size,
    )
    # validation set
    assert next(iter(example.val_loader))[0].shape == (  # input
        batch_size,
        example.s,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.val_loader))[1].shape == (  # output
        batch_size,
        example.s,
        example.s,
        output_size,
    )


#### Test cases for valid examples 1D
num_test_cases = 4  # Number of test cases
random_batch_sizes = [random.randint(1, 50) for _ in range(num_test_cases)]
random_retrain = [random.choice([1, -1]) for _ in range(num_test_cases)]
random_fourierf = [random.randint(0, 10) for _ in range(num_test_cases)]


@pytest.mark.parametrize(
    "example_name, expected_class, batch_size, retrain, fourierf, output_size",
    [
        (
            "burgers_zongyi",
            Burgers_Zongyi,
            random_batch_sizes[0],
            random_retrain[0],
            random_fourierf[0],
            1,
        ),
        (
            "fhn",
            FitzHughNagumo,
            random_batch_sizes[1],
            random_retrain[1],
            random_fourierf[1],
            2,
        ),
        (
            "hh",
            HodgkinHuxley,
            random_batch_sizes[2],
            random_retrain[2],
            random_fourierf[2],
            4,
        ),
        (
            "ord",
            OHaraRudy,
            random_batch_sizes[3],
            random_retrain[3],
            random_fourierf[3],
            41,
        ),
    ],
)
def test_valid_example_1d(
    example_name, expected_class, batch_size, retrain, fourierf, output_size
):
    example = NO_load_data_model(
        which_example=example_name,
        no_architecture={
            "FourierF": fourierf,
            "retrain": retrain,
        },
        batch_size=batch_size,
        training_samples=60,
    )

    assert isinstance(example, expected_class)

    # train set
    assert next(iter(example.train_loader))[0].shape == (  # input
        batch_size,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.train_loader))[1].shape == (  # output
        batch_size,
        example.s,
        output_size,
    )
    # test set
    assert next(iter(example.test_loader))[0].shape == (  # input
        batch_size,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.test_loader))[1].shape == (  # output
        batch_size,
        example.s,
        output_size,
    )
    # validation set
    assert next(iter(example.val_loader))[0].shape == (  # input
        batch_size,
        example.s,
        1 + 2 * fourierf,
    )
    assert next(iter(example.val_loader))[1].shape == (  # output
        batch_size,
        example.s,
        output_size,
    )


#### test case with invalid name
@pytest.mark.parametrize(
    "example_name",
    [
        ("Airfoil"),
        ("navier"),
        ("wave"),
    ],
)
def test_invalid_example(example_name):
    with pytest.raises(ValueError, match="The variable which_example is typed wrong"):
        NO_load_data_model(
            which_example=example_name,
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=32,
            training_samples=10,
        )


#### test case without out_dist
@pytest.mark.parametrize(
    "example_name",
    [
        ("fhn"),
        ("hh"),
        ("ord"),
        ("burgers_zongyi"),
        ("darcy_zongyi"),
        ("crosstruss"),
    ],
)
def test_invalid_out_dist(example_name):
    with pytest.raises(
        AssertionError, match="Out-of-distribution testing samples are not available"
    ):
        NO_load_data_model(
            which_example=example_name,
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=32,
            training_samples=10,
            in_dist=False,
        )


def test_afieti_dataset():
    batch_size = 100
    training_samples = 1500
    example = NO_load_data_model(
        which_example="afieti_homogeneous_neumann",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        in_dist=True,
        filename="dataset_homogeneous_Neumann_rhs_fixed_l_3_deg_3.mat",
    )

    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (batch_size, example.s_in)
    assert train_batch_output.shape == (batch_size, example.s_out)

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (batch_size, example.s_in)
    assert test_batch_output.shape == (batch_size, example.s_out)

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in)
    assert val_batch_output.shape == (batch_size, example.s_out)

    assert example.input_normalizer.mean.shape[0] == example.s_in
    assert example.input_normalizer.std.shape[0] == example.s_in
    assert example.output_normalizer.mean.shape[0] == example.s_out
    assert example.output_normalizer.std.shape[0] == example.s_out


def test_bampno_dataset():
    n_patch = 6
    batch_size = 100
    training_samples = 600
    example = NO_load_data_model(
        which_example="bampno",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="Darcy_8_chebyshev_60pts.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        n_patch,
        example.s_in,
        example.s_in,
        1,
    )
    assert train_batch_output.shape == (
        batch_size,
        n_patch,
        example.s_out,
        example.s_out,
        1,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        n_patch,
        example.s_in,
        example.s_in,
        1,
    )
    assert test_batch_output.shape == (
        batch_size,
        n_patch,
        example.s_out,
        example.s_out,
        1,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, n_patch, example.s_in, example.s_in, 1)
    assert val_batch_output.shape == (
        batch_size,
        n_patch,
        example.s_out,
        example.s_out,
        1,
    )

    # Check for the dimensions of the physical tensors
    X = example.X_phys
    Y = example.Y_phys
    assert X.shape == (n_patch, example.s_in, example.s_in)
    assert Y.shape == (n_patch, example.s_in, example.s_in)
    assert X.shape == Y.shape

    # # Check for the boundary conditions
    # assert abs(sum(train_batch_output[:, 0, 0, :, :].reshape(-1, 1))) < 1e-8
    # assert abs(sum(train_batch_output[:, 0, -1, :, :].reshape(-1, 1))) < 1e-8
    # assert abs(sum(train_batch_output[:, 0, :, 0, :].reshape(-1, 1))) < 1e-8

    # assert abs(sum(train_batch_output[:, 1, 0, :, :].reshape(-1, 1))) < 1e-8
    # assert abs(sum(train_batch_output[:, 1, :, -1, :].reshape(-1, 1))) < 1e-8

    # assert abs(sum(train_batch_output[:, 2, :, -1, :].reshape(-1, 1))) < 1e-8
    # assert abs(sum(train_batch_output[:, 2, -1, :, :].reshape(-1, 1))) < 1e-8
    # assert abs(sum(train_batch_output[:, 2, :, 0, :].reshape(-1, 1))) < 1e-8

    # # check for continuity condition for input
    # assert torch.allclose(
    #     train_batch_input[:, 0, :, -1, :], train_batch_input[:, 1, :, 0, :], atol=1e-6
    # )
    # assert torch.allclose(
    #     train_batch_input[:, 1, -1, :, :], train_batch_input[:, 2, 0, :, :], atol=1e-6
    # )

    # # check for continuity condition for output
    # assert torch.allclose(
    #     train_batch_output[:, 0, :, -1, :], train_batch_output[:, 1, :, 0, :], atol=1e-6
    # )
    # assert torch.allclose(
    #     train_batch_output[:, 1, -1, :, :], train_batch_output[:, 2, 0, :, :], atol=1e-6
    # )


def test_bampno_continuation_dataset():
    n_patch = 6
    batch_size = 100
    training_samples = 600
    example = NO_load_data_model(
        which_example="bampno_continuation",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="Darcy_8_uniform_95pts_fourier_continuation.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        1,
    )
    assert train_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        1,
    )
    assert test_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, example.s_in, 1)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    # Check for the dimensions of the physical tensors
    X = example.X_phys
    Y = example.Y_phys
    mask = example.mask
    assert X.shape == (example.s_in, example.s_in)
    assert Y.shape == (example.s_in, example.s_in)
    assert mask.shape == (example.s_in, example.s_in)
    assert X.shape == Y.shape
    assert X.shape == mask.shape


def test_eig_dataset():
    n_eig = 50
    batch_size = 50
    training_samples = 1200
    example = NO_load_data_model(
        which_example="eig",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="Darcy_eig_Square_uniform_60pts.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        1,
    )
    assert train_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        n_eig,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        1,
    )
    assert test_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        n_eig,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, example.s_in, 1)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        n_eig,
    )

    # Check for the dimensions of the physical tensors
    X = example.X_phys
    Y = example.Y_phys
    assert X.shape == (example.s_in, example.s_in)
    assert Y.shape == (example.s_in, example.s_in)
    assert X.shape == Y.shape


def test_coeff_rhs_dataset():
    batch_size = 100
    training_samples = 1200
    example = NO_load_data_model(
        which_example="coeff_rhs",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="Darcy_Square_uniform_60pts_coeff_rhs.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        2,
    )
    assert train_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
        2,
    )
    assert test_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, example.s_in, 2)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1,
    )

    # Check for the dimensions of the physical tensors
    X = example.X_phys
    Y = example.Y_phys
    assert X.shape == (example.s_in, example.s_in)
    assert Y.shape == (example.s_in, example.s_in)
    assert X.shape == Y.shape


def test_coeff_rhs_1d_dataset():
    batch_size = 100
    training_samples = 1200
    example = NO_load_data_model(
        which_example="coeff_rhs_1d",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="Darcy_uniform_1d_100pts_coeff_rhs.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        example.s_in,
        2,
    )
    assert train_batch_output.shape == (
        batch_size,
        example.s_out,
        1,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        example.s_in,
        2,
    )
    assert test_batch_output.shape == (
        batch_size,
        example.s_out,
        1,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, 2)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        1,
    )

    # Check for the dimensions of the physical tensors
    X = example.X_phys
    assert X.shape == (example.s_in,)


def test_afieti_fno_dataset():
    batch_size = 100
    training_samples = 1600
    example = NO_load_data_model(
        which_example="afieti_fno",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="dataset_homogeneous_Neumann_FNO.mat",
    )

    # Check for the dimensions of the input and output tensors
    train_batch_input, train_batch_output = next(iter(example.train_loader))
    assert train_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
    )
    assert train_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
    )

    test_batch_input, test_batch_output = next(iter(example.test_loader))
    assert test_batch_input.shape == (
        batch_size,
        example.s_in,
        example.s_in,
    )
    assert test_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, example.s_in)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
    )
