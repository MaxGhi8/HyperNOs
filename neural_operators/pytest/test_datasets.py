import random
import sys

import pytest

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


@pytest.mark.parametrize(
    "example_name",
    [
        ("afieti_homogeneous_neumann"),
    ],
)
def test_afieti(example_name):
    batch_size = 100
    training_samples = 1500
    example = NO_load_data_model(
        which_example=example_name,
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        in_dist=True,
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
