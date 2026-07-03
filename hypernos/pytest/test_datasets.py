import os
import random

import numpy as np
import pandas as pd
import pytest
import torch

from hypernos.datasets import (
    AFIETI,
    AFIETI_transformer,
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
    YetiSchurTransformer,
)

random.seed(42)  # Set a seed for reproducibility

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YETI_CSV_PATH = os.path.join(REPO_ROOT, "data", "mp_afieti", "yeti_dataset.csv")


def check_schur_complement_consistency(
    dataset_csv: str,
    patch_id: int,
    dirichlet: np.ndarray,
    output: np.ndarray,
    n_reference_samples: int = 200,
    seed: int = 0,
    tol: float = 1e-4,
) -> dict:
    """Checks whether a (dirichlet, output) pair for `patch_id` is consistent
    with output = S_k @ dirichlet for *some* symmetric positive
    (semi-)definite operator S_k -- the two properties any true Schur
    complement must have -- using only the OTHER samples already stored for
    the same patch as reference material. No access to the actual stiffness
    matrix / S_k is used.

    Two checks:
      1. Energy positivity: dirichlet . output should be >= -tol (a Schur
         complement of an SPD stiffness block is PSD, so v^T S_k v >= 0).
      2. Symmetry: for reference samples (v_j, y_j) of the SAME patch,
         v_j . output should equal dirichlet . y_j (both equal
         dirichlet^T S_k v_j when S_k is symmetric).

    This is a necessary-condition check, not a proof the pair is exactly
    correct: it would catch e.g. a sign error, a mismatched patch, or
    garbled/shuffled data, but not every possible corruption (an operator
    that happens to still be symmetric PSD but numerically wrong would
    pass). An exact per-row check would require reconstructing S_k itself,
    e.g. by exporting it from the C++ generator.
    """
    df = pd.read_csv(dataset_csv)
    sub = df[df["patch_id"] == patch_id]
    assert len(sub) > 0, f"no samples found for patch_id={patch_id}"

    n_skeleton = int(sub["n_skeleton"].iloc[0])
    dirichlet = np.asarray(dirichlet, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    assert dirichlet.shape == (n_skeleton,), (
        f"dirichlet must have shape ({n_skeleton},) for patch {patch_id}, got {dirichlet.shape}"
    )
    assert output.shape == (n_skeleton,), (
        f"output must have shape ({n_skeleton},) for patch {patch_id}, got {output.shape}"
    )

    dirichlet_cols = [f"dirichlet_{i}" for i in range(n_skeleton)]
    output_cols = [f"output_{i}" for i in range(n_skeleton)]

    rng = np.random.default_rng(seed)
    n_ref = min(n_reference_samples, len(sub))
    ref_idx = rng.choice(len(sub), size=n_ref, replace=False)
    v_ref = sub.iloc[ref_idx][dirichlet_cols].to_numpy()
    y_ref = sub.iloc[ref_idx][output_cols].to_numpy()

    energy = float(dirichlet @ output)

    lhs = v_ref @ output    # v_j . output, for each reference v_j
    rhs = y_ref @ dirichlet  # y_j . dirichlet, for each reference y_j
    abs_err = np.abs(lhs - rhs)
    denom = np.maximum(np.abs(lhs), np.abs(rhs))
    denom[denom == 0] = 1.0
    rel_err = abs_err / denom

    passed_psd = energy >= -tol
    passed_symmetry = bool(rel_err.max() < tol)

    return {
        "patch_id": patch_id,
        "n_reference_samples": n_ref,
        "energy": energy,
        "passed_psd": passed_psd,
        "max_relative_symmetry_error": float(rel_err.max()),
        "mean_relative_symmetry_error": float(rel_err.mean()),
        "passed_symmetry": passed_symmetry,
        "passed": passed_psd and passed_symmetry,
    }


@pytest.mark.parametrize("patch_id", [0, 5, 10, 20])
def test_yeti_schur_transformer_schur_complement_consistency(patch_id):
    df = pd.read_csv(YETI_CSV_PATH)
    sub = df[df["patch_id"] == patch_id]
    n_skeleton = int(sub["n_skeleton"].iloc[0])
    dirichlet_cols = [f"dirichlet_{i}" for i in range(n_skeleton)]
    output_cols = [f"output_{i}" for i in range(n_skeleton)]

    row = sub.iloc[0]
    dirichlet = row[dirichlet_cols].to_numpy(dtype=np.float64)
    output = row[output_cols].to_numpy(dtype=np.float64)

    result = check_schur_complement_consistency(YETI_CSV_PATH, patch_id, dirichlet, output)

    assert result["passed_psd"], result
    assert result["passed_symmetry"], result


def test_yeti_schur_transformer_energy_positivity_all_samples():
    """Verify dirichlet · output > 0 for every row in the dataset."""
    df = pd.read_csv(YETI_CSV_PATH)
    for pid in df["patch_id"].unique():
        sub = df[df["patch_id"] == pid]
        n = int(sub["n_skeleton"].iloc[0])
        dcols = [f"dirichlet_{i}" for i in range(n)]
        ocols = [f"output_{i}" for i in range(n)]
        D = sub[dcols].to_numpy(dtype=np.float64)
        O = sub[ocols].to_numpy(dtype=np.float64)
        energy = (D * O).sum(axis=1)
        assert (energy > 0).all(), (
            f"patch_id={pid}: {int((energy <= 0).sum())}/{len(energy)} samples "
            f"have non-positive energy (min={energy.min():.6e})"
        )


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


def test_afieti_dataset_zero_mean_rhs():
    batch_size = 100
    training_samples = 1500
    for filename in [
        "dataset_homogeneous_Neumann_rhs_fixed_l_4_deg_3.mat",
        "dataset_homogeneous_Neumann_rhs_fixed_l_5_deg_3_NEW.mat",
    ]:
        example = NO_load_data_model(
            which_example="afieti_homogeneous_neumann",
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=batch_size,
            training_samples=training_samples,
            in_dist=True,
            filename=filename,
        )
        train_batch_input, _ = next(iter(example.train_loader))

        # Check if the sum of each sample is zero
        assert torch.allclose(
            train_batch_input.sum(dim=1),
            torch.zeros_like(train_batch_input.sum(dim=1)),
            atol=1e-6,
        )

    for filename in [
        "dataset_homogeneous_Neumann_l_3_deg_3.mat",
        "dataset_homogeneous_Neumann_l_4_deg_3.mat",
    ]:
        example = NO_load_data_model(
            which_example="afieti_homogeneous_neumann",
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=batch_size,
            training_samples=training_samples,
            in_dist=True,
            filename=filename,
        )
        train_batch_input, _ = next(iter(example.train_loader))
        train_batch_input = train_batch_input[:, :-32]

        # Check if the sum of each sample is zero
        assert torch.allclose(
            train_batch_input.sum(dim=1),
            torch.zeros_like(train_batch_input.sum(dim=1)),
            atol=1e-6,
        )


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
    training_samples = 500
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
        1
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
        1
    )

    val_batch_input, val_batch_output = next(iter(example.val_loader))
    assert val_batch_input.shape == (batch_size, example.s_in, example.s_in)
    assert val_batch_output.shape == (
        batch_size,
        example.s_out,
        example.s_out,
        1
    )


def test_darcy_don():
    batch_size = 10
    training_samples = 160
    example = NO_load_data_model(
        which_example="darcy_don",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
    )

    # Check for the dimensions of the input and output tensors
    (train_branch_input, train_trunk_input), train_batch_output = next(
        iter(example.train_loader)
    )

    assert train_branch_input.shape == (batch_size, example.s, example.s, 1)
    assert train_trunk_input.shape == (example.s * example.s, 2)
    assert train_batch_output.shape == (batch_size, example.s, example.s, 1)

    (test_branch_input, test_trunk_input), test_batch_output = next(
        iter(example.test_loader)
    )
    assert test_branch_input.shape == (batch_size, example.s, example.s, 1)
    assert test_trunk_input.shape == (example.s * example.s, 2)
    assert test_batch_output.shape == (batch_size, example.s, example.s, 1)

    (val_branch_input, val_trunk_input), val_batch_output = next(
        iter(example.val_loader)
    )
    assert val_branch_input.shape == (batch_size, example.s, example.s, 1)
    assert val_trunk_input.shape == (example.s * example.s, 2)
    assert val_batch_output.shape == (batch_size, example.s, example.s, 1)


def test_afieti_transformer():
    batch_size = 10
    training_samples = 160
    example = NO_load_data_model(
        which_example="afieti_homogeneous_neumann_transformer",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="dataset_homogeneous_Neumann_l_0_deg_2_crazygeom.mat",
    )

    # Check for the dimensions of the input and output tensors
    (rhs, geom), output = next(iter(example.train_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_rhs)

    (rhs, geom), output = next(iter(example.val_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_rhs)

    (rhs, geom), output = next(iter(example.test_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_rhs)


def test_yeti_schur_transformer_dataset():
    batch_size = 100
    training_samples = 16000  # dataset has 20000 rows: 16000/2000/2000 split
    example = NO_load_data_model(
        which_example="mp_afieti",
        no_architecture={
            "FourierF": 0,
            "retrain": -1,
        },
        batch_size=batch_size,
        training_samples=training_samples,
        filename="yeti_dataset.csv",
    )

    assert isinstance(example, YetiSchurTransformer)
    assert example.s_geo == 304  # 1216 geom_* columns / 4 (x, y, z, w)
    assert example.s_rhs == 48  # 48 dirichlet_* columns
    assert example.s_out == 48  # 48 output_* columns

    # Check for the dimensions of the input and output tensors
    (rhs, geom), output = next(iter(example.train_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_out)

    (rhs, geom), output = next(iter(example.val_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_out)

    (rhs, geom), output = next(iter(example.test_loader))
    assert rhs.shape == (batch_size, example.s_rhs)
    assert geom.shape == (batch_size, example.s_geo, 4)
    assert output.shape == (batch_size, example.s_out)

    # Check for the dimensions of the normalizers
    assert example.input_normalizer.mean.shape == (example.s_geo, 4)
    assert example.input_normalizer.std.shape == (example.s_geo, 4)
    assert example.output_normalizer.mean.shape == (example.s_out,)
    assert example.output_normalizer.std.shape == (example.s_out,)


def test_yeti_schur_transformer_invalid_out_dist():
    with pytest.raises(
        AssertionError, match="Out-of-distribution testing samples are not available"
    ):
        NO_load_data_model(
            which_example="mp_afieti",
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=32,
            training_samples=10,
            in_dist=False,
            filename="yeti_dataset.csv",
        )


def test_yeti_schur_transformer_invalid_sampling_rate():
    with pytest.raises(AssertionError, match="Sampling rate must be 1"):
        NO_load_data_model(
            which_example="mp_afieti",
            no_architecture={
                "FourierF": 0,
                "retrain": -1,
            },
            batch_size=32,
            training_samples=10,
            s=2,
            filename="yeti_dataset.csv",
        )


def test_yeti_schur_transformer_shuffle_seed():
    def first_test_batch(shuffle_seed):
        example = YetiSchurTransformer(
            filename="yeti_dataset.csv",
            network_properties={"FourierF": 0, "retrain": -1},
            batch_size=200,
            training_samples=16000,
            search_path=REPO_ROOT,
            shuffle_seed=shuffle_seed,
        )
        (rhs, geom), output = next(iter(example.test_loader))
        return rhs, geom, output

    rhs_a, geom_a, output_a = first_test_batch(shuffle_seed=0)
    rhs_b, geom_b, output_b = first_test_batch(shuffle_seed=0)
    rhs_c, geom_c, output_c = first_test_batch(shuffle_seed=123)

    # The same shuffle_seed must yield an identical, reproducible split
    assert torch.equal(rhs_a, rhs_b)
    assert torch.equal(geom_a, geom_b)
    assert torch.equal(output_a, output_b)

    # A different shuffle_seed must yield a different split
    assert not torch.equal(rhs_a, rhs_c)
