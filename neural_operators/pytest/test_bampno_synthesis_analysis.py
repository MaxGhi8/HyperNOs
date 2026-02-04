import torch
import sys
import os

sys.path.append("..")
from datasets import NO_load_data_model
from architectures.BAMPNO import chebyshev_utilities as cheb


def get_M(n: int):
    M = torch.eye(n)
    M[0, 0] = 1 / 2
    M[0, 1] = -1 / 2
    M[1, 0] = 1 / 2
    M[1, 1] = 1 / 2
    for i in range(2, n):
        if i % 2 == 0:
            M[i, 0] = -1
        else:
            M[i, 1] = -1
    return M

def get_M_1(n: int):
    M_1 = torch.eye(n)
    M_1[0, 1] = 1
    M_1[1, 0] = -1
    for i in range(2, n):
        if i % 2 == 0:
            M_1[i, 0] = 1
            M_1[i, 1] = 1
        else:
            M_1[i, 0] = -1
            M_1[i, 1] = 1
    return M_1



def test_spectral_identity():
    # Setup
    batch_size = 2
    n_patch = 2
    n_x = 10
    n_y = 10
    modes = 10
    channels = 1
    device = "cpu"
    
    # Random input
    x = torch.randn(batch_size, n_patch, n_x, n_y, channels, device=device)
    
    # Enforce continuity
    x[:, 0, 0, :, :] = x[:, 1, -1, :, :]
    
    # Get Matrices
    M = get_M(modes)
    M_1 = get_M_1(modes)
    
    # 1. CFT
    coeffs = cheb.patched_values_to_coefficients(x)
    
    # For the test, we do not do any weight multiplication
    out_ft = coeffs.clone()
    
    # 2. Boundary adapted transform
    out_ft = cheb.patched_change_basis(out_ft, M_1)
    
    # 3. Identifications (Continuity)
    continuity_condition = {"horizontal": [(0, 1)]}
    
    for p1, p2 in continuity_condition["horizontal"]:
        tmp = (
            out_ft[:, p1, 0, 2 : modes, :]
            + out_ft[:, p2, 1, 2 : modes, :]
        ) / 2
        out_ft[:, p1, 0, 2 : modes, :] = tmp
        out_ft[:, p2, 1, 2 : modes, :] = tmp
        
    # 4. Inverse boundary adapted transform
    out_ft = cheb.patched_change_basis(out_ft, M)
    
    # 5. ICFT
    x_rec = cheb.patched_coefficients_to_values(out_ft)
    
    # --- Check ---
    diff = torch.norm(x - x_rec) / torch.norm(x)
    print(f"Relative difference: {diff.item()}")
    
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5), f"Identity failed, diff={diff.item()}"


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


if __name__ == "__main__":
    test_spectral_identity()
