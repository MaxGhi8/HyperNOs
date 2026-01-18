import os
import sys

import pytest
import torch

# Add parent directory to path to allow imports if needed, though mostly we rely on relative or package imports if installed.
sys.path.append("..")
from architectures import G_FNO


def test_gfno_2d():
    print("Testing G_FNO 2D (Forward Pass)...")
    batch_size = 2
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 8
    depth = 4

    # Input shape: (Batch, In_Channels, X, Y)
    dummy_input = torch.randn(batch_size, in_dim, 32, 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = G_FNO(
        problem_dim=2,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        reflection=False,
    ).to(device)

    out = model(dummy_input.to(device))
    print(f"2D Output shape: {out.shape}")
    assert out.shape == (batch_size, out_dim, 32, 32)


def test_gfno_3d():
    print("Testing G_FNO 3D (Forward Pass)...")
    batch_size = 2
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 4
    time_modes = 4
    depth = 4

    # Input shape: (Batch, In_Channels, X, Y, Z)
    dummy_input = torch.randn(batch_size, in_dim, 16, 16, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = G_FNO(
        problem_dim=3,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        time_modes=time_modes,
        reflection=False,
    ).to(device)

    out = model(dummy_input.to(device))
    print(f"3D Output shape: {out.shape}")
    assert out.shape == (batch_size, out_dim, 16, 16, 16)


def test_equivariance_2d():
    print("\nTesting G_FNO 2D Equivariance...")
    batch_size = 1
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 8
    depth = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Rotation Equivariance (Reflection=False)
    print("  Checking Rotation Equivariance (p4)...")
    model = G_FNO(
        problem_dim=2,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        reflection=False,
    ).to(device)
    model.eval()

    x = torch.randn(batch_size, in_dim, 32, 32).to(device)

    # Apply model to original
    y = model(x)

    # Apply model to rotated input (90 degrees)
    x_rot = x.rot90(1, dims=[-2, -1])
    y_rot_pred = model(x_rot)

    # Rotate the original output
    y_rot_gt = y.rot90(1, dims=[-2, -1])

    # Check error
    error = torch.norm(y_rot_pred - y_rot_gt) / torch.norm(y_rot_gt)
    print(f"    Rotation Error (Relative L2): {error.item():.6e}")
    assert error < 1e-4, f"Rotation Equivariance violation. Error: {error.item()}"

    # 2. Rotation + Reflection Equivariance (Reflection=True)
    print("  Checking Reflection Equivariance (p4m)...")
    model_refl = G_FNO(
        problem_dim=2,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        reflection=True,
    ).to(device)
    model_refl.eval()

    # Flip input (along X axis)
    x_flip = x.flip(dims=[-2])
    y_flip_pred = model_refl(x_flip)
    y_flip_gt = model_refl(x).flip(dims=[-2])

    error_flip = torch.norm(y_flip_pred - y_flip_gt) / torch.norm(y_flip_gt)
    print(f"    Reflection Error (Relative L2): {error_flip.item():.6e}")
    assert (
        error_flip < 1e-4
    ), f"Reflection Equivariance violation. Error: {error_flip.item()}"


def test_equivariance_3d():
    print("\nTesting G_FNO 3D Equivariance...")
    batch_size = 1
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 4
    time_modes = 4
    depth = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Rotation Equivariance (Rotation in spatial dimensions X, Y)
    print("  Checking Rotation Equivariance (p4 in XY plane)...")
    model = G_FNO(
        problem_dim=3,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        time_modes=time_modes,
        reflection=False,
    ).to(device)
    model.eval()

    x = torch.randn(batch_size, in_dim, 16, 16, 16).to(device)

    y = model(x)

    # Rotate in XY plane (dims -3 and -2)
    x_rot = x.rot90(1, dims=[-3, -2])
    y_rot_pred = model(x_rot)
    y_rot_gt = y.rot90(1, dims=[-3, -2])

    error = torch.norm(y_rot_pred - y_rot_gt) / torch.norm(y_rot_gt)
    assert error < 1e-4, f"Rotation Equivariance violation. Error: {error.item()}"

    # 2. Reflection Equivariance (Reflection in XY plane)
    print("  Checking Reflection Equivariance (p4m in XY plane)...")
    model_refl = G_FNO(
        problem_dim=3,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        time_modes=time_modes,
        reflection=True,
    ).to(device)
    model_refl.eval()

    # Flip input (along X axis, dim -3)
    x_flip = x.flip(dims=[-3])
    y_flip_pred = model_refl(x_flip)
    y_flip_gt = model_refl(x).flip(dims=[-3])

    error_flip = torch.norm(y_flip_pred - y_flip_gt) / torch.norm(y_flip_gt)
    print(f"    Reflection Error (Relative L2): {error_flip.item():.6e}")
    assert (
        error_flip < 1e-4
    ), f"Reflection Equivariance violation. Error: {error_flip.item()}"


def test_translation_equivariance():
    print("\nTesting G_FNO Translation Equivariance...")
    batch_size = 1
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 8
    depth = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 2D Translation
    print("  Checking 2D Translation Equivariance...")
    model_2d = G_FNO(
        problem_dim=2,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        reflection=False,
    ).to(device)
    model_2d.eval()

    # Create input constant everywhere except a subsquare
    x_2d = torch.zeros(batch_size, in_dim, 32, 32).to(device)
    x_2d[..., 10:22, 10:22] = 1.0

    # Translate input (circular shift)
    shift_x, shift_y = 5, 5
    x_2d_shifted = torch.roll(x_2d, shifts=(shift_x, shift_y), dims=(-2, -1))

    y_2d = model_2d(x_2d)
    y_2d_shifted_pred = model_2d(x_2d_shifted)
    y_2d_shifted_gt = torch.roll(y_2d, shifts=(shift_x, shift_y), dims=(-2, -1))

    error_2d = torch.norm(y_2d_shifted_pred - y_2d_shifted_gt) / torch.norm(
        y_2d_shifted_gt
    )
    print(f"    2D Translation Error (Relative L2): {error_2d.item():.6e}")
    assert (
        error_2d < 2e-4
    ), f"2D Translation Equivariance violation. Error: {error_2d.item()}"

    # 2. 3D Translation
    print("  Checking 3D Translation Equivariance...")
    modes_3d = 4
    time_modes = 4
    model_3d = G_FNO(
        problem_dim=3,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes_3d,
        time_modes=time_modes,
        reflection=False,
    ).to(device)
    model_3d.eval()

    # Create input constant everywhere except a sub-cube
    x_3d = torch.zeros(batch_size, in_dim, 16, 16, 16).to(device)
    x_3d[..., 4:12, 4:12, 4:12] = 1.0

    # Translate input
    shift_x, shift_y, shift_z = 3, 3, 3
    x_3d_shifted = torch.roll(
        x_3d, shifts=(shift_x, shift_y, shift_z), dims=(-3, -2, -1)
    )

    y_3d = model_3d(x_3d)
    y_3d_shifted_pred = model_3d(x_3d_shifted)
    y_3d_shifted_gt = torch.roll(
        y_3d, shifts=(shift_x, shift_y, shift_z), dims=(-3, -2, -1)
    )

    error_3d = torch.norm(y_3d_shifted_pred - y_3d_shifted_gt) / torch.norm(
        y_3d_shifted_gt
    )
    print(f"    3D Translation Error (Relative L2): {error_3d.item():.6e}")
    assert (
        error_3d < 1e-4
    ), f"3D Translation Equivariance violation. Error: {error_3d.item()}"
