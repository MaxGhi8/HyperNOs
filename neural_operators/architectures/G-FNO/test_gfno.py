import os
import sys

import torch

# Add parent directory to path to allow imports if needed, though mostly we rely on relative or package imports if installed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from GroupEquivariantFNO import G_FNO


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
    print("G_FNO 2D forward pass test passed!")


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
    print("G_FNO 3D forward pass test passed!")


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
    if error < 1e-4:
        print("    PASS: Rotation Equivariance holds.")
    else:
        print("    FAIL: Rotation Equivariance violation.")

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
    if error_flip < 1e-4:
        print("    PASS: Reflection Equivariance holds.")
    else:
        print("    FAIL: Reflection Equivariance violation.")


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
    # Note: GConv3d typically rotates in the spatial dims (dims -3, -2 corresponds to Y, X usually or X, Y depending on convention)
    # Based on GConv3d implementation: rot90(..., dims=[-3, -2]) suggests rotation in the first two dimensions of the 3 spatial dims.
    # Input is (B, C, X, Y, T). The dims are [-3, -2, -1] -> X, Y, T.

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
    if error < 1e-4:
        print("    PASS: Rotation Equivariance holds.")
    else:
        print("    FAIL: Rotation Equivariance violation.")

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
    if error_flip < 1e-4:
        print("    PASS: Reflection Equivariance holds.")
    else:
        print("    FAIL: Reflection Equivariance violation.")


if __name__ == "__main__":
    test_gfno_2d()
    test_gfno_3d()
    test_equivariance_2d()
    test_equivariance_3d()
