import os
import sys

import torch

# Add parent directory to path to allow imports if needed, though mostly we rely on relative or package imports if installed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from GroupEquivariantFNO import G_FNO


def test_gfno_2d():
    print("Testing G_FNO 2D...")
    batch_size = 2
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 8
    width = 16
    depth = 4

    # Input shape: (Batch, In_Channels, X, Y)
    # FNO.py expects (Batch, In_Channels, X, Y)
    dummy_input = torch.randn(batch_size, in_dim, 32, 32)

    model = G_FNO(
        problem_dim=2,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        reflection=False,
    )

    out = model(dummy_input)
    print(f"2D Output shape: {out.shape}")
    assert out.shape == (batch_size, out_dim, 32, 32)
    print("G_FNO 2D test passed!")


def test_gfno_3d():
    print("Testing G_FNO 3D...")
    batch_size = 2
    in_dim = 1
    out_dim = 1
    d_v = 16
    modes = 4
    time_modes = 4
    depth = 4

    # Input shape: (Batch, In_Channels, X, Y, Z)
    dummy_input = torch.randn(batch_size, in_dim, 16, 16, 16)

    model = G_FNO(
        problem_dim=3,
        in_dim=in_dim,
        d_v=d_v,
        out_dim=out_dim,
        L=depth,
        modes=modes,
        time_modes=time_modes,
        reflection=False,
    )

    out = model(dummy_input)
    print(f"3D Output shape: {out.shape}")
    assert out.shape == (batch_size, out_dim, 16, 16, 16)
    print("G_FNO 3D test passed!")


if __name__ == "__main__":
    test_gfno_2d()
    test_gfno_3d()
