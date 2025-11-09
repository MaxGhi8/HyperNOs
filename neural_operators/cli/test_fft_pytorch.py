"""
Test to isolate the 2D FFT/IFFT issue in MATLAB vs PyTorch
"""

import numpy as np
import torch
from scipy.io import savemat

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # default tensor dtype

# Create a simple test case
n_samples = 2
in_channels = 3
out_channels = 4
n_x = 64
n_y = 64
modes = 20

# Random input
torch.manual_seed(42)
x = torch.randn(n_samples, in_channels, n_x, n_y)

# Random weights (complex) - using torch.cdouble (complex128) for 64-bit compatibility
weights1 = (
    torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cdouble) * 0.1
)
weights2 = (
    torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cdouble) * 0.1
)

# PyTorch forward pass
x_ft = torch.fft.rfft2(x, norm=None)
print(f"After rfft2: {x_ft.shape}")
print(f"Expected: [{n_samples}, {in_channels}, {n_x}, {n_y//2 + 1}]")

# Multiply relevant Fourier modes
out_ft = torch.zeros(
    n_samples,
    out_channels,
    x.size(-2),
    x.size(-1) // 2 + 1,
    dtype=torch.cdouble,  # Use complex128 for 64-bit
)
out_ft[:, :, :modes, :modes] = torch.einsum(
    "bixy,ioxy->boxy", x_ft[:, :, :modes, :modes], weights1
)
out_ft[:, :, -modes:, :modes] = torch.einsum(
    "bixy,ioxy->boxy", x_ft[:, :, -modes:, :modes], weights2
)

# Inverse Fourier transform
out_pytorch = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=None)
print(f"PyTorch output shape: {out_pytorch.shape}")
print(
    f"PyTorch output stats: mean={out_pytorch.mean():.6f}, std={out_pytorch.std():.6f}"
)

# Save for MATLAB testing
savemat(
    "fft_test_data.mat",
    {
        "x": x.numpy(),
        "weights1_real": weights1.real.numpy(),
        "weights1_imag": weights1.imag.numpy(),
        "weights2_real": weights2.real.numpy(),
        "weights2_imag": weights2.imag.numpy(),
        "modes": modes,
        "out_pytorch": out_pytorch.numpy(),
        "x_ft_real": x_ft.real.numpy(),
        "x_ft_imag": x_ft.imag.numpy(),
        "out_ft_real": out_ft.real.numpy(),
        "out_ft_imag": out_ft.imag.numpy(),
    },
)

print("\nSaved test data to fft_test_data.mat")
print("You can now test the MATLAB implementation against this.")
