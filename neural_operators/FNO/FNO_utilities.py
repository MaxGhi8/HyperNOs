"""
In this file there are some utilities functions that are used in the main file.
"""

import math

import torch
import torch.nn as nn


def count_params_fno(config, accurate=True):
    """
    function to approximate the number of parameters for the FNO model and classical architecture
    """
    latent = 128
    P_Q = (
        config["in_dim"] + 2 * config["width"] + config["out_dim"] + 2
    ) * latent + config["width"] * config["out_dim"]

    hidden = (
        config["n_layers"]
        * (config["width"] ** 2)
        * config["modes"] ** config["problem_dim"]
        * 2 ** config["problem_dim"]
    )

    if accurate:
        return (
            hidden + P_Q + config["n_layers"] * (config["width"] ** 2 + config["width"])
        )
    else:
        return hidden


def compute_modes(total_param, maximum, config):
    modes = min(
        max(
            int(
                (
                    total_param
                    / (
                        2 ** config["problem_dim"]
                        * config["n_layers"]
                        * config["width"] ** 2
                    )
                )
                ** (1 / config["problem_dim"])
            ),
            1,
        ),
        maximum,
    )

    return modes


class MatrixRFFT2(nn.Module):
    """
    GPU-optimized Matrix-based implementation of 2D Real FFT using pre-computed DFT matrices.
    This is ONNX-compatible as it only uses matrix multiplications with efficient batching.
    """

    def __init__(self, height, width, device=None):
        super().__init__()
        self.height = height
        self.width = width

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Pre-compute DFT matrices
        self.register_buffer("dft_matrix_h", self._create_dft_matrix(height))
        self.register_buffer("dft_matrix_w", self._create_real_dft_matrix(width))

    def _create_dft_matrix(self, n):
        """Create complex DFT matrix of size n x n"""
        k = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(0)

        angles = -2 * math.pi * k * j / n
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)

        return torch.stack([real_part, imag_part], dim=-1)

    def _create_real_dft_matrix(self, n):
        """Create DFT matrix for real input (only compute positive frequencies)"""
        n_freqs = n // 2 + 1

        k = torch.arange(n_freqs, device=self.device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(0)

        angles = -2 * math.pi * k * j / n
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)

        return torch.stack([real_part, imag_part], dim=-1)

    def batch_complex_matmul(self, matrix, x_complex):
        """
        Efficient batched complex matrix multiplication using einsum
        matrix: [freq, time, 2] (real, imag)
        x_complex: [batch, spatial, time, 2] (real, imag)
        Returns: [batch, spatial, freq, 2] (real, imag)
        """
        # Extract real and imaginary parts
        m_real, m_imag = matrix[..., 0], matrix[..., 1]  # [freq, time]
        x_real, x_imag = x_complex[..., 0], x_complex[..., 1]  # [batch, spatial, time]

        # Batched complex multiplication using einsum for efficiency
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        result_real = torch.einsum("ft,bst->bsf", m_real, x_real) - torch.einsum(
            "ft,bst->bsf", m_imag, x_imag
        )
        result_imag = torch.einsum("ft,bst->bsf", m_real, x_imag) + torch.einsum(
            "ft,bst->bsf", m_imag, x_real
        )

        return torch.stack([result_real, result_imag], dim=-1)

    def batch_real_matmul(self, matrix, x_real):
        """
        Efficient batched real-to-complex matrix multiplication
        matrix: [freq, time, 2] (real, imag)
        x_real: [batch, spatial, time]
        Returns: [batch, spatial, freq, 2] (real, imag)
        """
        # For real input, imaginary part is zero
        batch_size, spatial_size, time_size = x_real.shape

        # Create complex representation with zero imaginary part
        x_zeros = torch.zeros_like(x_real, device=self.device)
        x_complex = torch.stack([x_real, x_zeros], dim=-1)  # [batch, spatial, time, 2]

        return self.batch_complex_matmul(matrix, x_complex)

    def forward(self, x):
        """
        GPU-optimized 2D Real FFT using batched matrix multiplication
        Input: [..., height, width] (real)
        Output: [..., height, width//2 + 1, 2] (complex)
        """
        x = x.to(self.device)

        # Store original shape for final reshape
        original_shape = x.shape
        batch_dims = original_shape[:-2]

        # Flatten all batch dimensions for efficient processing
        x_flat = x.reshape(-1, self.height, self.width)
        batch_size = x_flat.shape[0]

        # Step 1: FFT along width dimension (vectorized across all rows and batches)
        # Reshape to [batch_size * height, width] for efficient batching
        x_rows = x_flat.reshape(batch_size * self.height, self.width)

        # Apply real DFT along width - reshape for batch processing
        x_rows_batch = x_rows.unsqueeze(1)  # [batch*height, 1, width]
        x_freq_w = self.batch_real_matmul(self.dft_matrix_w, x_rows_batch)
        x_freq_w = x_freq_w.squeeze(1)  # [batch*height, width//2+1, 2]

        # Reshape back to [batch, height, width//2+1, 2]
        x_freq_w = x_freq_w.reshape(batch_size, self.height, self.width // 2 + 1, 2)

        # Step 2: FFT along height dimension (vectorized across all columns and batches)
        # Transpose to process columns efficiently: [batch, width//2+1, height, 2]
        x_freq_w_t = x_freq_w.permute(0, 2, 1, 3)

        # Reshape for batch processing: [batch * (width//2+1), height, 2]
        x_cols = x_freq_w_t.reshape(batch_size * (self.width // 2 + 1), self.height, 2)
        x_cols_batch = x_cols.unsqueeze(1)  # [batch*(width//2+1), 1, height, 2]

        # Apply complex DFT along height
        result = self.batch_complex_matmul(self.dft_matrix_h, x_cols_batch)
        result = result.squeeze(1)  # [batch*(width//2+1), height, 2]

        # Reshape back to [batch, width//2+1, height, 2] then transpose to [batch, height, width//2+1, 2]
        result = result.reshape(batch_size, self.width // 2 + 1, self.height, 2)
        result = result.permute(0, 2, 1, 3)

        # Reshape to original batch dimensions
        output_shape = batch_dims + (self.height, self.width // 2 + 1, 2)
        result = result.reshape(output_shape)

        return result


class MatrixIRFFT2(nn.Module):
    """
    GPU-optimized Matrix-based implementation of 2D Inverse Real FFT.
    """

    def __init__(self, height, width, device=None):
        super().__init__()
        self.height = height
        self.width = width

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Pre-compute IDFT matrices
        self.register_buffer("idft_matrix_h", self._create_idft_matrix(height))
        self.register_buffer("idft_matrix_w", self._create_real_idft_matrix(width))

    def _create_idft_matrix(self, n):
        """Create complex IDFT matrix of size n x n"""
        k = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(0)

        angles = 2 * math.pi * k * j / n
        real_part = torch.cos(angles) / n
        imag_part = torch.sin(angles) / n

        return torch.stack([real_part, imag_part], dim=-1)

    def _create_real_idft_matrix(self, n):
        """Create IDFT matrix for real output (from positive frequencies only)"""
        n_freqs = n // 2 + 1

        j = torch.arange(n, device=self.device, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(n_freqs, device=self.device, dtype=torch.float32).unsqueeze(0)

        angles = 2 * math.pi * k * j / n
        real_part = torch.cos(angles) / n
        imag_part = torch.sin(angles) / n

        # Handle scaling for real IFFT
        scaling = torch.ones(n_freqs, device=self.device, dtype=torch.float32)
        scaling[1 : n_freqs - 1 if n % 2 == 0 else n_freqs] *= 2
        if n % 2 == 0 and n_freqs > 1:
            scaling[-1] = 1  # Nyquist frequency

        scaling = scaling.unsqueeze(0)  # Broadcast along time dimension
        real_part = real_part * scaling
        imag_part = imag_part * scaling

        return torch.stack([real_part, imag_part], dim=-1)

    def batch_complex_matmul(self, matrix, x_complex):
        """Efficient batched complex matrix multiplication using einsum"""
        m_real, m_imag = matrix[..., 0], matrix[..., 1]
        x_real, x_imag = x_complex[..., 0], x_complex[..., 1]

        result_real = torch.einsum("ft,bst->bsf", m_real, x_real) - torch.einsum(
            "ft,bst->bsf", m_imag, x_imag
        )
        result_imag = torch.einsum("ft,bst->bsf", m_real, x_imag) + torch.einsum(
            "ft,bst->bsf", m_imag, x_real
        )

        return torch.stack([result_real, result_imag], dim=-1)

    def batch_complex_to_real_matmul(self, matrix, x_complex):
        """
        Efficient batched complex-to-real matrix multiplication
        Returns only the real part of the result
        """
        result_complex = self.batch_complex_matmul(matrix, x_complex)
        return result_complex[..., 0]  # Take only real part

    def forward(self, x):
        """
        GPU-optimized 2D Inverse Real FFT
        Input: [..., height, width//2 + 1, 2] (complex)
        Output: [..., height, width] (real)
        """
        x = x.to(self.device)

        original_shape = x.shape
        batch_dims = original_shape[:-3]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.height, self.width // 2 + 1, 2)
        batch_size = x_flat.shape[0]

        # Step 1: IFFT along height dimension
        # Transpose to [batch, width//2+1, height, 2] for column processing
        x_t = x_flat.permute(0, 2, 1, 3)

        # Reshape for batch processing
        x_cols = x_t.reshape(batch_size * (self.width // 2 + 1), self.height, 2)
        x_cols_batch = x_cols.unsqueeze(1)

        # Apply complex IDFT along height
        x_time_h = self.batch_complex_matmul(self.idft_matrix_h, x_cols_batch)
        x_time_h = x_time_h.squeeze(1)

        # Reshape and transpose back
        x_time_h = x_time_h.reshape(batch_size, self.width // 2 + 1, self.height, 2)
        x_time_h = x_time_h.permute(0, 2, 1, 3)  # [batch, height, width//2+1, 2]

        # Step 2: IFFT along width dimension (convert to real)
        # Reshape to [batch*height, width//2+1, 2] for row processing
        x_rows = x_time_h.reshape(batch_size * self.height, self.width // 2 + 1, 2)
        x_rows_batch = x_rows.unsqueeze(1)

        # Apply real IDFT along width
        result = self.batch_complex_to_real_matmul(self.idft_matrix_w, x_rows_batch)
        result = result.squeeze(1)  # [batch*height, width]

        # Reshape back to [batch, height, width]
        result = result.reshape(batch_size, self.height, self.width)

        # Reshape to original batch dimensions
        output_shape = batch_dims + (self.height, self.width)
        result = result.reshape(output_shape)

        return result


# Example usage and test
if __name__ == "__main__":
    # Test the implementation
    batch_size = 32
    d_v = 32
    height = 64
    width = 64

    # Test with different devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create test input
    x_real = torch.randn(batch_size, d_v, height, width, device=device)

    # Create matrix-based FFT modules
    rfft2 = MatrixRFFT2(height, width, device=device)
    irfft2 = MatrixIRFFT2(height, width, device=device)

    # Forward transform
    x_freq = rfft2(x_real)
    print(f"Input shape: {x_real.shape}")
    print(f"FFT output shape: {x_freq.shape}")

    # Inverse transform
    x_reconstructed = irfft2(x_freq)
    print(f"Reconstructed shape: {x_reconstructed.shape}")

    # Check reconstruction error
    error = torch.mean(torch.abs(x_real - x_reconstructed))
    print(f"Reconstruction error: {error.item():.6f}")

    # Compare with PyTorch's built-in FFT (if available)
    x_freq_torch = torch.fft.rfft2(x_real)
    x_freq_torch_stacked = torch.stack([x_freq_torch.real, x_freq_torch.imag], dim=-1)

    fft_error = torch.mean(torch.abs(x_freq - x_freq_torch_stacked))
    print(f"FFT comparison error: {fft_error.item():.6f}")

    x_reconstructed_torch = torch.fft.irfft2(x_freq_torch, s=(height, width))
    ifft_error = torch.mean(torch.abs(x_reconstructed - x_reconstructed_torch))
    print(f"IFFT comparison error: {ifft_error.item():.6f}")

    print("\nThis implementation is ONNX-compatible!")
