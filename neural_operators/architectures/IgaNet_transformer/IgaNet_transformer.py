import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

torch.set_default_dtype(torch.float32)


#########################################
# Utils
#########################################
def activation_fun(activation_str: str) -> nn.Module:
    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_str == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Not implemented activation function: {activation_str}")


@jaxtyped(typechecker=beartype)
def zero_mean_imposition(
    x: Float[Tensor, "n_samples d"],
) -> Float[Tensor, "n_samples d"]:
    """
    Impose the zero mean constraint (Orthogonal projection).
    """
    return x - x.mean(dim=1, keepdim=True)


#########################################
# Positional Encoding
#########################################
class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding to inject order information
    into the NURBS control points sequence.
    """

    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )

        pe = torch.zeros(max_len, 1, hidden_dim)
        # Sine
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Cosine
        cos_term = torch.cos(position * div_term)
        pe[:, 0, 1::2] = cos_term[:, : pe[:, 0, 1::2].size(1)]
        # Save
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [seq_len, batch_size, embedding_dim] for standard Transformer
        # We will permute inputs to match this expectation inside the blocks
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


#########################################
# Geometry Branch: Encoder + Resampler
#########################################
class GeometryEncoderResampler(nn.Module):
    """
    This module processes the raw geometry (NURBS control points) and produces
    a latent representation aligned with the solution basis size (d).
    So is from R^{p x 4} -> R^{d x k}, where d is number of solution DOFs and k is latent dimension.

    Architecture:
    1. Input Projection (4 -> hidden_dim)
    2. Self-Attention Encoder (Process global geometry interactions among p points)
    3. Cross-Attention Resampling (Interpolate from p points to d latent vectors)
    """

    def __init__(
        self,
        n_control_points_input: int,  # p
        n_dofs_output: int,  # d
        hidden_dim: int,  # hidden dimension
        n_layers_encoder: int,
        n_heads: int,
        dropout: float = 0.0,
        activation_str: str = "gelu",
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_dofs_output = n_dofs_output

        # 1. Input Embedding (x, y, z, w) -> hidden_dim
        self.input_projection = nn.Linear(4, hidden_dim)
        self.pos_encoder = PositionalEncoding(
            hidden_dim, max_len=n_control_points_input + 100, dropout=dropout
        )

        # 2. Transformer Encoder (Self-Attention on geometry)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,  # number of attention heads
            dim_feedforward=4
            * hidden_dim,  # hidden layer size in MLP between attention layers
            dropout=dropout,
            activation=activation_str,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers_encoder
        )  # n_layers_encoder is the number of attention layers

        # 3. Learnable Queries for Resampling
        self.target_queries = nn.Parameter(torch.randn(n_dofs_output, 1, hidden_dim))
        self.resampling_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout
        )  # Cross-Attention layer for resampling (p -> d)

        self._init_weights()

    def _init_weights(self):
        """
        All layers in the TransformerEncoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder instance.
        """
        # Initialize learnable queries
        nn.init.normal_(self.target_queries, mean=0, std=0.02)
        # Xavier init for linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, g: Float[Tensor, "n_samples p 4"]
    ) -> Float[Tensor, "n_samples d hidden_dim"]:

        # 1. Embedding & Positional Encoding
        src = self.input_projection(g).permute(1, 0, 2)  # (p, n_samples, hidden_dim)
        src = self.pos_encoder(src)

        # 2. Self-Attention Encoder
        memory = self.transformer_encoder(
            src
        )  # (p, n_samples, hidden_dim) for Transformer batch is in the second dim

        # 3. Cross-Attention Resampling
        queries = self.target_queries.repeat(
            1, g.size(0), 1
        )  # (d, n_samples, hidden_dim)

        # MultiheadAttention(query, key, value)
        latent_geometry, _ = self.resampling_attn(
            queries, memory, memory
        )  # (d, n_samples, hidden_dim)

        return latent_geometry.permute(1, 0, 2)  # (n_samples, d, hidden_dim)


#########################################
# Main Architecture: GCLO
#########################################
class input_normalizer_class(nn.Module):
    def __init__(self, input_normalizer) -> None:
        super(input_normalizer_class, self).__init__()
        self.input_normalizer = input_normalizer

    def forward(self, x):
        return self.input_normalizer.encode(x)


class output_denormalizer_class(nn.Module):
    def __init__(self, output_normalizer) -> None:
        super(output_denormalizer_class, self).__init__()
        self.output_normalizer = output_normalizer

    def forward(self, x):
        return self.output_normalizer.decode(x)


class GeometryConditionedLinearOperator(nn.Module):
    """
    Geometry-Conditioned Linear Operator (GCLO).

    It learns to approximate the solution operator u = A(g) * f.
    1. Processes geometry g (p x 4) into latent features Z (d x k).
    2. Constructs matrix A(g) via Attention mechanism on Z.
    3. Applies A(g) to f linearly and enforces constraints.
    """

    def __init__(
        self,
        n_dofs: int,  # d (Size of RHS and Solution)
        n_control_points: int,  # p (Size of Geometry input)
        hidden_dim: int,  # k (Latent dimension)
        n_heads: int = 4,
        n_heads_A: int = 1,
        n_layers_geo: int = 2,
        dropout_rate: float = 0.0,
        activation_str: str = "gelu",
        zero_mean: bool = True,
        example_input_normalizer=None,
        example_output_normalizer=None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.n_dofs = n_dofs
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_heads_A = n_heads_A

        # Normalizer
        self.input_normalizer = (
            nn.Identity()
            if example_input_normalizer is None
            else input_normalizer_class(example_input_normalizer)
        )

        # 1. Geometry Processing Branch
        self.geo_branch = GeometryEncoderResampler(
            n_control_points_input=n_control_points,
            n_dofs_output=n_dofs,
            hidden_dim=hidden_dim,
            n_layers_encoder=n_layers_geo,
            n_heads=n_heads,
            dropout=dropout_rate,
            activation_str=activation_str,
        )

        # 2. Linear Operator Projections (to build A(g))
        self.W_Q = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_heads_A)]
        )
        self.W_K = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_heads_A)]
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Denormalizer
        self.output_denormalizer = (
            nn.Identity()
            if example_output_normalizer is None
            else output_denormalizer_class(example_output_normalizer)
        )

        # Post processing
        self.post_processing = zero_mean_imposition if zero_mean else nn.Identity()

        self.to(device)

    @jaxtyped(typechecker=beartype)
    def construct_matrix(
        self, g: Float[Tensor, "n_samples p 4"]
    ) -> Float[Tensor, "n_samples d d"]:
        """
        Helper method to visualize or extract the learned operator matrix A(g).
        """
        # Process Geometry: g -> Z
        Z = self.geo_branch(g)  # Z shape: (n_samples, d, hidden_dim)

        # 2. Compute Q and K
        Q = torch.zeros_like(Z, device=Z.device)  # (n_samples, d, hidden_dim)
        K = torch.zeros_like(Z, device=Z.device)  # (n_samples, d, hidden_dim)
        for idx in range(self.n_heads_A):
            Q += self.W_Q[idx](Z)
            K += self.W_K[idx](Z)
        Q = Q / self.n_heads_A
        K = K / self.n_heads_A

        # 3. Compute Attention Matrix (Plain Form)
        # (n_samples, d, k) @ (n_samples, k, d) -> (n_samples, d, d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn_scores = torch.matmul(Z, Z.transpose(-2, -1)) * self.scale #! For the identity alternative

        # Apply Softmax over the last dimension
        # A = F.softmax(attn_scores, dim=-1)

        return attn_scores

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x  #: Float[Tensor, "n_samples d"], g: Float[Tensor, "n_samples p 4"]
    ) -> Float[Tensor, "n_samples d"]:
        """
        Forward pass: u = A(g) * f
        """
        # Unpack the input
        f, g = x
        g = self.input_normalizer(g)

        # 1-2. Process geometry and construct A(g)
        A = self.construct_matrix(g)

        # 2. Apply Linear Operator
        # A: (n_samples, d, d), f: (n_samples, d)
        u = torch.bmm(A, f.unsqueeze(-1)).squeeze(-1)

        # 3. Denormalize and enforce constraints
        u = self.output_denormalizer(u)
        u = self.post_processing(u)

        return u  # shape: (n_samples, d)


#########################################
# Main Execution Block
#########################################
if __name__ == "__main__":
    print("--- Testing GCLO Architecture ---")

    ## 1. Define Problem Dimensions
    BATCH_SIZE = 8
    N_DOFS = 128  # d: Number of basis functions for solution/RHS
    N_CONTROL_POINTS = 50  # p: Number of control points for geometry (can be != d)
    LATENT_DIM = 64  # k: Latent dimension for internal features

    ## 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GeometryConditionedLinearOperator(
        n_dofs=N_DOFS,
        n_control_points=N_CONTROL_POINTS,
        hidden_dim=LATENT_DIM,
        n_heads=4,
        n_layers_geo=2,
        zero_mean=True,
        device=device,
    )

    ## 3. Create Dummy Data
    # RHS f: Random vector in R^d
    f_input = torch.randn(BATCH_SIZE, N_DOFS).to(device)
    f_input = zero_mean_imposition(f_input)

    # Geometry g: Random control points (x, y, z, w)
    g_input = torch.randn(BATCH_SIZE, N_CONTROL_POINTS, 4).to(device)

    print(f"\nInput Shapes:")
    print(f"  RHS (f):      {f_input.shape}")
    print(f"  Geometry (g): {g_input.shape}")

    ## 4. Forward Pass
    u_pred = model((f_input, g_input))

    print(f"\nOutput Shape:")
    print(f"  Solution (u): {u_pred.shape}")

    ## 5. Verification
    # Check 1: Does output match expected shape?
    assert u_pred.shape == (BATCH_SIZE, N_DOFS), "Output shape mismatch!"

    # Check 2: Zero mean constraint
    sums = u_pred.sum(dim=1)
    print(f"\nVerification (Zero Mean Constraint):")
    print(f"  Sums of output vectors (should be close to 0):")
    print(f"  {sums.detach().cpu().numpy()}")

    # Numerical tolerance check
    assert torch.allclose(
        sums, torch.zeros_like(sums), atol=1e-5
    ), "Zero mean constraint NOT satisfied!"

    # Check 3: Check Matrix A internals
    A_matrix = model.construct_matrix(g_input)
    print(f"\nInternal Matrix A(g) shape: {A_matrix.shape}")
    assert A_matrix.shape == (BATCH_SIZE, N_DOFS, N_DOFS), "Matrix A shape mismatch!"

    print("\n--- Test finished! ---")
