# filepath: /home/max/Documents/PhD/HyperNOs/neural_operators/architectures/BAMPNO/BAMPNO_parallel.py
"""
Parallel (multi-GPU) version of BAMPNO with patch-dimension sharding.

Key idea:
- Shard the patch dimension across ranks (model-parallel over patches).
- Each rank computes Chebyshev transforms and local weight multiplications
  only for its subset of patches.
- We gather boundary-adapted coefficients across ranks, enforce continuity
  (and optional zero BC) globally, then broadcast and continue.

Two usage modes:
1. Single GPU / no torch.distributed launch: behaves like the original model.
2. Multi GPU (torchrun --nproc_per_node=N ...): automatic distributed init.

Minimal user changes:
- Import ParallelBAMPNO instead of BAMPNO.
- Construct model the same way; it auto-detects world size and sharding.
- In forward, you may pass either the FULL patch tensor on every rank
  (will be sliced locally) OR only the local patch slice (advanced usage).

Optimization note:
Currently continuity enforcement uses an all_gather of all (truncated) patch
coefficients, modifies interface modes once (rank 0), and broadcasts back.
This is simple & correct. A more bandwidth-efficient variant would exchange
only boundary coefficient slices. A placeholder method is included for future
improvement.
"""
from __future__ import annotations

import math
import os
from functools import cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # beartype / jaxtyping are optional for runtime, keep graceful fallback
    from beartype import beartype
    from jaxtyping import Float, jaxtyped
except Exception:  # pragma: no cover
    def jaxtyped(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap
    def beartype(fn):  # type: ignore
        return fn
    Float = torch.Tensor  # type: ignore

import mat73
from hypernos.utilities import find_file  # path search utility

# Import utilities from existing BAMPNO implementation
from . import chebyshev_utilities as cheb  # type: ignore
from .BAMPNO import activation  # reuse existing activation


#########################################
# Helper MLP (same as original, reduced duplication)
#########################################
class MLP(nn.Module):
    def __init__(
        self,
        problem_dim: int,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        fun_act: str,
        bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.problem_dim = problem_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp1 = nn.Linear(in_channels, mid_channels, bias=bias, device=device)
        self.mlp2 = nn.Linear(mid_channels, out_channels, bias=bias, device=device)
        self.fun_act = fun_act

    # Removed jaxtyping annotations for simplicity in parallel version
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = activation(self.mlp1(x), self.fun_act)
        return self.mlp2(x)

#########################################
# Distributed helpers
#########################################

def _maybe_init_distributed(backend: str = "nccl") -> Tuple[int, int, int, torch.device]:
    """Initialize torch.distributed if environment indicates multi-GPU run.

    Returns (rank, world_size, local_rank, device)
    """
    if not torch.distributed.is_available():  # single process fallback
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device

    if torch.distributed.is_initialized():  # already initialized externally
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())) if torch.cuda.is_available() else 0
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        return rank, world_size, local_rank, device

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", str(rank_env)))

    if world_size_env > 1:
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"  # fallback
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = local_rank_env
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return rank, world_size, local_rank, device
    else:  # single process
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device


def _patch_range(n_patch: int, world_size: int, rank: int) -> Tuple[int, int]:
    per = math.ceil(n_patch / world_size)
    start = rank * per
    end = min(n_patch, start + per)
    return start, end

#########################################
# Sharded Chebyshev Layer
#########################################
class ShardedChebyshevLayer(nn.Module):
    def __init__(
        self,
        n_patch: int,
        continuity_condition: Dict,
        in_channels: int,
        out_channels: int,
        modes: int,
        M: torch.Tensor,
        M_1: torch.Tensor,
        fun_act: str,
        zero_BC: Optional[Dict] = None,
        weights_norm: Optional[str] = None,
        same_params: bool = False,
        enable_parallel: bool = True,
        backend: str = "nccl",
    ):
        super().__init__()
        self.n_patch = n_patch
        self.continuity_condition = continuity_condition
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.M = M
        self.M_1 = M_1
        self.fun_act = fun_act
        self.zero_BC = zero_BC
        self.weights_norm = weights_norm
        self.same_params = same_params
        self.enable_parallel = enable_parallel

        # Distributed context
        if enable_parallel:
            self.rank, self.world_size, self.local_rank, self.device = _maybe_init_distributed(backend=backend)
        else:
            self.rank, self.world_size, self.local_rank, self.device = 0, 1, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.patch_start, self.patch_end = _patch_range(n_patch, self.world_size, self.rank)
        self.local_n_patch = max(0, self.patch_end - self.patch_start)
        self.max_local_patches = math.ceil(n_patch / self.world_size)

        # Weight initialization
        if self.same_params:
            shape = (self.modes, self.modes, self.in_channels, self.out_channels)
            if self.weights_norm == "Xavier":
                self.weights = nn.Parameter(nn.init.xavier_normal_(torch.empty(*shape), gain=1 / (self.in_channels * self.out_channels)))
            elif self.weights_norm == "Kaiming":
                self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty(*shape), a=0, mode="fan_in", nonlinearity=self.fun_act))
            else:
                scale = 1 / (self.in_channels * self.out_channels)
                self.weights = nn.Parameter(scale * torch.rand(*shape))
        else:
            shape = (self.n_patch, self.modes, self.modes, self.in_channels, self.out_channels)
            if self.weights_norm == "Xavier":
                self.weights = nn.Parameter(nn.init.xavier_normal_(torch.empty(*shape), gain=1 / (self.in_channels * self.out_channels)))
            elif self.weights_norm == "Kaiming":
                self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty(*shape), a=0, mode="fan_in", nonlinearity=self.fun_act))
            else:
                scale = 1 / (self.in_channels * self.out_channels)
                self.weights = nn.Parameter(scale * torch.rand(*shape))

        self.to(self.device)

    # Local tensor mul (non-sharded vs shared weights)
    def _tensor_mul_local(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (b, local_n_patch, modes, modes, in_channels)
        if self.same_params:
            return torch.einsum("blxyi,xyio->blxyo", x, self.weights)
        else:
            w_local = self.weights[self.patch_start:self.patch_end]  # (local_n_patch, m, m, in, out)
            return torch.einsum("blxyi,lxyio->blxyo", x, w_local)

    @staticmethod
    def _enforce_continuity(out_ft: torch.Tensor, continuity_condition: Dict, modes: int) -> torch.Tensor:
        # out_ft shape: (b, n_patch, n_x, n_y, out_channels) in boundary-adapted coeff basis
        for p1, p2 in continuity_condition.get("horizontal", []):
            tmp = (out_ft[:, p1, 0, 2:modes, :] + out_ft[:, p2, 1, 2:modes, :]) / 2
            out_ft[:, p1, 0, 2:modes, :] = tmp
            out_ft[:, p2, 1, 2:modes, :] = tmp
        for p1, p2 in continuity_condition.get("vertical", []):
            tmp = (out_ft[:, p1, 2:modes, 0, :] + out_ft[:, p2, 2:modes, 1, :]) / 2
            out_ft[:, p1, 2:modes, 0, :] = tmp
            out_ft[:, p2, 2:modes, 1, :] = tmp
        return out_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for local or (optionally) global patch tensor.

        Accepts either:
        - Local tensor: (b, local_n_patch, n_x, n_y, in_channels)
        - Global tensor: (b, n_patch, n_x, n_y, in_channels) -> will slice locally if parallel.
        """
        if self.world_size == 1:  # single GPU, fall back to original semantics
            return self._forward_single(x)

        if x.shape[1] == self.n_patch:  # global given, slice locally
            x = x[:, self.patch_start:self.patch_end]
        assert x.shape[1] == self.local_n_patch, "Input patch slice mismatch with local range"

        if self.local_n_patch == 0:  # some ranks may be empty if n_patch < world_size
            # Create an empty tensor path (still need to participate in collectives)
            # Build a dummy shape using first two dims from x
            b = x.shape[0]
            return torch.zeros(b, 0, *x.shape[2:-1], self.out_channels, device=self.device)

        b, n_local, n_x, n_y, _ = x.shape
        # CFT
        x_coeff = cheb.patched_values_to_coefficients(x)
        # Multiply relevant Chebyshev modes
        local_mul = torch.zeros(
            b, n_local, n_x, n_y, self.out_channels, device=x.device, dtype=x.dtype
        )
        local_mul[:, :, : self.modes, : self.modes, :] = self._tensor_mul_local(
            x_coeff[:, :, : self.modes, : self.modes, :]
        )
        # Boundary adapted transform
        local_mul = cheb.patched_change_basis(local_mul, self.M_1)

        # Zero BC (local patches only)
        if self.zero_BC:
            zero_vec = torch.zeros(b, self.modes, self.out_channels, device=x.device, dtype=x.dtype)
            for patch_global, edges in self.zero_BC.items():
                if self.patch_start <= patch_global < self.patch_end:
                    p_local = patch_global - self.patch_start
                    local_mul[:, p_local, :2, :2, :] = 0.0
                    for edge in edges:
                        if edge == 1:
                            local_mul[:, p_local, 0, : self.modes, :] = zero_vec
                        elif edge == 2:
                            local_mul[:, p_local, 1, : self.modes, :] = zero_vec
                        elif edge == 3:
                            local_mul[:, p_local, : self.modes, 0, :] = zero_vec
                        elif edge == 4:
                            local_mul[:, p_local, : self.modes, 1, :] = zero_vec

        # Gather all patches (pad to uniform size for all_gather)
        pad_patches = self.max_local_patches - n_local
        if pad_patches > 0:
            pad = torch.zeros(b, pad_patches, n_x, n_y, self.out_channels, device=x.device, dtype=x.dtype)
            local_padded = torch.cat([local_mul, pad], dim=1)
        else:
            local_padded = local_mul

        gather_list = [torch.zeros_like(local_padded) for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, local_padded)
        global_padded = torch.cat(gather_list, dim=1)  # (b, world_size*max_local_patches,...)
        global_out = global_padded[:, : self.n_patch]  # trim padded extra patches

        if self.rank == 0:
            global_out = self._enforce_continuity(global_out, self.continuity_condition, self.modes)
        torch.distributed.broadcast(global_out, src=0)

        # Inverse boundary adapted transform & ICFT (local slice only)
        local_slice = global_out[:, self.patch_start:self.patch_end]
        local_slice = cheb.patched_change_basis(local_slice, self.M)
        local_vals = cheb.patched_coefficients_to_values(local_slice)
        return local_vals

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # Expect shape (b, n_patch, n_x, n_y, in_channels)
        b, n_patch, n_x, n_y, _ = x.shape
        x_coeff = cheb.patched_values_to_coefficients(x)
        out_ft = torch.zeros(b, n_patch, n_x, n_y, self.out_channels, device=x.device, dtype=x.dtype)
        if self.same_params:
            out_ft[:, :, : self.modes, : self.modes, :] = torch.einsum(
                "bpxyi,xyio->bpxyo", x_coeff[:, :, : self.modes, : self.modes, :], self.weights
            )
        else:
            out_ft[:, :, : self.modes, : self.modes, :] = torch.einsum(
                "bpxyi,pxyio->bpxyo", x_coeff[:, :, : self.modes, : self.modes, :], self.weights
            )
        out_ft = cheb.patched_change_basis(out_ft, self.M_1)
        if self.zero_BC:
            zero_vec = torch.zeros(b, self.modes, self.out_channels, device=x.device, dtype=x.dtype)
            for patch_global, edges in self.zero_BC.items():
                out_ft[:, patch_global, :2, :2, :] = 0.0
                for edge in edges:
                    if edge == 1:
                        out_ft[:, patch_global, 0, : self.modes, :] = zero_vec
                    elif edge == 2:
                        out_ft[:, patch_global, 1, : self.modes, :] = zero_vec
                    elif edge == 3:
                        out_ft[:, patch_global, : self.modes, 0, :] = zero_vec
                    elif edge == 4:
                        out_ft[:, patch_global, : self.modes, 1, :] = zero_vec
        out_ft = self._enforce_continuity(out_ft, self.continuity_condition, self.modes)
        out_ft = cheb.patched_change_basis(out_ft, self.M)
        return cheb.patched_coefficients_to_values(out_ft)

#########################################
# Output denormalizer wrapper
#########################################
class output_denormalizer_class(nn.Module):
    def __init__(self, output_normalizer) -> None:
        super().__init__()
        self.output_normalizer = output_normalizer
    def forward(self, x):
        return self.output_normalizer.decode(x)

#########################################
# Parallel BAMPNO Architecture
#########################################
class ParallelBAMPNO(nn.Module):
    def __init__(
        self,
        problem_dim: int,
        n_patch: int,
        continuity_condition: Dict,
        n_pts: int,
        grid_filename: str,
        in_dim: int,
        d_v: int,
        out_dim: int,
        L: int,
        modes: int,
        fun_act: str,
        weights_norm: str,
        zero_BC: Optional[Dict] = None,
        arc: str = "Classic",
        RNN: bool = False,
        same_params: bool = False,
        FFTnorm: Optional[str] = None,
        example_output_normalizer=None,
        retrain_seed: int = -1,
        enable_parallel: bool = True,
        backend: str = "nccl",
    ):
        super().__init__()
        self.problem_dim = problem_dim
        self.n_patch = n_patch
        self.continuity_condition = continuity_condition
        self.n_pts = n_pts
        self.grid_filename = grid_filename
        self.in_dim = in_dim + problem_dim
        self.d_v = d_v
        self.out_dim = out_dim
        self.L = L
        self.modes = modes
        self.fun_act = fun_act
        self.weights_norm = weights_norm
        self.zero_BC = zero_BC
        self.arc = arc
        self.RNN = RNN
        self.same_params = same_params
        self.FFTnorm = FFTnorm
        self.retrain_seed = retrain_seed
        self.enable_parallel = enable_parallel

        self.rank, self.world_size, self.local_rank, self.device = _maybe_init_distributed(backend=backend if enable_parallel else "nccl")

        self.bias = False if self.zero_BC is not None else True
        self.M = self.get_M(n_pts).to(self.device)
        self.M_1 = self.get_M_1(n_pts).to(self.device)

        self.output_denormalizer = (
            nn.Identity() if example_output_normalizer is None else output_denormalizer_class(example_output_normalizer)
        )

        assert self.problem_dim == 2, "This implementation supports 2D only."

        # Lifting
        self.p = MLP(self.problem_dim, self.in_dim, self.d_v, 128, self.fun_act, self.bias, self.device)

        # Integral layers / linear maps
        ChebLayer = lambda: ShardedChebyshevLayer(
            self.n_patch,
            self.continuity_condition,
            self.d_v,
            self.d_v,
            self.modes,
            self.M,
            self.M_1,
            self.fun_act,
            zero_BC=self.zero_BC,
            weights_norm=self.weights_norm,
            same_params=self.same_params,
            enable_parallel=self.enable_parallel,
        )

        if self.arc == "Tran":
            if self.RNN:
                self.ws1 = nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device)
                self.ws2 = nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device)
                self.integrals = ChebLayer()
            else:
                self.ws1 = nn.ModuleList([nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device) for _ in range(self.L)])
                self.ws2 = nn.ModuleList([nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device) for _ in range(self.L)])
                self.integrals = nn.ModuleList([ChebLayer() for _ in range(self.L)])
        elif self.arc == "Zongyi":
            if self.RNN:
                self.ws = nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device)
                self.mlps = MLP(self.problem_dim, self.d_v, self.d_v, self.d_v, self.fun_act, self.bias, self.device)
                self.integrals = ChebLayer()
            else:
                self.ws = nn.ModuleList([nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device) for _ in range(self.L)])
                self.mlps = nn.ModuleList([MLP(self.problem_dim, self.d_v, self.d_v, self.d_v, self.fun_act, self.bias, self.device) for _ in range(self.L)])
                self.integrals = nn.ModuleList([ChebLayer() for _ in range(self.L)])
        elif self.arc in ("Classic", "Residual"):
            if self.RNN:
                self.ws = nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device)
                self.integrals = ChebLayer()
            else:
                self.ws = nn.ModuleList([nn.Linear(self.d_v, self.d_v, bias=self.bias, device=self.device) for _ in range(self.L)])
                self.integrals = nn.ModuleList([ChebLayer() for _ in range(self.L)])
        else:
            raise ValueError(f"Unknown architecture type: {self.arc}")

        # Projection
        self.q = MLP(self.problem_dim, self.d_v, self.out_dim, 128, self.fun_act, self.bias, self.device)
        self.to(self.device)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If parallel & local patches only, we must still reconstruct global grid for concatenation.
        is_global = x.shape[1] == self.n_patch
        if not is_global and self.enable_parallel and self.world_size > 1:
            pass
        grid = self.get_grid(
            x.size(0),
            self.grid_filename,
            x.device,
            s=2,
            search_path=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
        )
        if x.shape[1] != grid.shape[1]:
            start, end = _patch_range(self.n_patch, self.world_size, self.rank)
            grid = grid[:, start:end]
        x = torch.cat((grid, x), dim=-1)
        x = self.p(x)
        for i in range(self.L):
            if self.arc == "Tran":
                if self.RNN:
                    x1 = self.integrals(x)
                    x1 = activation(self.ws1(x1), self.fun_act)
                    x1 = self.ws2(x1)
                else:
                    x1 = self.integrals[i](x)
                    x1 = activation(self.ws1[i](x1), self.fun_act)
                    x1 = self.ws2[i](x1)
                x1 = activation(x1, self.fun_act)
                x = x + x1
            elif self.arc == "Classic":
                if self.RNN:
                    x1 = self.integrals(x)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                    x = activation(x, self.fun_act)
            elif self.arc == "Zongyi":
                if self.RNN:
                    x1 = self.integrals(x)
                    x1 = self.mlps(x1)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x1 = self.mlps[i](x1)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                    x = activation(x, self.fun_act)
            elif self.arc == "Residual":
                if self.RNN:
                    x1 = self.integrals(x)
                    x2 = self.ws(x)
                else:
                    x2 = self.integrals[i](x)
                    x1 = self.ws[i](x)
                if i < self.L - 1:
                    x = x + activation(x1 + x2, self.fun_act)
            else:
                raise RuntimeError("Unknown architecture kind")
        out = self.q(x)
        return self.output_denormalizer(out)

    # Caching utilities duplicated (cannot import private cached ones reliably)
    @cache
    def get_grid(
        self,
        batch_size: int,
        grid_filename: str,
        device: torch.device,
        s: int = 1,
        search_path: str = "/",
    ) -> torch.Tensor:
        grid_path = find_file(grid_filename, search_path)
        reader = mat73.loadmat(grid_path)
        X_phys = torch.from_numpy(reader["X_phys"]).type(torch.float32)
        Y_phys = torch.from_numpy(reader["Y_phys"]).type(torch.float32)
        X_phys = X_phys[:, ::s, ::s]
        Y_phys = Y_phys[:, ::s, ::s]
        n_patch = X_phys.shape[0]
        assert n_patch == self.n_patch, "Mismatch n_patch vs grid file"
        X_phys = X_phys.reshape(1, n_patch, *X_phys.shape[1:], 1).repeat(batch_size, 1, 1, 1, 1)
        Y_phys = Y_phys.reshape(1, n_patch, *Y_phys.shape[1:], 1).repeat(batch_size, 1, 1, 1, 1)
        return torch.cat((X_phys, Y_phys), dim=-1).to(device)

    @cache
    def get_M(self, n: int) -> torch.Tensor:
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

    @cache
    def get_M_1(self, n: int) -> torch.Tensor:
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