"""Matrix utilities for moment propagation.

Provides phi_1 functions (matrix exponential-based) used in
Euler-Rosenbrock solvers for stiff ODE systems.
"""

import torch


def phi_1(M: torch.Tensor) -> torch.Tensor:
    """Compute phi_1(M) = (exp(M) - I) / M via augmented matrix exponential.

    Uses the identity that phi_1(M) appears in the top-right block of
    exp([[M, I], [0, 0]]).

    Args:
        M: Square matrix or batch of square matrices (..., d, d).

    Returns:
        phi_1(M) with same shape as M.
    """
    shape = M.shape
    if len(shape) < 3:
        M = M.unsqueeze(0)
        shape = M.shape

    eye = torch.eye(shape[-2], shape[-1], device=M.device, dtype=M.dtype)
    eye = eye.expand(shape[:-2] + (-1, -1))
    first_row = torch.cat((M, eye), dim=-1)
    last_row = torch.zeros(
        *shape[:-2], shape[-2], 2 * shape[-1],
        device=M.device, dtype=M.dtype,
    )
    M_aug = torch.cat((first_row, last_row), dim=-2)
    exp_M_aug = torch.matrix_exp(M_aug)
    return exp_M_aug[..., :shape[-2], shape[-1]:]


def phi_1_pade(M: torch.Tensor) -> torch.Tensor:
    """Compute phi_1(M) via (2,1) Pade approximation.

    Approximates phi_1(M) = (I - M/3)^{-1} (I + M/6), which is accurate
    for small ||M|| and more efficient than the full matrix exponential.

    Args:
        M: Square matrix or batch of square matrices (..., d, d).

    Returns:
        phi_1(M) with same shape as M.
    """
    shape = M.shape
    if len(shape) < 3:
        M = M.unsqueeze(0)
        shape = M.shape

    I = torch.eye(
        shape[-2], shape[-1], device=M.device, dtype=M.dtype,
    ).unsqueeze(0)
    return torch.linalg.solve(I - M / 3, I + M / 6)
