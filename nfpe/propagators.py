"""Gaussian moment propagators for the Fokker-Planck equation.

Given an SDE dX = F(X)dt + B(X)dW, the Fokker-Planck equation governs the
evolution of the probability density. Under Gaussian closure, the moments
evolve as:
    dmu/dt = F(mu)
    dSigma/dt = DF * Sigma + Sigma * DF^T + B * B^T

These propagators integrate these moment dynamics as Neural ODEs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.func import vmap, jacfwd

from .utils import phi_1_pade


class JacobianAuxWrapper(nn.Module):
    """Wrapper for efficient joint Jacobian + function evaluation.

    Returns (f(y), f(y)) so that jacfwd with has_aux=True computes
    both the Jacobian and function value in a single forward pass.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def f(self, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.model.f(t, y)
        return result, result

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model.g(t, y)


class EulerGaussianPropagator(nn.Module):
    """Propagates Gaussian moments (mu, Sigma) using Jacobian-based dynamics.

    Computes the RHS of the moment ODEs:
        dmu/dt = F(mu)
        dSigma/dt = Sigma @ DF^T + DF @ Sigma + B @ B^T

    where DF = Jacobian of F evaluated at mu, computed via forward-mode AD.

    Compatible with torchdiffeq ODE solvers. The state vector is
    [mu; vec(Sigma)] where vec flattens the covariance matrix.

    Args:
        dim: State dimension d.
        model: SDE model with .f(t, y) and .g(t, y) methods.
    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, dim: int, model: nn.Module):
        super().__init__()
        self.dim = dim
        self.model = JacobianAuxWrapper(model)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(t, y)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute moment dynamics RHS."""
        mu = y[..., :self.dim]
        sigma = torch.unflatten(y[..., self.dim:], -1, (self.dim, self.dim))

        # Compute Jacobian and drift simultaneously
        jac, mu_hat = vmap(
            jacfwd(self.model.f, argnums=1, has_aux=True),
            in_dims=(None, 0),
        )(torch.tensor(0.0), mu)

        # Diffusion matrix B(mu)
        b = self.model.g(torch.tensor(0.0), mu)

        # Covariance dynamics: Sigma @ DF^T + DF @ Sigma + B @ B^T
        sigma_hat = (
            torch.matmul(sigma, jac.transpose(-1, -2))
            + torch.matmul(jac, sigma)
            + torch.matmul(b, b.transpose(-1, -2))
        )

        return torch.cat(
            (mu_hat, sigma_hat.flatten(start_dim=-2)), dim=-1,
        )

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """No stochastic term — this is a deterministic ODE."""
        return torch.tensor(0.0, device=y.device)


class UnscentedPropagator(nn.Module):
    """Propagates moments using the Unscented Transform.

    Instead of computing Jacobians explicitly, uses sigma points
    (unscented transform) to approximate the mean and covariance
    propagation through nonlinear dynamics.

    The state is stored as a (d+1, d) matrix where:
        - Row 0: mean mu
        - Rows 1..d: columns of sqrt(Sigma) (Cholesky factor)

    Args:
        dim: State dimension d.
        model: SDE model with .f(t, y) method.
    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, dim: int, model: nn.Module):
        super().__init__()
        self.dim = dim
        self.model = model

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(t, y)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shape = y.shape
        expected_size = self.dim + self.dim * self.dim
        assert shape[-1] == expected_size, (
            f"Input dimension mismatch: got {shape[-1]}, "
            f"expected {expected_size}"
        )
        y = y.reshape(-1, self.dim + 1, self.dim)
        y_hat = self.model.f(t, y)
        return y_hat.reshape(shape)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """No stochastic term."""
        return torch.zeros(1, device=y.device)


class EulerRosenbrockModel(nn.Module):
    """Euler-Rosenbrock integrator for stiff moment dynamics.

    Uses the Pade approximation of phi_1 for implicit-like stability
    with explicit-like cost. Suitable for stiff systems where the
    standard Euler method would require very small time steps.

    Args:
        dim: State dimension d.
        model: SDE model with .forward(y) and .g(t, y) methods.
        h: Fixed step size for the Rosenbrock scheme.
    """

    def __init__(self, dim: int, model: nn.Module, h: float):
        super().__init__()
        self.dim = dim
        self.model = JacobianAuxWrapper(model)
        self.h = h

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_mu = y[..., :self.dim]
        y_sigma = y[..., self.dim:]

        jac, y_mu_new = vmap(
            jacfwd(self.model.f, argnums=1, has_aux=True),
            in_dims=(None, 0),
        )(torch.tensor(0.0), y_mu)
        phi_1_jac = phi_1_pade(self.h * jac)

        combined = torch.cat((y_mu_new, y_sigma), dim=-1).unsqueeze(-1)
        return torch.matmul(phi_1_jac, combined).squeeze(-1)
