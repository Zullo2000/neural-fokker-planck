"""SDE model definitions for Neural Fokker-Planck Equations.

Provides parametric SDE models (Linear, CIR) and a neural MLP-based SDE
for learning nonlinear drift and diffusion from data.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDepWrapper(nn.Module):
    """Wraps a time-independent model f(y) as f(t, y) for ODE solvers."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(y)


class SDE(nn.Module):
    """Abstract base class for Ito SDEs: dX = f(t,X)dt + g(t,X)dW."""

    noise_type = "general"
    sde_type = "ito"

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift coefficient."""
        raise NotImplementedError

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient."""
        raise NotImplementedError


class LinearSDE(nn.Module):
    """Affine SDE: f(x) = Lx + b, g(x) = Lx + b.

    Used for systems like Black-Scholes / Geometric Brownian Motion
    where drift and diffusion are linear in the state.

    Args:
        f_linear: Drift matrix (state_size, state_size).
        f_bias: Drift bias (state_size,).
        g_linear: Diffusion matrix (state_size, state_size * brownian_size).
        g_bias: Diffusion bias (state_size * brownian_size,).
        learnable: If True, parameters are trainable.
    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(
        self,
        f_linear: torch.Tensor,
        f_bias: torch.Tensor,
        g_linear: torch.Tensor,
        g_bias: torch.Tensor,
        learnable: bool = True,
    ):
        super().__init__()
        if learnable:
            self.f_linear = nn.Parameter(f_linear)
            self.f_bias = nn.Parameter(f_bias)
            self.g_linear = nn.Parameter(g_linear)
            self.g_bias = nn.Parameter(g_bias)
        else:
            self.register_buffer("f_linear", f_linear)
            self.register_buffer("f_bias", f_bias)
            self.register_buffer("g_linear", g_linear)
            self.register_buffer("g_bias", g_bias)

        self.register_buffer("state_size", torch.tensor(f_linear.shape[0]))
        self.register_buffer(
            "brownian_size",
            torch.tensor(g_linear.shape[1] // f_linear.shape[0]),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return F.linear(y, self.f_linear, self.f_bias)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: f(x) = f_linear @ x + f_bias."""
        return F.linear(y, self.f_linear, self.f_bias)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion: g(x) = g_linear @ x + g_bias, reshaped for torchsde."""
        batch_size = y.shape[0] if len(y.shape) > 1 else 1
        return F.linear(y, self.g_linear, self.g_bias).view(
            batch_size, self.state_size, self.brownian_size,
        )


class CIRSDE(nn.Module):
    """Cox-Ingersoll-Ross SDE: dX = kappa*(theta - X)dt + sigma*sqrt(X)dW.

    Classic model for mean-reverting positive processes (interest rates, etc.).

    Args:
        kappa: Mean-reversion speed.
        theta: Long-term mean.
        sigma: Volatility of volatility.
        learnable: If True, parameters are trainable.
    """

    noise_type = "scalar"
    sde_type = "ito"

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        learnable: bool = False,
    ):
        super().__init__()
        self.kappa = nn.Parameter(
            torch.tensor(kappa, dtype=torch.float32), requires_grad=learnable,
        )
        self.theta = nn.Parameter(
            torch.tensor(theta, dtype=torch.float32), requires_grad=learnable,
        )
        self.sigma = nn.Parameter(
            torch.tensor(sigma, dtype=torch.float32), requires_grad=learnable,
        )

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: kappa * (theta - y)."""
        return self.kappa * (self.theta - y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion: sigma * sqrt(max(y, 0))."""
        return self.sigma * torch.sqrt(torch.clamp(y, min=0.0))


class MLPSDE(nn.Module):
    """Neural SDE with MLP-parameterized drift and diffusion.

    Drift F_theta: R^d -> R^d via a standard MLP.
    Diffusion G_theta: R^d -> R^{d x d} (positive semi-definite) via
    Cholesky parameterization: outputs lower-triangular L, G = LL^T.

    Args:
        state_size: Dimension of the state space.
        hidden_sizes: List of hidden layer widths.
        activation: Activation function class.
        brownian_size: Dimension of the Wiener process (defaults to state_size).
    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(
        self,
        state_size: int,
        hidden_sizes: Optional[List[int]] = None,
        activation: type = nn.Tanh,
        brownian_size: Optional[int] = None,
    ):
        super().__init__()
        self.state_size = state_size
        self.brownian_size = brownian_size or state_size

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        # Drift network: R^d -> R^d
        drift_layers: List[nn.Module] = []
        in_dim = state_size
        for h in hidden_sizes:
            drift_layers.append(nn.Linear(in_dim, h))
            drift_layers.append(activation())
            in_dim = h
        drift_layers.append(nn.Linear(in_dim, state_size))
        self.drift_net = nn.Sequential(*drift_layers)

        # Diffusion network: R^d -> R^{d * brownian_size}
        # Outputs lower-triangular Cholesky factor L, so G = LL^T is PSD
        diff_layers: List[nn.Module] = []
        in_dim = state_size
        for h in hidden_sizes:
            diff_layers.append(nn.Linear(in_dim, h))
            diff_layers.append(activation())
            in_dim = h
        # Output: d * brownian_size entries for the diffusion matrix
        diff_layers.append(nn.Linear(in_dim, state_size * self.brownian_size))
        self.diff_net = nn.Sequential(*diff_layers)

        # Initialize diffusion output small but nonzero
        nn.init.normal_(self.diff_net[-1].weight, std=0.01)
        nn.init.constant_(self.diff_net[-1].bias, 0.1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Drift only (for Jacobian computation)."""
        return self.drift_net(y)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift F_theta(x)."""
        return self.drift_net(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion B_theta(x), reshaped to (batch, d, m)."""
        batch_size = y.shape[0] if len(y.shape) > 1 else 1
        raw = self.diff_net(y)
        return raw.view(batch_size, self.state_size, self.brownian_size)

    def diffusion_matrix(self, y: torch.Tensor) -> torch.Tensor:
        """Compute G(x) = B(x) B(x)^T (positive semi-definite).

        Args:
            y: State tensor (..., d).

        Returns:
            G: Diffusion tensor (..., d, d).
        """
        B = self.g(0, y)
        return torch.matmul(B, B.transpose(-1, -2))
