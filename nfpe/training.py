"""Training utilities for Neural Fokker-Planck Equations.

Implements the forward-backward moment matching loss and training loops
for learning SDE drift and diffusion from observed moment dynamics.

Supports both single-component (means shape: T x d) and multi-component
(means shape: T x K x d) Gaussian Mixture representations.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.func import vmap, jacfwd


def compute_moment_derivatives(
    model: nn.Module,
    means: torch.Tensor,
    covariances: torch.Tensor,
    use_jacobian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dmu/dt and dSigma/dt from the model at given moments.

    For the Fokker-Planck moment equations:
        dmu/dt = F(mu)
        dSigma/dt = DF @ Sigma + Sigma @ DF^T + G(mu)

    where G(mu) = B(mu) @ B(mu)^T.

    Args:
        model: SDE model with .f(t, y) and .g(t, y) methods.
        means: Mean vectors (n_times, state_size) or (n_times, K, state_size).
        covariances: Covariance matrices (n_times, d, d) or (n_times, K, d, d).
        use_jacobian: If True, compute DF via AD. If False, use the model's
                      f_linear attribute directly (for linear models).

    Returns:
        mu_dot: Time derivative of means, same shape as means.
        sigma_dot: Time derivative of covariances, same shape as covariances.
    """
    # Flatten multi-component into batch for vectorized computation
    orig_shape = means.shape
    if means.dim() == 3:
        # Multi-component: (T, K, d) -> (T*K, d)
        T, K, d = means.shape
        means_flat = means.reshape(T * K, d)
        covs_flat = covariances.reshape(T * K, d, d)
    else:
        means_flat = means
        covs_flat = covariances

    mu_dot_flat = model.f(torch.tensor(0.0), means_flat)

    if use_jacobian:
        def f_only(y):
            return model.f(torch.tensor(0.0), y)

        jac_fn = vmap(jacfwd(f_only))
        n = means_flat.shape[0]
        chunk_size = 64
        if n <= chunk_size:
            jac = jac_fn(means_flat)
        else:
            jacs = []
            for i in range(0, n, chunk_size):
                jacs.append(jac_fn(means_flat[i:i + chunk_size]))
            jac = torch.cat(jacs, dim=0)

        b = model.g(torch.tensor(0.0), means_flat)
        if b.dim() == 2:
            b = b.unsqueeze(-1)
        diffusion_term = torch.matmul(b, b.transpose(-1, -2))
    else:
        jac = model.f_linear.unsqueeze(0).expand(
            means_flat.shape[0], -1, -1,
        )

        if hasattr(model, "g_linear"):
            b = (
                torch.matmul(model.g_linear, means_flat.unsqueeze(-1))
                + model.g_bias.unsqueeze(-1)
            )
            diffusion_term = torch.matmul(b, b.transpose(-1, -2))
        else:
            b = model.g(torch.tensor(0.0), means_flat)
            if b.dim() == 2:
                b = b.unsqueeze(-1)
            diffusion_term = torch.matmul(b, b.transpose(-1, -2))

    sigma_dot_flat = (
        torch.matmul(jac, covs_flat)
        + torch.matmul(covs_flat, jac.transpose(-1, -2))
        + diffusion_term
    )

    # Reshape back to original layout
    if means.dim() == 3:
        mu_dot = mu_dot_flat.reshape(orig_shape)
        sigma_dot = sigma_dot_flat.reshape(covariances.shape)
    else:
        mu_dot = mu_dot_flat
        sigma_dot = sigma_dot_flat

    return mu_dot, sigma_dot


def forward_backward_loss(
    model: nn.Module,
    means: torch.Tensor,
    covariances: torch.Tensor,
    dt: float,
    cov_weight: float = 100.0,
    use_jacobian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward-backward moment matching loss.

    Uses three-point stencil: for each interior time step t_n, the model
    must predict both the forward step (t_n -> t_{n+1}) and backward step
    (t_n -> t_{n-1}), which disentangles drift from diffusion.

    Supports both single-component and multi-component Gaussian Mixtures.

    Args:
        model: SDE model with .f() and .g() methods.
        means: Observed means (n_times, d) or (n_times, K, d).
        covariances: Observed covariances (n_times, d, d) or (n_times, K, d, d).
        dt: Time step between observations.
        cov_weight: Weight for the covariance loss relative to mean loss.
        use_jacobian: If True, compute Jacobian via AD (for nonlinear models).

    Returns:
        total_loss: Combined weighted loss.
        loss_means: MSE loss on mean predictions.
        loss_covs: MSE loss on covariance predictions.
    """
    mse = nn.MSELoss()

    mu_dot, sigma_dot = compute_moment_derivatives(
        model, means, covariances, use_jacobian=use_jacobian,
    )

    # Forward predictions: mu(t) + dt * dmu/dt ≈ mu(t+1)
    means_forward = means[:-1] + dt * mu_dot[:-1]
    covs_forward = covariances[:-1] + dt * sigma_dot[:-1]

    # Backward predictions: mu(t) - dt * dmu/dt ≈ mu(t-1)
    means_backward = means[1:] - dt * mu_dot[1:]
    covs_backward = covariances[1:] - dt * sigma_dot[1:]

    loss_means = (
        mse(means_forward, means[1:])
        + mse(means_backward, means[:-1])
    )
    loss_covs = (
        mse(covs_forward, covariances[1:])
        + mse(covs_backward, covariances[:-1])
    )

    total_loss = loss_means + cov_weight * loss_covs
    return total_loss, loss_means, loss_covs


def train_nfpe(
    model: nn.Module,
    means: torch.Tensor,
    covariances: torch.Tensor,
    dt: float,
    epochs: int = 1000,
    lr: float = 1e-2,
    cov_weight: float = 100.0,
    use_jacobian: bool = True,
    log_interval: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """Train an NFPE model to match observed moment dynamics.

    Args:
        model: SDE model to train.
        means: Observed means (n_times, d) or (n_times, K, d).
        covariances: Observed covariances (n_times, d, d) or (n_times, K, d, d).
        dt: Time step between observations.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        cov_weight: Weight for covariance loss.
        use_jacobian: If True, use AD Jacobian (required for nonlinear models).
        log_interval: How often to log losses.
        verbose: If True, print losses.

    Returns:
        History of losses as list of dicts with keys
        'epoch', 'total', 'means', 'covs'.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss, loss_means, loss_covs = forward_backward_loss(
            model, means, covariances, dt,
            cov_weight=cov_weight,
            use_jacobian=use_jacobian,
        )

        total_loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "total": total_loss.item(),
            "means": loss_means.item(),
            "covs": loss_covs.item(),
        }
        history.append(record)

        if verbose and epoch % log_interval == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"loss_means: {loss_means.item():.6f} | "
                f"loss_covs: {loss_covs.item():.6f}"
            )

    return history
