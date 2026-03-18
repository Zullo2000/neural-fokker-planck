"""Shared ground-truth SDEs, data generation, evaluation, and training helpers.

Used by pfi_benchmark.py and timing_comparison.py to benchmark NFPE
against PFI/UPFI systems and Neural SDE baselines.

Systems:
- Multi-dimensional OU: dX = -Theta @ X dt + sigma dW
- Bistable potential: dX = x(1 - |x|^2) dt + sigma dW
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torchsde
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import MLPSDE, simulate_sde
from nfpe.models import SDE
from nfpe.training import forward_backward_loss


# ---------------------------------------------------------------------------
# Ground-truth SDEs
# ---------------------------------------------------------------------------

class MultiDimOUSDE(SDE):
    """Multi-dimensional OU process: dX = -Theta @ X dt + sigma * dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, theta: torch.Tensor, sigma: float = 0.3):
        super().__init__()
        self.register_buffer("theta", theta)
        self.sigma = sigma

    def f(self, t, y):
        return -torch.matmul(y, self.theta.T)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


class BistableSDE(SDE):
    """Bistable potential: dX = x(1 - |x|^2) dt + sigma * dW.

    Radially symmetric potential V(x) = (|x|^2 - 1)^2 / 4.
    Equilibrium on the unit sphere |x| = 1.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma

    def f(self, t, y):
        r_sq = (y ** 2).sum(dim=-1, keepdim=True)
        return y * (1.0 - r_sq)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


# ---------------------------------------------------------------------------
# System factories
# ---------------------------------------------------------------------------

def make_ou_system(dim, sigma=0.3, seed=42, device=None):
    """Create a multi-D OU system with random positive-definite Theta.

    Returns: (sde_true, theta_true)
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed + 1)
    A = torch.randn(dim, dim, device=device) * 0.3
    theta = torch.matmul(A.T, A) + 0.5 * torch.eye(dim, device=device)
    sde = MultiDimOUSDE(theta, sigma=sigma).to(device)
    return sde, theta


def make_bistable_system(dim, sigma=0.5, device=None):
    """Create a bistable potential system.

    Returns: sde_true
    """
    if device is None:
        device = torch.device("cpu")
    return BistableSDE(sigma=sigma).to(device)


# ---------------------------------------------------------------------------
# Data generation (snapshot-based, matching PFI setting)
# ---------------------------------------------------------------------------

def generate_displaced_ics(n_ics, dim, displacement=2.0, seed=0):
    """Generate well-separated IC centers on a hypersphere shell."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_ics, dim)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / norms * displacement
    return torch.tensor(centers, dtype=torch.float32)


def generate_bistable_ics(n_ics, dim, seed=0):
    """Generate IC centers at mixed radii for the bistable system.

    Places ICs inside (r=0.5), on (r=1.0), and outside (r=1.5) the
    equilibrium shell to cover different dynamical regimes.
    """
    rng = np.random.RandomState(seed)
    radii = [0.5, 1.0, 1.5]
    centers = []
    for i in range(n_ics):
        direction = rng.randn(dim)
        direction = direction / np.linalg.norm(direction)
        r = radii[i % len(radii)]
        centers.append(direction * r)
    return torch.tensor(np.array(centers), dtype=torch.float32)


def simulate_snapshot_multi_ic(
    sde, ic_centers, batch_per_ic, ts, dt, ic_spread=0.1, device=None,
):
    """Generate snapshot moments from multiple IC clusters.

    Simulates one trajectory batch per IC over all timesteps, then extracts
    empirical moments at each time. NFPE only uses the moments (mean and
    covariance), which are permutation-invariant — the training pipeline
    never sees individual particle identities.

    Returns:
        means: (T, K, d)
        covariances: (T, K, d, d)
    """
    if device is None:
        device = torch.device("cpu")
    n_ics = ic_centers.shape[0]
    dim = ic_centers.shape[1]
    T = len(ts)

    all_means = torch.zeros(T, n_ics, dim)
    all_covs = torch.zeros(T, n_ics, dim, dim)

    for k in range(n_ics):
        center = ic_centers[k].to(device)
        y0 = center.unsqueeze(0) + torch.randn(
            batch_per_ic, dim, device=device,
        ) * ic_spread

        with torch.no_grad():
            traj = torchsde.sdeint(
                sde, y0, ts.to(device), dt=dt, method="euler",
            )  # (T, batch_per_ic, dim)

        for t_idx in range(T):
            snapshot = traj[t_idx]
            mu = snapshot.mean(dim=0)
            centered = snapshot - mu.unsqueeze(0)
            cov = (centered.T @ centered) / (batch_per_ic - 1)
            all_means[t_idx, k] = mu.cpu()
            all_covs[t_idx, k] = cov.cpu()

        del traj
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_means, all_covs


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_nfpe_multi_ic(
    model, means, covariances, dt, epochs=3000, lr=1e-3,
    cov_weight=100.0, log_interval=200, grad_clip=1.0,
    lr_schedule=True, weight_decay=0.0, use_jacobian=True,
    verbose=True, device=None,
):
    """Train NFPE with gradient clipping, LR scheduling, and timing.

    Returns: (history, total_time_seconds)
    """
    if device is None:
        device = torch.device("cpu")
    means = means.to(device)
    covariances = covariances.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

    history = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss, loss_means, loss_covs = forward_backward_loss(
            model, means, covariances, dt,
            cov_weight=cov_weight, use_jacobian=use_jacobian,
        )

        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - start_time
        record = {
            "epoch": epoch,
            "total": total_loss.item(),
            "means": loss_means.item(),
            "covs": loss_covs.item(),
            "wall_clock": elapsed,
        }
        history.append(record)

        if verbose and epoch % log_interval == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"loss_means: {loss_means.item():.6f} | "
                f"loss_covs: {loss_covs.item():.6f} | "
                f"time: {elapsed:.1f}s"
            )

    total_time = time.perf_counter() - start_time
    return history, total_time


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_drift_error(model, true_drift_fn, dim, n_eval=200,
                         eval_radius=2.0, device=None):
    """Evaluate learned drift vs true drift at random test points.

    Args:
        model: learned MLPSDE
        true_drift_fn: callable(x) -> drift, where x is (N, d)
        dim: state dimension
        n_eval: number of test points
        eval_radius: scale of test points

    Returns: dict with drift_mse, drift_relative_error
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(999)
    x_eval = (torch.randn(n_eval, dim) * eval_radius).to(device)

    with torch.no_grad():
        f_learned = model.drift_net(x_eval)
        f_true = true_drift_fn(x_eval)

    mse = ((f_learned - f_true) ** 2).mean().item()
    rel = (torch.norm(f_learned - f_true) / torch.norm(f_true)).item()

    return {"drift_mse": mse, "drift_relative_error": rel}


def evaluate_diffusion_error(model, true_G_value, dim, n_eval=200,
                             eval_radius=2.0, device=None):
    """Evaluate learned diffusion G = BB^T vs true (constant) G.

    Args:
        model: learned MLPSDE with .diffusion_matrix()
        true_G_value: scalar sigma^2 (assumes true G = sigma^2 * I)
        dim: state dimension

    Returns: dict with diffusion_mse
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(999)
    x_eval = (torch.randn(n_eval, dim) * eval_radius).to(device)

    G_true = torch.eye(dim, device=device) * true_G_value

    with torch.no_grad():
        G_learned = model.diffusion_matrix(x_eval)  # (N, d, d)

    G_true_batch = G_true.unsqueeze(0).expand_as(G_learned)
    mse = ((G_learned - G_true_batch) ** 2).mean().item()

    return {"diffusion_mse": mse}
