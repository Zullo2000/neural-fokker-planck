"""Data generation and processing for NFPE experiments.

Provides utilities for:
- Simulating SDE trajectories via torchsde
- Fitting Gaussian Mixture Models to trajectory snapshots
- Extracting moment tensors (means, covariances) for training
- Generating cross-sectional snapshot data (no trajectory tracking)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torchsde
import numpy as np
from sklearn.mixture import GaussianMixture


def simulate_sde(
    sde: torch.nn.Module,
    y0: torch.Tensor,
    ts: torch.Tensor,
    dt: float = 1e-3,
    method: str = "euler",
) -> torch.Tensor:
    """Simulate SDE trajectories using torchsde.

    Args:
        sde: SDE model with .f(t, y) and .g(t, y) methods,
             plus noise_type and sde_type attributes.
        y0: Initial conditions (batch_size, state_size).
        ts: Time points at which to record the solution.
        dt: Integration step size.
        method: SDE solver method ('euler', 'milstein', 'srk').

    Returns:
        Trajectories tensor (n_times, batch_size, state_size).
    """
    with torch.no_grad():
        ys = torchsde.sdeint(sde, y0, ts, dt=dt, method=method)
    return ys


def fit_gmm_to_snapshots(
    trajectories: torch.Tensor,
    n_components: int = 1,
    sort_by_mean: bool = True,
) -> tuple[list[GaussianMixture], torch.Tensor, torch.Tensor]:
    """Fit Gaussian Mixture Models to trajectory snapshots at each time step.

    Args:
        trajectories: Tensor of shape (n_times, batch_size, state_size).
        n_components: Number of Gaussian components per snapshot.
        sort_by_mean: If True, sort components by mean for consistency
                      across time steps.

    Returns:
        gmm_models: List of fitted GaussianMixture objects.
        means: Tensor of means (n_times, n_components, state_size).
        covariances: Tensor of covariances
                     (n_times, n_components, state_size, state_size).
    """
    n_times, batch_size, state_size = trajectories.shape
    data = trajectories.detach().cpu().numpy()

    gmm_models = []
    means_list = []
    covs_list = []

    for t_idx in range(n_times):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
        )
        gmm.fit(data[t_idx])
        gmm_models.append(gmm)

        means = gmm.means_  # (n_components, state_size)
        covs = gmm.covariances_  # (n_components, state_size, state_size)

        if sort_by_mean and n_components > 1:
            # Sort by first coordinate of mean for consistency
            order = np.argsort(means[:, 0])
            means = means[order]
            covs = covs[order]

        means_list.append(means)
        covs_list.append(covs)

    means_tensor = torch.tensor(
        np.array(means_list), dtype=torch.float32,
    )
    covs_tensor = torch.tensor(
        np.array(covs_list), dtype=torch.float32,
    )

    # Squeeze component dim if n_components == 1
    if n_components == 1:
        means_tensor = means_tensor.squeeze(1)
        covs_tensor = covs_tensor.squeeze(1)

    return gmm_models, means_tensor, covs_tensor


def shuffle_snapshots(
    trajectories: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Destroy trajectory correspondence by shuffling particles at each time step.

    The marginal distribution at each time step is preserved exactly,
    but the mapping between particles across time steps is destroyed.
    This simulates cross-sectional snapshot data where individual
    particles are not tracked.

    Args:
        trajectories: Tensor of shape (n_times, batch_size, state_size).
        seed: Optional random seed for reproducibility.

    Returns:
        Shuffled trajectories with same shape and same per-time marginals.
    """
    if seed is not None:
        torch.manual_seed(seed)
    n_times, batch_size, _state_size = trajectories.shape
    shuffled = trajectories.clone()
    for t in range(n_times):
        perm = torch.randperm(batch_size)
        shuffled[t] = trajectories[t, perm]
    return shuffled


def simulate_independent_snapshots(
    sde: torch.nn.Module,
    y0_sampler: Callable[[int], torch.Tensor],
    ts: torch.Tensor,
    n_particles: int,
    dt: float = 1e-3,
    method: str = "euler",
) -> torch.Tensor:
    """Generate cross-sectional snapshots with no particle correspondence.

    At each observation time t_n, simulates a fresh batch of particles
    from t=0 to t_n, keeping only the final state. The particles at
    different time steps are completely independent — there is no
    trajectory tracking.

    This models real-world scenarios like flow cytometry where samples
    are destroyed by measurement.

    Args:
        sde: SDE model with .f(t, y) and .g(t, y) methods.
        y0_sampler: Callable that takes n_particles and returns initial
                    conditions of shape (n_particles, state_size).
        ts: Observation time points (must start at 0 or near 0).
        n_particles: Number of particles to draw at each snapshot.
        dt: Integration step size.
        method: SDE solver method.

    Returns:
        Snapshot tensor (n_times, n_particles, state_size) where particles
        at different time steps are independent.
    """
    n_times = len(ts)
    y0_sample = y0_sampler(1)
    state_size = y0_sample.shape[-1]
    snapshots = torch.zeros(n_times, n_particles, state_size)

    with torch.no_grad():
        for t_idx in range(n_times):
            if t_idx == 0 or ts[t_idx].item() == 0.0:
                # At t=0, just sample initial conditions
                snapshots[t_idx] = y0_sampler(n_particles)
            else:
                # Simulate fresh particles from t=0 to t_n
                y0 = y0_sampler(n_particles)
                ts_segment = torch.tensor([0.0, ts[t_idx].item()])
                ys = torchsde.sdeint(
                    sde, y0, ts_segment, dt=dt, method=method,
                )
                snapshots[t_idx] = ys[-1]  # Keep only final state

    return snapshots
