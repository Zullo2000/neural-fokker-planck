"""Gaussian propagation experiment.

Propagates Gaussian distributions through a known SDE using deterministic
moment dynamics (Neural ODE on mean + covariance). Validates that the
predicted Gaussian matches the empirical distribution of sample trajectories.

System: Damped spiral with additive noise.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchdiffeq
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import (
    LinearSDE,
    TimeDepWrapper,
    EulerGaussianPropagator,
    UnscentedPropagator,
)


def create_spiral_sde() -> LinearSDE:
    """Create a damped spiral SDE with additive noise."""
    return LinearSDE(
        f_linear=torch.tensor([[-0.1, 0.5], [-0.5, -0.1]]),
        f_bias=torch.tensor([0.0, 0.0]),
        g_linear=torch.zeros(2, 2),
        g_bias=torch.tensor([0.05, 0.05]),
        learnable=False,
    )


def run_experiment(args):
    """Run the Gaussian propagation experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sde = create_spiral_sde()
    ts = torch.linspace(0, 2 * np.pi, args.t_size)
    dim = 2

    # Initial Gaussian: mu0, Sigma0
    mu0 = torch.tensor([1.0, 1.0])
    sigma0 = 0.1 * torch.eye(dim)

    # --- Propagate moments using UnscentedPropagator ---
    print("Propagating Gaussian moments...")
    propagator = UnscentedPropagator(dim, sde)

    # State: [mu; vec(sqrt(Sigma))]
    # For UnscentedPropagator, state is reshaped as (d+1, d) matrix
    sigma0_sqrt = torch.linalg.cholesky(sigma0)
    x0 = torch.cat([mu0, sigma0_sqrt.flatten()]).unsqueeze(0)  # (1, d+d*d)

    x_prop = torchdiffeq.odeint(
        propagator, x0, ts, method="rk4",
    )  # (T, 1, d+d*d)

    # Extract means and covariance square roots
    means_prop = x_prop[:, 0, :dim].detach()
    c_sqrt = x_prop[:, 0, dim:].reshape(-1, dim, dim).detach()
    covs_prop = torch.matmul(c_sqrt, c_sqrt.transpose(-1, -2))

    # --- Propagate using EulerGaussianPropagator (Jacobian-based) ---
    print("Propagating with Jacobian-based propagator...")
    euler_prop = EulerGaussianPropagator(dim, sde)

    x0_euler = torch.cat([mu0, sigma0.flatten()]).unsqueeze(0)
    x_euler = torchdiffeq.odeint(euler_prop, x0_euler, ts, method="rk4")

    means_euler = x_euler[:, 0, :dim].detach()
    covs_euler = x_euler[:, 0, dim:].reshape(-1, dim, dim).detach()

    # --- Validate with sample trajectories ---
    print(f"Validating with {args.n_samples} sample trajectories...")
    y0_samples = torch.distributions.MultivariateNormal(
        mu0, sigma0,
    ).sample((args.n_samples,))

    # Integrate samples through the deterministic ODE (drift only)
    y_samples = torchdiffeq.odeint(
        TimeDepWrapper(sde), y0_samples, ts, method="rk4",
    )  # (T, n_samples, d)

    # Empirical means and covariances
    means_emp = y_samples.mean(dim=1)
    centered = y_samples - means_emp.unsqueeze(1)
    covs_emp = torch.matmul(
        centered.transpose(1, 2), centered,
    ) / (args.n_samples - 1)

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    theta_grid = np.linspace(0, 2 * np.pi, 100)

    # Plot 1: Trajectory samples + predicted Gaussian contours
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot sample trajectories (light)
    y_np = y_samples.numpy()
    for i in range(min(args.n_samples, 50)):
        ax.plot(y_np[:, i, 0], y_np[:, i, 1], alpha=0.1, color="gray", linewidth=0.5)

    # Plot Gaussian contours at select times
    for t_idx in range(0, args.t_size, max(1, args.t_size // 12)):
        mu = means_euler[t_idx].numpy()
        cov = covs_euler[t_idx].numpy()

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)

        ellipse = (
            mu[:, None]
            + eigvecs @ np.diag(np.sqrt(eigvals)) @ np.array(
                [np.cos(theta_grid), np.sin(theta_grid)],
            )
        )
        alpha = 0.3 + 0.7 * t_idx / args.t_size
        ax.plot(ellipse[0], ellipse[1], color="blue", alpha=alpha, linewidth=1.5)
        ax.plot(mu[0], mu[1], "bo", markersize=4, alpha=alpha)

    # Plot mean trajectory
    ax.plot(
        means_euler[:, 0].numpy(), means_euler[:, 1].numpy(),
        "b-", linewidth=2, label="Predicted mean",
    )
    ax.plot(
        means_emp[:, 0].numpy(), means_emp[:, 1].numpy(),
        "r--", linewidth=2, label="Empirical mean",
    )

    ax.set_title("Gaussian Propagation: Predicted vs Empirical")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "propagation_contours.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Mean and covariance comparison over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t_np = ts.numpy()

    axes[0, 0].plot(t_np, means_euler[:, 0].numpy(), "b-", label="Predicted")
    axes[0, 0].plot(t_np, means_emp[:, 0].numpy(), "r--", label="Empirical")
    axes[0, 0].set_title("Mean x₁")
    axes[0, 0].legend()

    axes[0, 1].plot(t_np, means_euler[:, 1].numpy(), "b-", label="Predicted")
    axes[0, 1].plot(t_np, means_emp[:, 1].numpy(), "r--", label="Empirical")
    axes[0, 1].set_title("Mean x₂")
    axes[0, 1].legend()

    axes[1, 0].plot(t_np, covs_euler[:, 0, 0].numpy(), "b-", label="Predicted Σ₁₁")
    axes[1, 0].plot(t_np, covs_emp[:, 0, 0].numpy(), "r--", label="Empirical Σ₁₁")
    axes[1, 0].set_title("Variance x₁")
    axes[1, 0].legend()

    axes[1, 1].plot(t_np, covs_euler[:, 1, 1].numpy(), "b-", label="Predicted Σ₂₂")
    axes[1, 1].plot(t_np, covs_emp[:, 1, 1].numpy(), "r--", label="Empirical Σ₂₂")
    axes[1, 1].set_title("Variance x₂")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Time")

    plt.suptitle("Moment Propagation: Predicted vs Empirical")
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "propagation_moments.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Compute error metrics
    mean_err = torch.norm(means_euler - means_emp).item()
    cov_err = torch.norm(covs_euler - covs_emp).item()
    print(f"\nMean trajectory error (L2): {mean_err:.6f}")
    print(f"Covariance trajectory error (Frobenius): {cov_err:.6f}")
    print(f"\nFigures saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian propagation experiment",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=str, default="results/propagation",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
