"""2D Harmonic Oscillator identification experiment.

Learns the drift and diffusion of a 2D harmonic oscillator with additive
noise from simulated trajectory data using NFPE forward-backward training.

True system:
    dx1 = x2 dt
    dx2 = -x1 dt + 0.05 dW
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import LinearSDE, simulate_sde, fit_gmm_to_snapshots, train_nfpe


def create_oscillator_sde() -> LinearSDE:
    """Create the ground-truth 2D harmonic oscillator SDE."""
    return LinearSDE(
        f_linear=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        f_bias=torch.tensor([0.0, 0.0]),
        g_linear=torch.zeros(2, 2),  # No multiplicative noise
        g_bias=torch.tensor([0.0, 0.05]),
        learnable=False,
    )


def create_learnable_sde() -> LinearSDE:
    """Create a learnable 2D linear SDE with random initialization."""
    return LinearSDE(
        f_linear=torch.randn(2, 2) * 0.1,
        f_bias=torch.randn(2) * 0.1,
        g_linear=torch.randn(2, 2) * 0.1,
        g_bias=torch.randn(2) * 0.1,
        learnable=True,
    )


def run_experiment(args):
    """Run the oscillator identification experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Ground truth ---
    sde_true = create_oscillator_sde()
    ts = torch.linspace(0, 2 * np.pi, args.t_size)
    y0 = torch.randn(args.batch_size, 2) * 0.5 + torch.tensor([1.0, 0.0])

    # --- Simulate ---
    print("Simulating 2D oscillator trajectories...")
    trajectories = simulate_sde(sde_true, y0, ts, dt=args.dt)

    # --- Fit GMMs ---
    print("Fitting Gaussian Mixtures...")
    gmm_models, means, covariances = fit_gmm_to_snapshots(
        trajectories, n_components=1,
    )

    dt_obs = (ts[1] - ts[0]).item()

    # --- Train NFPE ---
    print("\nTraining NFPE (forward-backward)...")
    sde_estim = create_learnable_sde()
    history = train_nfpe(
        sde_estim, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        use_jacobian=False,
        log_interval=args.log_interval,
    )

    # --- Report ---
    print("\n" + "=" * 60)
    print("Learned Parameters")
    print("=" * 60)
    print(f"True f_linear:\n{sde_true.f_linear}")
    print(f"Learned f_linear:\n{sde_estim.f_linear.data}")
    print(f"\nTrue g_bias: {sde_true.g_bias}")
    print(f"Learned g_bias: {sde_estim.g_bias.data}")

    f_err = torch.norm(sde_estim.f_linear.data - sde_true.f_linear).item()
    g_err = torch.norm(sde_estim.g_bias.data - sde_true.g_bias).item()
    print(f"\nf_linear error (Frobenius): {f_err:.6f}")
    print(f"g_bias error (L2): {g_err:.6f}")

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: Phase space - original vs estimated
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    traj_np = trajectories.numpy()

    for i in range(min(args.batch_size, 15)):
        axes[0].plot(
            traj_np[:, i, 0], traj_np[:, i, 1],
            alpha=0.4, linewidth=0.8,
        )
    axes[0].set_title("Original System")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[0].set_aspect("equal")

    # Simulate estimated
    traj_est = simulate_sde(sde_estim, y0, ts, dt=args.dt)
    traj_est_np = traj_est.numpy()
    for i in range(min(args.batch_size, 15)):
        axes[1].plot(
            traj_est_np[:, i, 0], traj_est_np[:, i, 1],
            alpha=0.4, linewidth=0.8,
        )
    axes[1].set_title("NFPE Estimated System")
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "oscillator_phase_space.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Evolving Gaussian contours
    fig, ax = plt.subplots(figsize=(8, 8))
    theta_grid = np.linspace(0, 2 * np.pi, 100)

    for t_idx in range(0, args.t_size, max(1, args.t_size // 10)):
        mu = means[t_idx].detach().numpy()
        cov = covariances[t_idx].detach().numpy()

        # Eigendecomposition for ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)

        # 1-sigma ellipse
        ellipse = (
            mu[:, None]
            + eigvecs @ np.diag(np.sqrt(eigvals)) @ np.array(
                [np.cos(theta_grid), np.sin(theta_grid)],
            )
        )
        alpha = 0.3 + 0.7 * t_idx / args.t_size
        ax.plot(ellipse[0], ellipse[1], alpha=alpha, color="blue")
        ax.plot(mu[0], mu[1], "ko", markersize=3, alpha=alpha)

    ax.set_title("Evolving Gaussian Contours (1σ)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "oscillator_contours.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_arr = [h["epoch"] for h in history]
    ax.semilogy(epochs_arr, [h["means"] for h in history], label="Means loss")
    ax.semilogy(epochs_arr, [h["covs"] for h in history], label="Covs loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "oscillator_loss.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Save results
    results = {
        "f_linear_true": sde_true.f_linear.tolist(),
        "f_linear_learned": sde_estim.f_linear.data.tolist(),
        "g_bias_true": sde_true.g_bias.tolist(),
        "g_bias_learned": sde_estim.g_bias.data.tolist(),
        "f_error": f_err,
        "g_error": g_err,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "oscillator_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="2D Oscillator NFPE identification",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--t-size", type=int, default=128)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cov-weight", type=float, default=128.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=str, default="results/oscillator",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
