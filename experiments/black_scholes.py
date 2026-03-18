"""Black-Scholes / Geometric Brownian Motion experiment.

Reproduces Table 1 from the NLDL abstract: learns drift and diffusion
coefficients of a 1D GBM from simulated trajectories using NFPE with
forward-backward moment matching.

Reference parameters: f0=0, f1=2.5, b0=0, b1=0.4
RESS baseline [Iannacone & Gardoni 2024]: f1=2.420, b1=0.361
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


def create_true_sde(
    f1: float = 2.5, f0: float = 0.0,
    b1: float = 0.4, b0: float = 0.0,
) -> LinearSDE:
    """Create the ground-truth Black-Scholes SDE."""
    return LinearSDE(
        f_linear=torch.tensor([[f1]]),
        f_bias=torch.tensor([f0]),
        g_linear=torch.tensor([[b1]]),
        g_bias=torch.tensor([b0]),
        learnable=False,
    )


def create_learnable_sde(state_size: int = 1) -> LinearSDE:
    """Create a learnable affine SDE with random initialization."""
    return LinearSDE(
        f_linear=torch.randn(state_size, state_size) * 0.1,
        f_bias=torch.randn(state_size) * 0.1,
        g_linear=torch.randn(state_size, state_size) * 0.1,
        g_bias=torch.randn(state_size) * 0.1,
        learnable=True,
    )


def run_experiment(args):
    """Run the Black-Scholes identification experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Ground truth SDE ---
    sde_true = create_true_sde(f1=args.f1, b1=args.b1)
    ts = torch.linspace(0, args.t_end, args.t_size)
    y0 = torch.ones(args.batch_size, 1) * 0.1  # Initial price

    # --- Simulate trajectories ---
    print("Simulating SDE trajectories...")
    trajectories = simulate_sde(sde_true, y0, ts, dt=args.dt)

    # --- Fit GMMs ---
    print("Fitting Gaussian Mixtures...")
    gmm_models, means, covariances = fit_gmm_to_snapshots(
        trajectories, n_components=1,
    )

    # Ensure covariances are 2D matrices for 1D state
    if covariances.dim() == 1:
        covariances = covariances.unsqueeze(-1).unsqueeze(-1)
    elif covariances.dim() == 2:
        covariances = covariances.unsqueeze(-1)

    if means.dim() == 1:
        means = means.unsqueeze(-1)

    dt_obs = (ts[1] - ts[0]).item()

    # --- Train NFPE (forward-backward) ---
    print("\n--- Training NFPE (forward-backward) ---")
    sde_nfpe = create_learnable_sde()
    history_nfpe = train_nfpe(
        sde_nfpe, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        use_jacobian=False,
        log_interval=args.log_interval,
    )

    # --- Train NFPE forward-only (ablation) ---
    print("\n--- Training NFPE (forward-only) ---")
    sde_fwd = create_learnable_sde()
    history_fwd = train_nfpe_forward_only(
        sde_fwd, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
    )

    # --- Report results ---
    print("\n" + "=" * 60)
    print("RESULTS: Parameter Recovery")
    print("=" * 60)

    true_params = {"f0": 0.0, "f1": args.f1, "b0": 0.0, "b1": args.b1}
    ress_params = {"f0": 0.0, "f1": 2.420, "b0": 0.0, "b1": 0.361}

    nfpe_params = {
        "f0": sde_nfpe.f_bias.item(),
        "f1": sde_nfpe.f_linear.item(),
        "b0": sde_nfpe.g_bias.item(),
        "b1": sde_nfpe.g_linear.item(),
    }
    fwd_params = {
        "f0": sde_fwd.f_bias.item(),
        "f1": sde_fwd.f_linear.item(),
        "b0": sde_fwd.g_bias.item(),
        "b1": sde_fwd.g_linear.item(),
    }

    header = f"{'Method':<12} {'f0':>8} {'f1':>8} {'b0':>8} {'b1':>8}"
    print(header)
    print("-" * len(header))

    for name, params in [
        ("True", true_params),
        ("RESS [5]", ress_params),
        ("NFPE-fwd", fwd_params),
        ("NFPE", nfpe_params),
    ]:
        print(
            f"{name:<12} "
            f"{params['f0']:>8.3f} {params['f1']:>8.3f} "
            f"{params['b0']:>8.3f} {params['b1']:>8.3f}"
        )

    # Parameter errors (Table 1 format)
    print("\n" + "=" * 60)
    print("Parameter Errors (|estimated - true|)")
    print("=" * 60)
    print(header)
    print("-" * len(header))

    for name, params in [
        ("RESS [5]", ress_params),
        ("NFPE-fwd", fwd_params),
        ("NFPE", nfpe_params),
    ]:
        errs = {k: abs(params[k] - true_params[k]) for k in true_params}
        print(
            f"{name:<12} "
            f"{errs['f0']:>8.3f} {errs['f1']:>8.3f} "
            f"{errs['b0']:>8.3f} {errs['b1']:>8.3f}"
        )

    # --- Generate plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: Trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    traj_np = trajectories.numpy()
    for i in range(min(args.batch_size, 20)):
        axes[0].plot(ts.numpy(), traj_np[:, i, 0], alpha=0.5, linewidth=0.8)
    axes[0].set_title("Original System")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("X(t)")

    # Simulate estimated system
    y0_est = torch.ones(args.batch_size, 1) * 0.1
    traj_est = simulate_sde(sde_nfpe, y0_est, ts, dt=args.dt)
    traj_est_np = traj_est.numpy()
    for i in range(min(args.batch_size, 20)):
        axes[1].plot(ts.numpy(), traj_est_np[:, i, 0], alpha=0.5, linewidth=0.8)
    axes[1].set_title("NFPE Estimated System")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("X(t)")

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "black_scholes_trajectories.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Moments over time
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ts.numpy(), means.detach().numpy(), "b-", label="Observed")
    axes[0].set_title("Means over Time")
    axes[0].set_xlabel("Time")
    axes[0].legend()

    cov_norms = torch.norm(covariances.flatten(start_dim=-2), dim=-1)
    axes[1].plot(ts.numpy(), cov_norms.detach().numpy(), "r-", label="Observed")
    axes[1].set_title("Covariance Norm over Time")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "black_scholes_moments.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Training loss curves
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_arr = [h["epoch"] for h in history_nfpe]
    ax.semilogy(epochs_arr, [h["means"] for h in history_nfpe], label="NFPE means")
    ax.semilogy(epochs_arr, [h["covs"] for h in history_nfpe], label="NFPE covs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "black_scholes_loss.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Save results
    results = {
        "true": true_params,
        "ress": ress_params,
        "nfpe": nfpe_params,
        "nfpe_fwd": fwd_params,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "black_scholes_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


def train_nfpe_forward_only(
    model, means, covariances, dt,
    epochs=1000, lr=1e-2, cov_weight=100.0,
):
    """Train using only forward differences (ablation baseline)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        mu_dot = model.f(torch.tensor(0.0), means)

        # Covariance dynamics (linear model)
        drift_term = (
            torch.matmul(model.f_linear, covariances)
            + torch.matmul(covariances, model.f_linear.transpose(-1, -2))
        )
        b = (
            torch.matmul(model.g_linear, means.unsqueeze(-1))
            + model.g_bias.unsqueeze(-1)
        )
        diffusion_term = torch.matmul(b, b.transpose(-1, -2))

        means_forward = means[:-1] + dt * mu_dot[:-1]
        covs_forward = covariances[:-1] + dt * (drift_term[:-1] + diffusion_term[:-1])

        loss_means = mse(means_forward, means[1:])
        loss_covs = mse(covs_forward, covariances[1:])
        loss = loss_means + cov_weight * loss_covs

        loss.backward()
        optimizer.step()

        history.append({
            "epoch": epoch,
            "total": loss.item(),
            "means": loss_means.item(),
            "covs": loss_covs.item(),
        })

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"loss_means: {loss_means.item():.6f} | "
                f"loss_covs: {loss_covs.item():.6f}"
            )

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes NFPE experiment",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--t-size", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--f1", type=float, default=2.5)
    parser.add_argument("--b1", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cov-weight", type=float, default=100.0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=str, default="results/black_scholes",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
