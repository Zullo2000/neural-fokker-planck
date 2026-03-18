"""Nonlinear SDE identification experiments.

Learns nonlinear drift and diffusion using an MLP-parameterized NFPE.
This is the key Stage 2 result: NFPEs can learn nonlinear, non-parametric
drift and diffusion from data.

Available systems:
- cubic: dX = -X^3 dt + sigma dW  (single stable point, nonlinear damping)
- double_well: dX = (X - X^3) dt + sigma dW  (two stable points)
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchsde
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import MLPSDE, fit_gmm_to_snapshots, train_nfpe
from nfpe.models import SDE


class CubicDampingSDE(SDE):
    """Cubic damping SDE: dX = -X^3 dt + sigma*dW.

    Single stable equilibrium at x=0 with nonlinear restoring force.
    Good test case for NFPE: unimodal distribution with nonlinear drift.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma

    def f(self, t, y):
        return -y ** 3

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


class DoubleWellSDE(SDE):
    """Ground-truth double-well SDE: dX = (X - X^3)dt + sigma*dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma

    def f(self, t, y):
        return y - y ** 3

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


def run_experiment(args):
    """Run the double-well nonlinear identification experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Ground truth ---
    if args.system == "cubic":
        sde_true = CubicDampingSDE(sigma=args.sigma)
        true_drift_fn = lambda x: -x ** 3
        drift_label = "True F(x) = -x³"
    else:
        sde_true = DoubleWellSDE(sigma=args.sigma)
        true_drift_fn = lambda x: x - x ** 3
        drift_label = "True F(x) = x - x³"

    ts = torch.linspace(0, args.t_end, args.t_size)

    # Initial conditions: start far from equilibrium for observable dynamics
    # Concentrate ICs so the mean trajectory sweeps through the nonlinear region
    y0 = torch.randn(args.batch_size, 1) * 0.1 + 1.5

    # --- Simulate ---
    print("Simulating double-well SDE trajectories...")
    with torch.no_grad():
        trajectories = torchsde.sdeint(
            sde_true, y0, ts, dt=args.dt, method="euler",
        )

    # --- Fit GMMs ---
    print(f"Fitting Gaussian Mixtures (K={args.n_components})...")
    gmm_models, means, covariances = fit_gmm_to_snapshots(
        trajectories, n_components=args.n_components,
    )

    # Ensure correct dimensions for 1D state
    if args.n_components == 1:
        if means.dim() == 1:
            means = means.unsqueeze(-1)
        if covariances.dim() == 1:
            covariances = covariances.unsqueeze(-1).unsqueeze(-1)
        elif covariances.dim() == 2:
            covariances = covariances.unsqueeze(-1)
    else:
        # Multi-component: (T, K, d) and (T, K, d, d)
        if means.dim() == 2 and covariances.dim() == 2:
            # 1D state: means (T, K), covs (T, K) -> add state dims
            means = means.unsqueeze(-1)
            covariances = covariances.unsqueeze(-1).unsqueeze(-1)

    dt_obs = (ts[1] - ts[0]).item()

    # --- Train NFPE with MLP ---
    print(f"\nTraining MLP-NFPE (hidden={args.hidden_sizes})...")
    sde_mlp = MLPSDE(
        state_size=1,
        hidden_sizes=args.hidden_sizes,
        brownian_size=1,
    )

    history = train_nfpe(
        sde_mlp, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        use_jacobian=True,
        log_interval=args.log_interval,
    )

    # --- Evaluate learned drift ---
    print("\nEvaluating learned drift function...")
    # Evaluate over the range covered by observations
    x_min = means.min().item() - 0.3
    x_max = means.max().item() + 0.3
    x_eval = torch.linspace(x_min, x_max, 200).unsqueeze(-1)
    x_eval_full = torch.linspace(-2.0, 2.0, 200).unsqueeze(-1)

    with torch.no_grad():
        f_true = true_drift_fn(x_eval)
        f_learned = sde_mlp.drift_net(x_eval)
        f_true_full = true_drift_fn(x_eval_full)
        f_learned_full = sde_mlp.drift_net(x_eval_full)

    # Evaluate learned diffusion
    with torch.no_grad():
        b_learned = sde_mlp.diff_net(x_eval)
        g_learned = b_learned ** 2  # G = BB^T, 1D so just b^2

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: Learned vs true drift
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_eval_full.numpy(), f_true_full.numpy(), "b-", linewidth=2, label=drift_label)
    axes[0].plot(x_eval_full.numpy(), f_learned_full.numpy(), "r--", linewidth=2, label="Learned F(x)")
    # Shade the observed region
    axes[0].axvspan(x_min, x_max, alpha=0.1, color="green", label="Observed range")
    axes[0].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("F(x)")
    axes[0].set_title("Drift Function")
    axes[0].legend()

    # Plot diffusion
    axes[1].axhline(
        y=args.sigma ** 2, color="b", linewidth=2, label=f"True G = {args.sigma**2:.2f}",
    )
    axes[1].plot(x_eval.numpy(), g_learned.numpy(), "r--", linewidth=2, label="Learned G(x)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("G(x)")
    axes[1].set_title("Diffusion Coefficient G = BB^T")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "double_well_functions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    traj_np = trajectories.numpy()
    for i in range(min(args.batch_size, 30)):
        axes[0].plot(ts.numpy(), traj_np[:, i, 0], alpha=0.3, linewidth=0.5)
    axes[0].set_title("Original System")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("X(t)")

    # Simulate from learned model
    class LearnedSDE(SDE):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, mlp_sde):
            super().__init__()
            self.mlp = mlp_sde

        def f(self, t, y):
            return self.mlp.drift_net(y)

        def g(self, t, y):
            return self.mlp.diff_net(y)

    sde_learned_wrapper = LearnedSDE(sde_mlp)
    with torch.no_grad():
        traj_est = torchsde.sdeint(
            sde_learned_wrapper, y0, ts, dt=args.dt, method="euler",
        )
    traj_est_np = traj_est.numpy()
    for i in range(min(args.batch_size, 30)):
        axes[1].plot(ts.numpy(), traj_est_np[:, i, 0], alpha=0.3, linewidth=0.5)
    axes[1].set_title("NFPE Estimated System")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("X(t)")

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "double_well_trajectories.png"),
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
        os.path.join(args.output_dir, "double_well_loss.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Compute drift error in observed region only
    drift_mse = torch.mean((f_true - f_learned) ** 2).item()
    drift_mse_full = torch.mean((f_true_full - f_learned_full) ** 2).item()
    print(f"\nDrift MSE (observed range [{x_min:.1f}, {x_max:.1f}]): {drift_mse:.6f}")
    print(f"Drift MSE (full range [-2, 2]): {drift_mse_full:.6f}")
    print(f"Figures saved to {args.output_dir}/")

    # Save results
    results = {
        "drift_mse_observed": drift_mse,
        "drift_mse_full": drift_mse_full,
        "observed_range": [x_min, x_max],
        "sigma_true": args.sigma,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "double_well_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Double-well NFPE nonlinear experiment",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--t-size", type=int, default=50)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument(
        "--system", type=str, default="cubic",
        choices=["cubic", "double_well"],
        help="Which nonlinear SDE to learn",
    )
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[32, 32],
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cov-weight", type=float, default=100.0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=str, default="results/cubic_damping",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
