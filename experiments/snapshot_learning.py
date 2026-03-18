"""Stage 3: Snapshot Learning — Learning SDEs from cross-sectional data.

Demonstrates that NFPE can learn SDE parameters equally well from
cross-sectional snapshots (no trajectory tracking) as from tracked
trajectories. This is the key novelty claim: NFPE only needs distribution
moments, not individual particle correspondences across time.

Three pipelines are compared:
  A) Tracked:     standard trajectory simulation + GMM fitting
  B) Shuffled:    same trajectories with particles permuted per time step
  C) Independent: fresh particles simulated independently at each snapshot

Pipeline B proves the mathematical point (moments are permutation-invariant).
Pipeline C proves the practical point (works with truly independent samples).

Available systems:
  --system oscillator: 2D harmonic oscillator (linear, parametric)
  --system cubic:      1D cubic damping (nonlinear, MLP)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchsde
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import (
    LinearSDE,
    MLPSDE,
    simulate_sde,
    fit_gmm_to_snapshots,
    shuffle_snapshots,
    simulate_independent_snapshots,
    train_nfpe,
)
from nfpe.models import SDE


# ---------------------------------------------------------------------------
# Ground-truth SDE definitions
# ---------------------------------------------------------------------------

class CubicDampingSDE(SDE):
    """Cubic damping SDE: dX = -X^3 dt + sigma*dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, sigma: float = 0.2):
        super().__init__()
        self.sigma = sigma

    def f(self, t, y):
        return -y ** 3

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


def create_oscillator_sde() -> LinearSDE:
    """Create the ground-truth 2D harmonic oscillator SDE."""
    return LinearSDE(
        f_linear=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        f_bias=torch.tensor([0.0, 0.0]),
        g_linear=torch.zeros(2, 2),
        g_bias=torch.tensor([0.0, 0.05]),
        learnable=False,
    )


# ---------------------------------------------------------------------------
# Learnable model factories
# ---------------------------------------------------------------------------

def create_learnable_oscillator(model_seed: int) -> LinearSDE:
    """Create a learnable 2D linear SDE with deterministic initialization."""
    torch.manual_seed(model_seed)
    return LinearSDE(
        f_linear=torch.randn(2, 2) * 0.1,
        f_bias=torch.randn(2) * 0.1,
        g_linear=torch.randn(2, 2) * 0.1,
        g_bias=torch.randn(2) * 0.1,
        learnable=True,
    )


def create_learnable_cubic(model_seed: int) -> MLPSDE:
    """Create a learnable 1D MLP SDE with deterministic initialization."""
    torch.manual_seed(model_seed)
    return MLPSDE(
        state_size=1,
        hidden_sizes=[32, 32],
        brownian_size=1,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fix_1d_shapes(means, covariances):
    """Ensure correct tensor shapes for 1D state with K=1."""
    if means.dim() == 1:
        means = means.unsqueeze(-1)
    if covariances.dim() == 1:
        covariances = covariances.unsqueeze(-1).unsqueeze(-1)
    elif covariances.dim() == 2:
        covariances = covariances.unsqueeze(-1)
    return means, covariances


def plot_loss_comparison(histories, output_dir, prefix="snapshot"):
    """Plot overlaid training loss curves for all pipelines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, history, color in histories:
        epochs_arr = [h["epoch"] for h in history]
        axes[0].semilogy(
            epochs_arr, [h["means"] for h in history],
            label=label, color=color, alpha=0.8,
        )
        axes[1].semilogy(
            epochs_arr, [h["covs"] for h in history],
            label=label, color=color, alpha=0.8,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Means Loss")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Covariance Loss")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{prefix}_loss_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Oscillator experiment
# ---------------------------------------------------------------------------

def run_oscillator(args):
    """Run snapshot learning on 2D harmonic oscillator."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sde_true = create_oscillator_sde()
    ts = torch.linspace(0, 2 * np.pi, args.t_size)
    y0_mean = torch.tensor([1.0, 0.0])
    y0 = torch.randn(args.batch_size, 2) * 0.5 + y0_mean
    dt_obs = (ts[1] - ts[0]).item()

    print("Simulating 2D oscillator trajectories...")
    trajectories = simulate_sde(sde_true, y0, ts, dt=args.dt)

    # --- Three pipelines ---
    _, means_tracked, covs_tracked = fit_gmm_to_snapshots(
        trajectories, n_components=1,
    )

    sde_tracked = create_learnable_oscillator(args.model_seed)
    print("\n--- Pipeline A (Tracked) ---")
    hist_tracked = train_nfpe(
        sde_tracked, means_tracked, covs_tracked, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=False, log_interval=args.log_interval,
    )

    shuffled = shuffle_snapshots(trajectories, seed=args.seed + 1)
    _, means_shuffled, covs_shuffled = fit_gmm_to_snapshots(
        shuffled, n_components=1,
    )
    moment_diff = torch.norm(means_tracked - means_shuffled).item()
    cov_diff = torch.norm(covs_tracked - covs_shuffled).item()
    print(f"\nMoment diff (tracked vs shuffled): {moment_diff:.2e}")
    print(f"Cov diff (tracked vs shuffled): {cov_diff:.2e}")

    sde_shuffled = create_learnable_oscillator(args.model_seed)
    print("\n--- Pipeline B (Shuffled) ---")
    hist_shuffled = train_nfpe(
        sde_shuffled, means_shuffled, covs_shuffled, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=False, log_interval=args.log_interval,
    )

    print("\nSimulating independent snapshots...")
    def y0_sampler(n):
        return torch.randn(n, 2) * 0.5 + y0_mean

    independent = simulate_independent_snapshots(
        sde_true, y0_sampler, ts,
        n_particles=args.batch_size, dt=args.dt,
    )
    _, means_indep, covs_indep = fit_gmm_to_snapshots(
        independent, n_components=1,
    )
    indep_moment_diff = torch.norm(means_tracked - means_indep).item()
    print(f"Moment diff (tracked vs independent): {indep_moment_diff:.2e}")

    sde_indep = create_learnable_oscillator(args.model_seed)
    print("\n--- Pipeline C (Independent) ---")
    hist_indep = train_nfpe(
        sde_indep, means_indep, covs_indep, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=False, log_interval=args.log_interval,
    )

    # --- Errors ---
    def param_errors(sde_l):
        f_err = torch.norm(sde_l.f_linear.data - sde_true.f_linear).item()
        g_err = torch.norm(sde_l.g_bias.data - sde_true.g_bias).item()
        return {"f_error": f_err, "g_error": g_err}

    err_t, err_s, err_i = param_errors(sde_tracked), param_errors(sde_shuffled), param_errors(sde_indep)

    print("\n" + "=" * 70)
    print("Results Comparison (Oscillator)")
    print("=" * 70)
    print(f"{'Pipeline':<20} {'f_linear err':>14} {'g_bias err':>14}")
    print("-" * 50)
    for name, err in [("A (Tracked)", err_t), ("B (Shuffled)", err_s), ("C (Independent)", err_i)]:
        print(f"{name:<20} {err['f_error']:>14.6f} {err['g_error']:>14.6f}")

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    plot_loss_comparison([
        ("Tracked", hist_tracked, "blue"),
        ("Shuffled", hist_shuffled, "orange"),
        ("Independent", hist_indep, "green"),
    ], args.output_dir)

    # Parameter bar chart
    true_params = [
        sde_true.f_linear[0, 0].item(), sde_true.f_linear[0, 1].item(),
        sde_true.f_linear[1, 0].item(), sde_true.f_linear[1, 1].item(),
        sde_true.g_bias[0].item(), sde_true.g_bias[1].item(),
    ]
    labels = ["A[0,0]", "A[0,1]", "A[1,0]", "A[1,1]", "g[0]", "g[1]"]

    def get_params(sde):
        return [
            sde.f_linear.data[0, 0].item(), sde.f_linear.data[0, 1].item(),
            sde.f_linear.data[1, 0].item(), sde.f_linear.data[1, 1].item(),
            sde.g_bias.data[0].item(), sde.g_bias.data[1].item(),
        ]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, true_params, width, label="True", color="black", alpha=0.7)
    ax.bar(x - 0.5 * width, get_params(sde_tracked), width, label="Tracked", color="blue", alpha=0.7)
    ax.bar(x + 0.5 * width, get_params(sde_shuffled), width, label="Shuffled", color="orange", alpha=0.7)
    ax.bar(x + 1.5 * width, get_params(sde_indep), width, label="Independent", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Recovery: Tracked vs Shuffled vs Independent")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "snapshot_parameter_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Phase space
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    y0_plot = torch.randn(15, 2) * 0.5 + y0_mean
    for ax, sde, title in [
        (axes[0], sde_tracked, "Tracked"),
        (axes[1], sde_shuffled, "Shuffled"),
        (axes[2], sde_indep, "Independent"),
    ]:
        traj = simulate_sde(sde, y0_plot, ts, dt=args.dt).numpy()
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.4, linewidth=0.8)
        ax.set_title(f"Learned from {title} Data")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "snapshot_phase_space.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Save results
    results = {
        "system": "oscillator",
        "tracked": {"f_linear": sde_tracked.f_linear.data.tolist(),
                     "g_bias": sde_tracked.g_bias.data.tolist(), **err_t},
        "shuffled": {"f_linear": sde_shuffled.f_linear.data.tolist(),
                      "g_bias": sde_shuffled.g_bias.data.tolist(), **err_s,
                      "moment_diff": moment_diff, "cov_diff": cov_diff},
        "independent": {"f_linear": sde_indep.f_linear.data.tolist(),
                         "g_bias": sde_indep.g_bias.data.tolist(), **err_i,
                         "moment_diff": indep_moment_diff},
        "true": {"f_linear": sde_true.f_linear.tolist(),
                  "g_bias": sde_true.g_bias.tolist()},
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "snapshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


# ---------------------------------------------------------------------------
# Cubic damping experiment
# ---------------------------------------------------------------------------

def run_cubic(args):
    """Run snapshot learning on 1D cubic damping with MLP."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sigma = args.sigma
    sde_true = CubicDampingSDE(sigma=sigma)
    ts = torch.linspace(0, args.t_end, args.t_size)
    y0_mean = torch.tensor([1.5])
    y0 = torch.randn(args.batch_size, 1) * 0.1 + 1.5
    dt_obs = (ts[1] - ts[0]).item()
    true_drift_fn = lambda x: -x ** 3

    print(f"Simulating cubic damping SDE (sigma={sigma})...")
    trajectories = simulate_sde(sde_true, y0, ts, dt=args.dt)

    # --- Pipeline A: Tracked ---
    _, means_tracked, covs_tracked = fit_gmm_to_snapshots(
        trajectories, n_components=1,
    )
    means_tracked, covs_tracked = fix_1d_shapes(means_tracked, covs_tracked)

    sde_tracked = create_learnable_cubic(args.model_seed)
    print("\n--- Pipeline A (Tracked) ---")
    hist_tracked = train_nfpe(
        sde_tracked, means_tracked, covs_tracked, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=True, log_interval=args.log_interval,
    )

    # --- Pipeline B: Shuffled ---
    shuffled = shuffle_snapshots(trajectories, seed=args.seed + 1)
    _, means_shuffled, covs_shuffled = fit_gmm_to_snapshots(
        shuffled, n_components=1,
    )
    means_shuffled, covs_shuffled = fix_1d_shapes(means_shuffled, covs_shuffled)

    moment_diff = torch.norm(means_tracked - means_shuffled).item()
    cov_diff = torch.norm(covs_tracked - covs_shuffled).item()
    print(f"\nMoment diff (tracked vs shuffled): {moment_diff:.2e}")
    print(f"Cov diff (tracked vs shuffled): {cov_diff:.2e}")

    sde_shuffled = create_learnable_cubic(args.model_seed)
    print("\n--- Pipeline B (Shuffled) ---")
    hist_shuffled = train_nfpe(
        sde_shuffled, means_shuffled, covs_shuffled, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=True, log_interval=args.log_interval,
    )

    # --- Pipeline C: Independent ---
    print("\nSimulating independent snapshots...")
    def y0_sampler(n):
        return torch.randn(n, 1) * 0.1 + 1.5

    independent = simulate_independent_snapshots(
        sde_true, y0_sampler, ts,
        n_particles=args.batch_size, dt=args.dt,
    )
    _, means_indep, covs_indep = fit_gmm_to_snapshots(
        independent, n_components=1,
    )
    means_indep, covs_indep = fix_1d_shapes(means_indep, covs_indep)

    indep_moment_diff = torch.norm(means_tracked - means_indep).item()
    print(f"Moment diff (tracked vs independent): {indep_moment_diff:.2e}")

    sde_indep = create_learnable_cubic(args.model_seed)
    print("\n--- Pipeline C (Independent) ---")
    hist_indep = train_nfpe(
        sde_indep, means_indep, covs_indep, dt_obs,
        epochs=args.epochs, lr=args.lr, cov_weight=args.cov_weight,
        use_jacobian=True, log_interval=args.log_interval,
    )

    # --- Evaluate drift MSE ---
    x_min = min(means_tracked.min().item(), means_indep.min().item()) - 0.3
    x_max = max(means_tracked.max().item(), means_indep.max().item()) + 0.3
    x_eval = torch.linspace(x_min, x_max, 200).unsqueeze(-1)

    with torch.no_grad():
        f_true = true_drift_fn(x_eval)
        f_tracked = sde_tracked.drift_net(x_eval)
        f_shuffled = sde_shuffled.drift_net(x_eval)
        f_indep = sde_indep.drift_net(x_eval)

    mse_tracked = torch.mean((f_true - f_tracked) ** 2).item()
    mse_shuffled = torch.mean((f_true - f_shuffled) ** 2).item()
    mse_indep = torch.mean((f_true - f_indep) ** 2).item()

    print("\n" + "=" * 70)
    print("Results Comparison (Cubic Damping)")
    print("=" * 70)
    print(f"{'Pipeline':<20} {'Drift MSE':>14}")
    print("-" * 36)
    print(f"{'A (Tracked)':<20} {mse_tracked:>14.6f}")
    print(f"{'B (Shuffled)':<20} {mse_shuffled:>14.6f}")
    print(f"{'C (Independent)':<20} {mse_indep:>14.6f}")

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    plot_loss_comparison([
        ("Tracked", hist_tracked, "blue"),
        ("Shuffled", hist_shuffled, "orange"),
        ("Independent", hist_indep, "green"),
    ], args.output_dir)

    # Drift function comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_eval.numpy(), f_true.numpy(), "k-", linewidth=2.5, label="True F(x) = -x^3")
    ax.plot(x_eval.numpy(), f_tracked.numpy(), "b--", linewidth=2, label=f"Tracked (MSE={mse_tracked:.4f})")
    ax.plot(x_eval.numpy(), f_shuffled.numpy(), color="orange", linestyle="--",
            linewidth=2, label=f"Shuffled (MSE={mse_shuffled:.4f})")
    ax.plot(x_eval.numpy(), f_indep.numpy(), "g--", linewidth=2, label=f"Independent (MSE={mse_indep:.4f})")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("Learned Drift: Tracked vs Shuffled vs Independent")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "snapshot_drift_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Trajectory comparison (3 panels)
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    y0_plot = torch.randn(30, 1) * 0.1 + 1.5
    for ax, sde_mlp, title in [
        (axes[0], sde_tracked, "Tracked"),
        (axes[1], sde_shuffled, "Shuffled"),
        (axes[2], sde_indep, "Independent"),
    ]:
        wrapper = LearnedSDE(sde_mlp)
        with torch.no_grad():
            traj = torchsde.sdeint(
                wrapper, y0_plot, ts, dt=args.dt, method="euler",
            ).numpy()
        for i in range(traj.shape[1]):
            ax.plot(ts.numpy(), traj[:, i, 0], alpha=0.3, linewidth=0.5)
        ax.set_title(f"Learned from {title} Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "snapshot_trajectories.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Save results
    results = {
        "system": "cubic",
        "tracked": {"drift_mse": mse_tracked},
        "shuffled": {"drift_mse": mse_shuffled,
                      "moment_diff": moment_diff, "cov_diff": cov_diff},
        "independent": {"drift_mse": mse_indep,
                         "moment_diff": indep_moment_diff},
        "true": {"drift": "F(x) = -x^3", "sigma": sigma},
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "snapshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Snapshot Learning comparison",
    )
    parser.add_argument(
        "--system", type=str, default="oscillator",
        choices=["oscillator", "cubic"],
        help="Which SDE system to test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--t-size", type=int, default=128)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cov-weight", type=float, default=128.0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/snapshot_{args.system}"

    if args.system == "oscillator":
        run_oscillator(args)
    else:
        run_cubic(args)


if __name__ == "__main__":
    main()
