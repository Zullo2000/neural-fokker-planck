"""Multi-dimensional Ornstein-Uhlenbeck experiment.

Learns the drift matrix and diffusion of a multi-dimensional OU process
using an MLP-parameterized NFPE. Serves as a sanity check that the MLP
can recover linear dynamics, and tests scaling to higher dimensions.

True system: dX = -Theta @ X dt + Sigma dW

Key design choices:
- Multiple displaced initial conditions provide drift information at
  different state-space locations (critical for MLP to learn F(x) globally)
- Short time horizon captures the transient (where mean/cov dynamics are
  informative) rather than equilibrium (where signals vanish)
- Each IC cluster is treated as a "component" in the multi-component
  moment matching framework: means shape (T, K, d)
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

from nfpe import MLPSDE, LinearSDE, fit_gmm_to_snapshots, train_nfpe
from nfpe.models import SDE
from nfpe.training import forward_backward_loss

from torch.func import jacfwd


class OrnsteinUhlenbeckSDE(SDE):
    """Multi-dimensional OU process: dX = -Theta @ X dt + Sigma dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, theta: torch.Tensor, sigma: torch.Tensor):
        super().__init__()
        self.register_buffer("theta", theta)
        self.register_buffer("sigma_val", sigma)

    def f(self, t, y):
        return -torch.matmul(y, self.theta.T)

    def g(self, t, y):
        return self.sigma_val.unsqueeze(0).expand_as(y)


def generate_displaced_ics(n_ics, dim, displacement=2.0, seed=0):
    """Generate well-separated initial condition centers.

    Places ICs on a hypercube shell so they cover different directions
    in state space — ensuring the MLP sees drift at varied locations.
    """
    rng = np.random.RandomState(seed)
    # Random directions, normalized and scaled
    centers = rng.randn(n_ics, dim)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / norms * displacement
    return torch.tensor(centers, dtype=torch.float32)


def simulate_multi_ic(sde, ic_centers, batch_per_ic, ts, dt, ic_spread=0.1):
    """Simulate from multiple IC clusters, return per-cluster moments.

    Returns:
        all_means: (T, K, d) — mean trajectory for each IC cluster
        all_covs: (T, K, d, d) — covariance trajectory for each IC cluster
    """
    n_ics = ic_centers.shape[0]
    dim = ic_centers.shape[1]
    T = len(ts)

    all_means = torch.zeros(T, n_ics, dim)
    all_covs = torch.zeros(T, n_ics, dim, dim)

    for k in range(n_ics):
        # Tight cluster around each IC center
        y0 = ic_centers[k].unsqueeze(0) + torch.randn(batch_per_ic, dim) * ic_spread

        with torch.no_grad():
            traj = torchsde.sdeint(sde, y0, ts, dt=dt, method="euler")

        # Compute empirical moments at each time step
        for t_idx in range(T):
            snapshot = traj[t_idx]  # (batch_per_ic, dim)
            all_means[t_idx, k] = snapshot.mean(dim=0)
            centered = snapshot - all_means[t_idx, k].unsqueeze(0)
            all_covs[t_idx, k] = (centered.T @ centered) / (batch_per_ic - 1)

    return all_means, all_covs


def train_nfpe_multi_ic(
    model, means, covariances, dt, epochs=3000, lr=1e-3,
    cov_weight=100.0, log_interval=200, grad_clip=1.0,
    lr_schedule=True, weight_decay=0.0, use_jacobian=True,
    verbose=True,
):
    """Train NFPE with gradient clipping and optional LR scheduling.

    Args:
        means: (T, K, d) — multi-IC moment data
        covariances: (T, K, d, d)
        grad_clip: Max gradient norm (0 to disable)
        lr_schedule: If True, use cosine annealing
        weight_decay: L2 regularization (prevents overfitting moment noise)
        use_jacobian: If True, compute Jacobian via AD (required for nonlinear).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

    history = []
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

        record = {
            "epoch": epoch,
            "total": total_loss.item(),
            "means": loss_means.item(),
            "covs": loss_covs.item(),
        }
        history.append(record)

        if verbose and epoch % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:5d} | "
                f"loss_means: {loss_means.item():.6f} | "
                f"loss_covs: {loss_covs.item():.6f} | "
                f"lr: {current_lr:.6f}"
            )

    return history


def evaluate_drift(sde_mlp, theta_true, dim, n_eval=50, eval_radius=2.0):
    """Evaluate learned drift vs true drift at random points in state space.

    Returns dict with Jacobian error at origin and drift MSE over eval points.
    """
    # Jacobian at origin
    x_origin = torch.zeros(1, dim)

    def f_eval(x):
        return sde_mlp.drift_net(x)

    jac_mlp = jacfwd(f_eval)(x_origin).squeeze()
    jac_err = torch.norm(jac_mlp - (-theta_true)).item()

    # Drift MSE over random evaluation points
    torch.manual_seed(999)
    x_eval = torch.randn(n_eval, dim) * eval_radius
    with torch.no_grad():
        f_learned = sde_mlp.drift_net(x_eval)
    f_true = -x_eval @ theta_true.T
    drift_mse = ((f_learned - f_true) ** 2).mean().item()
    drift_rel_err = (
        torch.norm(f_learned - f_true) / torch.norm(f_true)
    ).item()

    return {
        "jac_mlp": jac_mlp,
        "jac_error": jac_err,
        "drift_mse": drift_mse,
        "drift_relative_error": drift_rel_err,
    }


def run_experiment(args):
    """Run the multi-D OU identification experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim = args.dim
    ts = torch.linspace(0, args.t_end, args.t_size)

    # Ground-truth parameters
    # Theta: positive definite (stable OU)
    torch.manual_seed(args.seed + 1)
    A = torch.randn(dim, dim) * 0.3
    theta_true = torch.matmul(A.T, A) + 0.5 * torch.eye(dim)

    eigvals = torch.linalg.eigvalsh(theta_true)
    print(f"Theta eigenvalues: {eigvals.numpy()}")
    print(f"Relaxation timescales: {(1.0 / eigvals).numpy()}")

    # Sigma: constant diagonal diffusion
    sigma_true = torch.ones(dim) * args.sigma

    sde_true = OrnsteinUhlenbeckSDE(theta_true, sigma_true)

    # --- Generate multi-IC data ---
    ic_centers = generate_displaced_ics(
        args.n_ics, dim, displacement=args.displacement, seed=args.seed,
    )
    print(f"\nSimulating {dim}D OU from {args.n_ics} displaced ICs "
          f"(displacement={args.displacement})...")
    for k in range(args.n_ics):
        print(f"  IC {k}: {ic_centers[k].numpy()}")

    means, covariances = simulate_multi_ic(
        sde_true, ic_centers, args.batch_size, ts, args.dt,
        ic_spread=args.ic_spread,
    )
    # means: (T, K, d), covariances: (T, K, d, d)

    dt_obs = (ts[1] - ts[0]).item()
    print(f"Moment data shape: means {list(means.shape)}, "
          f"covs {list(covariances.shape)}")
    print(f"dt_obs = {dt_obs:.4f}")

    # --- Train MLP NFPE ---
    print(f"\nTraining MLP-NFPE (dim={dim}, hidden={args.hidden_sizes})...")
    sde_mlp = MLPSDE(
        state_size=dim,
        hidden_sizes=args.hidden_sizes,
        brownian_size=dim,
    )

    history = train_nfpe_multi_ic(
        sde_mlp, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
    )

    # --- Also train a linear NFPE for comparison ---
    print(f"\nTraining Linear NFPE (dim={dim})...")

    # Note: LinearSDE only supports brownian_size=1, so it can only
    # represent rank-1 diffusion. Use lower cov_weight to compensate.
    sde_linear = LinearSDE(
        f_linear=torch.randn(dim, dim) * 0.1,
        f_bias=torch.randn(dim) * 0.1,
        g_linear=torch.zeros(dim, dim),
        g_bias=torch.randn(dim) * 0.1,
        learnable=True,
    )

    history_linear = train_nfpe_multi_ic(
        sde_linear, means, covariances, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=10.0,  # Lower: rank-1 diffusion can't match true G
        log_interval=args.log_interval,
        grad_clip=0,
        weight_decay=0,
        use_jacobian=False,
    )

    # --- Evaluate ---
    os.makedirs(args.output_dir, exist_ok=True)
    print("\nEvaluating learned dynamics...")

    eval_results = evaluate_drift(
        sde_mlp, theta_true, dim,
        eval_radius=args.displacement,
    )

    print(f"\nTrue -Theta:\n{-theta_true}")
    print(f"\nMLP Jacobian at origin:\n{eval_results['jac_mlp']}")
    print(f"\nJacobian error (Frobenius): {eval_results['jac_error']:.4f}")
    print(f"Drift MSE (over state space):  {eval_results['drift_mse']:.6f}")
    print(f"Drift relative error:          {eval_results['drift_relative_error']:.4f}")

    if hasattr(sde_linear, "f_linear"):
        linear_err = torch.norm(sde_linear.f_linear.data - (-theta_true)).item()
        print(f"Linear model f_linear error:   {linear_err:.4f}")

    # --- Plots ---
    # Plot 1: Training loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    epochs_arr = [h["epoch"] for h in history]
    axes[0].semilogy(epochs_arr, [h["total"] for h in history], "b-", label="MLP-NFPE")
    epochs_lin = [h["epoch"] for h in history_linear]
    axes[0].semilogy(epochs_lin, [h["total"] for h in history_linear], "r--", label="Linear NFPE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title(f"{dim}D OU: Training Loss")
    axes[0].legend()

    # Loss breakdown for MLP
    axes[1].semilogy(epochs_arr, [h["means"] for h in history], "b-", label="Mean loss")
    axes[1].semilogy(epochs_arr, [h["covs"] for h in history], "r-", label="Cov loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("MLP-NFPE Loss Breakdown")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "ou_loss.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Mean trajectories per IC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_np = ts.numpy()
    means_np = means.detach().numpy()

    colors = plt.cm.tab10(np.linspace(0, 1, args.n_ics))
    for k in range(args.n_ics):
        for d_idx in range(min(dim, 3)):
            linestyle = ["-", "--", ":"][d_idx]
            axes[0].plot(
                t_np, means_np[:, k, d_idx],
                color=colors[k], linestyle=linestyle, alpha=0.8,
                label=f"IC{k} x_{d_idx+1}" if d_idx == 0 else None,
            )
    axes[0].set_title("Observed Means (per IC)")
    axes[0].set_xlabel("Time")
    axes[0].legend(fontsize=7)

    # Variances
    cov_diag = torch.diagonal(covariances, dim1=-2, dim2=-1).detach().numpy()
    for k in range(args.n_ics):
        for d_idx in range(min(dim, 3)):
            linestyle = ["-", "--", ":"][d_idx]
            axes[1].plot(
                t_np, cov_diag[:, k, d_idx],
                color=colors[k], linestyle=linestyle, alpha=0.8,
            )
    axes[1].set_title("Observed Variances (per IC)")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "ou_moments.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Learned vs true drift field (first 2 dims)
    if dim >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Create grid in first 2 dims
        grid_range = args.displacement * 1.2
        x1 = np.linspace(-grid_range, grid_range, 12)
        x2 = np.linspace(-grid_range, grid_range, 12)
        X1, X2 = np.meshgrid(x1, x2)
        grid_pts = np.stack([X1.ravel(), X2.ravel()], axis=1)

        # Pad remaining dims with zeros
        full_pts = np.zeros((grid_pts.shape[0], dim))
        full_pts[:, :2] = grid_pts
        x_grid = torch.tensor(full_pts, dtype=torch.float32)

        # True drift
        f_true = (-x_grid @ theta_true.T).detach().numpy()
        axes[0].quiver(
            grid_pts[:, 0], grid_pts[:, 1],
            f_true[:, 0], f_true[:, 1],
            color="blue", alpha=0.7,
        )
        axes[0].set_title("True Drift Field")
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")
        axes[0].set_aspect("equal")

        # Learned drift
        with torch.no_grad():
            f_learned = sde_mlp.drift_net(x_grid).numpy()
        axes[1].quiver(
            grid_pts[:, 0], grid_pts[:, 1],
            f_learned[:, 0], f_learned[:, 1],
            color="red", alpha=0.7,
        )
        axes[1].set_title("MLP-NFPE Learned Drift")
        axes[1].set_xlabel("x₁")
        axes[1].set_ylabel("x₂")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        fig.savefig(
            os.path.join(args.output_dir, "ou_drift_field.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # Plot 4: Phase space — true vs learned trajectories
    if dim >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Simulate true trajectories from IC centers
        y0_eval = ic_centers
        with torch.no_grad():
            traj_true = torchsde.sdeint(
                sde_true, y0_eval, ts, dt=args.dt, method="euler",
            )

        for k in range(args.n_ics):
            axes[0].plot(
                traj_true[:, k, 0].numpy(), traj_true[:, k, 1].numpy(),
                color=colors[k], alpha=0.8, linewidth=1.5,
            )
            axes[0].plot(
                ic_centers[k, 0], ic_centers[k, 1],
                "o", color=colors[k], markersize=8,
            )
        axes[0].set_title("True Trajectories (dims 1-2)")
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")

        # Simulate from MLP model
        class LearnedOUSDE(SDE):
            noise_type = "diagonal"
            sde_type = "ito"

            def __init__(self, mlp_sde, state_dim):
                super().__init__()
                self.mlp = mlp_sde
                self.state_dim = state_dim

            def f(self, t, y):
                return self.mlp.drift_net(y)

            def g(self, t, y):
                batch = y.shape[0]
                raw = self.mlp.diff_net(y)
                B = raw.view(batch, self.state_dim, self.state_dim)
                return torch.diagonal(B, dim1=-2, dim2=-1)

        sde_wrap = LearnedOUSDE(sde_mlp, dim)
        with torch.no_grad():
            traj_est = torchsde.sdeint(
                sde_wrap, y0_eval, ts, dt=args.dt, method="euler",
            )

        for k in range(args.n_ics):
            axes[1].plot(
                traj_est[:, k, 0].numpy(), traj_est[:, k, 1].numpy(),
                color=colors[k], alpha=0.8, linewidth=1.5,
            )
            axes[1].plot(
                ic_centers[k, 0], ic_centers[k, 1],
                "o", color=colors[k], markersize=8,
            )
        axes[1].set_title("MLP-NFPE Trajectories (dims 1-2)")
        axes[1].set_xlabel("x₁")
        axes[1].set_ylabel("x₂")

        plt.tight_layout()
        fig.savefig(
            os.path.join(args.output_dir, "ou_phase_space.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # Save results
    results = {
        "dim": dim,
        "n_ics": args.n_ics,
        "displacement": args.displacement,
        "theta_true": theta_true.tolist(),
        "jac_mlp_at_origin": eval_results["jac_mlp"].detach().tolist(),
        "jac_error": eval_results["jac_error"],
        "drift_mse": eval_results["drift_mse"],
        "drift_relative_error": eval_results["drift_relative_error"],
        "final_loss": history[-1]["total"],
        "config": vars(args),
    }
    if hasattr(sde_linear, "f_linear"):
        results["linear_f_error"] = torch.norm(
            sde_linear.f_linear.data - (-theta_true)
        ).item()

    with open(os.path.join(args.output_dir, "ou_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-D Ornstein-Uhlenbeck NFPE experiment",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--n-ics", type=int, default=8,
                        help="Number of displaced initial conditions")
    parser.add_argument("--displacement", type=float, default=2.0,
                        help="Distance of IC centers from origin")
    parser.add_argument("--ic-spread", type=float, default=0.1,
                        help="Spread of particles around each IC center")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Particles per IC cluster")
    parser.add_argument("--t-size", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 64],
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cov-weight", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=300)
    parser.add_argument(
        "--output-dir", type=str, default="results/ou_3d",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
