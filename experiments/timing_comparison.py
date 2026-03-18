"""Wall-clock timing comparison: NFPE vs Neural SDE training.

Compares NFPE (moment-based, no SDE simulation during training) against a
standard Neural SDE baseline (trajectory-matching with torchsde backprop)
on the same system and architecture.

The Neural SDE baseline gets trajectory data (stronger signal); NFPE gets
snapshot data (weaker signal). Despite this asymmetry, NFPE is expected to
be dramatically faster per epoch.

Usage:
    python experiments/timing_comparison.py --dim 5 --epochs 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torchsde
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import MLPSDE
from nfpe.models import SDE
from benchmark_systems import (
    make_ou_system,
    generate_displaced_ics,
    simulate_snapshot_multi_ic,
    train_nfpe_multi_ic,
    evaluate_drift_error,
    evaluate_diffusion_error,
)


# ---------------------------------------------------------------------------
# Neural SDE baseline
# ---------------------------------------------------------------------------

class NeuralSDEWrapper(SDE):
    """Wraps MLPSDE for use with torchsde.sdeint (diagonal noise)."""

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
        # For diagonal noise, return diagonal of B
        return torch.diagonal(B, dim1=-2, dim2=-1)


def train_neural_sde(
    model,
    sde_true,
    y0_pool,
    ts_train,
    n_trajectories=32,
    epochs=500,
    lr=1e-3,
    dt=0.01,
    log_interval=50,
    verbose=True,
    device=None,
):
    """Train a Neural SDE via trajectory matching.

    Pre-generates ground-truth trajectories, then each epoch:
    1. Sample a mini-batch of trajectories
    2. Simulate the learned model forward (with gradients)
    3. MSE loss vs ground truth
    4. Backprop through the SDE solver

    Returns: (history, total_time_seconds)
    """
    if device is None:
        device = torch.device("cpu")

    state_dim = y0_pool.shape[1]
    wrapper = NeuralSDEWrapper(model, state_dim).to(device)
    ts_train = ts_train.to(device)

    # Pre-generate ground-truth trajectories
    print("  Pre-generating ground-truth trajectories...")
    n_pool = y0_pool.shape[0]
    y0_pool_dev = y0_pool.to(device)
    with torch.no_grad():
        traj_pool = torchsde.sdeint(
            sde_true, y0_pool_dev, ts_train, dt=dt, method="euler",
        )  # (T, N_pool, d)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample mini-batch
        idx = torch.randint(0, n_pool, (n_trajectories,))
        y0_batch = y0_pool_dev[idx]
        traj_true = traj_pool[:, idx, :]

        # Forward simulate with gradients (the expensive part)
        traj_pred = torchsde.sdeint(
            wrapper, y0_batch, ts_train, dt=dt, method="euler",
        )

        loss = torch.nn.functional.mse_loss(traj_pred, traj_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        elapsed = time.perf_counter() - start_time
        record = {
            "epoch": epoch,
            "loss": loss.item(),
            "wall_clock": elapsed,
        }
        history.append(record)

        if verbose and epoch % log_interval == 0:
            print(
                f"  Epoch {epoch:5d} | loss: {loss.item():.6f} | "
                f"time: {elapsed:.1f}s"
            )

    total_time = time.perf_counter() - start_time
    return history, total_time


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    """Run NFPE vs Neural SDE timing comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim = args.dim
    sigma = args.sigma

    # --- Ground truth ---
    sde_true, theta_true = make_ou_system(
        dim, sigma=sigma, seed=args.seed, device=device,
    )
    true_drift_fn = lambda x: -torch.matmul(x, theta_true.T.to(x.device))
    true_G = sigma ** 2

    ts_nfpe = torch.linspace(0, args.t_end, args.t_size)
    ts_nsde = torch.linspace(0, args.t_end, args.nsde_t_size)
    n_ics = max(args.n_ics, 2 * dim)

    # --- NFPE data (snapshots) ---
    print("\n--- NFPE: Generating snapshot data ---")
    ic_centers = generate_displaced_ics(
        n_ics, dim, displacement=args.displacement, seed=args.seed,
    )
    means, covs = simulate_snapshot_multi_ic(
        sde_true, ic_centers, args.batch_size, ts_nfpe, args.dt,
        ic_spread=args.ic_spread, device=device,
    )
    dt_obs = (ts_nfpe[1] - ts_nfpe[0]).item()

    # --- Neural SDE data (trajectories) ---
    print("\n--- Neural SDE: Generating trajectory pool ---")
    y0_pool = torch.randn(args.nsde_pool_size, dim) * args.displacement
    y0_pool = y0_pool.to(device)

    # --- Train NFPE ---
    print("\n--- Training NFPE ---")
    torch.manual_seed(args.model_seed)
    sde_nfpe = MLPSDE(
        state_size=dim, hidden_sizes=args.hidden_sizes, brownian_size=dim,
    )

    hist_nfpe, time_nfpe = train_nfpe_multi_ic(
        sde_nfpe, means, covs, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        grad_clip=1.0, weight_decay=1e-5,
        log_interval=args.log_interval,
        device=device,
    )

    # --- Train Neural SDE ---
    print("\n--- Training Neural SDE ---")
    torch.manual_seed(args.model_seed)
    sde_nsde = MLPSDE(
        state_size=dim, hidden_sizes=args.hidden_sizes, brownian_size=dim,
    )

    hist_nsde, time_nsde = train_neural_sde(
        sde_nsde, sde_true, y0_pool,
        ts_nsde,
        n_trajectories=args.nsde_batch,
        epochs=args.epochs, lr=args.lr,
        dt=args.nsde_dt,
        log_interval=args.log_interval,
        device=device,
    )

    # --- Evaluate both ---
    print("\n--- Evaluation ---")
    # Move models to CPU for evaluation
    sde_nfpe = sde_nfpe.cpu()
    sde_nsde = sde_nsde.cpu()
    eval_device = torch.device("cpu")

    nfpe_drift = evaluate_drift_error(
        sde_nfpe, true_drift_fn, dim,
        eval_radius=args.displacement, device=eval_device,
    )
    nsde_drift = evaluate_drift_error(
        sde_nsde, true_drift_fn, dim,
        eval_radius=args.displacement, device=eval_device,
    )
    nfpe_diff = evaluate_diffusion_error(
        sde_nfpe, true_G, dim,
        eval_radius=args.displacement, device=eval_device,
    )
    nsde_diff = evaluate_diffusion_error(
        sde_nsde, true_G, dim,
        eval_radius=args.displacement, device=eval_device,
    )

    # Timing stats
    nfpe_times = [hist_nfpe[i+1]["wall_clock"] - hist_nfpe[i]["wall_clock"]
                 for i in range(len(hist_nfpe)-1)]
    nsde_times = [hist_nsde[i+1]["wall_clock"] - hist_nsde[i]["wall_clock"]
                  for i in range(len(hist_nsde)-1)]
    nfpe_per_epoch = np.median(nfpe_times) if nfpe_times else 0
    nsde_per_epoch = np.median(nsde_times) if nsde_times else 0

    # --- Report ---
    print("\n" + "=" * 70)
    print(f"Timing Comparison (d={dim}, {args.epochs} epochs, device={device})")
    print("=" * 70)
    print(f"{'':>20} {'NFPE':>15} {'Neural SDE':>15} {'Speedup':>10}")
    print("-" * 62)
    print(f"{'Total time (s)':>20} {time_nfpe:>15.1f} {time_nsde:>15.1f} "
          f"{time_nsde/max(time_nfpe, 0.01):>10.1f}x")
    print(f"{'Per epoch (ms)':>20} {nfpe_per_epoch*1000:>15.1f} "
          f"{nsde_per_epoch*1000:>15.1f} "
          f"{nsde_per_epoch/max(nfpe_per_epoch, 1e-6):>10.1f}x")
    print(f"{'Drift MSE':>20} {nfpe_drift['drift_mse']:>15.6f} "
          f"{nsde_drift['drift_mse']:>15.6f}")
    print(f"{'Diffusion MSE':>20} {nfpe_diff['diffusion_mse']:>15.6f} "
          f"{nsde_diff['diffusion_mse']:>15.6f}")
    print(f"\nNote: Neural SDE uses trajectory data; NFPE uses snapshot data.")

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: Time per epoch bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["NFPE", "Neural SDE"],
        [nfpe_per_epoch * 1000, nsde_per_epoch * 1000],
        color=["blue", "red"], alpha=0.7,
    )
    ax.set_ylabel("Time per epoch (ms)")
    ax.set_title(f"Training Speed (d={dim}, {device})")
    for bar, val in zip(bars, [nfpe_per_epoch*1000, nsde_per_epoch*1000]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "timing_per_epoch.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Loss vs wall-clock time (key plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    nfpe_wc = [h["wall_clock"] for h in hist_nfpe]
    nfpe_loss = [h["total"] for h in hist_nfpe]
    nsde_wc = [h["wall_clock"] for h in hist_nsde]
    nsde_loss = [h["loss"] for h in hist_nsde]

    ax.semilogy(nfpe_wc, nfpe_loss, "b-", label="NFPE (snapshots)", alpha=0.8)
    ax.semilogy(nsde_wc, nsde_loss, "r-", label="Neural SDE (trajectories)",
                alpha=0.8)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence vs Time (d={dim})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "timing_loss_vs_wallclock.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 3: Accuracy comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric, nfpe_val, nsde_val, title in [
        (axes[0], "Drift MSE", nfpe_drift["drift_mse"],
         nsde_drift["drift_mse"], "Drift Recovery"),
        (axes[1], "Diffusion MSE", nfpe_diff["diffusion_mse"],
         nsde_diff["diffusion_mse"], "Diffusion Recovery"),
    ]:
        bars = ax.bar(
            ["NFPE", "Neural SDE"], [nfpe_val, nsde_val],
            color=["blue", "red"], alpha=0.7,
        )
        ax.set_ylabel(metric)
        ax.set_title(title)
        for bar, val in zip(bars, [nfpe_val, nsde_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "timing_accuracy_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # --- Save results ---
    results = {
        "dim": dim,
        "device": str(device),
        "epochs": args.epochs,
        "nfpe": {
            "total_time": time_nfpe,
            "per_epoch_ms": nfpe_per_epoch * 1000,
            "drift_mse": nfpe_drift["drift_mse"],
            "drift_relative_error": nfpe_drift["drift_relative_error"],
            "diffusion_mse": nfpe_diff["diffusion_mse"],
            "data_type": "snapshots",
        },
        "neural_sde": {
            "total_time": time_nsde,
            "per_epoch_ms": nsde_per_epoch * 1000,
            "drift_mse": nsde_drift["drift_mse"],
            "drift_relative_error": nsde_drift["drift_relative_error"],
            "diffusion_mse": nsde_diff["diffusion_mse"],
            "data_type": "trajectories",
        },
        "speedup": time_nsde / max(time_nfpe, 0.01),
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "timing_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="NFPE vs Neural SDE timing comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--n-ics", type=int, default=8)
    parser.add_argument("--displacement", type=float, default=2.0)
    parser.add_argument("--ic-spread", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--t-size", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[64, 64],
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cov-weight", type=float, default=100.0)
    parser.add_argument("--log-interval", type=int, default=100)
    # Neural SDE specific
    parser.add_argument("--nsde-batch", type=int, default=32)
    parser.add_argument("--nsde-pool-size", type=int, default=256)
    parser.add_argument("--nsde-dt", type=float, default=0.01)
    parser.add_argument("--nsde-t-size", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/timing")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
