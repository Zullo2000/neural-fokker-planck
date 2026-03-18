"""PFI/UPFI accuracy benchmark — NFPE on multi-D OU and bistable systems.

Benchmarks NFPE on the same systems used by PFI (Zhang et al., NeurIPS 2025)
across dimensions d=2,5,10. Unlike PFI, NFPE jointly learns BOTH drift AND
diffusion from distribution snapshots.

Usage:
    python experiments/pfi_benchmark.py --system ou --dims 2 5 10
    python experiments/pfi_benchmark.py --system bistable --dims 2 5 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nfpe import MLPSDE
from benchmark_systems import (
    make_ou_system,
    make_bistable_system,
    generate_displaced_ics,
    generate_bistable_ics,
    simulate_snapshot_multi_ic,
    train_nfpe_multi_ic,
    evaluate_drift_error,
    evaluate_diffusion_error,
)
from torch.func import jacfwd


def run_single_dim(dim, args, device):
    """Run benchmark for a single dimension. Returns results dict."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sigma = args.sigma
    ts = torch.linspace(0, args.t_end, args.t_size)
    dt_obs = (ts[1] - ts[0]).item()
    n_ics = max(args.n_ics, 2 * dim)

    # --- Ground truth ---
    if args.system == "ou":
        sde_true, theta_true = make_ou_system(
            dim, sigma=sigma, seed=args.seed, device=device,
        )
        true_drift_fn = lambda x: -torch.matmul(x, theta_true.T.to(x.device))
        ic_centers = generate_displaced_ics(
            n_ics, dim, displacement=args.displacement, seed=args.seed,
        )
    else:
        sde_true = make_bistable_system(dim, sigma=sigma, device=device)
        theta_true = None
        true_drift_fn = lambda x: x * (1.0 - (x ** 2).sum(dim=-1, keepdim=True))
        ic_centers = generate_bistable_ics(n_ics, dim, seed=args.seed)

    true_G = sigma ** 2  # True G = sigma^2 * I for both systems

    print(f"\n{'='*60}")
    print(f"Dimension d={dim}, system={args.system}, n_ics={n_ics}")
    print(f"{'='*60}")

    # --- Generate snapshot data ---
    print(f"Generating snapshot data ({n_ics} ICs x {args.batch_size} particles)...")
    means, covs = simulate_snapshot_multi_ic(
        sde_true, ic_centers, args.batch_size, ts, args.dt,
        ic_spread=args.ic_spread, device=device,
    )
    print(f"Moment data: means {list(means.shape)}, covs {list(covs.shape)}")

    # --- Train NFPE ---
    sde_mlp = MLPSDE(
        state_size=dim,
        hidden_sizes=args.hidden_sizes,
        brownian_size=dim,
    )

    history, train_time = train_nfpe_multi_ic(
        sde_mlp, means, covs, dt_obs,
        epochs=args.epochs, lr=args.lr,
        cov_weight=args.cov_weight,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        device=device,
    )

    # --- Evaluate ---
    drift_results = evaluate_drift_error(
        sde_mlp, true_drift_fn, dim,
        eval_radius=args.displacement, device=device,
    )
    diff_results = evaluate_diffusion_error(
        sde_mlp, true_G, dim,
        eval_radius=args.displacement, device=device,
    )

    # Jacobian error at origin (OU only) — compute on CPU
    jac_error = None
    if args.system == "ou" and theta_true is not None:
        sde_mlp_cpu = sde_mlp.cpu()
        x_origin = torch.zeros(1, dim)
        def f_eval(x):
            return sde_mlp_cpu.drift_net(x)
        jac_mlp = jacfwd(f_eval)(x_origin).squeeze()
        jac_error = torch.norm(jac_mlp - (-theta_true.cpu())).item()
        sde_mlp.to(device)

    print(f"\nResults (d={dim}):")
    print(f"  Drift MSE:          {drift_results['drift_mse']:.6f}")
    print(f"  Drift relative err: {drift_results['drift_relative_error']:.4f}")
    print(f"  Diffusion MSE:      {diff_results['diffusion_mse']:.6f}")
    if jac_error is not None:
        print(f"  Jacobian error:     {jac_error:.4f}")
    print(f"  Training time:      {train_time:.1f}s")

    return {
        "dim": dim,
        "drift_mse": drift_results["drift_mse"],
        "drift_relative_error": drift_results["drift_relative_error"],
        "diffusion_mse": diff_results["diffusion_mse"],
        "jac_error": jac_error,
        "train_time": train_time,
        "final_loss": history[-1]["total"],
        "history": history,
        "sde_mlp": sde_mlp,
        "ic_centers": ic_centers,
        "true_drift_fn": true_drift_fn,
        "theta_true": theta_true,
    }


def run_experiment(args):
    """Run PFI benchmark across all requested dimensions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    all_histories = {}

    for dim in args.dims:
        res = run_single_dim(dim, args, device)
        all_results[dim] = {
            "dim": dim,
            "drift_mse": res["drift_mse"],
            "drift_relative_error": res["drift_relative_error"],
            "diffusion_mse": res["diffusion_mse"],
            "jac_error": res["jac_error"],
            "train_time": res["train_time"],
            "final_loss": res["final_loss"],
        }
        all_histories[dim] = res["history"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Plots ---
    os.makedirs(args.output_dir, exist_ok=True)
    dims = args.dims

    # Plot 1: Drift MSE vs dimension
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    drift_mses = [all_results[d]["drift_mse"] for d in dims]
    diff_mses = [all_results[d]["diffusion_mse"] for d in dims]

    axes[0].semilogy(dims, drift_mses, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Drift MSE")
    axes[0].set_title(f"Drift Recovery ({args.system.upper()})")
    axes[0].set_xticks(dims)

    axes[1].semilogy(dims, diff_mses, "rs-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Diffusion MSE")
    axes[1].set_title(f"Diffusion Recovery ({args.system.upper()})")
    axes[1].set_xticks(dims)

    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "benchmark_scaling.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Training loss curves per dimension
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    for ax, dim in zip(axes, dims):
        hist = all_histories[dim]
        epochs_arr = [h["epoch"] for h in hist]
        ax.semilogy(epochs_arr, [h["means"] for h in hist], label="Means")
        ax.semilogy(epochs_arr, [h["covs"] for h in hist], label="Covs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"d={dim}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, "benchmark_loss_curves.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"PFI Benchmark Summary ({args.system.upper()})")
    print("=" * 70)
    print(f"{'Dim':>5} {'Drift MSE':>12} {'Diff MSE':>12} "
          f"{'Drift Rel':>10} {'Jac Err':>10} {'Time(s)':>10}")
    print("-" * 60)
    for dim in dims:
        r = all_results[dim]
        jac_str = f"{r['jac_error']:.4f}" if r["jac_error"] is not None else "—"
        print(f"{dim:>5} {r['drift_mse']:>12.6f} {r['diffusion_mse']:>12.6f} "
              f"{r['drift_relative_error']:>10.4f} {jac_str:>10} "
              f"{r['train_time']:>10.1f}")

    # Save JSON
    results_json = {
        "system": args.system,
        "results": {str(d): all_results[d] for d in dims},
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nFigures and results saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="PFI/UPFI accuracy benchmark",
    )
    parser.add_argument(
        "--system", type=str, default="ou",
        choices=["ou", "bistable"],
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=[2, 5, 10],
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cov-weight", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/pfi_benchmark_{args.system}"

    run_experiment(args)


if __name__ == "__main__":
    main()
