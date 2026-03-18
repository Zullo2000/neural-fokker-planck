"""Tests for NFPE data utilities."""

import torch
from nfpe import LinearSDE, simulate_sde, fit_gmm_to_snapshots, shuffle_snapshots


def test_simulate_sde_shape():
    sde = LinearSDE(
        f_linear=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        f_bias=torch.zeros(2),
        g_linear=torch.zeros(2, 2),
        g_bias=torch.tensor([0.0, 0.05]),
        learnable=False,
    )
    y0 = torch.randn(10, 2)
    ts = torch.linspace(0, 1, 5)
    traj = simulate_sde(sde, y0, ts)
    assert traj.shape == (5, 10, 2)


def test_fit_gmm_shapes():
    traj = torch.randn(5, 20, 2)  # T=5, N=20, d=2
    gmms, means, covs = fit_gmm_to_snapshots(traj, n_components=2)
    assert len(gmms) == 5
    assert means.shape == (5, 2, 2)
    assert covs.shape == (5, 2, 2, 2)


def test_shuffle_preserves_moments():
    torch.manual_seed(0)
    traj = torch.randn(5, 100, 2)
    traj_shuffled = shuffle_snapshots(traj)
    # Marginal moments should be identical
    for t in range(5):
        assert torch.allclose(
            traj[t].mean(0), traj_shuffled[t].mean(0), atol=1e-5,
        )
