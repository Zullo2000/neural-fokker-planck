"""Tests for NFPE model definitions."""

import torch
from nfpe import LinearSDE, CIRSDE, MLPSDE


def test_linear_sde_shapes():
    sde = LinearSDE(
        f_linear=torch.randn(2, 2),
        f_bias=torch.randn(2),
        g_linear=torch.randn(2, 2),
        g_bias=torch.randn(2),
    )
    y = torch.randn(5, 2)
    assert sde.f(0, y).shape == (5, 2)
    assert sde.g(0, y).shape == (5, 2, 1)


def test_cir_sde_shapes():
    sde = CIRSDE(kappa=1.0, theta=0.5, sigma=0.3)
    y = torch.abs(torch.randn(5, 1))
    assert sde.f(0, y).shape == (5, 1)
    assert sde.g(0, y).shape == (5, 1)


def test_mlpsde_shapes():
    sde = MLPSDE(state_size=3, hidden_sizes=[16, 16], brownian_size=3)
    y = torch.randn(5, 3)
    assert sde.f(0, y).shape == (5, 3)
    assert sde.g(0, y).shape == (5, 3, 3)


def test_mlpsde_diffusion_matrix_psd():
    sde = MLPSDE(state_size=3, hidden_sizes=[16, 16], brownian_size=3)
    y = torch.randn(10, 3)
    G = sde.diffusion_matrix(y)
    assert G.shape == (10, 3, 3)
    # G = B @ B^T must be positive semi-definite
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals >= -1e-6).all(), f"Negative eigenvalue: {eigvals.min()}"
