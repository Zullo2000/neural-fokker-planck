"""Tests for NFPE training utilities."""

import torch
from nfpe import LinearSDE, forward_backward_loss, train_nfpe


def _make_toy_data():
    """Create minimal moment data for testing."""
    T, d = 10, 2
    means = torch.randn(T, d) * 0.5
    covs = torch.eye(d).unsqueeze(0).expand(T, d, d) * 0.1
    dt = 0.1
    return means, covs, dt


def test_forward_backward_loss_returns():
    means, covs, dt = _make_toy_data()
    model = LinearSDE(
        f_linear=torch.randn(2, 2) * 0.1,
        f_bias=torch.zeros(2),
        g_linear=torch.zeros(2, 2),
        g_bias=torch.randn(2) * 0.1,
    )
    total, lm, lc = forward_backward_loss(model, means, covs, dt)
    assert total.dim() == 0  # scalar
    assert lm.dim() == 0
    assert lc.dim() == 0
    total.backward()  # gradients flow


def test_train_nfpe_loss_decreases():
    means, covs, dt = _make_toy_data()
    model = LinearSDE(
        f_linear=torch.randn(2, 2) * 0.1,
        f_bias=torch.zeros(2),
        g_linear=torch.zeros(2, 2),
        g_bias=torch.randn(2) * 0.1,
    )
    history = train_nfpe(
        model, means, covs, dt,
        epochs=50, lr=0.01, verbose=False,
        use_jacobian=False,
    )
    assert len(history) == 50
    assert history[-1]["total"] < history[0]["total"]
