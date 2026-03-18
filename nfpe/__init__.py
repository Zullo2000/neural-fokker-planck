"""Neural Fokker-Planck Equations (NFPE).

A framework for learning stochastic differential equations from data
via deterministic Fokker-Planck moment dynamics.

Key components:
    - models: SDE definitions (Linear, CIR, MLP-based)
    - propagators: Gaussian moment propagators (Euler, Unscented, Rosenbrock)
    - training: Forward-backward moment matching loss and training loop
    - data: SDE simulation and GMM fitting
    - utils: Matrix utilities (phi_1)
"""

from .models import SDE, LinearSDE, CIRSDE, MLPSDE, TimeDepWrapper
from .propagators import (
    EulerGaussianPropagator,
    UnscentedPropagator,
    EulerRosenbrockModel,
    JacobianAuxWrapper,
)
from .training import forward_backward_loss, train_nfpe
from .data import (
    simulate_sde,
    fit_gmm_to_snapshots,
    shuffle_snapshots,
    simulate_independent_snapshots,
)
from .utils import phi_1, phi_1_pade

__all__ = [
    # Models
    "SDE",
    "LinearSDE",
    "CIRSDE",
    "MLPSDE",
    "TimeDepWrapper",
    # Propagators
    "EulerGaussianPropagator",
    "UnscentedPropagator",
    "EulerRosenbrockModel",
    "JacobianAuxWrapper",
    # Training
    "forward_backward_loss",
    "train_nfpe",
    # Data
    "simulate_sde",
    "fit_gmm_to_snapshots",
    "shuffle_snapshots",
    "simulate_independent_snapshots",
    # Utils
    "phi_1",
    "phi_1_pade",
]
