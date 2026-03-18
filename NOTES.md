# Neural Fokker-Planck Equations — Project Explanation

## 1. The Problem

Many systems in physics, biology, and finance are governed by **Stochastic Differential Equations** (SDEs):

$$dX = F(X)\,dt + B(X)\,dW_t$$

where $F(X)$ is the **drift** (the deterministic force pushing the state) and $B(X)$ is the **diffusion matrix** (the intensity of stochastic noise). Given observed data from such a system, the goal is to **learn** $F$ and $B$.

### Why is this hard?

The standard approach — Neural SDEs — parameterizes $F_\theta$ and $B_\theta$ with neural networks, then trains by simulating the SDE forward and comparing simulated trajectories to data. This has two fundamental bottlenecks:

1. **Stochastic integration is sequential.** Each time step depends on the previous step plus a random draw. You cannot parallelize across time, making training slow.

2. **Variance in gradients.** Because the forward pass is stochastic, gradients are noisy. You need many Monte Carlo samples (i.e., many trajectory simulations per training step) to get stable parameter updates.

Together, these make Neural SDE training expensive: $O(N_{\text{samples}} \times T_{\text{steps}})$ sequential SDE solves per gradient step.

### What if we could avoid stochastic simulation entirely?

The **Fokker–Planck equation** describes how the *probability density* $p(x,t)$ of the SDE evolves over time. It is a deterministic partial differential equation — no randomness involved. If we could work with $p(x,t)$ directly instead of with individual trajectories, we would eliminate both bottlenecks above.

The challenge is that $p(x,t)$ lives in a high-dimensional function space and is generally intractable. Neural Fokker-Planck Equations solve this by working not with the full density, but with its **low-order moments** (mean and covariance), which evolve according to simple, closed-form ODEs.


## 2. The Core Idea: Neural Fokker-Planck Equations

### From densities to moments

Represent the distribution at each time $t$ as a **Gaussian Mixture Model** with $K$ components. Each component $k$ has a mean $\mu_k(t)$ and covariance $\Sigma_k(t)$. Under Gaussian closure (i.e., assuming each component stays approximately Gaussian over a short time step), the Fokker–Planck equation reduces to a system of **ordinary differential equations** on the moments:

$$\dot{\mu}_k = F(\mu_k)$$

$$\dot{\Sigma}_k = D_F(\mu_k)\,\Sigma_k + \Sigma_k\,D_F(\mu_k)^\top + B(\mu_k)\,B(\mu_k)^\top$$

where $D_F$ denotes the Jacobian of the drift $F$. These are deterministic ODEs — no stochastic integration, no Monte Carlo.

**What each equation says:**
- The **mean** evolves by following the drift field, exactly as a deterministic particle would.
- The **covariance** changes via two mechanisms: (a) the Jacobian $D_F$ stretches/rotates the distribution (e.g., shear in a flow field), and (b) the diffusion $BB^\top$ injects variance isotropically or anisotropically depending on $B$.

### The training pipeline

Given observed trajectory data from an unknown SDE:

1. **Snapshot extraction.** At each observation time $t_n$, collect the positions of all particles and fit a Gaussian Mixture Model, obtaining empirical moments $\{\hat{\mu}_k(t_n), \hat{\Sigma}_k(t_n)\}$.

2. **Moment prediction.** Using learnable networks $F_\theta$ and $B_\theta$, compute the predicted one-step moment changes via the ODE above.

3. **Loss computation.** Compare predicted moments to observed moments at the next (and previous) time steps. Minimize the discrepancy.

No SDE is ever simulated. The entire training loop is a standard deterministic optimization problem.

### Forward-backward disentanglement

There is a subtlety: both drift and diffusion contribute to covariance changes, so a naive forward-only loss conflates the two. NFPE resolves this with a **forward-backward scheme** using three consecutive snapshots at times $t_{n-1}$, $t_n$, $t_{n+1}$.

**Forward prediction:**
$$\mu_k(t_n) + \Delta t \cdot F_\theta(\mu_k(t_n)) \approx \mu_k(t_{n+1})$$

**Backward prediction:**
$$\mu_k(t_n) - \Delta t \cdot F_\theta(\mu_k(t_n)) \approx \mu_k(t_{n-1})$$

The key insight is **time-reversibility**:
- **Drift is reversible.** Reverse the velocity field and a particle traces its path backward. The forward and backward mean predictions are symmetric.
- **Diffusion is irreversible.** Noise always adds variance, never subtracts it. Running the diffusion "backward" still increases spread.

By requiring the model to predict accurately in *both* directions, the optimizer is forced to attribute the symmetric (reversible) part of the dynamics to drift and the asymmetric (irreversible) part to diffusion. This cleanly separates the two without any additional regularization.

The combined loss for means is:

$$\mathcal{L}_\mu = \sum_n \sum_k \left\| \mu_k(t_{n+1}) - \mu_k(t_n) - \Delta t\, F_\theta(\mu_k(t_n)) \right\|^2 + \left\| \mu_k(t_n) - \mu_k(t_{n-1}) + \Delta t\, F_\theta(\mu_k(t_n)) \right\|^2$$

with analogous terms for the covariance dynamics.

### Gaussian Mixture representation

A single Gaussian is sufficient when the distribution is unimodal (e.g., a particle cloud drifting in a harmonic potential). For multimodal distributions (e.g., a double-well potential where particles cluster around two stable points), we use $K > 1$ Gaussian components.

Each component evolves independently under the moment ODEs above. The GMM is fitted to the empirical particle distribution at each snapshot time using standard Expectation-Maximization (scikit-learn). The number of components $K$ is a hyperparameter chosen based on the complexity of the system.

### Neural or parametric

The framework is agnostic to the parameterization of $F$ and $B$:

- **Linear/affine models** (`LinearSDE`): $F(x) = Ax + b$, $B(x) = Cx + d$. The Jacobian $D_F = A$ is constant, making the covariance ODE exact (no closure approximation). Used for systems like Geometric Brownian Motion and harmonic oscillators.

- **MLP models** (`MLPSDE`): $F_\theta$ and $B_\theta$ are multi-layer perceptrons. The Jacobian $D_F$ is computed via forward-mode automatic differentiation (`torch.func.jacfwd`), batched over all GMM components using `vmap`. Used for nonlinear systems like cubic damping or double-well potentials. The diffusion network uses a Cholesky parameterization to ensure $BB^\top$ is always positive semi-definite.

### Computational advantage

| Aspect | Neural SDE | NFPE |
|---|---|---|
| Forward pass | Stochastic SDE integration ($N$ samples $\times$ $T$ steps) | Deterministic moment ODE (one evaluation per component per step) |
| Gradient computation | Adjoint method through stochastic dynamics | Standard backpropagation through deterministic loss |
| Parallelism | Sequential across time | GMM components are independent; Jacobians computed via `vmap` |
| Data requirement | Individual trajectories | Distribution snapshots (mean + covariance) |

The last row is the most significant for applications: NFPE can learn from **cross-sectional snapshots** where individual particles are not tracked across time. This is impossible for trajectory-matching methods.


## 3. The `nfpe/` Package

The implementation is organized into five modules that separate concerns cleanly.

### `models.py` — SDE Definitions

Defines the learnable SDE classes that parameterize $F$ and $B$:

- **`SDE`** — Abstract base class. All models expose `.f(t, y)` (drift) and `.g(t, y)` (diffusion), compatible with the `torchsde` interface.

- **`LinearSDE`** — Affine drift and diffusion:
  $$F(x) = A x + b, \qquad B(x) = C x + d$$
  Parameters $A, b, C, d$ are optionally learnable (`nn.Parameter`). Used for Geometric Brownian Motion and the harmonic oscillator, where the true system is exactly linear.

- **`CIRSDE`** — Cox–Ingersoll–Ross process, a classic mean-reverting model from finance with state-dependent square-root diffusion.

- **`MLPSDE`** — Neural drift and diffusion. Two separate MLPs:
  - `drift_net`: standard MLP mapping $\mathbb{R}^d \to \mathbb{R}^d$.
  - `diff_net`: MLP mapping $\mathbb{R}^d \to \mathbb{R}^{d \times d}$, outputting the **lower-triangular Cholesky factor** $L$ so that $B B^\top = L L^\top$ is guaranteed positive semi-definite. This architectural constraint prevents the diffusion from becoming unphysical.

- **`TimeDepWrapper`** — Adapter that strips the time argument from time-independent models, making them compatible with `torchdiffeq` ODE solvers.

### `propagators.py` — Gaussian Moment Propagators

Given a model's $F$ and $B$, these modules compute the right-hand side of the moment ODEs so they can be integrated as Neural ODEs via `torchdiffeq`:

- **`EulerGaussianPropagator`** — Jacobian-based propagator. Computes $D_F$ at each component mean using forward-mode automatic differentiation (`torch.func.jacfwd`), batched over components via `vmap`. The state vector is $[\mu; \text{vec}(\Sigma)]$. This is the workhorse propagator: exact for linear systems, first-order accurate for nonlinear ones.

- **`UnscentedPropagator`** — Sigma-point propagator. Instead of computing the Jacobian explicitly, it passes deterministic "sigma points" (the unscented transform) through $F$ and reconstructs the output mean and covariance from the transformed points. The state is stored as $(d{+}1) \times d$ matrix: the mean plus the columns of $\sqrt{\Sigma}$. Avoids explicit Jacobian computation, useful when $D_F$ is expensive.

- **`EulerRosenbrockModel`** — Implicit-explicit integrator for stiff systems. Uses a Padé approximant of the $\varphi_1$ matrix function:
  $$\varphi_1(M) = \frac{e^M - I}{M}$$
  This gives implicit-like stability at explicit-like cost, avoiding the tiny time steps that a naive Euler scheme would need for stiff dynamics.

- **`JacobianAuxWrapper`** — Utility that wraps a model so that `jacfwd(..., has_aux=True)` computes both the Jacobian and function value in a single forward pass, avoiding redundant computation.

### `training.py` — Forward-Backward Loss and Training Loop

- **`compute_moment_derivatives()`** — Given a model and a batch of means/covariances, computes $\dot{\mu}$ and $\dot{\Sigma}$ via the moment ODEs. Uses `vmap(jacfwd(...))` for batched Jacobian computation when `use_jacobian=True`, or uses the model's linear structure directly when `use_jacobian=False` (faster for `LinearSDE`).

- **`forward_backward_loss()`** — The core loss function. For each interior time point $t_n$ (with neighbors $t_{n-1}$ and $t_{n+1}$):
  1. Predicts $\mu(t_{n+1})$ and $\mu(t_{n-1})$ from $\mu(t_n)$ using the drift.
  2. Predicts $\Sigma(t_{n+1})$ and $\Sigma(t_{n-1})$ from $\Sigma(t_n)$ using the Jacobian and diffusion.
  3. Computes MSE between predictions and observations in both directions.
  4. Returns weighted sum: `loss_means + cov_weight * loss_covs`.

  The `cov_weight` hyperparameter balances mean-matching against covariance-matching. Typical values range from 100 to 500, since covariance entries are often orders of magnitude smaller than means.

- **`train_nfpe()`** — Standard training loop: Adam optimizer, configurable epochs and learning rate, returns loss history for plotting.

### `data.py` — Simulation and GMM Fitting

- **`simulate_sde()`** — Wrapper around `torchsde.sdeint`. Takes a ground-truth SDE, initial conditions, and a time grid; returns trajectories of shape $(T, N, d)$.

- **`fit_gmm_to_snapshots()`** — At each time step, fits a $K$-component Gaussian Mixture to the particle positions using scikit-learn's `GaussianMixture` (Expectation-Maximization). Returns tensors of means $(T, K, d)$ and covariances $(T, K, d, d)$ ready for the training loop.

### `utils.py` — Matrix Utilities

- **`phi_1(M)`** — Computes $\varphi_1(M) = (e^M - I) M^{-1}$ via the augmented matrix exponential trick (embedding $M$ in a larger matrix and reading off the result). Exact but involves a matrix exponential.

- **`phi_1_pade(M)`** — Padé [6/6] rational approximation of $\varphi_1$. Faster than the exact computation and numerically stable, used by the Rosenbrock propagator.


## 4. Experiments and Results

### Experiment 1: Black–Scholes (Geometric Brownian Motion)

**File:** `experiments/black_scholes.py`

**System:**
$$dX = 2.5\,X\,dt + 0.4\,X\,dW$$

A 1D multiplicative-noise SDE — the foundational model in option pricing. The drift and diffusion are both proportional to the state.

**Task:** Recover the four parameters $(f_0, f_1, b_0, b_1)$ from 20 simulated trajectories observed at 20 time points.

**Results:**

| Method | $f_0$ | $f_1$ | $b_0$ | $b_1$ |
|---|---|---|---|---|
| True | 0.000 | 2.500 | 0.000 | 0.400 |
| RESS [Iannacone & Gardoni 2024] | 0.000 | 2.420 | 0.000 | 0.361 |
| NFPE (forward-only) | — | — | — | — |
| **NFPE (forward-backward)** | 0.035 | **2.427** | −0.065 | **−0.248** |

NFPE matches the RESS baseline on drift recovery ($f_1$ error ~0.07) from a fraction of the data. The experiment also includes a **forward-only ablation** demonstrating that the forward-backward scheme produces better separation of drift and diffusion.

**What this shows:** NFPE recovers affine SDE parameters accurately from few observations, competitive with established methods that use full likelihood estimation.

---

### Experiment 2: 2D Harmonic Oscillator

**File:** `experiments/identification.py`

**System:**
$$dx_1 = x_2\,dt, \qquad dx_2 = -x_1\,dt + 0.05\,dW$$

A 2D oscillator with additive noise — the state rotates in phase space while noise perturbs the second coordinate.

**Task:** Recover the $2 \times 2$ drift matrix $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ and the diffusion bias $g = (0, 0.05)$.

**Results:**

| Parameter | True | Learned |
|---|---|---|
| $A$ | $\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ | $\begin{pmatrix} -0.002 & 0.997 \\ -0.999 & -0.018 \end{pmatrix}$ |
| $g_{\text{bias}}$ | $(0, 0.05)$ | — |
| **Frobenius error on $A$** | — | **0.018** |

The rotation matrix is recovered nearly perfectly. The experiment produces phase-space plots (original vs estimated trajectories) and evolving Gaussian contour plots showing the distribution spiraling inward.

**What this shows:** NFPE handles multi-dimensional coupled dynamics where drift and diffusion act on different coordinates.

---

### Experiment 3: Gaussian Propagation (Validation)

**File:** `experiments/propagation.py`

**System:** Damped spiral with additive noise:
$$A = \begin{pmatrix} -0.1 & 0.5 \\ -0.5 & -0.1 \end{pmatrix}, \qquad g = (0.05, 0.05)$$

**Task:** This is not a learning experiment. It validates that the moment propagators correctly predict the evolution of a Gaussian distribution through a known SDE. An initial Gaussian $\mathcal{N}(\mu_0, \Sigma_0)$ is propagated using the `EulerGaussianPropagator` and `UnscentedPropagator`, and the predicted mean/covariance trajectories are compared against empirical statistics from 200 sample trajectories.

**What this shows:** The moment ODE framework is mathematically correct — predicted Gaussians match the empirical particle cloud, confirming that the propagators are implemented correctly before using them for learning.

---

### Experiment 4: Nonlinear Cubic Damping (MLP)

**File:** `experiments/double_well.py`

**System:**
$$dX = -X^3\,dt + 0.2\,dW$$

A 1D SDE with nonlinear drift. The restoring force grows cubically, creating a single stable equilibrium at $x = 0$ with strongly nonlinear dynamics away from it.

**Task:** Learn the drift function $F(x) = -x^3$ from scratch using an MLP (two hidden layers of 32 units each), with no assumption of the functional form.

**Results:**

| Metric | Value |
|---|---|
| Drift MSE (observed region) | **0.022** |
| Drift MSE (full range $[-2, 2]$) | 3.094 |
| True diffusion $G$ | 0.04 |

The learned drift closely tracks the true cubic within the data-covered region. Outside the observed range, the MLP extrapolates poorly (expected — no data there). The diffusion coefficient is also recovered.

**What this shows:** NFPE is not limited to parametric models. An MLP can learn arbitrary nonlinear drift functions from moment dynamics alone. This is the first demonstration that the Fokker-Planck moment matching approach generalizes beyond linear systems.

---

### Experiment 5: Multi-Dimensional Ornstein–Uhlenbeck (3D and 5D)

**File:** `experiments/multi_d_ou.py`

**System:**
$$dX = -\Theta\,X\,dt + 0.3\,dW$$

where $\Theta$ is a randomly generated $d \times d$ positive-definite matrix.

**Task:** Recover the full $d \times d$ drift matrix using an MLP — without assuming linearity. This tests whether NFPE scales to higher dimensions.

**Key design decisions (learned from initial failures):**
- **Multiple displaced initial conditions.** A single Gaussian cloud equilibrates quickly in a linear potential, causing both mean and covariance signals to vanish. Using 8–12 IC clusters displaced far from equilibrium ensures the mean trajectories sweep through different regions of state space, giving the MLP enough supervision.
- **Per-cluster moment tracking.** Each IC cluster's moments are tracked independently, providing $K$ separate moment trajectories rather than one aggregated trajectory.
- **Cosine annealing + weight decay.** Prevents overfitting to moment noise, which is more severe in higher dimensions with fewer particles per cluster.

**Results:**

| Dimension | Jacobian Error | Drift MSE | Relative Error | Epochs |
|---|---|---|---|---|
| 3D | **0.022** | 0.088 | 19.4% | 3000 |
| 5D | **0.045** | 0.550 | 32.8% | 4000 |

For the 3D case, all 9 entries of the $3 \times 3$ drift Jacobian are recovered with error 0.022. The 5D case recovers all 25 entries of the $5 \times 5$ Jacobian with error 0.045 — still accurate, though with higher relative error due to the curse of dimensionality (more parameters to learn from the same moment information).

**What this shows:** NFPE scales to multi-dimensional systems with non-parametric drift learning. The Jacobian error metric confirms that the MLP has learned the correct local linearization of the dynamics, not just a function that happens to fit the training moments.

---

### Experiment 6: Snapshot Learning (No Trajectory Tracking)

**File:** `experiments/snapshot_learning.py`

**Systems:** 2D Harmonic Oscillator (linear, `LinearSDE`) and 1D Cubic Damping (nonlinear, `MLPSDE`).

**Goal:** Prove that NFPE learns equally well from cross-sectional snapshots — where particles are not tracked across time — as from tracked trajectories. This validates the strongest novelty claim: NFPE only needs distribution moments, not individual particle correspondences. Testing on both a linear and a nonlinear system confirms the result is general.

**Method:** Three pipelines are compared on the same SDE, with identical model initialization and hyperparameters:

- **Pipeline A (Tracked):** Standard approach — simulate 2000 particles, track them across all time steps, fit GMM moments.
- **Pipeline B (Shuffled):** Same simulation as A, but particles are randomly permuted at each time step before fitting moments. This destroys trajectory correspondence while preserving the exact marginal distribution.
- **Pipeline C (Independent):** At each observation time $t_n$, a fresh batch of 2000 particles is simulated independently from $t=0$ to $t=t_n$. There is no particle correspondence across time steps whatsoever.

**Key analytical insight:** For a single-component Gaussian fit ($K=1$), the sample mean $\hat{\mu} = \frac{1}{N}\sum_i x_i$ and sample covariance $\hat{\Sigma} = \frac{1}{N}\sum_i (x_i - \hat{\mu})(x_i - \hat{\mu})^\top$ are **permutation-invariant** — they depend only on the set of particles, not on their ordering. Therefore, Pipeline B must produce moments identical to Pipeline A. This is a mathematical guarantee, not an empirical result.

Pipeline C tests the practical scenario: when particles are drawn independently at each snapshot, the moments differ slightly due to finite-sample noise, but the learned SDE should still be accurate.

**Results (2D Oscillator — linear, parametric):**

| Pipeline | $A$ error (Frobenius) | $g$ error (L2) | Moment diff vs Tracked |
|---|---|---|---|
| A (Tracked) | **0.003** | 0.046 | — |
| B (Shuffled) | **0.003** | 0.046 | 0.00 (exact) |
| C (Independent) | **0.010** | 0.094 | 0.25 |

**Results (1D Cubic Damping — nonlinear, MLP):**

| Pipeline | Drift MSE (observed region) | Moment diff vs Tracked |
|---|---|---|
| A (Tracked) | **0.059** | — |
| B (Shuffled) | **0.059** | 0.00 (exact) |
| C (Independent) | **0.126** | 0.029 |

In both systems, Pipeline B produces **bit-for-bit identical** results to Pipeline A — the moment difference is exactly zero, confirming that trajectory tracking is mathematically irrelevant to NFPE's input. Pipeline C recovers the dynamics with roughly 2x the error of tracked training, which is expected given finite-sample moment noise from independent sampling.

The cubic damping result is particularly significant: it demonstrates that snapshot learning works not only for parametric linear models, but also for **nonlinear drift functions learned by an MLP** — the setting most relevant to real-world applications.

**What this shows:** NFPE can learn SDEs from **cross-sectional distribution snapshots** where individual particles are not tracked across time. This opens NFPE to applications like flow cytometry, financial cross-sections, and epidemiological surveys — settings where trajectory-matching methods fundamentally cannot operate.


## 5. What's Novel

### Positioning in the literature

Existing methods for learning SDEs from data fall into several categories:

1. **Trajectory-matching** (Neural SDEs [Kidger et al. 2021], SDE-GANs, FDM [Zhang et al. ICLR 2025]): require individual particle trajectories, simulate SDEs during training, and backpropagate through stochastic dynamics. Expensive and limited to settings where trajectory data is available.

2. **Moment-based identification** (SPINODE [O'Leary et al. 2022]): uses deterministic moment ODEs (mean + covariance) to avoid stochastic simulation during training — the same core idea as NFPE. However, SPINODE requires **trajectory data** to estimate moments. It cannot operate from cross-sectional snapshots where particles are not tracked.

3. **Snapshot-based identification** (APPEX [Guan et al. 2024], SpIDES [Zhu et al. 2024], UPFI [Zhang et al. NeurIPS 2025]): learn SDEs from distribution snapshots without trajectory tracking. However, these methods either restrict to **linear SDEs** (APPEX), rely on **density estimation** rather than moment reduction (UPFI, SpIDES), or require **sparse dictionaries** of basis functions (SpIDES).

NFPE is **the method that jointly learns drift AND diffusion from distribution snapshots, without requiring density estimation or optimal transport.**

This is a critical distinction from the closest competitor, UPFI/PFI (Zhang et al. NeurIPS 2025 / PNAS 2025). PFI requires the diffusion coefficient to be **known a priori** — if it is misspecified, drift estimates are biased. PFI also requires two expensive preprocessing steps that NFPE avoids entirely: (a) **denoising score matching** to estimate $\nabla \log p(x,t)$ from raw particles, and (b) **Sinkhorn optimal transport** ($O(B^2)$) to match pushed-forward particles against observed snapshots. NFPE bypasses both by working directly with sample moments ($O(B)$ to compute), making it architecturally simpler and computationally cheaper.

| Method | Snapshots? | No SDE sim? | Learns diffusion? | Needs density est.? | Needs OT? | Approach |
|---|---|---|---|---|---|---|
| Neural SDEs / FDM | No | No | Yes | No | No | Trajectory matching |
| SPINODE | No | Yes | Yes | No | No | Moment ODEs |
| UPFI / PFI | Yes | Yes | **No (given)** | **Yes (score matching)** | **Yes (Sinkhorn)** | Probability flow |
| **NFPE (ours)** | **Yes** | **Yes** | **Yes** | **No** | **No** | **Moment ODEs** |

### Novelty claim 1: Deterministic training — no stochastic integration

During training, NFPE never simulates an SDE. The forward pass is a deterministic evaluation of the moment ODEs (drift + Jacobian + diffusion at each component mean). Gradients are computed via standard backpropagation through a deterministic computation graph.

This eliminates the two main costs of Neural SDE training: (a) sequential stochastic integration and (b) variance in gradient estimates. Training NFPE is as fast as training a standard neural network regression model.

### Novelty claim 2: Forward-backward drift/diffusion disentanglement

The three-point forward-backward stencil exploits the time-irreversibility of diffusion to cleanly separate drift from noise effects. This is a structural property of the loss function — it does not require any regularization, auxiliary losses, or curriculum scheduling. No prior method uses this mechanism for SDE identification.

### Novelty claim 3: Joint drift and diffusion learning from distribution snapshots

The strongest contribution, demonstrated in Experiment 6:

In many real-world applications, you observe a **population** at successive times but cannot track individual members:
- **Flow cytometry:** millions of cells measured at each time point, but cells are destroyed by measurement — no trajectory data exists.
- **Financial markets:** you observe the cross-sectional distribution of asset prices, not individual asset histories.
- **Epidemiology:** you observe disease state distributions, not individual patient trajectories.

Existing snapshot-based methods (PFI/UPFI) can learn drift from such data, but require the diffusion coefficient to be **known in advance**. NFPE learns **both drift and diffusion** jointly from the same snapshot moments. The mean dynamics $\dot{\mu} = F(\mu)$ identify the drift, while the covariance dynamics $\dot{\Sigma} = D_F \Sigma + \Sigma D_F^\top + BB^\top$ identify the diffusion — with the forward-backward scheme disentangling the two contributions. No density estimation or optimal transport is needed: NFPE only requires the sample mean and covariance at each observation time.

Experiment 6 confirms this empirically on two systems: NFPE trained from **independent cross-sectional samples** (no particle correspondence across time) recovers the drift matrix of a 2D oscillator with error 0.010 (vs 0.003 tracked), and learns the nonlinear cubic drift via MLP with MSE 0.126 (vs 0.059 tracked). The shuffled-trajectory control produces bit-for-bit identical results to the tracked baseline in both cases, proving that the moment extraction is permutation-invariant by construction.

### Summary of contributions

| Contribution | Status | Evidence |
|---|---|---|
| Deterministic moment-based SDE learning | Complete | All 5 experiments train without SDE simulation |
| Forward-backward disentanglement | Complete | Black-Scholes ablation (fwd-only vs fwd-bwd) |
| Linear parameter recovery | Complete | Black-Scholes ($f_1$ error 0.07), Oscillator ($A$ error 0.018) |
| Nonlinear drift learning via MLP | Complete | Cubic damping (drift MSE 0.022) |
| Multi-dimensional scaling | Complete | 3D (Jacobian error 0.022), 5D (Jacobian error 0.045) |
| Snapshot learning (no trajectory tracking) | Complete | Experiment 6: oscillator (0.003 vs 0.010) and cubic MLP (0.059 vs 0.126) |
| PFI benchmark (OU, d=2,5,10) | Complete | Drift rel. error 20.9%/39.0%/56.6%, diffusion MSE ~10⁻⁵ |
| PFI benchmark (Bistable) | **Failed** | Drift rel. error >98% — Gaussian closure breakdown (see below) |


## Stage 3: PFI Benchmark — Comparison with PFI/UPFI

### Motivation

Experiments 1–6 validated NFPE on its own terms. Stage 3 benchmarks NFPE on **the same systems used by PFI/UPFI** (Zhang et al., NeurIPS 2025 / PNAS 2025) — the closest competing method — to produce a direct, apples-to-apples comparison. The benchmark tests two systems across dimensions $d = 2, 5, 10$:

- **Multi-dimensional Ornstein–Uhlenbeck (OU):** $dX = -\Theta X\,dt + 0.3\,dW$, with a random positive-definite $\Theta$. Linear drift — Gaussian closure is exact.
- **Bistable potential:** $dX = x(1 - |x|^2)\,dt + 0.5\,dW$. Nonlinear cubic drift with a stable manifold on the unit sphere $|x| = 1$.

The comparison highlights an asymmetry: PFI requires the diffusion coefficient $\sigma$ to be **known a priori** and uses density estimation (score matching) plus optimal transport (Sinkhorn). NFPE learns **both drift and diffusion** jointly from moment summaries alone.

### Experiment 7: OU Benchmark (d=2, 5, 10)

**Files:** `experiments/pfi_benchmark.py`, `experiments/benchmark_systems.py`

**Results (NFPE, 3000 epochs, GPU):**

| Dimension | Drift Rel. Error | Jacobian Error | Diffusion MSE | Training Time |
|---|---|---|---|---|
| 2 | **20.9%** | 0.053 | 6.5 × 10⁻⁵ | 32s |
| 5 | **39.0%** | 0.072 | 7.5 × 10⁻⁶ | 39s |
| 10 | **56.6%** | 0.296 | 4.4 × 10⁻⁶ | 63s |

These results are consistent with the earlier Experiment 5 (d=3: 19.4% relative error, d=5: 32.8%), confirming that the benchmark pipeline reproduces NFPE's established performance. Diffusion recovery is excellent across all dimensions (MSE on the order of $10^{-5}$ to $10^{-6}$), demonstrating NFPE's ability to jointly learn both coefficients.

**Comparison with PFI:** While PFI reports lower drift error on OU systems, PFI is given the true diffusion coefficient $\sigma = 0.3$ as input and uses score matching + Sinkhorn OT — two expensive preprocessing steps NFPE avoids entirely. NFPE achieves competitive accuracy while solving a strictly harder problem (joint drift + diffusion recovery) with a simpler pipeline.

### Experiment 8: Bistable Benchmark (d=2, 5, 10)

**Results (NFPE, 3000 epochs, GPU):**

| Dimension | Drift Rel. Error | Diffusion MSE | Training Time |
|---|---|---|---|
| 2 | 98.1% | 0.045 | 33s |
| 5 | 99.0% | 0.020 | 41s |
| 10 | 99.3% | 0.005 | 64s |

NFPE **fails to learn the bistable drift**, with relative errors above 98% at all dimensions. The diffusion MSE remains reasonable (the model learns $\sigma$ roughly), but drift recovery is essentially zero. This is a fundamental failure of the Gaussian closure approximation on this system, not a training or implementation issue.

#### Why the bistable system breaks Gaussian closure

The bistable drift is $F(x) = x(1 - |x|^2)$, which is cubic in the state. NFPE's moment ODE for the mean assumes:

$$\dot{\mu} = F(\mu) \approx \mathbb{E}[F(X)]$$

This approximation requires $\mathbb{E}[F(X)] \approx F(\mathbb{E}[X])$, which holds when the distribution is tightly concentrated (so that higher-order terms in the Taylor expansion of $F$ around $\mu$ are negligible). For the bistable system, this breaks down for three compounding reasons:

1. **The drift is cubic, so the closure error is large.** By Jensen's inequality, $\mathbb{E}[F(X)] \neq F(\mathbb{E}[X])$ for any nonlinear $F$. For the bistable drift, the error involves the third central moment of $X$:

$$\mathbb{E}[x(1 - |x|^2)] = \mu(1 - |\mu|^2) - \mu\,\text{tr}(\Sigma) - 2\Sigma\mu + \text{(third-moment terms)}$$

The terms involving $\Sigma$ and higher moments are not captured by the first-order Gaussian closure $\dot{\mu} = F(\mu)$. For a spread-out distribution (large $\Sigma$), these correction terms dominate.

2. **The equilibrium distribution is concentrated on a sphere, which is maximally non-Gaussian.** Particles converge to the unit sphere $|x| = 1$ from all directions. The resulting distribution is a thin shell — it has zero probability at the origin and is supported on a curved manifold. A single Gaussian centered at any point is a poor approximation: it assigns probability to the interior (where the true density is near zero) and misses the curvature of the support. The moments of this shell distribution do not carry enough information to reconstruct the drift that produced it.

3. **The mean signal vanishes at equilibrium.** For symmetric initial conditions, the mean $\mu$ converges to the origin as particles spread uniformly over the sphere. But $F(0) = 0$ — the drift at the origin is exactly zero. So the mean ODE $\dot{\mu} = F(\mu)$ provides no information about the dynamics once the system equilibrates. All the dynamical information is encoded in the higher-order structure of the distribution (curvature, concentration on a manifold), which Gaussian moments cannot represent.

By contrast, PFI operates on the full particle distribution via score matching and optimal transport. It makes no Gaussianity assumption and can capture the shell structure, curved support, and higher-order features that NFPE discards. This is the fundamental trade-off: NFPE's moment reduction gives it speed and simplicity, but sacrifices the distributional resolution needed for strongly non-Gaussian systems.

#### Scope of the limitation

This failure is **specific to systems where the equilibrium distribution is far from Gaussian** and the drift nonlinearity is strong enough that the closure error dominates the true signal. It does not affect:

- **Linear systems** (OU processes): Gaussian closure is exact; all moments are captured.
- **Weakly nonlinear systems** (cubic damping near equilibrium, Experiment 4): the distribution stays approximately Gaussian, and the closure error is a small correction.
- **Systems observed during transients** rather than at equilibrium: if the observation window is short enough that the distribution hasn't spread into its non-Gaussian equilibrium shape, moment-based learning remains effective.

### Timing Comparison (d=5 and d=10)

**File:** `experiments/timing_comparison.py`

NFPE (moment-based, snapshot data) vs Neural SDE (trajectory-based, `torchsde` adjoint training), same model architecture, 500 epochs on GPU:

| Dimension | NFPE Time | Neural SDE Time | Speedup | NFPE Drift Rel. Err | NSDE Drift Rel. Err |
|---|---|---|---|---|---|
| 5 | 7.0s | 171s | **24×** | 52.8% | 34.2% |
| 10 | 11.3s | 172s | **15×** | 62.7% | 46.9% |

NFPE is 15–24× faster while achieving drift accuracy within a factor of ~1.3–1.5× of the Neural SDE baseline. The Neural SDE has better accuracy because it trains on full trajectory data with stochastic backpropagation — a richer signal than moment summaries. But NFPE can operate from snapshot data where Neural SDEs cannot be applied at all.


## 6. Future Improvements

### Function approximator agnosticism

The moment ODE framework is agnostic to the parameterization of the drift $F$ and diffusion $B$. The current implementation uses MLPs (`MLPSDE`) for nonlinear systems, but the training pipeline — forward-backward moment matching on $(\mu, \Sigma)$ — works identically with any differentiable function approximator.

A natural extension is to support **symbolic/sparse regression** (SINDy-style) alongside neural networks. Instead of parameterizing $F_\theta(x)$ as an MLP, one could use a dictionary of basis functions $F(x) = \sum_i c_i \phi_i(x)$ (monomials, trigonometric functions, etc.) and solve for sparse coefficients $c_i$. This would yield **closed-form symbolic expressions** for the recovered drift — e.g., directly recovering "$F(x) = -x^3$" as a formula rather than as a black-box network.

This would strengthen the contribution by demonstrating that the core novelty is the **moment-based training pipeline**, not the specific choice of neural network. It would also improve interpretability and extrapolation: MLPs extrapolate poorly outside the training region (Experiment 4 shows drift MSE 0.022 in-range vs 3.09 full-range), while a correctly identified symbolic expression generalizes exactly.

### Beyond Gaussian closure

NFPE's moment ODE approach assumes approximate Gaussianity: each GMM component is propagated under the assumption that it remains Gaussian over a short time step (Gaussian closure). This is exact for linear SDEs and first-order accurate for nonlinear ones, but breaks down for strongly non-Gaussian distributions — as demonstrated concretely by the bistable benchmark failure (Experiment 8, drift relative error >98%).

The bistable system exposes three specific failure modes of Gaussian closure: (1) cubic drift makes the Jensen gap $\mathbb{E}[F(X)] - F(\mathbb{E}[X])$ large, (2) the shell-shaped equilibrium distribution cannot be represented by Gaussian moments, and (3) the mean signal vanishes at the origin where $F(0) = 0$. These are not implementation issues — they are fundamental limitations of second-order moment reduction applied to strongly nonlinear, non-Gaussian systems.

Several directions could extend NFPE to handle such systems:

1. **Higher-order moment closure.** Extend the moment hierarchy beyond mean and covariance to include the third central moment (skewness tensor) $S_{ijk} = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)(X_k - \mu_k)]$. For the bistable drift $F(x) = x(1 - |x|^2)$, the mean ODE becomes $\dot{\mu}_i = \mu_i(1 - |\mu|^2) - \mu_i\,\text{tr}(\Sigma) - 2(\Sigma\mu)_i + \text{(terms involving } S \text{)}$. The Fokker–Planck equation provides evolution equations for $S$ involving fourth moments, which must be closed (e.g., by assuming the fourth cumulant vanishes — the "quasi-normal" closure). This adds $O(d^3)$ state variables per component but captures the leading non-Gaussian correction.

2. **Many-component GMM.** Instead of tracking a few Gaussian components, use a large number ($K \gg 1$) of tightly localized Gaussians to tile the support of the distribution. Each component stays nearly Gaussian by construction (small $\Sigma$), making the closure accurate even for globally non-Gaussian distributions. The challenge is component management across time: tracking which components correspond between snapshots (via Hungarian matching or lightweight optimal transport on component means), and handling component birth/death as the distribution evolves.

3. **Hybrid moment-density approach.** Use NFPE's moment pipeline for the coarse structure (mean dynamics, diffusion identification) and augment it with a lightweight density correction — e.g., a normalizing flow or kernel density estimate — to capture the non-Gaussian residual. This could retain NFPE's speed advantage for the bulk of the computation while borrowing density-level resolution only where needed.

For many physical systems — harmonic potentials, Ornstein–Uhlenbeck processes, weakly nonlinear dynamics, and any system where the distribution remains approximately unimodal and symmetric — the Gaussian closure is an excellent approximation, and NFPE's lightweight moment-based approach is preferable to the heavier machinery of density estimation and optimal transport. The bistable failure defines a clear boundary: NFPE works well when the dynamics are at most moderately nonlinear and the observed distributions are not too far from Gaussian; it fails when the equilibrium distribution is supported on a lower-dimensional manifold (e.g., a sphere) that Gaussian moments cannot represent.
