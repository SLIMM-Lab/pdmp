# Moment Matching for Bayesian Inference and Forward UQ

## Overview

Sampling-based Bayesian inference (MCMC / PDMP) and Monte Carlo forward uncertainty quantification are accurate but expensive: each evaluation of a 3-D finite-element forward model can take tens of seconds, and obtaining well-converged statistics requires thousands of evaluations.

Moment matching offers a cheap analytical alternative. Instead of sampling, it represents the distribution of interest as a Gaussian whose first two moments (mean μ and covariance Σ) are computed via deterministic or local-linearisation approximations. The approximation is exact when the underlying distribution is Gaussian, and gives a useful first estimate otherwise.

Two complementary drivers are provided:

| Script | Purpose |
|---|---|
| `run_laplace.py` | Laplace approximation of the posterior p(θ\|data). Finds the MAP and approximates the posterior as N(MAP, −H⁻¹). |
| `forward_uq_moment.py` | Propagates a Gaussian input distribution through a forward model using both the unscented transform (UT) and first-order linearisation. |

---

## Laplace Approximation (`run_laplace.py`)

### Background

The Laplace approximation replaces a posterior p(θ\|data) ∝ exp(log p) by the Gaussian centred at the MAP estimate θ* with covariance equal to the negative inverse Hessian of the log-posterior evaluated there:

```
θ*   = arg max  log p(θ | data)          (BFGS optimisation)
Σ_L  = −[∇² log p(θ*)]⁻¹                (curvature at the MAP)
```

The approximation is exact when log p is a quadratic (i.e., the posterior is Gaussian), and provides a useful first estimate for near-Gaussian posteriors. For strongly non-Gaussian posteriors (heavy tails, multimodality) it will underestimate spread in non-quadratic directions.

### Usage

The script accepts the same YAML config as `run_inference.py`. An optional `laplace:` block controls how many synthetic samples are drawn from the Gaussian approximation:

```bash
python run_laplace.py --config examples/inverse_problem/itz/itz_noise_low/joint/rwm/config.yaml
```

### Configuration

```yaml
problem:
  name: Transformed             # outer Affine wrapper — stripped automatically
  transformation: Affine
  distribution:
    name: BayesianInverse       # the actual posterior is built from this
    prior: { ... }
    likelihood: { ... }
    model: { ... }

# Optional: override Laplace-specific settings.
laplace:
  n_samples: 10000              # draws written to samples.dat (default 10 000)
  x_0: [0, 0, 0, 0, 0, 0]      # BFGS starting point (default: sampler.x_0
                                #   if present, else zeros)

output:
  dir: .                        # all output files go here
  logging:
    log_file: inference.log
    level: INFO

seed: 0
```

The `laplace:` block is entirely optional. If omitted, `n_samples` defaults to 10 000 and `x_0` is read from `sampler.x_0` (if the config has a `sampler:` block) or set to zeros.

### Output Files

| File | Shape | Description |
|---|---|---|
| `mean.dat` | (d,) | MAP estimate in natural [θ, ψ] coordinates |
| `cov.dat` | (d, d) | Laplace covariance −H⁻¹ at the MAP |
| `samples.dat` | (n\_samples, d) | Synthetic draws from N(mean, cov); layout matches `run_inference.py` output for direct use by `analyze_results.py` |
| `config_used.yaml` | — | Copy of the resolved configuration |

### Notes on the Outer `Transformed` Wrapper

The inference configs for the ITZ problem wrap the inner `BayesianInverse` posterior in an outer `Transformed: Affine` distribution. This Affine map is built automatically from `find_mean` / `find_curvature` of the inner posterior — it is itself a Laplace approximation used to whiten the sampling space for MCMC efficiency.

If `run_laplace.py` operated on this outer target, it would find a MAP ≈ 0 and a Hessian ≈ −I (trivially, because the whitening was built from the Laplace). The output would be the uninformative N(0, I) regardless of the actual posterior shape.

The driver therefore **strips all outer `Transformed` wrappers** before running BFGS and curvature estimation. The result is the non-trivial Laplace approximation of the actual BayesianInverse posterior in natural [θ, ψ] coordinates. This output is directly consumable by `forward_uq_moment.py`.

---

## Moment-Matched Forward UQ (`forward_uq_moment.py`)

### Background

Given a Gaussian approximation N(μ, Σ) of the input (posterior over field parameters), the goal is to characterise the distribution of output quantities Q = f(θ) — e.g., RVE stresses and strains — without running thousands of forward model evaluations.

Two approximations are computed and compared in a single run:

1. **Unscented transform (UT)**: deterministic sigma points placed at strategic locations around μ; exact to 3rd order for Gaussian inputs on polynomial models.
2. **First-order linearisation (Gauss propagation)**: linearise f at μ, then propagate: μ_Q ≈ f(μ), Σ_Q ≈ J Σ J^T.

Running both simultaneously makes the difference between them a diagnostic: if UT and lin agree, the map is locally linear; if they diverge (especially on variance), the nonlinearity is significant and the linearisation should not be trusted.

### Unscented Transform

The scaled UT (Wan & van der Merwe, 2000) places 2d+1 sigma points around the input mean along the principal axes of the input covariance. For a d-dimensional input:

```
λ = α² (d + κ) − d
σ-points:    x₀ = μ
             xᵢ = μ + √(d+λ) · L[:, i]     i = 1…d    (L = chol(Σ))
             xᵢ = μ − √(d+λ) · L[:, i−d]   i = d+1…2d

Mean weights:    w₀ᵐ = λ/(d+λ),   wᵢᵐ = 1/(2(d+λ))
Cov weights:     w₀ᶜ = w₀ᵐ + (1 − α² + β),   wᵢᶜ = wᵢᵐ

μ_Q ≈ Σ wᵢᵐ f(xᵢ)
Σ_Q ≈ Σ wᵢᶜ (f(xᵢ) − μ_Q)(f(xᵢ) − μ_Q)ᵀ
```

The default parameters are `alpha=1.0, beta=2.0, kappa=0.0`, which places sigma points at ±√d σ along each axis — wide enough to probe genuine nonlinearity in outputs like `max_von_mises`. Setting `alpha` small (e.g. 1e-3) collapses the sigma points onto the mean and makes UT indistinguishable from a finite-difference Jacobian.

### First-Order Linearisation

The Jacobian J = ∂f/∂θ is estimated by **central finite differences** with step size `h` (default 1e-3):

```
J[:, i] ≈ (f(μ + h eᵢ) − f(μ − h eᵢ)) / (2h)
```

JAX autodiff is not used because the RVE model breaks the JAX trace at its output boundary (`float()` and `np.array()` casts in `_assemble_outputs`). Finite differences require 2d forward evaluations (6 for d=3), the same order as the UT (7 evaluations), and follow the precedent of `JaxFemModel.eval_hessian` which also uses FD.

Linearisation fails for **non-smooth outputs** like `max_von_mises` and `max_strain`. The argmax element in the mesh can switch under finite parameter perturbations, making the Jacobian path-dependent and the resulting Σ_Q unreliable (typically too small). This failure mode is expected and is one reason for including the UT alongside.

### Usage

```bash
python forward_uq_moment.py examples/forward_uq/rve_joint/config_joint_moment.yaml
```

### Configuration

```yaml
model:
  name: RVE
  fibers: [ ... ]
  quantities: [avg_stress, max_von_mises, max_stress, max_strain]
  components: [xx, yy]
  # ... remaining RVE settings ...

forward_uq_moment:
  # Directory produced by run_laplace.py — must contain mean.dat and cov.dat.
  posterior_dir: ../../inverse_problem/itz/itz_noise_low/joint/rwm

  # Inference config used to generate the posterior. Only the likelihood
  # transformation block is read (to reconstruct latent → physical map).
  # No forward model is loaded from this config.
  inference_config: ../../inverse_problem/itz/itz_noise_low/joint/rwm/config.yaml

  # Indices into the natural [θ, ψ] vector that are forwarded to the model.
  # For the ITZ case: [0]=rho, [1]=l, [2]=E_inf. Indices [3,4,5] are KO
  # discrepancy hyperparameters that the RVE does not use.
  param_indices: [0, 1, 2]

  ut:
    alpha: 1.0     # sigma-point spread; 1.0 → points at ±√d σ
    beta:  2.0     # optimal for Gaussian inputs
    kappa: 0.0

  # Central FD step size in natural latent space. The effective physical-space
  # step is scaled by the Jacobian of the transformation at the MAP.
  fd_step: 1.0e-3

  # Synthetic samples drawn from each moment-matched output Gaussian.
  # Written as samples_ut.dat and samples_lin.dat for use with the
  # existing forward_uq.py plotting utilities.
  n_synthetic_samples: 2000
  seed: 0

output:
  dir: results_moment
```

### Output Files

| File | Description |
|---|---|
| `mean_ut.dat` | UT-estimated output mean, shape (m,) |
| `cov_ut.dat` | UT-estimated output covariance, shape (m, m) |
| `mean_lin.dat` | Linearisation-estimated output mean, shape (m,) |
| `cov_lin.dat` | Linearisation-estimated output covariance, shape (m, m) |
| `samples_ut.dat` | Synthetic draws from N(mean\_ut, cov\_ut), shape (n\_syn, m) |
| `samples_lin.dat` | Synthetic draws from N(mean\_lin, cov\_lin), shape (n\_syn, m) |
| `sigma_points.dat` | UT sigma points in latent space, shape (2d+1, d) |
| `sigma_outputs.dat` | Forward model output at each sigma point, shape (2d+1, m) |
| `jacobian.dat` | FD Jacobian ∂f/∂θ at the MAP, shape (m, d) |
| `outputs_legend.txt` | Column labels for output arrays |

### When UT and Linearisation Agree vs. Diverge

| Output type | Expected agreement | Reason |
|---|---|---|
| `avg_stress` (linear in E) | UT ≈ lin | Map is nearly affine; variance is quadratic in J |
| `max_von_mises` (quadratic in σ) | UT > lin variance | Quadratic term omitted by lin; UT captures it |
| `max_stress`, `max_strain` | UT ≫ lin variance | Piecewise linear (kinks); lin uses local slope only |

If UT and lin agree on all quantities, the latent-to-output map is well-approximated as linear over the support of the posterior — which is itself a useful finding, validating the linearisation-based approach for that posterior.

---

## Integration with the ITZ Test Case

### Workflow

```
1. Generate measurements (once):
   python examples/inverse_problem/itz/itz_noise_low/generate_measurements.py

2. Run Laplace inference (joint):
   python run_laplace.py \
     --config examples/inverse_problem/itz/itz_noise_low/joint/rwm/config.yaml
   → writes  joint/rwm/mean.dat, joint/rwm/cov.dat, joint/rwm/samples.dat

3. Run Laplace inference (separate, per geometry):
   for i in 01 02 … 10; do
     python run_laplace.py \
       --config examples/inverse_problem/itz/itz_noise_low/separate/$i/rwm/config.yaml
   done
   → writes  separate/NN/rwm/mean.dat, cov.dat, samples.dat

4. Run MC forward UQ (reference):
   python forward_uq.py examples/forward_uq/rve_joint/config_joint.yaml
   → writes  rve_joint/results/outputs.dat, figures/

5. Run moment-matched forward UQ (joint Laplace):
   python forward_uq_moment.py \
     examples/forward_uq/rve_joint/config_joint_moment.yaml
   → writes  rve_joint/results_moment/{mean,cov,samples}_{ut,lin}.dat
```

### Example Config

The ready-to-use config for step 5 is at:

```
examples/forward_uq/rve_joint/config_joint_moment.yaml
```

---

## Verification

### Sanity Checks

The moment-matching functions satisfy the following properties, which can be checked analytically:

**1. Linear model — both methods exact:**

```python
from forward_uq_moment import unscented_sigma_points, fd_jacobian
import numpy as np

A = np.random.randn(4, 3)
b = np.random.randn(4)
f = lambda x: A @ x + b

mu    = np.array([0.5, 1.0, -0.3])
Sigma = np.diag([0.3, 0.2, 0.5])

pts, wm, wc = unscented_sigma_points(mu, Sigma)
outs = np.array([f(p) for p in pts])
mu_ut  = (wm[:, None] * outs).sum(axis=0)

# Should be zero to machine precision:
print(np.max(np.abs(mu_ut - (A @ mu + b))))   # ≈ 1e-15
```

**2. Quadratic model — UT corrects bias, lin does not:**

For f(x) = x², E[f(X)] = μ² + σ² (bias = σ²). UT captures this; linearisation returns μ² only.

```python
f2    = lambda x: x ** 2
mu1d  = np.array([1.0])
Sig1d = np.array([[0.25]])

pts, wm, wc = unscented_sigma_points(mu1d, Sig1d)
mu_ut = sum(wm[i] * f2(pts[i]) for i in range(3))
print(mu_ut)   # ≈ 1.25  (= 1.0² + 0.25, exact)

J, f0 = fd_jacobian(f2, mu1d)
print(f0)      # ≈ 1.0  (= mu², biased by -sigma²)
```

**3. Max function — diverge significantly:**

`f(x) = max(x[0], x[1])` has a kink along the diagonal. The FD Jacobian depends on which argument is larger at the evaluation point; the UT captures the full distribution through the kink.

### Comparison with MCMC Results

When the posterior is **nearly Gaussian** (small noise, mildly nonlinear model):
- `run_laplace.py` `samples.dat` and RWM `samples.dat` should show similar marginal histograms.
- `forward_uq_moment.py` UT and MC `forward_uq.py` results should agree on mean and variance for all quantities.

When the posterior is **non-Gaussian** (strong nonlinearity, high noise, or multi-modal likelihood):
- RWM `samples.dat` will show asymmetric or heavy-tailed marginals that differ from the Laplace Gaussian.
- UT output moments may still approximate MC forward UQ reasonably (UT is 3rd-order accurate for smooth f), but `cov_ut.dat` will underestimate output variance for strongly non-Gaussian inputs.
- `cov_lin.dat` will be unreliable for any non-smooth output quantity.

The three-way comparison — MC forward UQ / UT / linearisation — thus simultaneously reveals approximation error at two levels: the Laplace inference approximation and the moment-propagation approximation.
