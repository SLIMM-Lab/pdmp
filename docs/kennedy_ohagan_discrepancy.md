# Kennedy–O'Hagan Model Discrepancy

## Overview

The standard Bayesian calibration in this codebase assumes

$$y^{\text{obs}}(\mathbf{x}_j) = \eta(\mathbf{x}_j, \theta) + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, \sigma^2_\varepsilon)$$

where $\eta$ is the forward (simulator) model and $\theta$ are the calibration parameters. The **Kennedy–O'Hagan (K&O) extension** adds an explicit model discrepancy term that accounts for systematic structural bias in the simulator:

$$y^{\text{obs}}(\mathbf{x}_j) = \eta(\mathbf{x}_j, \theta) + \delta(\mathbf{x}_j) + \varepsilon_j$$

where $\delta(\mathbf{x})$ is a zero-mean Gaussian process with covariance $\sigma^2_\delta C_\delta(\rho_\delta)$. Marginalising $\delta$ analytically gives a multivariate Gaussian likelihood:

$$p(\mathbf{y} \mid \theta, \psi) = \mathcal{N}\!\left(\eta(\theta),\; \Sigma(\psi)\right), \qquad \Sigma = \sigma^2_\delta C_\delta(\rho) + \sigma^2_\varepsilon I$$

The hyperparameter vector $\psi = (\sigma^2_\delta, \sigma^2_\varepsilon, \rho_\delta)$ is sampled jointly with $\theta$.

### Multi-geometry (per-group) extension

When observations come from $G$ distinct measurement campaigns (e.g. different microstructure realizations), the discrepancy at the same spatial point in geometry $g$ and geometry $g'$ is physically unrelated — both are independent draws from the same prior GP, not a single shared realization. Using a single pooled covariance matrix in that setting causes the length-scale $\rho$ to collapse toward a nugget (see `JOINT_KO_DISCUSSION.md`).

The correct model assigns **independent GP realizations** $\delta_g$ per group while **sharing the hyperparameters** $\psi$. After marginalising all $\delta_g$, the likelihood covariance is block-diagonal over (output component $d$, geometry $g$):

$$\Sigma = \operatorname{block-diag}_{d=1,\ldots,D;\; g=1,\ldots,G}\!\bigl(\sigma^2_\delta\, C_g(\rho) + \sigma^2_\varepsilon\, I_P\bigr)$$

where $C_g(\rho)$ is the RBF kernel evaluated at the $P$ sensor coordinates of group $g$ (which may differ across groups). The parameter space remains $\psi = (\sigma^2_\delta, \sigma^2_\varepsilon, \rho)$ — three scalar hyperparameters shared across all $D \cdot G$ blocks. Enable this with `n_groups=G` in the constructor.

## Comparison with standard calibration

| Aspect | Standard | K&O (single group) | K&O (G groups) |
|---|---|---|---|
| Likelihood covariance | $\sigma^2_\varepsilon I$ | $\sigma^2_\delta C(\rho) + \sigma^2_\varepsilon I$ | block-diag, one $P \times P$ block per $(d,g)$ |
| Sampled parameters | $\theta$ | $(\theta,\, \psi)$ | $(\theta,\, \psi)$ — same three hyperparameters |
| Cholesky cost | $O(m)$ | $O(m^3)$ | $O(D \cdot G \cdot P^3)$ — much cheaper when $P \ll m$ |
| Config name | `GaussianLikelihood` | `KOGaussianLikelihood` | `KOGaussianLikelihood` with `n_groups: G` |

## Kernel

The discrepancy GP uses an ARD squared-exponential (RBF) kernel:

$$C_{ij} = \exp\!\left(-\sum_{k=1}^{d_x} \rho_k\,(x_{ik} - x_{jk})^2\right)$$

where $\rho_k > 0$ are inverse squared length-scale parameters, one per input dimension ($d_x$). Larger $\rho_k$ means faster spatial decay in dimension $k$.

## Parameter space and log-transform

All hyperparameters are positive. The sampler works in the unconstrained log-space:

$$\tilde\psi = \bigl(\log\sigma^2_\delta,\; \log\sigma^2_\varepsilon,\; \log\rho_1,\; \ldots,\; \log\rho_{d_x}\bigr)$$

The full sampler state is therefore

$$\mathbf{z} = [\theta_1, \ldots, \theta_p,\; \log\sigma^2_\delta,\; \log\sigma^2_\varepsilon,\; \log\rho_1, \ldots, \log\rho_{d_x}]$$

Priors are specified **directly on log-space** quantities (i.e. a `MultivariateNormal` prior on $\log\sigma^2_\delta$ is equivalent to a log-normal prior on $\sigma^2_\delta$). No Jacobian correction is required.

## YAML configuration

Set `name: KOGaussianLikelihood` in the likelihood block and add a `psi_prior` sub-block. The `x_locs` key is optional for models with a `x_obs_` attribute (e.g. `PiecewiseConstantModel`).

```yaml
problem:
  name: BayesianInverse
  prior:
    name: MultivariateNormal
    mean: [1.5, 2.5]               # prior on theta
    cov: [[4.0, 0.0], [0.0, 4.0]]
  likelihood:
    name: KOGaussianLikelihood
    observation_file: observations.dat
    # x_locs is optional; inferred from model.x_obs_ when omitted
    # x_locs: [0.1, 0.2, ..., 1.0]
    psi_prior:
      name: MultivariateNormal
      # entries: [log(sigma2_delta), log(sigma2_eps), log(rho_1), ...]
      mean: [-4.0, -6.0, 1.5]
      cov: [[4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0]]
  model:
    name: PiecewiseConstant
    F: [1.0]
    dim: 2
    n_obs_loc: 10

sampler:
  name: RandomWalkMetropolis
  n_samples: 100000
  # x_0 has length n_theta + n_psi = 2 + 3 = 5
  x_0: [1.5, 2.5, -4.0, -6.0, 1.5]
  sigma: 0.2

output:
  dir: ./results
  logging:
    level: INFO
    log_file: inference.log

seed: 42
```

### Multi-geometry YAML example

For joint inference over G=10 microstructure realizations with 3 displacement components and P=10 sensors per geometry:

```yaml
  likelihood:
    name: KOGaussianLikelihood
    observation_file: observations.dat   # shape (1, 3*10*10) = (1, 300)
    n_components: 3                      # ux, uy, uz treated independently
    n_groups: 10                         # one independent δ_g per geometry
    # x_locs shape: (n_groups * P, 3) = (100, 3), geom-major order
    # x_locs: loaded separately and passed via Python API (see below)
    psi_prior:
      name: MultivariateNormal
      # same three hyperparameters — shared across all 30 blocks
      mean: [-4.0, -9.2, 1.5]
      cov: [[4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 4.0]]
    # Optionally fix a hyperparameter that the data cannot identify:
    # fixed_psi:
    #   log_s2e: -9.2   # = 2 * log(0.01), fixes noise to sigma_eps = 0.01
```

When `fixed_psi` is used the fixed entries are removed from the sampler's parameter vector; `x_0` and the `psi_prior` must cover only the **free** parameters.

### `x_locs` resolution

| Situation | How `x_locs` is obtained |
|-----------|--------------------------|
| `x_locs` key present in config | Used directly |
| Model has `x_obs_` attribute (`PiecewiseConstantModel`) | Taken from `model.x_obs_` |
| Neither | `ValueError` at construction |

For `JaxFemModel` with 3D sensors, pass `x_locs` explicitly as an `(m, 3)` array of sensor coordinates. Each sensor point is treated as one location; the kernel is built over physical coordinates, not DOFs.

For `n_groups > 1` the array must have shape `(n_groups * P, d_x)` in **geom-major order**: rows `[g*P : (g+1)*P]` are the P sensor coordinates for group g. The total observation length must satisfy `m = n_components * n_groups * P`.

### `psi_prior` mean: practical guidance

| Parameter | Meaning | Suggested starting point |
|---|---|---|
| $\log\sigma^2_\delta$ | log discrepancy variance | set so $\sigma_\delta \sim$ expected simulator bias magnitude |
| $\log\sigma^2_\varepsilon$ | log noise variance | set to $2\log(\text{instrument noise std})$ |
| $\log\rho_k$ | log inverse length-scale | set so $1/\sqrt{\rho_k} \sim$ spatial length scale at which discrepancy varies |

Use a large prior variance (e.g. 4.0) to remain weakly informative.

## Python API

### Kernel utilities (`pdmp.discrepancy`)

```python
from pdmp.discrepancy import (
    rbf_kernel_matrix,
    rbf_kernel_matrix_drho,
    build_noise_covariance,
    KOGaussianLikelihood,
)
```

**`rbf_kernel_matrix(x_locs, rho)`**

Build the $m \times m$ kernel matrix.

```python
import numpy as np
x = np.linspace(0, 1, 5).reshape(-1, 1)  # (5, 1) sensor locations
rho = np.array([3.0])                      # inverse squared length-scale
C = rbf_kernel_matrix(x, rho)             # (5, 5) matrix
```

**`build_noise_covariance(x_locs, rho, sigma2_delta, sigma2_eps)`**

Assemble $\Sigma = \sigma^2_\delta C_\delta + \sigma^2_\varepsilon I$.

```python
Sigma = build_noise_covariance(x, rho, sigma2_delta=0.01, sigma2_eps=0.001)
```

**`KOGaussianLikelihood(model, u_obs, x_locs, psi_prior, n_components, n_groups, kernel, fixed_psi)`**

Single-geometry (baseline) usage:

```python
from pdmp.distributions import MultivariateNormal, JointDistribution, Posterior
from pdmp.forward_model import PiecewiseConstantModel

model   = PiecewiseConstantModel(F=np.array([1.0]), n_params=2,
                                 x_obs=np.linspace(0.1, 1.0, 10))
u_obs   = np.loadtxt("observations.dat").reshape(1, -1)  # (1, 10)
x_locs  = model.x_obs_                                    # (10,) → reshaped internally

psi_prior = MultivariateNormal(mean=np.array([-4., -6., 1.5]),
                               cov=4.0 * np.eye(3))
ko_lik = KOGaussianLikelihood(model=model, u_obs=u_obs,
                              x_locs=x_locs, psi_prior=psi_prior)

# Evaluate at [theta_1, theta_2, log_s2d, log_s2e, log_rho]
params = np.array([2.0, 3.0, -4.6, -6.9, 1.6])
ll      = ko_lik.log_density(params)
grad_ll = ko_lik.grad_log_density(params)   # shape (5,)
```

Multi-geometry usage (G=10 groups, D=3 components, P=10 sensors per geometry):

```python
G, D, P = 10, 3, 10

# u_obs shape: (1, D*G*P) = (1, 300), layout [all ux, all uy, all uz], geom-major within each DOF
u_obs = np.loadtxt("observations.dat").reshape(1, -1)

# x_locs shape: (G*P, 3) in geom-major order: rows [g*P:(g+1)*P] are group g's sensors
x_locs = np.vstack([sensor_coords_per_geom[g] for g in range(G)])  # (100, 3)

psi_prior = MultivariateNormal(mean=np.array([-4., -9.2, 1.5]),
                               cov=np.diag([4.0, 1.0, 4.0]))
ko_lik = KOGaussianLikelihood(
    model=model,
    u_obs=u_obs,
    x_locs=x_locs,
    psi_prior=psi_prior,
    n_components=D,
    n_groups=G,
)

# params = [theta ..., log_s2d, log_s2e, log_rho]
ll      = ko_lik.log_density(params)
grad_ll = ko_lik.grad_log_density(params)
```

To fix a hyperparameter (e.g. noise level is known):

```python
ko_lik = KOGaussianLikelihood(
    ...,
    fixed_psi={"log_s2e": 2 * np.log(0.01)},  # fixes sigma_eps = 0.01
    psi_prior=MultivariateNormal(mean=np.array([-4., 1.5]),
                                 cov=4.0 * np.eye(2)),  # only log_s2d and log_rho
)
# params = [theta ..., log_s2d, log_rho]  — log_s2e absent from vector
```

### Assembling the posterior manually

`get_target()` handles this automatically from YAML, but you can also build it by hand:

```python
prior_theta = MultivariateNormal(mean=np.array([1.5, 2.5]),
                                 cov=4.0 * np.eye(2))
joint_prior = JointDistribution([prior_theta, psi_prior])   # dim = 2 + 3 = 5
posterior   = Posterior(prior=joint_prior, likelihood=ko_lik)

posterior.dim          # 5
posterior.log_density(params)         # scalar
posterior.grad_log_density(params)    # (5,) gradient
```

## Posterior outputs

After sampling the chain contains rows of shape `(n_theta + n_psi,)`.

```python
samples = np.loadtxt("results/samples.dat")  # (N, 5)
theta_samples = samples[:, :2]               # theta marginal — no reweighting needed
psi_samples   = samples[:, 2:]               # [log_s2d, log_s2e, log_rho] marginal

# Convert back to original scale
sigma2_delta_samples = np.exp(psi_samples[:, 0])
sigma2_eps_samples   = np.exp(psi_samples[:, 1])
rho_samples          = np.exp(psi_samples[:, 2])
```

Check that $\sigma^2_\delta$ is not absorbing all signal relative to $\sigma^2_\varepsilon$ — if its posterior concentrates far above the noise floor, the model discrepancy and calibration parameters may be poorly identified for the given sensor layout.

## Working example

A complete runnable example is provided in `examples/kennedy_ohagan/test_piecewise/`:

```
examples/kennedy_ohagan/test_piecewise/
├── generate_data.py   # synthesises observations with GP discrepancy
├── config.yaml        # YAML config for K&O RWM calibration
├── run.sh             # convenience runner
└── analyze_results.py # trace plots and marginal posteriors
```

Run it from the repository root:

```bash
cd examples/kennedy_ohagan/test_piecewise
python generate_data.py   # writes observations.dat, ground_truth.dat
python ../../../run_inference.py --config config.yaml
python analyze_results.py # writes results/traces.pdf etc.
```

The example uses a 1D bar (`PiecewiseConstantModel`) with `n_params=2` and 10 sensors. The true parameters are $\theta = [2, 3]$, $\sigma^2_\delta = 0.01$, $\sigma^2_\varepsilon = 0.001$, $\rho = 5$.

## Implementation notes

- **Numerical stability**: each per-group block $\Sigma_g$ is factored via `_safe_cholesky` (with automatic jitter escalation and eigenvalue fallback), the same utility used by `MultivariateNormal`.
- **Gradient computation**: gradients w.r.t. $\theta$ use the model's VJP path (`linearize` → `eval_vjp` → `eval_grad` in priority order, matching `GaussianLikelihood`). Gradients w.r.t. $\tilde\psi$ are computed analytically using the trace identity $\operatorname{tr}(A B) = \sum_{ij} A_{ij} B_{ji}$ applied per-group and summed.
- **Hessian**: finite differences of `grad_log_density`, consistent with `GaussianLikelihood`.
- **Fixed-psi parameters**: removed from the sampler's parameter vector. The `psi_prior` and `x_0` must be dimensioned for only the **free** entries. Valid names for `fixed_psi`: `"log_s2d"`, `"log_s2e"`, `"log_rho"` (1-D or isotropic), or `"log_rho_0"`, `"log_rho_1"`, … (ARD).
- **Backwards compatibility**: `n_groups=1` (default) recovers the original single-geometry behavior exactly. `GaussianLikelihood` and all existing configs work unchanged.
- **σ²_ε identifiability with many groups**: when the inferred length-scale is comparable to the domain size, the per-group GP can absorb i.i.d. noise as a smooth component, causing σ²_ε to under-estimate the true noise. Mitigations: tighten the prior on `log_s2e`, use `fixed_psi` to pin it to the known noise level, increase `--n-sensors`, or use a shorter-scale prior on `log_rho`. See `JOINT_KO_DISCUSSION.md` for a detailed diagnosis.

## See also

- `pdmp/discrepancy.py` — kernel functions and `KOGaussianLikelihood` implementation
- `tests/test_discrepancy.py` — unit tests including gradient checks against finite differences
- `tests/test_grouped_ko_recovery.py` — MAP recovery test for the multi-geometry likelihood
- `examples/inverse_problem/itz/itz_noise_low/JOINT_KO_DISCUSSION.md` — motivation, pathology analysis, and post-implementation observations for the per-group extension
- `ko_calibration_impl.md` — design notes and statistical derivation
