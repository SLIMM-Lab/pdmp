# Kennedy–O'Hagan Model Discrepancy

## Overview

The standard Bayesian calibration in this codebase assumes

$$y^{\text{obs}}(\mathbf{x}_j) = \eta(\mathbf{x}_j, \theta) + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, \sigma^2_\varepsilon)$$

where $\eta$ is the forward (simulator) model and $\theta$ are the calibration parameters. The **Kennedy–O'Hagan (K&O) extension** adds an explicit model discrepancy term that accounts for systematic structural bias in the simulator:

$$y^{\text{obs}}(\mathbf{x}_j) = \eta(\mathbf{x}_j, \theta) + \delta(\mathbf{x}_j) + \varepsilon_j$$

where $\delta(\mathbf{x})$ is a zero-mean Gaussian process with covariance $\sigma^2_\delta C_\delta(\rho_\delta)$. Marginalising $\delta$ analytically gives a multivariate Gaussian likelihood:

$$p(\mathbf{y} \mid \theta, \psi) = \mathcal{N}\!\left(\eta(\theta),\; \Sigma(\psi)\right), \qquad \Sigma = \sigma^2_\delta C_\delta(\rho) + \sigma^2_\varepsilon I$$

The hyperparameter vector $\psi = (\sigma^2_\delta, \sigma^2_\varepsilon, \rho_\delta)$ is sampled jointly with $\theta$.

## Comparison with standard calibration

| Aspect | Standard | K&O |
|---|---|---|
| Likelihood covariance | $\sigma^2_\varepsilon I$ | $\sigma^2_\delta C_\delta(\rho) + \sigma^2_\varepsilon I$ |
| Sampled parameters | $\theta$ | $(\theta,\, \psi)$ |
| Log-likelihood cost | $O(m)$ | $O(m^3)$ via Cholesky ($m$ = sensor count) |
| Config name | `GaussianLikelihood` | `KOGaussianLikelihood` |

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

### `x_locs` resolution

| Situation | How `x_locs` is obtained |
|-----------|--------------------------|
| `x_locs` key present in config | Used directly |
| Model has `x_obs_` attribute (`PiecewiseConstantModel`) | Taken from `model.x_obs_` |
| Neither | `ValueError` at construction |

For `JaxFemModel` with 3D sensors, pass `x_locs` explicitly as an `(m, 3)` array of sensor coordinates. Each sensor point is treated as one location; the kernel is built over physical coordinates, not DOFs.

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

**`KOGaussianLikelihood(model, u_obs, x_locs, psi_prior)`**

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

- **Numerical stability**: the kernel covariance $\Sigma$ is factored via `_safe_cholesky` (with automatic jitter escalation and eigenvalue fallback), the same utility used by `MultivariateNormal`.
- **Gradient computation**: gradients w.r.t. $\theta$ use the model's VJP path (`linearize` → `eval_vjp` → `eval_grad` in priority order, matching `GaussianLikelihood`). Gradients w.r.t. $\tilde\psi$ are computed analytically using the trace identity $\operatorname{tr}(A B) = \sum_{ij} A_{ij} B_{ji}$.
- **Hessian**: finite differences of `grad_log_density`, consistent with `GaussianLikelihood`.
- **Backwards compatibility**: no existing classes are modified. `GaussianLikelihood` and all existing configs work unchanged.

## See also

- `pdmp/discrepancy.py` — kernel functions and `KOGaussianLikelihood` implementation
- `tests/test_discrepancy.py` — unit tests including gradient checks against finite differences
- `ko_calibration_impl.md` — design notes and statistical derivation
