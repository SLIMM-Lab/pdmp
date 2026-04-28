#!/usr/bin/env python3
"""
Smoke-test verifying that pdmp + jax-fem work correctly inside the container.

Runs from the /pdmp bind-mount (PYTHONPATH=/pdmp is set by the container).
Mesh generation is done inline by gmsh — no external .msh file needed.

Checks:
  1. Core library imports and versions
  2. JaxFemModel construction (gmsh mesh generation)
  3. Forward evaluation and VJP gradient
  4. Posterior log-density at a single point (one FEM solve)
  5. Short MCMC chain on an analytic Gaussian target (no FEM overhead)
"""

import numpy as np

SEP = "=" * 60


def _ok(msg):
    print(f"  [OK]  {msg}")


def _section(msg):
    print(f"\n{SEP}\n{msg}\n{SEP}")


# ── 1. Imports ────────────────────────────────────────────────────────────────
_section("1. Library imports")

import jax
_ok(f"jax {jax.__version__}")

import jax_fem
_ok(f"jax_fem {getattr(jax_fem, '__version__', 'installed')}")

import pdmp
_ok(f"pdmp {pdmp.__version__}")

# ── 2. JaxFemModel construction ───────────────────────────────────────────────
_section("2. JaxFemModel construction (gmsh mesh generation)")

from pdmp.forward_model import JaxFemModel

# h=0.5 → coarse mesh; the default sensor is at the mid-side face.
model = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, nu=0.3, n_params=1)

n_in = model.get_dim_in()
n_out = model.get_dim_out()
_ok(f"input dim: {n_in},  output dim: {n_out}")
assert n_in == 1
assert n_out >= 1

# ── 3. Forward evaluation and VJP gradient ────────────────────────────────────
_section("3. Forward evaluation and VJP gradient")

# theta is the Young's modulus directly — must be strictly positive
theta_true = np.array([1.3])
y_true = model.eval(theta_true)
_ok(f"model.eval({theta_true}) -> {y_true}")

v = np.ones(n_out)
y_lin, vjp_fn = model.linearize(theta_true)
g = vjp_fn(v)
_ok(f"VJP gradient -> {g}")
assert g.shape == (n_in,)
assert np.all(np.isfinite(g)), "gradient contains non-finite values"

# ── 4. Posterior log-density (single FEM solve) ───────────────────────────────
_section("4. Posterior log-density (single FEM solve)")

from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior

rng = np.random.default_rng(seed=0)
sigma_obs = 0.01
y_obs = (y_true + sigma_obs * rng.standard_normal(y_true.shape)).reshape(1, -1)

# Prior centred well away from zero so MCMC cannot wander to E ≤ 0
prior = MultivariateNormal(mean=np.array([1.3]), cov=np.array([[0.1**2]]))
likelihood = GaussianLikelihood(model, y_obs, sigma_obs)
target = Posterior(prior, likelihood)

# Evaluate at the prior mean — one FEM solve
theta_eval = np.array([1.3])
log_p = target.log_density(theta_eval)
_ok(f"log p(theta={theta_eval} | y) = {log_p:.4f}")
assert np.isfinite(log_p), "log posterior is not finite"

# ── 5. Short MCMC on an analytic target (no FEM overhead) ────────────────────
_section("5. RandomWalkMetropolis on analytic Gaussian (50 samples)")

from pdmp.loader import get_sampler

analytic_target = MultivariateNormal(mean=np.array([0.0]), cov=np.array([[1.0]]))
sampler_cfg = {
    'name': 'RandomWalkMetropolis',
    'sigma': 0.3,
    'x_0': np.array([0.0]),
    'n_samples': 50,
}
sampler = get_sampler(sampler_cfg, target=analytic_target, rng=rng)
sampler.run()
chain = sampler.chain
_ok(f"chain shape: {chain.shape}")
_ok(f"chain mean: {np.mean(chain):.4f}  (expect ≈ 0)")
assert chain.shape[0] == 50

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("All checks passed.")
print(SEP)
