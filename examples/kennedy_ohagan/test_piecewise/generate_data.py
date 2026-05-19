#!/usr/bin/env python3
"""Generate synthetic observations for the K&O piecewise-constant test case
and write sampler configs for both RWM and BPS.

Forward model: 1D bar with 2-segment piecewise-constant Young's modulus.
  theta = [E1, E2]  (Young's modulus in each segment)
  F = [1.0]         (applied force)
  x_obs: 10 sensor locations in (0, 1]

The observation model is:
  y = eta(x, theta_true) + delta(x) + eps

where
  delta(x) ~ GP(0, sigma2_delta * C_delta(rho))   (model discrepancy)
  eps       ~ N(0, sigma2_eps * I)                 (measurement noise)

Outputs (top-level):
  observations.dat   -- (1, 10) observation matrix
  ground_truth.dat   -- [theta_true, psi_true] in log-space for psi

BPS-specific output:
  bps/config.yaml    -- Transformed(Affine) config with precomputed M and b
"""

import os
import numpy as np

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.discrepancy import rbf_kernel_matrix, KOGaussianLikelihood
from pdmp.distributions import (
    MultivariateNormal,
    JointDistribution,
    Posterior,
    find_mean,
    find_curvature,
    get_prior,
    get_likelihood,
)
from pdmp.loader import numpy_to_yaml, dump_yaml_custom_format

# ── ground truth ────────────────────────────────────────────────────────────
THETA_TRUE = np.array([2.0, 5.0])
SIGMA2_DELTA = 0.01
SIGMA2_EPS = 0.001
RHO_TRUE = 5.0

N_OBS_LOC = 10
F = np.array([1.0])
SEED = 5

HERE = os.path.dirname(os.path.abspath(__file__))
BPS_DIR = os.path.join(HERE, "bps")


def _ko_prior():
    """Return (prior_theta, psi_prior) matching both configs."""
    prior_theta = MultivariateNormal(
        mean=np.array([1.5, 2.5]),
        cov=np.diag([4.0, 4.0]),
    )
    psi_prior = MultivariateNormal(
        mean=np.array([-4.0, -6.0, 1.5]),
        cov=4.0 * np.eye(3),
    )
    return prior_theta, psi_prior


def main():
    rng = np.random.default_rng(SEED)

    # ── generate observations ────────────────────────────────────────────────
    x_obs = np.linspace(0.1, 1.0, N_OBS_LOC)
    model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs)

    eta = model.eval(THETA_TRUE, idx=0)

    rho = np.array([RHO_TRUE])
    C_delta = rbf_kernel_matrix(x_obs.reshape(-1, 1), rho)
    L_delta = np.linalg.cholesky(SIGMA2_DELTA * C_delta)
    delta = L_delta @ rng.standard_normal(N_OBS_LOC)
    eps = rng.normal(0.0, np.sqrt(SIGMA2_EPS), N_OBS_LOC)

    y = eta + delta + eps
    u_obs = y.reshape(1, -1)

    obs_path = os.path.join(HERE, "observations.dat")
    gt_path = os.path.join(HERE, "ground_truth.dat")

    np.savetxt(obs_path, u_obs)
    np.savetxt(
        gt_path,
        np.concatenate([
            THETA_TRUE,
            [np.log(SIGMA2_DELTA),
             np.log(SIGMA2_EPS),
             np.log(RHO_TRUE)]
        ]))

    print(f"theta_true  = {THETA_TRUE}")
    print(f"sigma2_delta= {SIGMA2_DELTA}  (log = {np.log(SIGMA2_DELTA):.3f})")
    print(f"sigma2_eps  = {SIGMA2_EPS}  (log = {np.log(SIGMA2_EPS):.3f})")
    print(f"rho         = {RHO_TRUE}  (log = {np.log(RHO_TRUE):.3f})")
    print(f"\neta range   : [{eta.min():.4f}, {eta.max():.4f}]")
    print(f"delta std   : {delta.std():.4f}")
    print(f"obs range   : [{y.min():.4f}, {y.max():.4f}]")
    print(f"Written: {obs_path}")
    print(f"Written: {gt_path}")

    # ── build K&O posterior for MAP / curvature ──────────────────────────────
    print("\nBuilding K&O posterior for MAP / Laplace curvature...")
    prior_theta, psi_prior = _ko_prior()

    ko_lik = KOGaussianLikelihood(
        model=model,
        u_obs=u_obs,
        x_locs=x_obs,
        psi_prior=psi_prior,
    )
    joint_prior = JointDistribution([prior_theta, psi_prior])
    posterior = Posterior(prior=joint_prior, likelihood=ko_lik)

    x_0 = np.array([1.5, 2.5, -4.0, -6.0, 1.5])
    print("  Finding MAP (BFGS)...")
    b = find_mean(posterior, x_0=x_0)
    print(f"  b = {b}")

    print("  Computing Laplace covariance at MAP...")
    M = find_curvature(posterior, b)
    print(f"  M diagonal = {np.diag(M)}")

    # ── write bps/config.yaml ────────────────────────────────────────────────
    os.makedirs(BPS_DIR, exist_ok=True)

    bps_config = {
        "problem": {
            "name": "Transformed",
            "transformation": "Affine",
            "M": M,
            "b": b,
            "distribution": {
                "name": "BayesianInverse",
                "prior": {
                    "name": "MultivariateNormal",
                    "mean": np.array([2.0, 2.0]),
                    "cov": np.diag([4.0, 4.0]),
                },
                "likelihood": {
                    "name": "KOGaussianLikelihood",
                    "observation_file": "../observations.dat",
                    "psi_prior": {
                        "name": "MultivariateNormal",
                        "mean": np.array([-4.0, -6.0, 1.5]),
                        "cov": 4.0 * np.eye(3),
                    },
                },
                "model": {
                    "name": "PiecewiseConstant",
                    "F": np.array([1.0]),
                    "dim": 2,
                    "n_obs_loc": 10,
                },
            },
        },
        "sampler": {
            "name": "BouncyParticle",
            "t_max": 2000,
            "dt": 0.05,
            "offset_shrinkage": 0.01,
            "refreshment_rate": 0.1,
            # x_0 is at the origin: MAP in whitened space
            "x_0": np.zeros(5),
        },
        "surrogate": {
            "name": "GaussianProcess",
            # Whitened space is ~N(0, I), so standard normal mean/cov
            "mean": np.zeros(5),
            "cov": np.eye(5),
            "n_samples": 200,
            "n_restarts": 50,
            "lbfgs_steps": 5,
            "train_on_init": True,
            "lr": 0.5,
        },
        "output": {
            "dir": ".",
            "logging": {
                "level": "INFO",
                "log_file": "inference.log",
            },
        },
        "seed": 42,
    }

    bps_cfg_path = os.path.join(BPS_DIR, "config.yaml")
    dump_yaml_custom_format(numpy_to_yaml(bps_config), bps_cfg_path)
    print(f"\nWritten: {bps_cfg_path}")


if __name__ == "__main__":
    main()
