#!/usr/bin/env python3
"""Generate synthetic observations for the K&O piecewise-constant test case.

Forward model: 1D bar with 2-segment piecewise-constant Young's modulus.
  theta = [E1, E2]  (Young's modulus in each segment)
  F = [1.0]         (applied force)
  x_obs: 10 sensor locations in (0, 1]

The observation model is:
  y = eta(x, theta_true) + delta(x) + eps

where
  delta(x) ~ GP(0, sigma2_delta * C_delta(rho))   (model discrepancy)
  eps       ~ N(0, sigma2_eps * I)                 (measurement noise)

Outputs:
  observations.dat   -- (1, 10) observation matrix
  ground_truth.dat   -- theta_true and psi_true
"""

import os
import numpy as np

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.discrepancy import rbf_kernel_matrix

# ── ground truth ────────────────────────────────────────────────────────────
THETA_TRUE = np.array([2.0, 3.0])
SIGMA2_DELTA = 0.01
SIGMA2_EPS = 0.001
RHO_TRUE = 5.0

N_OBS_LOC = 10
F = np.array([1.0])
SEED = 0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(SEED)

    x_obs = np.linspace(0.1, 1.0, N_OBS_LOC)
    model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs)

    # Clean forward model output
    eta = model.eval(THETA_TRUE, idx=0)

    # GP discrepancy
    rho = np.array([RHO_TRUE])
    C_delta = rbf_kernel_matrix(x_obs.reshape(-1, 1), rho)
    L_delta = np.linalg.cholesky(SIGMA2_DELTA * C_delta)
    delta = L_delta @ rng.standard_normal(N_OBS_LOC)

    # Measurement noise
    eps = rng.normal(0.0, np.sqrt(SIGMA2_EPS), N_OBS_LOC)

    y = eta + delta + eps
    u_obs = y.reshape(1, -1)

    # Save
    obs_path = os.path.join(OUT_DIR, "observations.dat")
    gt_path = os.path.join(OUT_DIR, "ground_truth.dat")

    np.savetxt(obs_path, u_obs)
    np.savetxt(gt_path,
               np.concatenate([THETA_TRUE,
                                [np.log(SIGMA2_DELTA),
                                 np.log(SIGMA2_EPS),
                                 np.log(RHO_TRUE)]]))

    print(f"theta_true  = {THETA_TRUE}")
    print(f"sigma2_delta= {SIGMA2_DELTA}  (log = {np.log(SIGMA2_DELTA):.3f})")
    print(f"sigma2_eps  = {SIGMA2_EPS}  (log = {np.log(SIGMA2_EPS):.3f})")
    print(f"rho         = {RHO_TRUE}  (log = {np.log(RHO_TRUE):.3f})")
    print(f"\neta range   : [{eta.min():.4f}, {eta.max():.4f}]")
    print(f"delta std   : {delta.std():.4f}")
    print(f"obs range   : [{y.min():.4f}, {y.max():.4f}]")
    print(f"\nWritten: {obs_path}")
    print(f"Written: {gt_path}")


if __name__ == "__main__":
    main()
