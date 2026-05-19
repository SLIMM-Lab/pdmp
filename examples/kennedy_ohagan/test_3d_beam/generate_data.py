#!/usr/bin/env python3
"""Generate synthetic observations for the K&O 3D beam test case
and write sampler configs for RWM (with and without K&O discrepancy).

Forward model: 3D cantilever beam with exponentially recovering Young's modulus.
  theta = [rho, l]   (recovery parameter and length scale)
  5 sensors on x=0 face, each giving 3 DOFs → 15 observations

The observation model is:
  y = eta(x, theta_true) + delta(x) + eps

where
  delta(x) ~ GP(0, sigma2_delta * C_delta(rho_ko))  (model discrepancy)
  eps       ~ N(0, sigma2_eps * I)                    (measurement noise)

The KO discrepancy kernel uses z-coordinates of sensor locations (1D).
C_delta is block-diagonal: block_diag(C_z, C_z, C_z) where C_z is the (5×5)
RBF kernel over the 5 sensor z-coords.  Different DOF components at the same
sensor are exactly independent; spatial correlation is preserved within each
component.  x_locs has shape (N_SENSOR_PTS, 1) and n_components=N_DOF_PER_PT.

Sampling space (both configs): xi = [logit(rho), log(l)] and for KO additionally
psi = [log_s2d, log_s2e, log_rho_ko] — these are all already unconstrained, so
no reparametrisation of psi is needed.

Outputs (top-level):
  observations.dat   -- (1, 15) observation matrix
  ground_truth.dat   -- [theta_true, psi_true] with psi in log-space

RWM-specific outputs:
  rwm/config.yaml       -- RWM with KO discrepancy (5-D latent)
  rwm_no_ko/config.yaml -- RWM without KO (2-D latent)
"""

import os
import numpy as np
from scipy.linalg import block_diag

from pdmp.forward_model import get_model
from pdmp.random_field import get_jax_field
from pdmp.discrepancy import rbf_kernel_matrix
from pdmp.loader import numpy_to_yaml, dump_yaml_custom_format

# ── ground truth ─────────────────────────────────────────────────────────────
THETA_TRUE = np.array([0.7, 1.2])  # [rho, l] in physical space
SIGMA2_DELTA = 0.01  # KO discrepancy variance
SIGMA2_EPS = 0.001  # measurement noise variance
RHO_KO_TRUE = 2.0  # KO discrepancy lengthscale (inverse sq.)

SEED = 4
N_SAMPLES = 2000

# ── sensor configuration ──────────────────────────────────────────────────────
SENSOR_Z = [0.5, 1.0, 1.5, 2.0, 2.5]
SENSORS = [
    {
        "name": "sensors_x0",
        "location_fn": "side_faces",
        "points": [[0.0, 0.5, z] for z in SENSOR_Z],
    },
]
N_SENSOR_PTS = len(SENSOR_Z)
N_DOF_PER_PT = 3
N_OBS = N_SENSOR_PTS * N_DOF_PER_PT  # 15

# ── model configuration ───────────────────────────────────────────────────────
MODEL_CFG = {
    "name": "JaxFem",
    "d_x": 1.0,
    "d_y": 1.0,
    "d_z": 2.5,
    "h": 0.25,
    "nu": 0.3,
    "indenter_loc": 1.25,
    "total_load": [0.0, 0.001, 0.0],
    "field": {
        "name": "JaxExponentialRecoveryField",
        "f_infinity": 1.0,
        "idx": 2,
        "coefficient_distribution": {
            "name": "MultivariateNormal",
            "mean": [0.5, 0.9],
            "cov": [[2.0, 0.0], [0.0, 2.0]],
        },
    },
    "sensors": SENSORS,
}

HERE = os.path.dirname(os.path.abspath(__file__))
RWM_DIR = os.path.join(HERE, "rwm")
RWM_NO_KO_DIR = os.path.join(HERE, "rwm_no_ko")


def _build_z_locs() -> np.ndarray:
    """Base KO sensor locations: unique z-coords of the N_SENSOR_PTS sensors.

    Returns:
        x_locs: (N_SENSOR_PTS, 1) array of unique z-coordinates.
    """
    return np.array(SENSOR_Z).reshape(-1, 1)


def main():
    rng = np.random.default_rng(SEED)

    # ── build forward model ───────────────────────────────────────────────────
    print("Building JAX field...")
    field = get_jax_field(MODEL_CFG["field"], rng=rng)

    print("Building FEM model (mesh assembly may take a moment)...")
    model = get_model(MODEL_CFG, field=field)

    n_dofs = model.get_dim_out()
    print(
        f"  {N_SENSOR_PTS} sensor points × {N_DOF_PER_PT} DOF = {n_dofs} observations"
    )
    assert n_dofs == N_OBS, f"Expected {N_OBS} DOFs, got {n_dofs}"

    # ── generate observations ─────────────────────────────────────────────────
    print(f"\nGround truth: rho={THETA_TRUE[0]}, l={THETA_TRUE[1]}")
    eta = np.asarray(model.eval(THETA_TRUE, idx=0)).ravel()

    x_locs = _build_z_locs()  # (N_SENSOR_PTS, 1)
    rho_ko_vec = np.array([RHO_KO_TRUE])
    C_z = rbf_kernel_matrix(x_locs, rho_ko_vec)  # (5, 5)
    C_delta = block_diag(*[C_z] * N_DOF_PER_PT)  # (15, 15)
    L_delta = np.linalg.cholesky(SIGMA2_DELTA * C_delta)
    delta = L_delta @ rng.standard_normal(N_OBS)
    eps = rng.normal(0.0, np.sqrt(SIGMA2_EPS), N_OBS)

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
             np.log(RHO_KO_TRUE)]
        ]))

    print(f"eta range   : [{eta.min():.6f}, {eta.max():.6f}]")
    print(f"delta std   : {delta.std():.6f}")
    print(f"obs range   : [{y.min():.6f}, {y.max():.6f}]")
    print(f"sigma2_delta= {SIGMA2_DELTA}  (log = {np.log(SIGMA2_DELTA):.3f})")
    print(f"sigma2_eps  = {SIGMA2_EPS}  (log = {np.log(SIGMA2_EPS):.3f})")
    print(f"rho_ko      = {RHO_KO_TRUE}  (log = {np.log(RHO_KO_TRUE):.3f})")
    print(f"Written: {obs_path}")
    print(f"Written: {gt_path}")

    # ── write rwm/config.yaml (with K&O discrepancy) ─────────────────────────
    # Latent space: xi = [logit(rho), log(l), log_s2d, log_s2e, log_rho_ko]
    # The Composite transform maps:
    #   xi[0] --Sigmoid(0,1)--> rho
    #   xi[1] --Exponential--> l
    #   xi[2:5] --Identity--> [log_s2d, log_s2e, log_rho_ko]  (already unconstrained)
    # so KOGaussianLikelihood receives [rho, l, log_s2d, log_s2e, log_rho_ko].
    print("\nWriting RWM config with K&O discrepancy...")
    os.makedirs(RWM_DIR, exist_ok=True)

    psi_mean = np.array([-4.0, -7.0, 1.5])  # [log_s2d, log_s2e, log_rho_ko]
    psi_cov = np.diag([4.0, 2.0, 2.0])
    # Prior in latent theta space: loose Gaussian centred at logit(0.5)=0, log(0.9)≈-0.105
    latent_theta_mean = np.array([0.0, np.log(0.9)])
    latent_theta_cov = np.diag([2.0, 2.0])

    rwm_ko_config = {
        "problem": {
            "name": "BayesianInverse",
            "prior": {
                "name": "MultivariateNormal",
                "mean": np.concatenate([latent_theta_mean, psi_mean]),
                "cov": block_diag(latent_theta_cov, psi_cov),
            },
            "likelihood": {
                "name":
                "TransformedLikelihood",
                "transformation":
                "Composite",
                "indices": [[0], [1], [2, 3, 4]],
                "transformations": [
                    {
                        "type": "Sigmoid",
                        "a": 0.0,
                        "b": 1.0
                    },
                    "Exponential",
                    "Identity",
                ],
                "likelihood": {
                    "name": "KOGaussianLikelihood",
                    "observation_file": "../observations.dat",
                    "x_locs": x_locs,
                    "n_components": N_DOF_PER_PT,
                    "psi_prior": {
                        "name": "MultivariateNormal",
                        "mean": psi_mean,
                        "cov": psi_cov,
                    },
                },
            },
            "model": MODEL_CFG,
        },
        "sampler": {
            "name": "RandomWalkMetropolis",
            "n_samples": N_SAMPLES,
            "x_0": np.concatenate([latent_theta_mean, psi_mean]),
            "sigma": 0.2,
        },
        "output": {
            "dir": ".",
            "logging": {
                "level": "INFO",
                "log_file": "inference.log"
            },
        },
        "seed": 42,
    }

    rwm_ko_path = os.path.join(RWM_DIR, "config.yaml")
    dump_yaml_custom_format(numpy_to_yaml(rwm_ko_config), rwm_ko_path)
    print(f"Written: {rwm_ko_path}")

    # ── write rwm_no_ko/config.yaml (without K&O) ────────────────────────────
    # Latent space: xi = [logit(rho), log(l)]
    # Composite transform: xi[0] --Sigmoid--> rho, xi[1] --Exponential--> l
    # sigma: measurement noise only — the no-KO model is the *naive* inference
    # that ignores the discrepancy, mirroring test_piecewise/rwm_no_ko.
    # Prior must be specified directly in latent space (same as K&O): TransformedLikelihood
    # transforms xi → physical for the likelihood but evaluates the prior at xi directly.
    print("Writing RWM config without K&O discrepancy...")
    os.makedirs(RWM_NO_KO_DIR, exist_ok=True)

    sigma_no_ko = float(np.sqrt(SIGMA2_EPS))

    rwm_no_ko_config = {
        "problem": {
            "name": "BayesianInverse",
            "prior": {
                "name": "MultivariateNormal",
                "mean": latent_theta_mean,
                "cov": latent_theta_cov,
            },
            "likelihood": {
                "name":
                "TransformedLikelihood",
                "transformation":
                "Composite",
                "indices": [[0], [1]],
                "transformations": [
                    {
                        "type": "Sigmoid",
                        "a": 0.0,
                        "b": 1.0
                    },
                    "Exponential",
                ],
                "likelihood": {
                    "name": "GaussianLikelihood",
                    "observation_file": "../observations.dat",
                    "sigma": sigma_no_ko,
                },
            },
            "model": MODEL_CFG,
        },
        "sampler": {
            "name": "RandomWalkMetropolis",
            "n_samples": N_SAMPLES,
            "x_0": np.array(latent_theta_mean),
            "sigma": 1.0,
        },
        "output": {
            "dir": ".",
            "logging": {
                "level": "INFO",
                "log_file": "inference.log"
            },
        },
        "seed": 42,
    }

    rwm_no_ko_path = os.path.join(RWM_NO_KO_DIR, "config.yaml")
    dump_yaml_custom_format(numpy_to_yaml(rwm_no_ko_config), rwm_no_ko_path)
    print(f"Written: {rwm_no_ko_path}")

    print("\nDone. Run inference with:")
    print(f"  cd {RWM_DIR} && python ../../../run_inference.py config.yaml")
    print(
        f"  cd {RWM_NO_KO_DIR} && python ../../../run_inference.py config.yaml"
    )


if __name__ == "__main__":
    main()
