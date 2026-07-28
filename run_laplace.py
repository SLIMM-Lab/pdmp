#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver for moment-matching Bayesian inference via Laplace approximation.

Mirrors `run_inference.py` but instead of running an MCMC/PDMP sampler on the
target distribution, it produces the Laplace (Gaussian-at-MAP) approximation
N(mean, cov) in the natural [θ, ψ] coordinates and samples it into a
`samples.dat`, alongside `mean.dat` / `cov.dat` — the input expected by
`forward_uq_moment.py` for moment-matched forward UQ.

How the Laplace approximation is obtained
-----------------------------------------
There are two paths, in order of preference:

1. **Reuse the sampler's affine whitening (preferred).** When the config's
   problem is an ``Affine`` ``Transformed`` wrapper carrying ``b`` and ``M``
   (as written into a sampler run's ``config_used.yaml``), those *are* the
   Laplace approximation of the inner [θ, ψ] posterior: ``b`` is the mode
   (``find_mean``) and ``M`` the covariance Σ (``find_curvature``), computed
   once during sampler setup. We use ``mean = b``, ``cov = M`` directly — no
   forward-model evaluations, the same well-conditioned Gaussian the sampler
   used, and exactly the space ``forward_uq_moment.py`` works in. Point
   ``--config`` at the sampler's ``config_used.yaml`` to take this path.

2. **Recompute from scratch (fallback).** With no precomputed ``M``/``b``, the
   outer ``Transformed`` wrappers are stripped and the Gaussian is found on the
   inner target: ``mean = find_mean`` and ``cov`` from the log-density Hessian.

       cov = -inv(Hessian of log p at mean)

   For ill-conditioned inverse problems the Hessian is a *noisy finite-
   difference* estimate (see ``JaxFemModel.eval_hessian``), and its naive
   inverse can be **indefinite** — not a valid covariance, and useless for
   ``forward_uq_moment.py`` (``J Σ Jᵀ`` then yields negative variances, and the
   UT Cholesky needs heavy repair). We therefore regularise to the nearest SPD
   matrix (symmetrise, floor the precision's eigenvalues, invert), which keeps
   poorly-identified directions finite and positive. This is still less
   trustworthy than path 1; a warning is logged recommending a config with
   precomputed ``M``/``b``.

Usage:
    python run_laplace.py --config path/to/config.yaml

The same YAML used for `run_inference.py` is accepted. Optional keys:

    laplace:
      n_samples: 10000        # synthetic draws written to samples.dat
      x_0: [0, 0, 0, 0, 0, 0] # initial point for BFGS (defaults to sampler.x_0
                              # if present, else zeros)
      reuse_affine: true      # use precomputed affine b/M when present
                              # (default true); set false to always recompute
"""

import os
import argparse

import numpy as np

from pdmp import logger
from pdmp.logger_setup import setup_file_handler, suppress_external_loggers
from pdmp.loader import get_target, get_config, save_config
from pdmp.distributions import MultivariateNormal, find_mean


def find_affine_laplace(problem_cfg):
    """Return (mean, cov) from a precomputed affine whitening, or None.

    Walks nested ``Transformed`` wrappers looking for an ``Affine`` one that
    already carries ``b`` and ``M``. Those are the sampler's Laplace
    approximation of the inner posterior: ``b`` is the mode and ``M`` the
    covariance Σ in the inner [θ, ψ] space. Mirroring ``get_transformation``,
    a *symmetric* ``M`` is the covariance itself, while a non-symmetric ``M``
    (a user-supplied rotation/shear factor C) implies ``Σ = C·Cᵀ``.
    """
    prob = problem_cfg
    while isinstance(prob, dict) and prob.get('name') == 'Transformed':
        if (prob.get('transformation') == 'Affine'
                and prob.get('M') is not None and prob.get('b') is not None):
            b = np.asarray(prob['b'], dtype=float)
            M = np.asarray(prob['M'], dtype=float)
            cov = M if np.allclose(M, M.T) else M @ M.T
            return b, 0.5 * (cov + cov.T)
        prob = prob.get('distribution')
    return None


def nearest_spd_cov(hessian, rel_floor=1e-8):
    """Laplace covariance from a (possibly noisy/indefinite) log-density Hessian.

    The Laplace covariance is ``-inv(H)``; at a true maximum ``H`` is negative
    definite, so the precision ``P = -H`` is SPD. A finite-difference ``H`` of
    an ill-conditioned posterior can pick up small negative eigenvalues, making
    ``-inv(H)`` indefinite — not a covariance. We symmetrise, floor the
    eigenvalues of ``P`` to a small positive multiple of its largest eigenvalue,
    and invert. Poorly-identified directions get a large-but-finite, positive
    variance instead of a negative or divergent one.
    """
    H = 0.5 * (np.asarray(hessian, dtype=float)
               + np.asarray(hessian, dtype=float).T)
    w, V = np.linalg.eigh(-H)  # eigenvalues/vectors of the precision P = -H
    floor = rel_floor * max(float(w.max()), 0.0)
    if floor <= 0.0:
        floor = rel_floor
    n_bad = int((w < floor).sum())
    if n_bad:
        logger.warning(
            f"Laplace precision had {n_bad}/{w.size} eigenvalue(s) below the "
            f"floor ({floor:.3e}); clipped to keep the covariance SPD. The "
            f"finite-difference Hessian is poorly conditioned here.")
    w = np.maximum(w, floor)
    return (V / w) @ V.T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Laplace approximation of a posterior.")
    parser.add_argument(
        "--config",
        default='config.yaml',
        type=str,
        help="The path to the configuration file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve relative paths (observation_file, msh files, ...) against the
    # config file's own directory, matching run_inference.py's convention.
    os.chdir(os.path.dirname(os.path.abspath(args.config)))

    config = get_config(args.config)

    out_dir = config['output']['dir']
    os.makedirs(out_dir, exist_ok=True)
    setup_file_handler(logger, out_dir, **config['output']['logging'])

    rng = np.random.default_rng(config.get('seed', 0))

    try:
        laplace_cfg = config.get('laplace', {})

        # Path 1 (preferred): reuse the sampler's precomputed affine b/M — they
        # already are the Laplace approximation of the inner [θ, ψ] posterior,
        # in the space forward_uq_moment.py expects, and well-conditioned. No
        # forward-model evaluations needed. See the module docstring.
        affine = None
        if laplace_cfg.get('reuse_affine', True):
            affine = find_affine_laplace(config['problem'])

        if affine is not None:
            mean, cov = affine
            logger.warning(
                " ---- Reusing precomputed affine 'b'/'M' as the Laplace "
                "mean/covariance ---- ")
        else:
            # Path 2 (fallback): recompute on the inner target. Strip outer
            # Transformed wrappers — see module docstring.
            problem_cfg = config['problem']
            while isinstance(problem_cfg, dict) and problem_cfg.get(
                    'name') == 'Transformed':
                problem_cfg = problem_cfg['distribution']

            target = get_target(problem_cfg, rng=rng)
            suppress_external_loggers()

            # Initial point: prefer laplace.x_0, then sampler.x_0, then zeros.
            x_0 = laplace_cfg.get('x_0', None)
            if x_0 is None and 'sampler' in config:
                x_0 = config['sampler'].get('x_0', None)
            if x_0 is None:
                x_0 = np.zeros(target.dim)
            x_0 = np.asarray(x_0, dtype=float)

            logger.warning(' ---- Computing Laplace approximation ---- ')
            mean = find_mean(target, x_0=x_0)
            # Regularise the (noisy finite-difference) Hessian to the nearest
            # SPD covariance so the result is always a valid Gaussian.
            cov = nearest_spd_cov(target.hessian_log_density(mean))
            logger.warning(
                "Computed the Laplace covariance from the finite-difference "
                "Hessian (nearest-SPD regularised). For a result identical to "
                "the sampler's, run on a config carrying precomputed affine "
                "'M'/'b' (e.g. a completed run's config_used.yaml).")

        logger.info(f"Laplace mean: {mean}")
        logger.info(f"Laplace cov diag: {np.diag(cov)}")

        gaussian = MultivariateNormal(mean, cov, rng=rng)

        n_samples = int(laplace_cfg.get('n_samples', 10000))
        samples = gaussian.get_sample(n_samples)

        np.savetxt(os.path.join(out_dir, 'mean.dat'), mean)
        np.savetxt(os.path.join(out_dir, 'cov.dat'), cov)
        np.savetxt(os.path.join(out_dir, 'samples.dat'), samples)

        save_config(config, out_dir, 'config_used.yaml')

        logger.warning(
            f' ---- Laplace approximation written to {out_dir} ---- ')

    except Exception as e:
        logger.exception("An error occurred during Laplace approximation: %s",
                         str(e))


if __name__ == '__main__':
    main()
