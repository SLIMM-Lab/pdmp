#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver for moment-matching Bayesian inference via Laplace approximation.

Mirrors `run_inference.py` but instead of running an MCMC/PDMP sampler on the
target distribution, it computes the Laplace (Gaussian-at-MAP) approximation:

    mean = arg max log p(x | data)            (BFGS on -log_density)
    cov  = -inv(Hessian of log p at mean)

The Gaussian N(mean, cov) is then sampled to produce a `samples.dat` file
suitable for downstream analysis and forward UQ.

Outer ``Transformed`` wrappers in the config (typically the auto-built
Affine whitening used by the sampler) are stripped before running Laplace.
Reasons:
  * the auto-Affine wrapper's ``b`` and ``M`` are themselves computed via
    ``find_mean`` / ``find_curvature`` of the inner posterior (i.e. they
    already are a Laplace approximation), so running Laplace on top of that
    would do the same work twice and yield a trivial N(0, I)-ish output;
  * the natural [θ, ψ] coordinates produced here are the input space
    expected by ``forward_uq_moment.py`` for moment-matched forward UQ.

Usage:
    python run_laplace.py --config path/to/config.yaml

The same YAML used for `run_inference.py` is accepted. Optional keys:

    laplace:
      n_samples: 10000        # synthetic draws written to samples.dat
      x_0: [0, 0, 0, 0, 0, 0] # initial point for BFGS (defaults to sampler.x_0
                              # if present, else zeros)
"""

import os
import argparse

import numpy as np

from pdmp import logger
from pdmp.logger_setup import setup_file_handler, suppress_external_loggers
from pdmp.loader import get_target, get_config, save_config
from pdmp.distributions import MultivariateNormal, find_mean, find_curvature


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
        # Strip outer Transformed wrappers — see module docstring.
        problem_cfg = config['problem']
        while isinstance(problem_cfg, dict) and problem_cfg.get(
                'name') == 'Transformed':
            problem_cfg = problem_cfg['distribution']

        target = get_target(problem_cfg, rng=rng)
        suppress_external_loggers()

        laplace_cfg = config.get('laplace', {})

        # Initial point: prefer laplace.x_0, then sampler.x_0, then zeros.
        x_0 = laplace_cfg.get('x_0', None)
        if x_0 is None and 'sampler' in config:
            x_0 = config['sampler'].get('x_0', None)
        if x_0 is None:
            x_0 = np.zeros(target.dim)
        x_0 = np.asarray(x_0, dtype=float)

        logger.warning(' ---- Computing Laplace approximation ---- ')
        mean = find_mean(target, x_0=x_0)
        cov = find_curvature(target, mean=mean)

        logger.info(f"Laplace mean: {mean}")
        logger.info(f"Laplace cov diag: {np.diag(cov)}")

        gaussian = MultivariateNormal(mean, cov, rng=rng)

        n_samples = int(laplace_cfg.get('n_samples', 10000))
        samples = gaussian.get_sample(n_samples)

        np.savetxt(os.path.join(out_dir, 'mean.dat'), mean)
        np.savetxt(os.path.join(out_dir, 'cov.dat'), cov)
        np.savetxt(os.path.join(out_dir, 'samples.dat'), samples)

        save_config(config, out_dir, 'config_used.yaml')

        logger.warning(f' ---- Laplace approximation written to {out_dir} ---- ')

    except Exception as e:
        logger.exception("An error occurred during Laplace approximation: %s",
                         str(e))


if __name__ == '__main__':
    main()
