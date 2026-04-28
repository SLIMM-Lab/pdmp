#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import torch
from scipy.optimize import minimize

from pdmp import logger
from pdmp.logger_setup import setup_file_handler, suppress_external_loggers
from pdmp.loader import get_target, get_config, save_config
from pdmp.distributions import TransformedLikelihood


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deterministic MAP calibration.")
    parser.add_argument(
        "--config",
        default='config.yaml',
        type=str,
        help="The path to the configuration file.",
    )
    return parser.parse_args()


def get_physical_transformation(target):
    """Return the Composite transformation mapping xi -> physical params.

    The posterior is Posterior(prior, TransformedLikelihood(base, Composite)).
    We reach into the likelihood to reuse its transformation for reporting
    parameters on the physical scale.
    """
    likelihood = getattr(target, '_likelihood', None)
    if isinstance(likelihood, TransformedLikelihood):
        return likelihood._transformation
    return None


def run_optimization(target, x_0, method, options):
    n_log_post = lambda x: -target.log_density(x)
    n_grad_log_post = lambda x: -target.grad_log_density(x)
    return minimize(n_log_post,
                    x_0,
                    jac=n_grad_log_post,
                    method=method,
                    options=options)


def main():
    args = parse_args()
    config = get_config(args.config)

    setup_file_handler(logger, config['output']['dir'],
                       **config['output']['logging'])

    rng = np.random.default_rng(config.get('seed', 0))
    torch_seed = rng.integers(0, 2**32)
    torch.manual_seed(torch_seed)
    torch.set_default_dtype(torch.float64)

    try:
        target = get_target(config['problem'], rng=rng)
        suppress_external_loggers()

        opt_cfg = config['optimizer']
        method = opt_cfg.get('method', 'L-BFGS-B')
        options = opt_cfg.get('options', {})
        if isinstance(options, np.ndarray):
            options = options.tolist()
        n_restarts = int(opt_cfg.get('n_restarts', 0))

        x_0 = np.asarray(opt_cfg['x_0'], dtype=float)
        starts = [x_0]
        for _ in range(n_restarts):
            starts.append(np.asarray(target.prior.get_sample(), dtype=float))

        results = []
        for i, x_start in enumerate(starts):
            logger.info(f"Restart {i}: starting from {x_start}")
            res = run_optimization(target, x_start, method, options)
            logger.info(f"Restart {i}: fun={res.fun:.6e}, "
                        f"success={res.success}, nit={res.nit}, "
                        f"message={res.message}")
            results.append(res)

        best = min(results, key=lambda r: r.fun)
        xi_star = np.asarray(best.x, dtype=float)
        grad_norm = float(np.linalg.norm(best.jac)) if best.jac is not None \
            else float(np.linalg.norm(-target.grad_log_density(xi_star)))

        logger.info("==== MAP optimisation summary ====")
        logger.info(f"best neg_log_post: {best.fun:.6e}")
        logger.info(f"gradient norm at optimum: {grad_norm:.3e}")
        logger.info(f"iterations: {best.nit}")
        logger.info(f"message: {best.message}")
        logger.info(f"xi* = {xi_star}")

        out_dir = config['output']['dir']
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, 'map.dat'), xi_star, fmt='%.6e')

        trans = get_physical_transformation(target)
        if trans is not None:
            x_star = np.asarray(trans.transform(xi_star), dtype=float)
            np.savetxt(os.path.join(out_dir, 'map_physical.dat'),
                       x_star,
                       fmt='%.6e')
            logger.info(f"x* (physical) = {x_star}")

        restart_funs = np.array([r.fun for r in results], dtype=float)
        np.savetxt(os.path.join(out_dir, 'restart_objectives.dat'),
                   restart_funs,
                   fmt='%.6e')

        save_config(config, out_dir, 'config_used.yml')

    except Exception as e:
        logger.exception("An error occurred during the calibration: %s",
                         str(e))


if __name__ == '__main__':
    main()
