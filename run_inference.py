#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import jax
#
# jax.config.update("jax_log_compiles", True)
#
# import logging
#
# for name in ("jax", "jax._src.dispatch", "jax._src.interpreters.pxla",
#              "jax._src.compiler"):
#     logging.getLogger(name).setLevel(logging.DEBUG)
#     logging.getLogger(name).propagate = True
#
import os
import yaml
import argparse
import torch

import numpy as np

# Tighten PETSc KSP tolerances before any solver is constructed. jax-fem's
# `petsc_solve` calls ksp.setFromOptions(), which reads from this global
# options database. The defaults (rtol=1e-5) leave the absolute residual at
# ~rtol·||b||, which on this problem's traction-driven RHS can be O(10), well
# above the hard-coded `assert err < 0.1` in jax_fem/solver.py:135.
from petsc4py import PETSc
_petsc_opts = PETSc.Options()
_petsc_opts.setValue('ksp_rtol', '1e-10')
_petsc_opts.setValue('ksp_atol', '1e-50')
_petsc_opts.setValue('ksp_max_it', '1000')

from pdmp import logger
from pdmp.logger_setup import setup_file_handler, suppress_external_loggers
from pdmp.loader import get_target, get_sampler, get_surrogate, get_config, save_config
from pdmp.distributions import find_mean, find_curvature


def parse_args():
    """Parse the command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """

    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation.")
    parser.add_argument(
        "--config",
        default='config.yaml',
        type=str,
        help="The path to the configuration file.",
    )

    return parser.parse_args()


def precompute_affine_params(config: dict, config_path: str,
                             rng: np.random.Generator) -> bool:
    """Pre-compute and persist an auto affine transform's ``M``/``b``.

    When the problem is an ``Affine`` ``TransformedDistribution`` with no ``M``
    or ``b`` given, those are normally found by optimisation (``find_mean`` for
    the mode, ``find_curvature`` for the curvature) every time the target is
    built — i.e. on every resumed job. This computes them once from the base
    distribution and writes them back into the config file, so subsequent runs
    read them directly.

    A dedicated ``rng`` is used so the main sampler rng is untouched: the
    sample path is then identical whether or not this pre-computation ran on a
    given invocation (the persisted ``M``/``b`` make target construction draw no
    random numbers), keeping re-runs and resumes reproducible.

    Returns True if parameters were computed and written.
    """
    prob = config.get('problem')
    if not (isinstance(prob, dict)
            and prob.get('name') == 'Transformed'
            and prob.get('transformation') == 'Affine'
            and prob.get('M') is None
            and prob.get('b') is None
            and isinstance(prob.get('distribution'), dict)):
        return False

    logger.warning(
        "Affine transformation has no 'M'/'b'; pre-computing them from the "
        "base distribution.")
    base = get_target(prob['distribution'], rng=rng)
    b = find_mean(base, trace_file=None)
    M = find_curvature(base, mean=b)
    prob['b'] = b
    prob['M'] = M

    save_config(config, os.path.dirname(config_path),
                os.path.basename(config_path))
    logger.warning(
        f"Wrote affine 'M' and 'b' to {config_path} "
        "(file rewritten; comments are not preserved).")
    return True


def main():

    # parse input arguments
    args = parse_args()

    # Resolve all relative paths (observation_file, output dir, msh files)
    # against the config file's own directory, not the shell's CWD.
    config_path = os.path.abspath(args.config)
    os.chdir(os.path.dirname(config_path))

    config = get_config(config_path)

    # Checkpoint/resume settings. Enabled by default; a periodic snapshot lets
    # an interrupted (e.g. scheduler-killed) run resume with an identical path.
    # Checkpoint files live in their own subdirectory to keep the output dir
    # clean.
    output_dir = config['output']['dir']
    ckpt_cfg = config.get('checkpoint', {})
    ckpt_enabled = ckpt_cfg.get('enabled', False)
    ckpt_dir = os.path.join(output_dir, str(ckpt_cfg.get('dir', 'checkpoints')))
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pkl')
    done_path = os.path.join(ckpt_dir, 'completed.flag')
    ckpt_interval = float(ckpt_cfg.get('interval_minutes', 10)) * 60.0

    # A run is being resumed if a checkpoint (or completion flag) already
    # exists. On a fresh start we overwrite the log file; on a resume we append
    # to it, so the log is continuous across the chain of restarts.
    resuming = ckpt_enabled and (os.path.exists(ckpt_path)
                                 or os.path.exists(done_path))
    setup_file_handler(logger, output_dir, append=resuming,
                       **config['output']['logging'])

    # If a previous job already finished this run, do nothing. This lets a
    # self-resubmitting sbatch chain terminate cleanly.
    if ckpt_enabled and os.path.exists(done_path):
        logger.warning(f"Run already complete ({done_path} exists); nothing to do.")
        return

    # collect seed for rng
    rng = np.random.default_rng(config.get('seed', 0))

    # Generate a random seed for PyTorch from NumPy's RNG
    torch_seed = rng.integers(0, 2**32)  # Get a random 32-bit integer
    torch.manual_seed(torch_seed)
    torch.set_default_dtype(torch.float64)

    try:
        # If an auto Affine transform is requested (no M/b), compute and persist
        # them once so they are not re-optimised on every resumed job. A
        # dedicated rng keeps the main sampler stream untouched.
        precompute_affine_params(
            config, config_path,
            np.random.default_rng(config.get('seed', 0)))

        # load the problem configuration
        target = get_target(config['problem'], rng=rng)

        # Suppress verbose output from external libraries after they're imported
        suppress_external_loggers()

        if 'surrogate' in config:
            surrogate = get_surrogate(config.get('surrogate'),
                                      target=target,
                                      rng=rng)
            sampler = get_sampler(config['sampler'],
                                  target=target,
                                  surrogate=surrogate,
                                  rng=rng)
        else:
            sampler = get_sampler(config['sampler'], target=target, rng=rng)

        if ckpt_enabled:
            # Online surrogate retraining is not captured by sampler-only
            # checkpointing, so the identical-path guarantee would be violated.
            update_model = config.get('surrogate', {}).get('update_model')
            if update_model is not None and len(update_model) > 0:
                logger.warning(
                    "Surrogate has an online update_model schedule; its state "
                    "is NOT checkpointed, so a resumed run may diverge from an "
                    "uninterrupted one.")

            os.makedirs(ckpt_dir, exist_ok=True)
            sampler.enable_checkpointing(ckpt_path, ckpt_interval)
            if os.path.exists(ckpt_path):
                resumed_iter = sampler.load_checkpoint(ckpt_path)
                logger.warning(
                    f"Resumed from checkpoint at event {resumed_iter}.")

        sampler.run()
        sampler.write_data(config['output']['dir'])
        save_config(config, config['output']['dir'], 'config_used.yaml')

        # Mark completion and drop the checkpoint so a resubmitted job is a no-op.
        if ckpt_enabled:
            with open(done_path, 'w') as f:
                f.write('done\n')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    except Exception as e:
        logger.exception("An error occurred during the simulation: %s", str(e))


if __name__ == '__main__':

    main()
