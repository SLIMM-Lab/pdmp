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
import shutil
import traceback
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
from pdmp.surrogates import SURROGATE_REGISTRY, GaussianProcessBase


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


def precompute_affine_params(config: dict, sidecar_path: str,
                             rng: np.random.Generator) -> bool:
    """Pre-compute and persist an auto affine transform's ``M``/``b``.

    When the problem is an ``Affine`` ``TransformedDistribution`` with no ``M``
    or ``b`` given, those are normally found by optimisation (``find_mean`` for
    the mode, ``find_curvature`` for the curvature) every time the target is
    built — i.e. on every resumed job. This computes them once from the base
    distribution and caches them in ``sidecar_path`` (a small ``.npz``), so
    subsequent runs load them directly without re-optimising.

    The cache is written **atomically** (temp-then-rename) and is a sidecar
    file, never the input ``config.yaml``: rewriting the user's config is both
    lossy (comments are dropped) and unsafe under a scheduler kill or two
    concurrent jobs, where a half-written config drops ``M``/``b`` and triggers
    a needless (expensive) recompute. ``M``/``b`` are deterministic given the
    base distribution, so a concurrent recompute still writes identical bytes.

    A dedicated ``rng`` is used so the main sampler rng is untouched: the
    sample path is then identical whether or not this pre-computation ran on a
    given invocation (the cached ``M``/``b`` make target construction draw no
    random numbers), keeping re-runs and resumes reproducible.

    Returns True if parameters were applied to ``config`` (loaded or computed).
    """
    prob = config.get('problem')
    if not (isinstance(prob, dict)
            and prob.get('name') == 'Transformed'
            and prob.get('transformation') == 'Affine'
            and prob.get('M') is None
            and prob.get('b') is None
            and isinstance(prob.get('distribution'), dict)):
        return False

    # Reuse cached params if a previous run already computed them.
    if os.path.exists(sidecar_path):
        with np.load(sidecar_path) as data:
            prob['b'] = data['b']
            prob['M'] = data['M']
        logger.warning(f"Loaded cached affine 'M'/'b' from {sidecar_path}.")
        return True

    logger.warning(
        "Affine transformation has no 'M'/'b'; pre-computing them from the "
        "base distribution.")
    base = get_target(prob['distribution'], rng=rng)
    b = find_mean(base, trace_file=None)
    M = find_curvature(base, mean=b)
    prob['b'] = b
    prob['M'] = M

    os.makedirs(os.path.dirname(sidecar_path) or '.', exist_ok=True)
    tmp = sidecar_path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, M=M, b=b)
    os.replace(tmp, sidecar_path)
    logger.warning(f"Cached affine 'M'/'b' to {sidecar_path}.")
    return True


# Files a completed run writes into the output directory (union over all
# samplers, plus the persisted config). Listed explicitly -- never a blanket
# wipe -- because output.dir is frequently '.', the case directory that also
# holds the inputs (config.yaml, observations, meshes).
_RUN_OUTPUT_FILES = (
    'positions.dat', 'times.dat', 'velocities.dat',
    'times_all.dat', 'offset_history.dat', 'eval_times.dat',
    'rate_evals_per_event.dat', 'samples.dat', 'accepted.dat',
    'proposals.dat', 'n_evals.dat', 'other.json',
    'config_used.yaml', 'neural_network.th',
)


def clear_previous_outputs(output_dir: str, ckpt_dir: str,
                           surrogate_cfg: dict) -> None:
    """Remove artefacts from an earlier run so a fresh start is clean.

    Only known output files/dirs are deleted -- never the directory itself --
    so the inputs living alongside the outputs (config.yaml, observations,
    meshes) and the active log file are left untouched. Called only on a fresh
    start (never when resuming), so no live checkpoint is ever removed.
    """
    removed = []

    for name in _RUN_OUTPUT_FILES:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)

    # Surrogate persistence dirs (GP 'model_data' / configured data_path,
    # figure_path) and the whole checkpoint directory (which also holds the
    # checkpointed surrogate under checkpoints/surrogate).
    dirs = {ckpt_dir, 'model_data'}
    if surrogate_cfg.get('data_path'):
        dirs.add(surrogate_cfg['data_path'])
    if surrogate_cfg.get('figure_path'):
        dirs.add(surrogate_cfg['figure_path'])
    for d in dirs:
        if d and os.path.isdir(d):
            shutil.rmtree(d)
            removed.append(d + os.sep)

    if removed:
        logger.warning("Fresh run: cleared previous outputs:\n    " +
                       "\n    ".join(sorted(removed)))


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
    # Marker dropped on an unexpected failure so a self-resubmitting chain
    # halts instead of looping on a deterministic error. Delete it to retry.
    error_path = os.path.join(ckpt_dir, 'error.flag')
    ckpt_interval = float(ckpt_cfg.get('interval_minutes', 10)) * 60.0

    # Cache for an auto Affine transform's M/b. A persistent sidecar (not the
    # input config), so it survives a fresh-start cleanup and is reused instead
    # of re-optimised. Delete it to force a recompute (e.g. if the problem
    # changed).
    affine_sidecar = os.path.join(output_dir, 'affine_params.npz')

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

    # On a fresh start (not a resume), wipe artefacts a previous run left in
    # this directory so stale outputs don't linger. Disable via output.clean.
    if not resuming and config['output'].get('clean', True):
        clear_previous_outputs(output_dir, ckpt_dir, config.get('surrogate', {}))

    # collect seed for rng
    rng = np.random.default_rng(config.get('seed', 0))

    # Generate a random seed for PyTorch from NumPy's RNG
    torch_seed = rng.integers(0, 2**32)  # Get a random 32-bit integer
    torch.manual_seed(torch_seed)
    torch.set_default_dtype(torch.float64)

    try:
        # If an auto Affine transform is requested (no M/b), compute and cache
        # them once so they are not re-optimised on every resumed job. A
        # dedicated rng keeps the main sampler stream untouched.
        precompute_affine_params(
            config, affine_sidecar,
            np.random.default_rng(config.get('seed', 0)))

        # load the problem configuration
        target = get_target(config['problem'], rng=rng)

        # Suppress verbose output from external libraries after they're imported
        suppress_external_loggers()

        if 'surrogate' in config:
            # Work on a copy so injected checkpoint settings (data_path,
            # train_on_init) do not leak into the saved config_used.yaml.
            surrogate_cfg = dict(config['surrogate'])

            # GP surrogates can be reloaded from disk on resume instead of
            # retraining (expensive, and the trained model is already persisted
            # by GaussianProcessBase.train()). Other surrogates are cheap or
            # analytic, so they are simply rebuilt.
            surrogate_class = SURROGATE_REGISTRY.get(surrogate_cfg['name'])
            supports_reload = (
                ckpt_enabled and surrogate_class is not None
                and issubclass(surrogate_class, GaussianProcessBase))

            surrogate_data_path = None
            if supports_reload:
                # Persist the GP under the checkpoint dir (unless the config
                # pins data_path), so save and reload agree on the location.
                surrogate_data_path = config['surrogate'].get(
                    'data_path', os.path.join(ckpt_dir, 'surrogate'))
                surrogate_cfg['data_path'] = surrogate_data_path
                if resuming:
                    # Skip the hyperparameter fit; the model is reloaded below.
                    surrogate_cfg['train_on_init'] = False

            surrogate = get_surrogate(surrogate_cfg, target=target, rng=rng)

            if supports_reload and resuming:
                surrogate.load_model(surrogate_data_path)
                logger.warning(
                    f"Reloaded trained GP surrogate from {surrogate_data_path} "
                    "(skipped retraining).")

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
            else:
                # Snapshot the initial state (event 0 + post-construction rng)
                # right away. A kill before the first periodic snapshot then
                # still resumes -- and, crucially, reloads the just-trained GP
                # surrogate instead of retraining it. The run loop starts at
                # event 1 either way, so the path is unchanged.
                sampler.save_checkpoint(ckpt_path)
                logger.warning(
                    f"Wrote initial checkpoint at event {sampler._iter}.")

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
        # Fail loudly: drop an error marker so a self-resubmitting chain stops
        # instead of looping on a deterministic failure, then re-raise so the
        # process exits non-zero and the scheduler sees the failure.
        if ckpt_enabled:
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                with open(error_path, 'w') as f:
                    f.write(traceback.format_exc())
                logger.error(
                    f"Wrote error marker {error_path}; delete it to retry.")
            except Exception:
                logger.exception("Failed to write the error marker.")
        raise


if __name__ == '__main__':

    main()
