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

from pdmp import logger
from pdmp.logger_setup import setup_file_handler, suppress_external_loggers
from pdmp.loader import get_target, get_sampler, get_surrogate, get_config, save_config


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


def main():

    # parse input arguments
    args = parse_args()

    # Resolve all relative paths (observation_file, output dir, msh files)
    # against the config file's own directory, not the shell's CWD.
    os.chdir(os.path.dirname(os.path.abspath(args.config)))

    config = get_config(args.config)

    setup_file_handler(logger, config['output']['dir'],
                       **config['output']['logging'])

    # collect seed for rng
    rng = np.random.default_rng(config.get('seed', 0))

    # Generate a random seed for PyTorch from NumPy's RNG
    torch_seed = rng.integers(0, 2**32)  # Get a random 32-bit integer
    torch.manual_seed(torch_seed)
    torch.set_default_dtype(torch.float64)

    try:
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

        sampler.run()
        sampler.write_data(config['output']['dir'])
        save_config(config, config['output']['dir'], 'config_used.yml')

    except Exception as e:
        logger.exception("An error occurred during the simulation: %s", str(e))


if __name__ == '__main__':

    main()
