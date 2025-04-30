#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import argparse
import torch

import numpy as np

from pdmp import logger
from pdmp.logger_setup import setup_file_handler
from pdmp.loader import get_target, get_sampler, get_surrogate, yaml_to_numpy, save_config


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    argparse.Namespace: The command line arguments.
    """

    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation.")
    parser.add_argument(
        "--config",
        default='config.yml',
        type=str,
        help="The path to the configuration file.",
    )

    return parser.parse_args()


def main():

    # parse input arguments
    args = parse_args()

    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_file_handler(logger, config['output']['dir'],
                       **config['output']['logging'])

    # # convert the configuration to numpy arrays
    config = yaml_to_numpy(config,
                           exclude_keys={'hidden_layers', 'update_model'})

    # collect seed for rng
    rng = np.random.default_rng(config.get('seed', 0))

    # Generate a random seed for PyTorch from NumPy's RNG
    torch_seed = rng.integers(0, 2**32)  # Get a random 32-bit integer
    torch.manual_seed(torch_seed)
    torch.set_default_dtype(torch.float64)

    try:
        # load the problem configuration
        target = get_target(config['problem'], rng=rng)

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
