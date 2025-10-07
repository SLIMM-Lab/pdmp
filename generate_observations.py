#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import argparse
import os

import numpy as np

from typing import Any

from pdmp.loader import yaml_to_numpy
from pdmp.distributions import get_prior, AffineTransformtion, ExponentialTransformation
from pdmp.forward_model import get_model
from pdmp.random_field import get_field


def parse_args():
    """Parse the command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """

    parser = argparse.ArgumentParser(description="Run observation generation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The path to the configuration file.",
    )

    return parser.parse_args()


def generate_observations(config: dict[str, Any], rng: np.random.Generator):
    """Generate observations.

    Supports both legacy and new RandomField-based configuration. If the model config
    contains a 'field' entry, that field is instantiated once and shared between the
    prior (e.g. prior name 'FromField') and the forward model to ensure consistency.
    """

    # load the observation configuration
    observation_config = config['observations']
    n_obs = observation_config['n_obs']
    sigma = observation_config['sigma']

    # load the problem configuration
    problem_config = config['problem']

    # get prior and model from problem configuration
    if problem_config['name'] == 'BayesianInverse':
        model_cfg = problem_config['model']
        field = None
        if isinstance(model_cfg, dict) and 'field' in model_cfg:
            field = get_field(model_cfg['field'], rng=rng)
        prior = get_prior(problem_config['prior'], rng=rng, field=field)
        model = get_model(model_cfg, field=field)
        llh_config = problem_config['likelihood']

    elif problem_config['name'] == 'Transformed':
        dist_cfg = problem_config['distribution']
        model_cfg = dist_cfg['model']
        field = None
        if isinstance(model_cfg, dict) and 'field' in model_cfg:
            field = get_field(model_cfg['field'], rng=rng)
        prior = get_prior(dist_cfg['prior'], rng=rng, field=field)
        model = get_model(model_cfg, field=field)
        llh_config = dist_cfg['likelihood']
    else:
        raise ValueError("Problem must be BayesianInverse or Transformed.")

    # sample ground truth coefficients from prior and write to file
    ground_truth = prior.get_sample()
    np.savetxt(os.path.join('ground_truth.dat'), ground_truth)

    if llh_config['name'] == 'TransformedLikelihood':
        if llh_config['transformation'] == 'Exponential':
            ground_truth = ExponentialTransformation().transform(ground_truth)
        elif llh_config['transformation'] == 'AffineTransform':
            ground_truth = AffineTransformtion(
                llh_config['b'], llh_config['M']).transform(ground_truth)

    # init array
    obs = np.zeros((n_obs, model.get_dim_out()))

    for i in range(n_obs):
        obs[i] = model.eval(ground_truth, idx=i) + rng.normal(
            0, sigma, model.get_dim_out())

    # write observations to file
    np.savetxt(os.path.join('observations.dat'), obs)


def main():

    # parse input arguments
    args = parse_args()

    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # setup_file_handler(logger, level='INFO', log_dir='.', log_file='generate_observations.log')

    # convert the configuration to numpy arrays
    config = yaml_to_numpy(config)

    # init rng and generate observations
    rng = np.random.default_rng(config.get('seed', 0))
    generate_observations(config, rng)


if __name__ == '__main__':

    main()
