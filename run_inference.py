import yaml
import argparse

import numpy as np

from pdmp import logger
from pdmp.logger_setup import setup_file_handler
from pdmp.loader import get_target, get_sampler, yaml_to_numpy, save_config


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    argparse.Namespace: The command line arguments.
    """

    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The path to the configuration file.",
    )

    return parser.parse_args()

def main():

    # parse input arguments
    args = parse_args()

    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_file_handler(logger, config['output']['dir'], **config['output']['logging'])

    # convert the configuration to numpy arrays
    config = yaml_to_numpy(config)

    rng = np.random.default_rng(config.get('seed', 0))

    try:
        # load the problem configuration
        problem_config = config['problem']
        target = get_target(problem_config, rng=rng)
        sampler = get_sampler(config['sampler'], target=target, rng=rng)

        sampler.run()
        sampler.write_data(config['output']['dir'])
        save_config(config, config['output']['dir'], 'config_used.yml')

    except Exception as e:
        logger.exception("An error occurred during the simulation: %s", str(e))

if __name__ == '__main__':

    main()