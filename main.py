import yaml
import os
import argparse

import numpy as np
import seaborn as sns

from typing import Any

from pdmp import logger
from pdmp.logger_setup import setup_file_handler
from pdmp.distributions import Distribution, MultivariateNormal, CubicDistribution, Posterior
from pdmp.sampler import Sampler
from pdmp.zigzag import ZigZagSampler
from pdmp.mcmc import MetropolisHastingsSampler, HamiltonianMonteCarlo, LangevinDynamicsSampler

def get_problem(
        problem_config: dict[str, Any],
        rng: np.random.Generator
) -> Distribution:
    """
    Load the problem configuration.

    Parameters:
    problem_config (dict): The problem configuration.

    Returns:
    Distribution: The target distribution.
    """

    if problem_config['name'] == 'Cubic':
        return CubicDistribution.from_dict(problem_config, rng=rng)
    elif problem_config['name'] == 'Gaussian':
        return MultivariateNormal.from_dict(problem_config, rng=rng)
    elif problem_config['name'] == 'BayesianInverse':
        return Posterior.from_dict(problem_config, rng=rng)
    else:
        raise ValueError(f"Problem {problem_config['name']} not recognized.")

def get_sampler(
        sampler_config: dict[str, Any],
        target: Distribution,
        rng: np.random.Generator
) -> Sampler:
    """
    Load the sampler configuration.

    Parameters:
    sampler_config (dict): The sampler configuration.
    target (MultivariateNormal): The target distribution.
    rng (np.random.Generator): The random number generator.

    Returns:
    ZigZagSampler: The sampler.
    """

    if sampler_config['name'] == 'ZigZag':
        return ZigZagSampler.from_dict(sampler_config, target=target, rng=rng)
    else:
        raise ValueError(f"Sampler {sampler_config['name']} not recognized.")

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

def yaml_to_numpy(data: Any):
    """
    Convert the configuration dictionary to numpy arrays.

    Parameters:
    dictionary (dict): The dictionary.

    Returns:
    dict: The configuration dictionary with numpy arrays.
    """

    if isinstance(data, dict):
        # Process dictionaries recursively
        return {key: yaml_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Check if all elements are floats or integers
        if all(isinstance(x, (int, float)) for x in data):
            return np.array(data, dtype=type(data[0]))
        elif all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in data):
            # Handle 2D arrays
            return np.array(data, dtype=type(data[0][0]))
        else:
            # Leave other lists unchanged
            return [yaml_to_numpy(x) for x in data]
    else:
        # Return the data as is for non-list, non-dict types
        return data

def numpy_to_yaml(data: Any):
    """
    Convert numpy arrays in the configuration dictionary back to lists.

    Parameters:
    data (Any): The data to convert.

    Returns:
    Any: The data with numpy arrays converted to lists.
    """

    if isinstance(data, dict):
        # Process dictionaries recursively
        return {key: numpy_to_yaml(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        # Convert numpy arrays to lists
        return data.tolist()
    elif isinstance(data, list):
        # Process lists recursively
        return [numpy_to_yaml(x) for x in data]
    else:
        # Return the data as is for non-list, non-dict, non-numpy types
        return data


class CustomDumper(yaml.SafeDumper):
    """
    Custom YAML Dumper that forces lists to be displayed in bracket format [x, y, z]
    while keeping dictionaries in the usual indented structure.
    """

    def represent_list(self, data):
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Attach the new list representation to our dumper
CustomDumper.add_representer(list, CustomDumper.represent_list)

def dump_yaml_custom_format(data, file_path):
    """
    Dumps YAML data where lists are in bracket format but dictionaries use standard indentation.

    Args:
        data (dict): The data to be serialized.
        file_path (str): Path to the output YAML file.
    """
    with open(file_path, "w") as f:
        yaml.dump(data, f, Dumper=CustomDumper, sort_keys=False, default_flow_style=False)

def save_config(
        config: dict,
        save_dir: str,
        file_name: str = 'config_used.pickle'
):
    """
    Save the config to a file.

    Parameters:
    config (dict): The configuration dictionary.
    save_dir (str): The directory to the file.
    file_name (str): The name of the file.
    """

    save_path = os.path.join(save_dir, file_name)
    config_yaml = numpy_to_yaml(config)
    dump_yaml_custom_format(config_yaml, save_path)

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
        target = get_problem(problem_config, rng=rng)
        sampler = get_sampler(config['sampler'], target=target, rng=rng)

        sampler.run()
        sampler.write_data(config['output']['dir'])
        save_config(config, config['output']['dir'], 'config_used.yml')

    except Exception as e:
        logger.exception("An error occurred during the simulation: %s", str(e))

if __name__ == '__main__':

    main()