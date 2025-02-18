import os
from typing import Any, Union

import numpy as np
import yaml

from pdmp.distributions import Distribution, CubicDistribution, MultivariateNormal, Posterior, TransformedDistribution
from pdmp.distributions import get_prior, get_likelihood
from pdmp.forward_model import get_model
from pdmp.sampler import Sampler
from pdmp.zigzag import ZigZagSampler
from pdmp.mcmc import RandomWalkMetropolisSampler
from pdmp.surrogates import SurrogateModel, LaplaceSurrogate, NeuralNetwork, GaussianProcess

def get_target(
        config: dict[str, Any],
        rng: np.random.Generator
) -> Union[Distribution, Posterior]:
    """
    Load the problem configuration.

    Parameters:
    problem_config (dict): The problem configuration.

    Returns:
    Distribution: The target distribution.
    """

    if config['name'] == 'Cubic':
        return CubicDistribution.from_dict(config, rng=rng)

    elif config['name'] == 'Gaussian':
        return MultivariateNormal.from_dict(config, rng=rng)

    elif config['name'] == 'BayesianInverse':
        # check if config has the necessary keys
        if 'prior' not in config or 'likelihood' not in config:
            raise ValueError("Parameters must include 'prior' and 'likelihood'.")

        # get the prior distribution and the model
        prior = get_prior(config['prior'], rng=rng)
        model = get_model(config['model'])
        likelihood = get_likelihood(config['likelihood'], model=model, rng=rng)
        return Posterior(prior=prior, likelihood=likelihood, rng=rng)

    elif config['name'] == 'Transformed':
        target = get_target(config['distribution'], rng=rng)
        return TransformedDistribution(base_distribution=target, params=config)

    else:
        raise ValueError(f"Problem {config['name']} not recognized.")

def get_sampler(
        sampler_config: dict[str, Any],
        target: Distribution,
        surrogate: SurrogateModel = None,
        rng: np.random.Generator = None

) -> Sampler:
    """
    Load the sampler configuration.

    Parameters:
    sampler_config (dict): The sampler configuration.
    target (MultivariateNormal): The target distribution.
    rng (np.random.Generator): The random number generator. Default is None.

    Returns:
    ZigZagSampler: The sampler.
    """

    if sampler_config['name'] == 'ZigZag':
        return ZigZagSampler.from_dict(sampler_config, target=target, rng=rng, surrogate=surrogate)
    elif sampler_config['name'] == 'RandomWalkMetropolis':
        return RandomWalkMetropolisSampler.from_dict(sampler_config, target=target, rng=rng, surrogate=surrogate)
    else:
        raise ValueError(f"Sampler {sampler_config['name']} not recognized.")

def get_surrogate(
        config: dict[str, Any],
        target: Distribution,
        rng: np.random.Generator = None
) -> SurrogateModel:
    """
    Get a surrogate model from a dictionary.

    Parameters:
    config (dict): The configuration dictionary.
    target (Distribution): The target distribution.
    rng (np.random.Generator): The random number generator. Default is None.

    Returns:
    LaplaceSurrogate: A surrogate model for the target distribution.
    """

    if config['name'] == 'Laplace':
        return LaplaceSurrogate.from_dict(config, target=target, rng=rng)

    if config['name'] == 'NeuralNetwork':
        return NeuralNetwork.from_dict(config, target=target, rng=rng)

    if config['name'] == 'GaussianProcess':
        return GaussianProcess.from_dict(config, target=target, rng=rng)


def yaml_to_numpy(data: Any, exclude_keys: set = None):
    """
    Convert the configuration dictionary to numpy arrays.

    Parameters:
    dictionary (dict): The dictionary.

    Returns:
    dict: The configuration dictionary with numpy arrays.
    """

    # necessary to keep list of NN hidden layers as lists
    if exclude_keys is None:
        exclude_keys = set()

    if isinstance(data, dict):
        # Process dictionaries recursively
        return {key: yaml_to_numpy(value, exclude_keys) if key not in exclude_keys else value
                for key, value in data.items()}
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
