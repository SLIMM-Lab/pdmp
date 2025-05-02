import numpy as np

from pdmp.distributions import Distribution
from pdmp.surrogates import SurrogateModel
from pdmp import logger

SAMPLER_REGISTRY = {}


def register_sampler(name):
    """Register a sampler class.

    Args:
        name (str): The name of the sampler.
    """

    def decorator(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError(f"Sampler {name} is already registered.")
        SAMPLER_REGISTRY[name] = cls
        return cls

    return decorator


class Sampler:
    """Base class for samplers."""

    def __init__(self, *args, **kwargs):

        # check for unused keyword arguments
        unused_args = []

        for key in kwargs:
            if key != 'name':
                unused_args.append(key)

        if len(unused_args) > 0:
            logger.warning(f"Unused keyword arguments: {unused_args}")

    @classmethod
    def from_dict(cls,
                  target: Distribution,
                  rng: np.random.Generator,
                  surrogate: SurrogateModel = None,
                  **kwargs):
        """Create a sampler from a configuration dictionary.

        Args:
            target: The target distribution.
            rng: The random number generator. Defaults to None.
            surrogate (SurrogateModel): The surrogate model. Defaults to None.

        Returns:
            Sampler: The configured sampler instance.
        """
        return cls(target=target, rng=rng, surrogate=surrogate, **kwargs)

    def run(self):
        """Run the sampler."""
        raise NotImplementedError

    def write_data(self, folder: str, precision: int = 6):
        """Write the sampler data to a file.

        Args:
            folder: The folder to write the data to.
            precision: The precision of the data. Defaults to 6.
        """
        raise NotImplementedError
