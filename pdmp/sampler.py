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

    def __init__(self, *args, **kawrgs):
        pass

    @classmethod
    def from_dict(cls, target, rng, **kwargs):
        """Create a sampler from a configuration dictionary.

        Args:
            target (Distribution): The target distribution.
            rng (np.random.Generator): The random number generator. Defaults to None.
            surrogate (SurrogateModel): The surrogate model. Defaults to None.

        Returns:
            Sampler: The configured sampler instance.
        """
        return cls(target=target, rng=rng, **kwargs)

    def run(self):
        raise NotImplementedError

    def write_data(self, folder: str, precision: int = 6):
        raise NotImplementedError