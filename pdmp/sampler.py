import os
import time
import pickle

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

    # Names of the mutable-state attributes a subclass must snapshot to resume
    # an interrupted run with a bit-identical path. Overridden per sampler.
    # Attributes that are absent on a given instance (e.g. thinning-only state)
    # are skipped silently, so the same tuple works regardless of configuration.
    _CHECKPOINT_ATTRS = ()

    # Checkpointing is opt-in via enable_checkpointing(); disabled by default.
    _checkpoint_path = None
    _checkpoint_interval = None
    _last_ckpt = 0.0

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

    # ------------------------------------------------------------------ #
    # Checkpointing: snapshot/restore enough state to resume an           #
    # interrupted run with a bit-identical sample path.                   #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        """Collect the full mutable state needed to resume the run.

        Captures every attribute named in ``_CHECKPOINT_ATTRS`` (skipping any
        that are absent on this instance) plus the exact numpy ``Generator``
        bit-state. The rng state is the linchpin of the identical-path
        guarantee: restoring it reproduces every subsequent draw.
        """
        attrs = {
            name: getattr(self, name)
            for name in self._CHECKPOINT_ATTRS if hasattr(self, name)
        }
        return {
            'class': type(self).__name__,
            'dim': self._dim,
            'n_max': self._n_max,
            'rng_state': self._rng.bit_generator.state,
            'attrs': attrs,
        }

    def load_state_dict(self, d: dict):
        """Restore state captured by :meth:`state_dict` into this instance.

        The instance must already be constructed from the same config (same
        class, dimension and budget); the asserts guard against silently
        resuming into a mismatched run.
        """
        if d['class'] != type(self).__name__:
            raise ValueError(
                f"Checkpoint is for {d['class']}, but this sampler is "
                f"{type(self).__name__}.")
        if d['dim'] != self._dim or d['n_max'] != self._n_max:
            raise ValueError(
                f"Checkpoint dim/n_max ({d['dim']}, {d['n_max']}) does not "
                f"match this sampler ({self._dim}, {self._n_max}).")

        for name, value in d['attrs'].items():
            setattr(self, name, value)
        # Restore the rng last so it overrides any draws made during
        # construction (initial position/velocity etc.).
        self._rng.bit_generator.state = d['rng_state']

    def save_checkpoint(self, path: str):
        """Pickle the state dict atomically (write-temp-then-rename).

        Atomicity ensures a scheduler kill mid-write cannot corrupt an existing
        checkpoint. Full-precision binary is used deliberately: the human
        readable ``.dat`` outputs are lossy and must not be used to resume.
        """
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint file and restore it. Returns the resumed ``_iter``."""
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.load_state_dict(d)
        return self._iter

    def enable_checkpointing(self, path: str, interval_seconds: float):
        """Turn on periodic wall-clock checkpointing during the run loop."""
        self._checkpoint_path = path
        self._checkpoint_interval = interval_seconds
        self._last_ckpt = time.monotonic()

    def _maybe_checkpoint(self):
        """Write a checkpoint if the wall-clock interval has elapsed.

        Called from the run loop at event boundaries, so the snapshot is always
        consistent (rng state, ``_iter`` and the trajectory arrays all from the
        same moment).
        """
        if self._checkpoint_path is None:
            return
        now = time.monotonic()
        if now - self._last_ckpt >= self._checkpoint_interval:
            self.save_checkpoint(self._checkpoint_path)
            self._last_ckpt = now
            logger.debug(f"Checkpoint written at event {self._iter}")

    def run(self):
        """Run the sampler."""
        raise NotImplementedError

    def write_data(self, folder: str = '.', precision: int = 6):
        """Write the sampler data to a file.

        Args:
            folder: The folder to write the data to. Defaults to the current folder.
            precision: The precision of the data. Defaults to 6.
        """
        raise NotImplementedError
