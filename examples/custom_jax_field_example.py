"""Example: Extending JAX Random Fields

This file demonstrates how to create custom JAX-compatible random fields
by implementing the JaxRandomField protocol.
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from pdmp.random_field import JaxRandomField
from pdmp.distributions import MultivariateNormal


@dataclass
class JaxLayeredField:
    """Example: Layered material with different properties in z-direction.

    This field divides the domain into horizontal layers, each with its own
    material property parameter. Useful for modeling stratified materials
    or layered manufacturing.

    Attributes:
        n_layers: Number of layers in z-direction
        z_bounds: Tuple of (z_min, z_max) for the domain
        mean: Prior mean for each layer parameter
        std: Prior standard deviation for each layer parameter
    """
    n_layers: int
    z_bounds: tuple
    mean: float
    std: float

    @property
    def dim(self) -> int:
        """Number of parameters = number of layers."""
        return self.n_layers

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate layered field at spatial locations.

        Args:
            coeffs: Layer parameters, shape (n_layers,)
            x: Spatial coordinates, shape (n_points, 3) or (n_points, spatial_dim)
                Assumes last dimension is z-coordinate

        Returns:
            Field values at x, shape (n_points,)
        """
        coeffs = jnp.atleast_1d(coeffs)
        if coeffs.shape[0] != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} coefficients, got {coeffs.shape[0]}")

        # Extract z-coordinates
        x = jnp.atleast_2d(x)
        if x.ndim == 1:
            z = x  # Assume 1D input is z-coordinate
        else:
            z = x[:, -1]  # Last column is z

        # Determine which layer each point belongs to
        z_min, z_max = self.z_bounds
        layer_height = (z_max - z_min) / self.n_layers

        # Compute layer indices (0 to n_layers-1)
        layer_indices = jnp.floor((z - z_min) / layer_height).astype(int)
        layer_indices = jnp.clip(layer_indices, 0, self.n_layers - 1)

        # Look up coefficient for each point's layer
        return coeffs[layer_indices]

    def coefficient_distribution(self, rng: Optional[np.random.Generator] = None) -> MultivariateNormal:
        """Return independent Normal priors for each layer."""
        mean_vec = np.full(self.n_layers, self.mean)
        cov_mat = (self.std ** 2) * np.eye(self.n_layers)
        return MultivariateNormal(mean_vec, cov_mat, rng=rng)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], *, rng: Optional[np.random.Generator] = None):
        """Construct from configuration dictionary.

        Expected keys:
            name: must be 'JaxLayeredField'
            n_layers: number of layers (default 3)
            z_bounds: [z_min, z_max] (default [0.0, 2.5])
            mean: scalar mean value (default 1e6)
            std: scalar standard deviation (default 2e5)
        """
        if config.get('name', None) != 'JaxLayeredField':
            raise ValueError("JaxLayeredField config must have name 'JaxLayeredField'.")

        n_layers = int(config.get('n_layers', 3))
        z_bounds = tuple(config.get('z_bounds', [0.0, 2.5]))
        mean = float(config.get('mean', 1e6))
        std = float(config.get('std', 2e5))

        return cls(n_layers=n_layers, z_bounds=z_bounds, mean=mean, std=std)


# Example usage and test
if __name__ == '__main__':
    print("=" * 70)
    print("Example: JaxLayeredField")
    print("=" * 70)

    # Create a 3-layer field
    field = JaxLayeredField(n_layers=3, z_bounds=(0.0, 3.0), mean=1e6, std=2e5)

    print(f"Field dimension: {field.dim}")
    print(f"Number of layers: {field.n_layers}")
    print(f"Z-bounds: {field.z_bounds}")

    # Define layer parameters (bottom to top)
    coeffs = jnp.array([8e5, 1e6, 1.2e6])  # Stiffness increases with height

    # Evaluate at various z-coordinates
    z_coords = jnp.array([[0.0, 0.0, 0.5],   # Layer 0
                          [0.0, 0.0, 1.5],   # Layer 1
                          [0.0, 0.0, 2.5]])  # Layer 2

    values = field.evaluate(coeffs, z_coords)

    print(f"\nEvaluation:")
    for i, (z, val) in enumerate(zip(z_coords[:, 2], values)):
        print(f"  z={z:.1f}: E={val:.2e} (Layer {i})")

    # Test prior distribution
    prior = field.coefficient_distribution()
    print(f"\nPrior:")
    print(f"  Mean: {prior.mean}")
    print(f"  Std:  {np.sqrt(np.diag(prior.cov))}")

    # Sample from prior
    rng = np.random.default_rng(42)
    prior_with_rng = field.coefficient_distribution(rng=rng)
    sample = prior_with_rng.get_sample()
    print(f"\nPrior sample: {sample}")

    print("\n" + "=" * 70)
    print("To use with JaxFemModel, add to get_jax_field():")
    print("=" * 70)
    print("""
# In pdmp/random_field.py, update get_jax_field():

def get_jax_field(config: Dict[str, Any], rng: Optional[np.random.Generator] = None):
    name = config.get('name', None)
    if name == 'JaxConstantField':
        return JaxConstantField.from_dict(config, rng=rng)
    elif name == 'JaxLayeredField':
        return JaxLayeredField.from_dict(config, rng=rng)
    else:
        raise ValueError(f"Unknown JAX field type: {name}")
    """)

    print("\nThen in config YAML:")
    print("""
model:
  name: JaxFem
  d_x: 1.0
  d_y: 1.0
  d_z: 3.0
  field:
    name: JaxLayeredField
    n_layers: 3
    z_bounds: [0.0, 3.0]
    mean: 1.0e6
    std: 2.0e5
    """)

    print("\n✓ Example complete!")
