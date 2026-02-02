import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, Protocol

from .project_field import (
    PiecewiseConstantBasis,
    NormPiecewiseConstantBasis,
    compute_coefficients,
    compute_coefficients_norm,
    squared_exponential_kernel,
)
from .distributions import MultivariateNormal


@dataclass
class GaussianRandomField:
    """Encapsulates a Gaussian random field represented in a finite basis.

    This class stores a basis, mean vector, covariance (correlation) matrix of the
    coefficients, and kernel parameters used to construct the field. It provides
    utilities for evaluating the field at spatial locations given coefficient
    samples and for obtaining the coefficient distribution as a MultivariateNormal.
    """
    basis: PiecewiseConstantBasis
    mean: np.ndarray
    cov: np.ndarray
    kernel_params: Dict[str, float]

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def coefficient_distribution(self, rng: Optional[np.random.Generator] = None) -> MultivariateNormal:
        """Return a multivariate normal distribution over coefficients."""
        return MultivariateNormal(self.mean, self.cov, rng=rng)

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Return the design matrix Φ(x) where each column corresponds to a basis function evaluated at x.

        Args:
            x: 1D array of spatial points in the field domain.
        Returns:
            Φ(x) with shape (len(x), dim)
        """
        phi = self.basis(x)  # shape (len(x), dim)
        return phi

    def evaluate(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the random field realization at locations x.

        Args:
            coeffs: Coefficient vector (dim,)
            x: Spatial points
        Returns:
            Field values at x (len(x),)
        """
        Φ = self.design_matrix(x)
        return Φ @ coeffs

    @classmethod
    def from_dict(cls, config: Dict[str, Any], *, rng: Optional[np.random.Generator] = None) -> "GaussianRandomField":
        """Construct a GaussianRandomField from a configuration dictionary.

        Expected keys:
            name: must be 'GaussianRandomField'
            dim: number of coefficients
            mean: scalar or list/array (length dim)
            interval: (a,b) domain tuple (default (0,1))
            kernel_params: {'sigma': float, 'l': float}
            basis: one of {'PiecewiseConstant', 'NormPiecewiseConstant'} (default 'PiecewiseConstant')
        """
        if config.get('name', None) not in {'GaussianRandomField'}:
            raise ValueError("GaussianRandomField config must have name 'GaussianRandomField'.")
        interval: Tuple[float, float] = tuple(config.get('interval', (0.0, 1.0)))  # type: ignore
        d = int(config['dim'])
        kernel_params = config.get('kernel_params', {'sigma': 1.0, 'l': 0.3})
        mean_cfg = config.get('mean', 0.0)
        basis_name = config.get('basis', 'PiecewiseConstant')
        if basis_name == 'PiecewiseConstant':
            basis = PiecewiseConstantBasis(d, interval)
            cov = compute_coefficients(
                squared_exponential_kernel,
                basis,
                interval,
                kernel_params=kernel_params,
            )
        elif basis_name == 'NormPiecewiseConstant':
            basis = NormPiecewiseConstantBasis(d, interval)
            cov = compute_coefficients_norm(
                squared_exponential_kernel,
                basis,
                interval,
                kernel_params=kernel_params,
            )
        else:
            raise ValueError(f"Unsupported basis '{basis_name}'. Choose 'PiecewiseConstant' or 'NormPiecewiseConstant'.")
        if np.isscalar(mean_cfg):
            mean = np.ones(d) * float(mean_cfg)
        else:
            mean_arr = np.array(mean_cfg)
            if mean_arr.shape != (d,):
                raise ValueError("mean must be scalar or length equal to dim")
            mean = mean_arr
        # cov = compute_coefficients(
        #     squared_exponential_kernel,
        #     basis,
        #     interval,
        #     kernel_params=kernel_params,
        # )
        return cls(basis=basis, mean=mean, cov=cov, kernel_params=kernel_params)


def get_field(config: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> GaussianRandomField:
    """Factory for GaussianRandomField objects.

    Args:
        config: configuration dictionary for the field
        rng: optional random generator (not currently used)
    Returns:
        GaussianRandomField instance
    """
    return GaussianRandomField.from_dict(config, rng=rng)


# ============================================================================
# JAX-Compatible Random Fields
# ============================================================================


class JaxRandomField(Protocol):
    """Protocol (interface) for JAX-compatible random fields.

    A JAX random field must support:
    - Evaluation at arbitrary spatial coordinates given coefficients
    - Pure functional operations compatible with jax.vjp/jax.grad
    - Coefficient dimension and distribution specification
    """

    @property
    def dim(self) -> int:
        """Number of parameters/coefficients."""
        ...

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate field at spatial locations x given coefficients.

        Args:
            coeffs: Coefficient vector, shape (dim,)
            x: Spatial coordinates, shape (n_points, spatial_dim) or (n_points,)

        Returns:
            Field values at x, shape (n_points,)
        """
        ...

    def coefficient_distribution(self, rng: Optional[np.random.Generator] = None) -> MultivariateNormal:
        """Return the prior distribution over coefficients."""
        ...


@dataclass
class JaxConstantField:
    """JAX-compatible constant random field.

    This is the simplest random field: a single parameter that is constant
    throughout the entire domain. The field value is the same everywhere and
    follows a Normal distribution.

    Attributes:
        mean: Mean of the single parameter
        std: Standard deviation of the single parameter
    """
    mean: float
    std: float

    @property
    def dim(self) -> int:
        """Single parameter."""
        return 1

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate constant field.

        Args:
            coeffs: Single coefficient, shape (1,) or scalar
            x: Spatial coordinates (ignored for constant field)

        Returns:
            Constant field value broadcast to match x shape
        """
        coeffs = jnp.atleast_1d(coeffs)
        if coeffs.shape[0] != 1:
            raise ValueError(f"JaxConstantField expects 1 coefficient, got {coeffs.shape[0]}")

        # Infer output shape from x
        if isinstance(x, (int, float)):
            return coeffs[0]
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            n_points = x.shape[0]
        else:
            n_points = x.shape[0]

        # Return constant value broadcast to all points
        return jnp.full(n_points, coeffs[0])

    def coefficient_distribution(self, rng: Optional[np.random.Generator] = None) -> MultivariateNormal:
        """Return Normal(mean, std^2) distribution."""
        mean_vec = np.array([self.mean])
        cov_mat = np.array([[self.std**2]])
        return MultivariateNormal(mean_vec, cov_mat, rng=rng)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], *, rng: Optional[np.random.Generator] = None) -> "JaxConstantField":
        """Construct from configuration dictionary.

        Expected keys:
            name: must be 'JaxConstantField'
            mean: scalar mean value (default 0.0)
            std: scalar standard deviation (default 1.0)
        """
        if config.get('name', None) != 'JaxConstantField':
            raise ValueError("JaxConstantField config must have name 'JaxConstantField'.")
        mean = float(config.get('mean', 0.0))
        std = float(config.get('std', 1.0))
        return cls(mean=mean, std=std)


def get_jax_field(config: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> JaxRandomField:
    """Factory for JAX-compatible random field objects.

    Args:
        config: configuration dictionary with 'name' key specifying field type
        rng: optional random generator (for future use)

    Returns:
        JaxRandomField instance
    """
    name = config.get('name', None)
    if name == 'JaxConstantField':
        return JaxConstantField.from_dict(config, rng=rng)
    else:
        raise ValueError(f"Unknown JAX field type: {name}")
