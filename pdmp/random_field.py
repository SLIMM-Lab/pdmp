import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, Protocol

from .project_field import (
    ConstantBasis,
    PiecewiseConstantBasis,
    NormPiecewiseConstantBasis,
    compute_coefficients,
    compute_coefficients_norm,
    squared_exponential_kernel,
)
from .distributions import MultivariateNormal, Distribution


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
    distribution: MultivariateNormal

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def coefficient_distribution(self) -> MultivariateNormal:
        """Return a multivariate normal distribution over coefficients."""
        return self.distribution

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
    def from_dict(
            cls,
            config: Dict[str, Any],
            *,
            rng: Optional[np.random.Generator] = None
    ) -> "GaussianRandomField":
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
            raise ValueError(
                "GaussianRandomField config must have name 'GaussianRandomField'."
            )
        interval: Tuple[float,
                        float] = tuple(config.get('interval',
                                                  (0.0, 1.0)))  # type: ignore
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
            raise ValueError(
                f"Unsupported basis '{basis_name}'. Choose 'PiecewiseConstant' or 'NormPiecewiseConstant'."
            )
        if np.isscalar(mean_cfg):
            mean = np.ones(d) * float(mean_cfg)
        else:
            mean_arr = np.array(mean_cfg)
            if mean_arr.shape != (d, ):
                raise ValueError("mean must be scalar or length equal to dim")
            mean = mean_arr
        distribution = MultivariateNormal(mean, cov, rng=rng)
        return cls(basis=basis,
                   mean=mean,
                   cov=cov,
                   kernel_params=kernel_params,
                   distribution=distribution)


def get_field(
        config: Dict[str, Any],
        rng: Optional[np.random.Generator] = None) -> GaussianRandomField:
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

    @property
    def coefficient_distribution(self):
        """Return the distribution over coefficients."""
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


@dataclass
class JaxRandomFieldBase:
    """JAX-compatible random field with arbitrary coefficient distribution.

    This base class separates the spatial structure (basis functions) from the
    coefficient distribution, allowing flexible combinations. Use this for cases
    where coefficients follow non-Gaussian distributions (e.g., exponential, uniform).

    Attributes:
        basis: Basis function object (e.g., PiecewiseConstantBasis)
        coefficient_dist: Distribution object for the coefficients (any Distribution subclass)
    """
    basis: Any  # PiecewiseConstantBasis or similar
    coefficient_dist: Distribution  # Distribution object

    @property
    def dim(self) -> int:
        """Number of parameters/coefficients."""
        return self.coefficient_dist.dim

    @property
    def coefficient_distribution(self) -> Distribution:
        """Return the coefficient distribution."""
        return self.coefficient_dist

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate field at spatial locations x given coefficients.

        Args:
            coeffs: Coefficient vector, shape (dim,)
            x: Spatial coordinates, shape (n_points, spatial_dim) or (n_points,)

        Returns:
            Field values at x, shape (n_points,)
        """
        # Convert x to numpy for basis evaluation
        x_np = np.asarray(x)

        # Handle 1D case: basis expects either (n_points,) or (n_points, 1)
        # For multi-dimensional basis, keep shape as (n_points, spatial_dim)
        if x_np.ndim == 1:
            # 1D points - basis can handle this
            pass
        elif x_np.ndim == 2:
            # Multi-dimensional points - basis should handle (n_points, spatial_dim)
            pass
        else:
            # Flatten higher-dimensional arrays
            x_np = x_np.ravel()

        # Evaluate basis functions (returns shape (n_points, dim))
        phi = self.basis(x_np)  # (n_points, dim)

        # Convert to JAX array and multiply with coefficients
        phi_jax = jnp.array(phi)
        return phi_jax @ coeffs  # (n_points,)

    @classmethod
    def from_dict(
            cls,
            config: Dict[str, Any],
            *,
            rng: Optional[np.random.Generator] = None) -> "JaxRandomFieldBase":
        """Construct from configuration dictionary.

        Expected keys:
            name: must be 'JaxRandomField'
            basis: dictionary specifying basis configuration
                type: 'PiecewiseConstant' or 'NormPiecewiseConstant'
                dim: number of basis functions
                interval: (a, b) domain tuple (default (0, 1))
            coefficient_distribution: dictionary specifying distribution
                name: distribution type (e.g., 'MultivariateNormal', 'Independent', etc.)
                ... other distribution-specific parameters

        Example:
            config = {
                'name': 'JaxRandomField',
                'basis': {
                    'type': 'PiecewiseConstant',
                    'dim': 10,
                    'interval': [0.0, 1.0]
                },
                'coefficient_distribution': {
                    'name': 'MultivariateNormal',
                    'mean': [0.0] * 10,
                    'cov': np.eye(10)
                }
            }
        """
        if config.get('name', None) != 'JaxRandomField':
            raise ValueError(
                "JaxRandomFieldBase config must have name 'JaxRandomField'.")

        # Parse basis
        basis_config = config.get('basis', {})
        basis_type = basis_config.get('type', 'PiecewiseConstant')
        basis_dim = int(basis_config.get('dim', 1))
        basis_interval = tuple(basis_config.get('interval', (0.0, 1.0)))

        if basis_type == 'PiecewiseConstant':
            basis = PiecewiseConstantBasis(basis_dim, basis_interval)
        elif basis_type == 'NormPiecewiseConstant':
            basis = NormPiecewiseConstantBasis(basis_dim, basis_interval)
        elif basis_type == 'Constant':
            basis = ConstantBasis(basis_interval)
        else:
            raise ValueError(f"Unsupported basis type '{basis_type}'")

        # Parse coefficient distribution
        dist_config = config.get('coefficient_distribution', {})
        dist_name = dist_config.get('name', 'MultivariateNormal')

        if dist_name == 'MultivariateNormal':
            mean_cfg = dist_config.get('mean', 0.0)
            if np.isscalar(mean_cfg):
                mean = np.ones(basis_dim) * float(mean_cfg)
            else:
                mean = np.array(mean_cfg)
                if mean.shape != (basis_dim, ):
                    raise ValueError(
                        "mean must be scalar or length equal to basis dim")

            cov_cfg = dist_config.get('cov', None)
            if cov_cfg is None:
                cov = np.eye(basis_dim)
            else:
                cov = np.array(cov_cfg)
                if cov.shape != (basis_dim, basis_dim):
                    raise ValueError(f"cov must be ({basis_dim}, {basis_dim})")

            coefficient_dist = MultivariateNormal(mean, cov, rng=rng)
        else:
            raise ValueError(
                f"Unsupported coefficient distribution '{dist_name}'. "
                "Use JaxGaussianRandomField for kernel-based Gaussian fields.")

        return cls(basis=basis, coefficient_dist=coefficient_dist)


@dataclass
class JaxGaussianRandomField:
    """JAX-compatible Gaussian random field with kernel-based covariance structure.

    This specialized class represents Gaussian random fields derived from spatial
    covariance kernels (e.g., squared exponential). The covariance matrix is
    computed by projecting the kernel onto a finite basis.

    For non-Gaussian or arbitrary coefficient distributions, use JaxRandomFieldBase instead.

    Attributes:
        basis: Basis function object (e.g., PiecewiseConstantBasis)
        mean: Mean coefficient vector
        cov: Covariance matrix (computed from kernel projection)
        kernel_params: Parameters used to construct the covariance (e.g., {'sigma': 1.0, 'l': 0.3})
    """
    basis: Any  # PiecewiseConstantBasis or similar
    mean: np.ndarray
    cov: np.ndarray
    kernel_params: Dict[str, float]
    distribution: MultivariateNormal

    @property
    def dim(self) -> int:
        """Number of parameters/coefficients."""
        return int(self.mean.shape[0])

    @property
    def coefficient_distribution(self) -> MultivariateNormal:
        """Return Gaussian prior over coefficients."""
        return self.distribution

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate field at spatial locations x given coefficients.

        Args:
            coeffs: Coefficient vector, shape (dim,)
            x: Spatial coordinates, shape (n_points, spatial_dim) or (n_points,)

        Returns:
            Field values at x, shape (n_points,)
        """
        # Convert x to numpy for basis evaluation
        x_np = np.asarray(x)

        # Handle 1D case: basis expects either (n_points,) or (n_points, 1)
        # For multi-dimensional basis, keep shape as (n_points, spatial_dim)
        if x_np.ndim == 1:
            # 1D points - basis can handle this
            pass
        elif x_np.ndim == 2:
            # Multi-dimensional points - basis should handle (n_points, spatial_dim)
            pass
        else:
            # Flatten higher-dimensional arrays
            x_np = x_np.ravel()

        # Evaluate basis functions (returns shape (n_points, dim))
        phi = self.basis(x_np)  # (n_points, dim)

        # Convert to JAX array and multiply with coefficients
        phi_jax = jnp.array(phi)
        return phi_jax @ coeffs  # (n_points,)

    @classmethod
    def from_dict(
            cls,
            config: Dict[str, Any],
            *,
            rng: Optional[np.random.Generator] = None
    ) -> "JaxGaussianRandomField":
        """Construct Gaussian field from kernel parameters.

        Expected keys:
            name: must be 'JaxGaussianRandomField'
            dim: number of coefficients
            mean: scalar or list/array (length dim)
            interval: (a, b) domain tuple (default (0, 1))
            kernel_params: {'sigma': float, 'l': float}
            basis: 'PiecewiseConstant' or 'NormPiecewiseConstant' (default 'PiecewiseConstant')

        Example:
            config = {
                'name': 'JaxGaussianRandomField',
                'dim': 10,
                'mean': 0.0,
                'interval': [0.0, 1.0],
                'kernel_params': {'sigma': 1.0, 'l': 0.3},
                'basis': 'PiecewiseConstant'
            }
        """
        if config.get('name', None) != 'JaxGaussianRandomField':
            raise ValueError(
                "JaxGaussianRandomField config must have name 'JaxGaussianRandomField'."
            )

        interval = tuple(config.get('interval', (0.0, 1.0)))
        dim = int(config['dim'])
        kernel_params = config.get('kernel_params', {'sigma': 1.0, 'l': 0.3})
        mean_cfg = config.get('mean', 0.0)
        basis_name = config.get('basis', 'PiecewiseConstant')

        # Create basis and compute covariance from kernel
        if basis_name == 'PiecewiseConstant':
            basis = PiecewiseConstantBasis(dim, interval)
            cov = compute_coefficients(
                squared_exponential_kernel,
                basis,
                interval,
                kernel_params=kernel_params,
            )
        elif basis_name == 'NormPiecewiseConstant':
            basis = NormPiecewiseConstantBasis(dim, interval)
            cov = compute_coefficients_norm(
                squared_exponential_kernel,
                basis,
                interval,
                kernel_params=kernel_params,
            )
        else:
            raise ValueError(
                f"Unsupported basis '{basis_name}'. "
                "Choose 'PiecewiseConstant' or 'NormPiecewiseConstant'.")

        # Create mean vector
        if np.isscalar(mean_cfg):
            mean = np.ones(dim) * float(mean_cfg)
        else:
            mean = np.array(mean_cfg)
            if mean.shape != (dim, ):
                raise ValueError("mean must be scalar or length equal to dim")

        distribution = MultivariateNormal(mean, cov, rng=rng)

        return cls(basis=basis,
                   mean=mean,
                   cov=cov,
                   kernel_params=kernel_params,
                   distribution=distribution)


@dataclass
class JaxExponentialRecoveryField:
    """JAX-compatible field with exponential recovery profile.

    F(x) = F_inf * (1 - (1 - rho) * exp(-x/l))

    Parameters to infer: [rho, l] (and optionally F_inf when infer_f_infinity=True)
    Fixed parameter: F_inf (when infer_f_infinity=False)

    The field varies along the first spatial dimension and is constant along others.
    """
    f_infinity: float
    idx: int  # index of spatial dimension for recovery (default 0)
    coefficient_dist: Distribution
    infer_f_infinity: bool = False

    @property
    def dim(self) -> int:
        """Two parameters [rho, l], or three [rho, l, f_inf] when infer_f_infinity=True."""
        return 3 if self.infer_f_infinity else 2

    @property
    def coefficient_distribution(self) -> Distribution:
        return self.coefficient_dist

    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate field.

        Args:
            coeffs: [rho, l]
            x: coordinates

        Returns:
            Field values.
        """
        # coeffs[0] -> rho
        # coeffs[1] -> l
        rho = coeffs[0]
        l_scale = coeffs[1]
        f_inf = coeffs[2] if self.infer_f_infinity else self.f_infinity

        # Ensure x is JAX array
        x = jnp.atleast_1d(x)

        if x.ndim > 1:
            # Assume first dimension defines the variation
            x_val = x[:, self.idx]
        else:
            x_val = x

        # F(x) = F_inf * (1 - (1 - rho) * exp(-x/l))
        # Note: if l is 0 or negative, this might blow up physically,
        # but mathematically it evaluates. Distribution should constrain l > 0.

        return f_inf * (1.0 - (1.0 - rho) * jnp.exp(-x_val / l_scale))

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        *,
        rng: Optional[np.random.Generator] = None
    ) -> "JaxExponentialRecoveryField":
        """Construct from config.

        Expected keys:
            name: 'JaxExponentialRecoveryField'
            f_infinity: float
            idx: index of spatial dimension for recovery (default 0)
            coefficient_distribution: distribution config
        """
        if config.get('name') != 'JaxExponentialRecoveryField':
            raise ValueError(
                "Config name must be 'JaxExponentialRecoveryField'")

        f_infinity = float(config.get('f_infinity', 1.0))
        idx = int(config.get('idx', 0))
        infer_f_infinity = bool(config.get('infer_f_infinity', False))
        n_dim = 3 if infer_f_infinity else 2

        dist_config = config.get('coefficient_distribution', {})

        dist_name = dist_config.get('name', 'MultivariateNormal')
        if dist_name == 'MultivariateNormal':
            default_mean = [0.5, 1.0, 1.0] if infer_f_infinity else [0.5, 1.0]
            default_cov = (np.eye(n_dim) * 0.1).tolist()
            mean_cfg = dist_config.get('mean', default_mean)
            cov_cfg = dist_config.get('cov', default_cov)

            mean = np.array(mean_cfg)
            if mean.shape != (n_dim,):
                raise ValueError(
                    f"Mean must have length {n_dim} "
                    f"({'rho, l, f_inf' if infer_f_infinity else 'rho, l'})")

            cov = np.array(cov_cfg)
            if cov.shape != (n_dim, n_dim):
                raise ValueError(f"Covariance must be ({n_dim}, {n_dim})")

            dist = MultivariateNormal(mean, cov, rng=rng)
        else:
            raise ValueError(f"Unsupported distribution {dist_name}")

        return cls(f_infinity=f_infinity, idx=idx, coefficient_dist=dist,
                   infer_f_infinity=infer_f_infinity)


@dataclass
class JaxConstantField:
    """JAX-compatible constant random field.

    This is the simplest random field: a single parameter that is constant
    throughout the entire domain. The field value is the same everywhere and
    follows a Normal distribution.

    Attributes:
        distribution: MultivariateNormal distribution over the single parameter
    """
    distribution: MultivariateNormal  # Distribution over the single parameter

    @property
    def dim(self) -> int:
        """Single parameter."""
        return 1

    @property
    def coefficient_distribution(self) -> MultivariateNormal:
        """Return coefficient distribution."""
        return self.distribution

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
            raise ValueError(
                f"JaxConstantField expects 1 coefficient, got {coeffs.shape[0]}"
            )

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

    @classmethod
    def from_dict(
            cls,
            config: Dict[str, Any],
            *,
            rng: Optional[np.random.Generator] = None) -> "JaxConstantField":
        """Construct from configuration dictionary.

        Expected keys:
            name: must be 'JaxConstantField'
            mean: scalar mean value (default 0.0)
            std: scalar standard deviation (default 1.0)
        """
        if config.get('name', None) != 'JaxConstantField':
            raise ValueError(
                "JaxConstantField config must have name 'JaxConstantField'.")
        mean = float(config.get('mean', 0.0))
        std = float(config.get('std', 1.0))
        distribution = MultivariateNormal(np.array([mean]),
                                          np.array([[std**2]]),
                                          rng=rng)
        return cls(distribution=distribution)


def get_jax_field(config: Dict[str, Any], rng: Optional[np.random.Generator] = None) \
        -> JaxConstantField | JaxGaussianRandomField | JaxRandomFieldBase | JaxExponentialRecoveryField:
    """Factory for JAX-compatible random field objects.

    Supported field types:
        - JaxConstantField: Single constant parameter
        - JaxGaussianRandomField: Gaussian field with kernel-based covariance
        - JaxRandomField: Generic field with arbitrary coefficient distribution
        - JaxExponentialRecoveryField: Exponential recovery profile F(x) = F_inf(1 - (1-rho)exp(-x/l))

    Args:
        config: configuration dictionary with 'name' key specifying field type
        rng: optional random generator

    Returns:
        JaxRandomField instance

    Examples:
        Constant field:
            config = {'name': 'JaxConstantField', 'mean': 100.0, 'std': 20.0}

        Gaussian random field with kernel:
            config = {
                'name': 'JaxGaussianRandomField',
                'dim': 10,
                'mean': 0.0,
                'kernel_params': {'sigma': 1.0, 'l': 0.3},
                'basis': 'PiecewiseConstant',
                'interval': [0.0, 1.0]
            }

        Generic field with custom distribution:
            config = {
                'name': 'JaxRandomField',
                'basis': {'type': 'PiecewiseConstant', 'dim': 10, 'interval': [0.0, 1.0]},
                'coefficient_distribution': {'name': 'MultivariateNormal', 'mean': 0.0, 'cov': np.eye(10)}
            }
    """
    name = config.get('name', None)
    if name == 'JaxConstantField':
        return JaxConstantField.from_dict(config, rng=rng)
    elif name == 'JaxGaussianRandomField':
        return JaxGaussianRandomField.from_dict(config, rng=rng)
    elif name == 'JaxRandomField':
        return JaxRandomFieldBase.from_dict(config, rng=rng)
    elif name == 'JaxExponentialRecoveryField':
        return JaxExponentialRecoveryField.from_dict(config, rng=rng)
    else:
        raise ValueError(
            f"Unknown JAX field type: {name}. "
            "Supported types: 'JaxConstantField', 'JaxGaussianRandomField', 'JaxRandomField', 'JaxExponentialRecoveryField'"
        )
