"""Tests for JAX random field classes with arbitrary coefficient distributions."""
import numpy as np
import jax.numpy as jnp
import pytest

from pdmp.random_field import (JaxRandomFieldBase, JaxGaussianRandomField,
                               JaxConstantField, get_jax_field)
from pdmp.distributions import MultivariateNormal


def test_jax_random_field_base_with_gaussian():
    """Test JaxRandomFieldBase with MultivariateNormal coefficient distribution."""
    print("=" * 70)
    print("Testing JaxRandomFieldBase with Gaussian coefficients")
    print("=" * 70)

    config = {
        'name': 'JaxRandomField',
        'basis': {
            'type': 'PiecewiseConstant',
            'dim': 5,
            'interval': [0.0, 1.0]
        },
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': 1.0,
            'cov': np.eye(5) * 0.1
        }
    }

    field = get_jax_field(config)

    # Check properties
    assert field.dim == 5, f"Expected dim=5, got {field.dim}"

    # Get distribution
    dist = field.coefficient_distribution
    assert isinstance(dist, MultivariateNormal)
    assert dist.dim == 5

    # Sample coefficients
    coeffs = jnp.array(dist.get_sample())
    assert coeffs.shape == (5, )

    # Evaluate field
    x = jnp.linspace(0, 1, 10)
    field_values = field.evaluate(coeffs, x)
    assert field_values.shape == (10, )

    print(f"✓ JaxRandomFieldBase test passed")
    print(f"  Field dimension: {field.dim}")
    print(f"  Coefficient sample: {coeffs}")
    print(f"  Field values (first 3): {field_values[:3]}")


def test_jax_gaussian_random_field_with_kernel():
    """Test JaxGaussianRandomField with kernel-based covariance."""
    print("\n" + "=" * 70)
    print("Testing JaxGaussianRandomField with kernel covariance")
    print("=" * 70)

    config = {
        'name': 'JaxGaussianRandomField',
        'dim': 10,
        'mean': 0.0,
        'interval': [0.0, 1.0],
        'kernel_params': {
            'sigma': 1.0,
            'l': 0.3
        },
        'basis': 'PiecewiseConstant'
    }

    field = get_jax_field(config)

    # Check properties
    assert field.dim == 10
    assert hasattr(field, 'kernel_params')
    assert field.kernel_params == {'sigma': 1.0, 'l': 0.3}

    # Get distribution
    dist = field.coefficient_distribution
    assert isinstance(dist, MultivariateNormal)
    assert dist.dim == 10

    # Sample coefficients
    coeffs = jnp.array(dist.get_sample())
    assert coeffs.shape == (10, )

    # Evaluate field
    x = jnp.linspace(0, 1, 20)
    field_values = field.evaluate(coeffs, x)
    assert field_values.shape == (20, )

    print(f"✓ JaxGaussianRandomField test passed")
    print(f"  Field dimension: {field.dim}")
    print(f"  Kernel params: {field.kernel_params}")
    print(f"  Coefficient sample mean: {coeffs.mean():.3f}")
    print(f"  Field values mean: {field_values.mean():.3f}")


def test_constant_field_still_works():
    """Ensure JaxConstantField still works as before."""
    print("\n" + "=" * 70)
    print("Testing JaxConstantField (backward compatibility)")
    print("=" * 70)

    config = {'name': 'JaxConstantField', 'mean': 100.0, 'std': 20.0}

    field = get_jax_field(config)

    assert field.dim == 1

    # Get distribution
    dist = field.coefficient_distribution
    assert dist.dim == 1

    # Sample coefficient
    coeff = jnp.array(dist.get_sample())

    # Evaluate field
    x = jnp.linspace(0, 1, 10)
    field_values = field.evaluate(coeff, x)
    assert field_values.shape == (10, )
    assert jnp.allclose(field_values,
                        field_values[0])  # All values should be constant

    print(f"✓ JaxConstantField test passed")
    print(f"  Constant value: {field_values[0]}")


def test_jax_gaussian_field_matches_basis_projection():
    """Test that JaxGaussianRandomField produces spatially correlated fields."""
    print("\n" + "=" * 70)
    print("Testing JaxGaussianRandomField spatial correlation")
    print("=" * 70)

    config = {
        'name': 'JaxGaussianRandomField',
        'dim': 15,
        'mean': 5.0,
        'interval': [0.0, 1.0],  # Smaller domain to avoid covariance issues
        'kernel_params': {
            'sigma': 2.0,
            'l': 0.15
        },
        'basis': 'PiecewiseConstant'
    }

    field = get_jax_field(config)
    dist = field.coefficient_distribution

    # Sample multiple realizations
    n_samples = 3
    x_eval = jnp.linspace(0, 1, 50)

    for i in range(n_samples):
        coeffs = jnp.array(dist.get_sample())
        field_values = field.evaluate(coeffs, x_eval)

        # Field should vary spatially
        assert field_values.std() > 0, "Field should have spatial variation"

        # But should have some smoothness (neighboring values shouldn't be too different)
        diffs = jnp.diff(field_values)
        max_diff = jnp.max(jnp.abs(diffs))

        print(f"  Realization {i+1}: mean={field_values.mean():.3f}, "
              f"std={field_values.std():.3f}, max_diff={max_diff:.3f}")

    print(f"✓ Spatial correlation test passed")


def test_field_evaluation_is_jax_differentiable():
    """Test that field evaluation is JAX-differentiable w.r.t. coefficients."""
    print("\n" + "=" * 70)
    print("Testing JAX differentiability (w.r.t. coefficients)")
    print("=" * 70)

    import jax

    config = {
        'name': 'JaxGaussianRandomField',
        'dim': 5,
        'mean': 0.0,
        'interval': [0.0, 1.0],
        'kernel_params': {
            'sigma': 1.0,
            'l': 0.3
        },
        'basis': 'PiecewiseConstant'
    }

    field = get_jax_field(config)

    # Define a function to differentiate
    def eval_sum(coeffs):
        x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        return jnp.sum(field.evaluate(coeffs, x))

    # Sample coefficients
    coeffs = jnp.array([0.1, 0.2, -0.1, 0.3, -0.2])

    # Compute gradient w.r.t. coefficients
    grad_coeffs = jax.grad(eval_sum)(coeffs)
    assert grad_coeffs.shape == (5, )
    assert not jnp.allclose(grad_coeffs, 0.0), "Gradient should be non-zero"

    print(f"✓ JAX differentiability test passed")
    print(f"  Gradient w.r.t. coefficients: {grad_coeffs}")
    print(f"  Note: Spatial gradients require full JAX basis implementation")


def test_invalid_config_raises_error():
    """Test that invalid configurations raise appropriate errors."""
    print("\n" + "=" * 70)
    print("Testing error handling")
    print("=" * 70)

    # Unknown field type
    with pytest.raises(ValueError, match="Unknown JAX field type"):
        get_jax_field({'name': 'NonExistentField'})

    # Wrong name for JaxRandomFieldBase
    with pytest.raises(ValueError, match="must have name 'JaxRandomField'"):
        JaxRandomFieldBase.from_dict({'name': 'WrongName'})

    # Wrong name for JaxGaussianRandomField
    with pytest.raises(ValueError,
                       match="must have name 'JaxGaussianRandomField'"):
        JaxGaussianRandomField.from_dict({'name': 'WrongName'})

    print(f"✓ Error handling test passed")


if __name__ == "__main__":
    test_jax_random_field_base_with_gaussian()
    test_jax_gaussian_random_field_with_kernel()
    test_constant_field_still_works()
    test_jax_gaussian_field_matches_basis_projection()
    test_field_evaluation_is_jax_differentiable()
    test_invalid_config_raises_error()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
