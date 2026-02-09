#!/usr/bin/env python3
"""Test script for JAX-compatible random fields with JaxFemModel."""

import numpy as np
import jax.numpy as jnp
from pdmp.random_field import JaxConstantField, get_jax_field
from pdmp.forward_model import JaxFemModel
from pdmp.distributions import MultivariateNormal

def test_jax_constant_field():
    """Test the JaxConstantField implementation."""
    print("=" * 60)
    print("Testing JaxConstantField")
    print("=" * 60)

    # Create a constant field
    mean = 10.
    std = 1.5
    field_dist = MultivariateNormal(mean=np.array([mean]), cov=np.array([[std**2]]))
    field = JaxConstantField(distribution=field_dist)

    print(f"Field dimension: {field.dim}")
    assert field.dim == 1, "JaxConstantField should have dimension 1"

    # Test evaluation at various points
    coeffs = jnp.array([15.])
    x = jnp.array([[0.0, 0.0, 0.0],
                   [0.5, 0.5, 1.0],
                   [1.0, 1.0, 2.5]])

    values = field.evaluate(coeffs, x)
    print(f"Field values at {x.shape[0]} points: {values}")
    assert values.shape == (3,), f"Expected shape (3,), got {values.shape}"
    assert jnp.allclose(values, 15.), "All values should equal the coefficient"

    # Test coefficient distribution
    dist = field.coefficient_distribution
    print(f"Prior mean: {dist.mean}")
    print(f"Prior covariance: {dist.cov}")
    assert np.allclose(dist.mean, [mean]), "Mean should match"
    assert np.allclose(dist.cov, [[std**2]]), "Covariance should match"

    print("✓ JaxConstantField tests passed!\n")


def test_jax_field_from_config():
    """Test creating JAX field from configuration."""
    print("=" * 60)
    print("Testing get_jax_field factory")
    print("=" * 60)

    config = {
        'name': 'JaxConstantField',
        'mean': 20.,
        'std': 5.
    }

    field = get_jax_field(config)
    print(f"Created field with dim={field.dim}")
    print(f"Field type: {type(field).__name__}")

    # Test that it works
    coeffs = jnp.array([18.])
    x = jnp.array([[0.5, 0.5, 1.25]])
    value = field.evaluate(coeffs, x)
    print(f"Field evaluation: {value}")

    assert jnp.allclose(value, 18.), "Value should match coefficient"
    print("✓ get_jax_field tests passed!\n")


def test_jax_fem_model_with_field():
    """Test JaxFemModel with a constant random field."""
    print("=" * 60)
    print("Testing JaxFemModel with JaxConstantField")
    print("=" * 60)

    # Create a simple constant field for Young's modulus
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Create a small FEM model
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.5,  # coarse mesh for testing
        n_params=1,
        d_obs=1,
        field=field
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")

    assert model.get_dim_in() == 1, "Should have 1 parameter (from field)"

    # Test forward evaluation with a parameter value
    params = np.array([12.])
    print(f"Evaluating model with params={params}")

    try:
        y = model.eval(params)
        print(f"Model output shape: {y.shape}")
        print(f"Model output (first 3 values): {y[:3]}")
        print("✓ Forward evaluation successful!\n")
    except Exception as e:
        print(f"⚠ Forward evaluation failed: {e}\n")
        raise


def test_jax_fem_model_from_config():
    """Test creating JaxFemModel from configuration with field."""
    print("=" * 60)
    print("Testing JaxFemModel.from_dict with field")
    print("=" * 60)

    field_config = {
        'name': 'JaxConstantField',
        'mean': 10,
        'std': 2
    }
    field = get_jax_field(field_config)

    model_config = {
        'name': 'JaxFem',
        'd_x': 1.0,
        'd_y': 1.0,
        'd_z': 2.5,
        'h': 0.5,
        'nu': 0.3
    }

    model = JaxFemModel.from_dict(model_config, field=field)

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")

    assert model.get_dim_in() == 1, "Should infer n_params=1 from field"
    print("✓ Model creation from config successful!\n")


if __name__ == '__main__':
    # Run all tests
    test_jax_constant_field()
    test_jax_field_from_config()
    test_jax_fem_model_with_field()
    test_jax_fem_model_from_config()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
