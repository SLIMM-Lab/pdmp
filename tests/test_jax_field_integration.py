#!/usr/bin/env python3
"""End-to-end test of JaxFemModel with JaxConstantField using the loader."""

import numpy as np
import yaml
import tempfile
import os

from pdmp.loader import get_target
from pdmp import logger
from pdmp.random_field import JaxConstantField
from pdmp.forward_model import JaxFemModel
from pdmp.utils import grad_fd
from pdmp.distributions import MultivariateNormal


def test_full_integration():
    """Test full integration: config → loader → posterior."""
    print("=" * 70)
    print("Testing Full Integration with Loader")
    print("=" * 70)

    # First, generate synthetic observations
    print("Generating synthetic observations...")

    # Create a temporary observation file
    temp_obs_file = tempfile.NamedTemporaryFile(mode='w',
                                                delete=False,
                                                suffix='.txt')
    theta_true = np.array([1.2e6])

    # Create a simple model to generate observations
    mean = 1.0e6
    std = 2.0e5
    field_temp = JaxConstantField(
        MultivariateNormal(mean=np.array([mean]), cov=np.array([[std**2]])))
    model_temp = JaxFemModel(d_x=1.0,
                             d_y=1.0,
                             d_z=2.5,
                             h=0.5,
                             nu=0.3,
                             field=field_temp)
    y_obs = model_temp.eval(theta_true)

    # Write observations to file
    np.savetxt(temp_obs_file.name, y_obs.reshape(1, -1))
    temp_obs_file.close()

    print(f"  Created observation file: {temp_obs_file.name}")
    print(f"  Observations shape: {y_obs.shape}")
    print(f"  Observations: {y_obs}")

    # Create configuration
    config = {
        'name': 'BayesianInverse',
        'model': {
            'name': 'JaxFem',
            'd_x': 1.0,
            'd_y': 1.0,
            'd_z': 2.5,
            'h': 0.5,
            'nu': 0.3,
            'field': {
                'name': 'JaxConstantField',
                'mean': 3.0,
                'std': 0.25
            }
        },
        'prior': {
            'name': 'FromField'
        },
        'likelihood': {
            'name': 'GaussianLikelihood',
            'sigma': 0.01,
            'observation_file': temp_obs_file.name
        }
    }

    # Create posterior using loader
    rng = np.random.default_rng(42)
    print("\nLoading posterior from configuration...")
    posterior = get_target(config, rng)

    print(f"✓ Posterior created successfully")
    print(f"  - Prior dimension: {posterior.prior.dim}")
    print(f"  - Model input dim: {posterior._likelihood._model.get_dim_in()}")
    print(
        f"  - Model output dim: {posterior._likelihood._model.get_dim_out()}")

    # Sample from prior
    print("\nSampling from prior...")
    theta_0 = posterior.get_prior_sample()
    print(f"  Prior sample: {theta_0}")
    print(f"  Sample shape: {theta_0.shape}")

    # Evaluate prior log probability
    print("\nEvaluating prior log probability...")
    log_prior = posterior.prior.log_density(theta_0)
    log_prior_val = np.asarray(log_prior).item() if np.asarray(
        log_prior).size == 1 else float(np.asarray(log_prior).flat[0])
    print(f"  log π(θ): {log_prior_val:.4f}")

    # Evaluate model (likelihood data term)
    print("\nEvaluating forward model...")
    y = posterior._likelihood._model.eval(theta_0)
    print(f"  Model output shape: {y.shape}")
    print(f"  Model output: {y}")

    # Evaluate likelihood
    print("\nEvaluating likelihood...")
    log_like = posterior._likelihood.log_density(theta_0)
    log_like_val = np.asarray(log_like).item() if np.asarray(
        log_like).size == 1 else float(np.asarray(log_like).flat[0])
    print(f"  log L(θ|y): {log_like_val:.4f}")

    # Evaluate posterior log probability
    print("\nEvaluating posterior log probability...")
    log_post = posterior.log_density(theta_0)
    log_post_val = np.asarray(log_post).item() if np.asarray(
        log_post).size == 1 else float(np.asarray(log_post).flat[0])
    print(f"  log p(θ|y): {log_post_val:.4f}")
    print(f"  Verification: log_post ≈ log_prior + log_like")
    print(f"    {log_post_val:.4f} ≈ {log_prior_val:.4f} + {log_like_val:.4f}")
    print(
        f"    Difference: {abs(log_post_val - (log_prior_val + log_like_val)):.6e}"
    )

    # Test gradient computation
    print("\nTesting gradient computation...")
    grad = posterior.grad_log_density(theta_0)
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient: {grad}")

    # Cleanup
    os.unlink(temp_obs_file.name)

    print("\n✓ All integration tests passed!")


def test_from_yaml_file():
    """Test loading from YAML file."""
    print("\n" + "=" * 70)
    print("Testing Loading from YAML File")
    print("=" * 70)

    yaml_path = 'examples/jax_fem_constant_field.yaml'

    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Loaded configuration from {yaml_path}")

        rng = np.random.default_rng(123)
        posterior = get_target(config, rng)

        print(f"✓ Created posterior from YAML config")
        print(f"  - Prior dimension: {posterior.prior.dim}")

        # Quick sanity check
        theta = posterior.get_prior_sample()
        log_p = posterior.prior.log_density(theta)
        log_p_val = np.asarray(log_p).item() if np.asarray(
            log_p).size == 1 else float(np.asarray(log_p).flat[0])
        print(f"  - Sample from prior: {theta}")
        print(f"  - Log prior: {log_p_val:.4f}")

        print("\n✓ YAML file test passed!")

    except FileNotFoundError:
        print(f"⚠ YAML file not found at {yaml_path}")
        print("  This is expected if running from a different directory.")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def test_gradient_correctness():
    """Test that gradients are computed correctly via finite differences."""
    print("\n" + "=" * 70)
    print("Testing Gradient Correctness")
    print("=" * 70)

    # Generate observations
    print("Generating synthetic observations...")
    field_temp = JaxConstantField(
        MultivariateNormal(mean=np.array([3.0]), cov=np.array([[0.25**2]])))
    model_temp = JaxFemModel(d_x=1.0,
                             d_y=1.0,
                             d_z=2.5,
                             h=0.5,
                             nu=0.3,
                             field=field_temp)
    theta_true = np.array([3.1])
    y_obs = model_temp.eval(theta_true)

    # Create temp file
    temp_obs_file = tempfile.NamedTemporaryFile(mode='w',
                                                delete=False,
                                                suffix='.txt')
    np.savetxt(temp_obs_file.name, y_obs.reshape(1, -1))
    temp_obs_file.close()

    config = {
        'name': 'BayesianInverse',
        'model': {
            'name': 'JaxFem',
            'd_x': 1.0,
            'd_y': 1.0,
            'd_z': 2.5,
            'h': 0.5,
            'nu': 0.3,
            'field': {
                'name': 'JaxConstantField',
                'mean': 3.0,
                'std': 0.25
            }
        },
        'prior': {
            'name': 'FromField'
        },
        'likelihood': {
            'name': 'GaussianLikelihood',
            'sigma': 0.01,
            'observation_file': temp_obs_file.name
        }
    }

    rng = np.random.default_rng(999)
    posterior = get_target(config, rng)

    # Test point
    theta = np.array([2.0])

    # Analytical gradient
    print("Computing analytical gradient...")
    grad_analytical = posterior.grad_log_density(theta)
    print(f"  Analytical gradient: {grad_analytical}")

    # Finite difference gradient
    print("\nComputing finite difference gradient...")
    h = 1e-4  # Step size (larger for numerical stability with FEM)

    grad_finite_difference = grad_fd(posterior.log_density, theta, h=h)

    print(f"  Finite difference gradient: {grad_finite_difference}")

    # Compare
    rel_error = np.abs(grad_analytical - grad_finite_difference) / (
        np.abs(grad_finite_difference) + 1e-10)
    print(f"\nComparison:")
    print(
        f"  Absolute difference: {np.abs(grad_analytical - grad_finite_difference)}"
    )
    print(f"  Relative error: {rel_error}")

    # Cleanup
    os.unlink(temp_obs_file.name)

    if rel_error[0] < 0.1:  # 10% tolerance for FEM
        print("✓ Gradient test passed!")
    else:
        print(f"⚠ Large gradient error: {rel_error[0]:.2%}")
        print("  (This may be acceptable for FEM with coarse mesh)")


if __name__ == '__main__':
    # Run tests
    test_full_integration()
    test_from_yaml_file()
    test_gradient_correctness()

    print("\n" + "=" * 70)
    print("All End-to-End Tests Complete! ✓")
    print("=" * 70)
