"""Test TransformedLikelihood with LOGIT and COMPOSITE transformations."""

import numpy as np
from pdmp.distributions import (
    TransformedLikelihood,
    GaussianLikelihood,
    LOGIT,
    SIGMOID,
    COMPOSITE,
    EXPONENTIAL,
)
from pdmp.forward_model import LinearModel


def test_logit_transformation():
    """Test that TransformedLikelihood works with LOGIT transformation."""
    print("Testing LOGIT transformation...")

    rng = np.random.default_rng(42)

    # Create a simple identity model (1D -> 1D)
    A = np.array([[1.0]])
    b = np.array([0.0])
    model = LinearModel(A, b)

    # Create a simple Gaussian likelihood
    y_obs = np.array([[0.5]])
    noise_std = 0.1
    likelihood = GaussianLikelihood(model, y_obs, noise_std, rng=rng)

    # Create transformed likelihood with logit
    params = {
        'transformation': LOGIT,
        'a': 0.0,
        'b': 1.0,
    }

    transformed = TransformedLikelihood(likelihood, params)

    # Test evaluation at a point (in transformed space)
    xi = np.array([0.0])  # Logit of 0.5
    log_dens = transformed.log_density(xi, idx=0)
    grad = transformed.grad_log_density(xi, idx=0)
    hess = transformed.hessian_log_density(xi, idx=0)

    print(f"  log_density at xi=0.0: {log_dens}")
    print(f"  grad shape: {grad.shape}")
    print(f"  hessian shape: {hess.shape}")
    print("✓ LOGIT transformation test passed!\n")


def test_composite_transformation():
    """Test that TransformedLikelihood works with COMPOSITE transformation."""
    print("Testing COMPOSITE transformation...")

    rng = np.random.default_rng(42)

    # Create a 2D identity model (2D -> 2D)
    A = np.eye(2)
    b = np.zeros(2)
    model = LinearModel(A, b)

    # Create a 2D Gaussian likelihood
    y_obs = np.array([[0.5, 2.0]])
    noise_std = 0.1
    likelihood = GaussianLikelihood(model, y_obs, noise_std, rng=rng)

    # Create composite transformation: sigmoid on first dim, exponential on second
    params = {
        'transformation': COMPOSITE,
        'transformations': [
            {'type': SIGMOID, 'a': 0.0, 'b': 1.0},
            EXPONENTIAL,
        ],
        'indices': [np.array([0]), np.array([1])],
    }

    transformed = TransformedLikelihood(likelihood, params)

    # Test evaluation at a point
    xi = np.array([0.0, 0.5])
    log_dens = transformed.log_density(xi, idx=0)
    grad = transformed.grad_log_density(xi, idx=0)
    hess = transformed.hessian_log_density(xi, idx=0)

    print(f"  log_density at xi=[0.0, 0.5]: {log_dens}")
    print(f"  grad shape: {grad.shape}")
    print(f"  hessian shape: {hess.shape}")
    print("✓ COMPOSITE transformation test passed!\n")


def test_composite_with_dict_specs():
    """Test COMPOSITE with mixed dict/string specifications."""
    print("Testing COMPOSITE with mixed specifications...")

    rng = np.random.default_rng(42)

    # Create a 3D identity model (3D -> 3D)
    A = np.eye(3)
    b = np.zeros(3)
    model = LinearModel(A, b)

    # Create a 3D Gaussian likelihood
    y_obs = np.array([[0.5, 2.0, 1.0]])
    noise_std = 0.1
    likelihood = GaussianLikelihood(model, y_obs, noise_std, rng=rng)

    # Mix string and dict specifications
    params = {
        'transformation': COMPOSITE,
        'transformations': [
            SIGMOID,  # String spec
            {'type': LOGIT, 'a': 0.0, 'b': 2.0},  # Dict spec
            EXPONENTIAL,  # String spec
        ],
        'indices': [np.array([0]), np.array([1]), np.array([2])],
    }

    transformed = TransformedLikelihood(likelihood, params)

    # Test evaluation at a point
    xi = np.array([0.0, 0.5, 0.0])
    log_dens = transformed.log_density(xi, idx=0)
    grad = transformed.grad_log_density(xi, idx=0)
    hess = transformed.hessian_log_density(xi, idx=0)

    print(f"  log_density at xi=[0.0, 0.5, 0.0]: {log_dens}")
    print(f"  grad shape: {grad.shape}")
    print(f"  hessian shape: {hess.shape}")
    print("✓ COMPOSITE with mixed specs test passed!\n")


if __name__ == "__main__":
    test_logit_transformation()
    test_composite_transformation()
    test_composite_with_dict_specs()
    print("All tests passed! ✓")

