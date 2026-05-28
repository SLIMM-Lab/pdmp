"""Minimal example demonstrating TransformedLikelihood with LOGIT and COMPOSITE."""

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

# Example 1: LOGIT transformation (1D)
print("=" * 60)
print("Example 1: LOGIT Transformation")
print("=" * 60)

rng = np.random.default_rng(42)
A = np.array([[1.0]])
b = np.array([0.0])
model = LinearModel(A, b)
y_obs = np.array([[0.5]])
likelihood = GaussianLikelihood(model, y_obs, sigma=0.1, rng=rng)

# Transform with logit (bounded to [0, 1])
params_logit = {'transformation': LOGIT, 'a': 0.0, 'b': 1.0}
transformed_logit = TransformedLikelihood(likelihood, params_logit)

# Evaluate at xi=0 (which maps to x=0.5 in original space)
xi = np.array([0.0])
print(f"xi (transformed space): {xi}")
print(f"log_density: {transformed_logit.log_density(xi, idx=0):.4f}")
print(f"grad shape: {transformed_logit.grad_log_density(xi, idx=0).shape}")
print()

# Example 2: COMPOSITE transformation (3D)
print("=" * 60)
print("Example 2: COMPOSITE Transformation (3D)")
print("=" * 60)

A3d = np.eye(3)
b3d = np.zeros(3)
model3d = LinearModel(A3d, b3d)
y_obs3d = np.array([[0.5, 2.0, 1.0]])
likelihood3d = GaussianLikelihood(model3d, y_obs3d, sigma=0.1, rng=rng)

# Different transformation for each dimension:
# - Dimension 0: SIGMOID (bounded to [0, 1])
# - Dimension 1: LOGIT (bounded to [0, 2])
# - Dimension 2: EXPONENTIAL (positive)
params_composite = {
    'transformation':
    COMPOSITE,
    'transformations': [
        {
            'type': SIGMOID,
            'a': 0.0,
            'b': 1.0
        },
        {
            'type': LOGIT,
            'a': 0.0,
            'b': 2.0
        },
        EXPONENTIAL,  # String spec
    ],
    'indices': [
        np.array([0]),  # First dimension
        np.array([1]),  # Second dimension
        np.array([2]),  # Third dimension
    ],
}

transformed_composite = TransformedLikelihood(likelihood3d, params_composite)

xi3d = np.array([0.0, 0.5, 0.0])
print(f"xi (transformed space): {xi3d}")
print(f"log_density: {transformed_composite.log_density(xi3d, idx=0):.4f}")
print(
    f"grad shape: {transformed_composite.grad_log_density(xi3d, idx=0).shape}")
print(
    f"hessian shape: {transformed_composite.hessian_log_density(xi3d, idx=0).shape}"
)
print()

print("=" * 60)
print("✓ All examples completed successfully!")
print("=" * 60)
