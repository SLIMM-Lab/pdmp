"""Example: Using LogitTransformation to model bounded parameters with Gaussian priors.

This example demonstrates how to use LogitTransformation to work with parameters
that are naturally bounded (e.g., probabilities, fractions) but you want to use
a Gaussian prior in the unbounded space.
"""

import numpy as np
import matplotlib.pyplot as plt
from pdmp.distributions import (MultivariateNormal, TransformedDistribution,
                                LogitTransformation, LOGIT)

# Example 1: Simple 1D case
print("=" * 60)
print("Example 1: Bounded parameter with Gaussian prior")
print("=" * 60)

# We have a parameter p that must be in [0, 1] (e.g., a probability)
# But we want to use a Gaussian prior in the unbounded space

# Define Gaussian in unbounded space (ξ space)
mean_xi = np.array([0.0])  # logit(0.5) = 0, so this centers at p=0.5
cov_xi = np.array([[2.0]])
base_dist = MultivariateNormal(mean=mean_xi, cov=cov_xi, seed=42)

# Transform to bounded [0, 1]
params = {'transformation': LOGIT, 'a': 0.0, 'b': 1.0}
p_dist = TransformedDistribution(base_dist, params)

# Sample parameters (will be in [0, 1])
p_samples = p_dist.get_sample(n=5000)

print(f"\nSampled {len(p_samples)} values of p ∈ [0, 1]")
print(f"  Range: [{p_samples.min():.4f}, {p_samples.max():.4f}]")
print(f"  Mean: {p_samples.mean():.4f}")
print(f"  Median: {np.median(p_samples):.4f}")
print(f"  Std: {p_samples.std():.4f}")

# Evaluate log density at a point
p_test = np.array([0.7])
log_p = p_dist.log_density(p_test)
print(f"\nlog p(p={p_test[0]}) = {log_p:.4f}")

# Example 2: Multivariate case with different bounds
print("\n" + "=" * 60)
print("Example 2: Multiple bounded parameters with different bounds")
print("=" * 60)

# Suppose we have:
# - θ₁ ∈ [0, 1]: a probability
# - θ₂ ∈ [0.5, 2.0]: a rate parameter with known bounds

# Define Gaussian in unbounded space
mean_xi = np.array([0.0, 0.0])
cov_xi = np.array([[1.0, 0.3], [0.3, 0.5]])
base_dist = MultivariateNormal(mean=mean_xi, cov=cov_xi, seed=123)

# Transform with vector bounds
a = np.array([0.0, 0.5])
b = np.array([1.0, 2.0])
params = {'transformation': LOGIT, 'a': a, 'b': b}
theta_dist = TransformedDistribution(base_dist, params)

# Sample
theta_samples = theta_dist.get_sample(n=5000)

print(f"\nSampled {len(theta_samples)} parameter vectors")
print(
    f"  θ₁ range: [{theta_samples[:, 0].min():.4f}, {theta_samples[:, 0].max():.4f}]"
)
print(
    f"  θ₂ range: [{theta_samples[:, 1].min():.4f}, {theta_samples[:, 1].max():.4f}]"
)
print(f"  θ₁ mean: {theta_samples[:, 0].mean():.4f} (expected ~0.5)")
print(f"  θ₂ mean: {theta_samples[:, 1].mean():.4f} (expected ~1.25)")

# Example 3: Direct use of LogitTransformation
print("\n" + "=" * 60)
print("Example 3: Direct transformation usage")
print("=" * 60)

trans = LogitTransformation(a=0.0, b=1.0)

# Convert between bounded and unbounded spaces
p_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
xi_values = trans.transform(p_values)

print("\nBounded → Unbounded:")
print("  p (bounded):  ", p_values)
print("  ξ (unbounded):", xi_values)

# Convert back
p_recovered = trans.inverse_transform(xi_values)
print("\nUnbounded → Bounded:")
print("  ξ (unbounded):", xi_values)
print("  p (bounded):  ", p_recovered)
print(f"  Match original: {np.allclose(p_values, p_recovered)}")

# Visualization
print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Samples from Example 1
ax = axes[0, 0]
ax.hist(p_samples.flatten(),
        bins=50,
        density=True,
        alpha=0.7,
        edgecolor='black')
ax.axvline(0.5, color='red', linestyle='--', label='p=0.5 (mean in xi space)')
ax.set_xlabel('p (probability parameter)')
ax.set_ylabel('Density')
ax.set_title('Example 1: Samples from p ~ Logit(N(0, 1))')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: 2D scatter from Example 2
ax = axes[0, 1]
ax.scatter(theta_samples[:, 0], theta_samples[:, 1], alpha=0.1, s=1)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axhline(1.25, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('θ₁ ∈ [0, 1]')
ax.set_ylabel('θ₂ ∈ [0.5, 2.0]')
ax.set_title('Example 2: Joint samples')
ax.grid(True, alpha=0.3)

# Plot 3: Transformation function
ax = axes[1, 0]
p_grid = np.linspace(0.01, 0.99, 100)
xi_grid = trans.transform(p_grid)
ax.plot(p_grid, xi_grid, linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('p (bounded)')
ax.set_ylabel('ξ (unbounded)')
ax.set_title('Logit Transformation: p → ξ')
ax.grid(True, alpha=0.3)

# Plot 4: Jacobian (derivative)
ax = axes[1, 1]
jac_vals = np.array([trans.jacobian(np.array([p]))[0, 0] for p in p_grid])
ax.plot(p_grid, jac_vals, linewidth=2, label='dξ/dp')
ax.set_xlabel('p (bounded)')
ax.set_ylabel('Jacobian dξ/dp')
ax.set_title('Logit Jacobian (rate of change)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✅ Examples completed successfully!")
print("=" * 60)
