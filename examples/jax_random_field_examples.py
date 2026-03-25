"""Example: Using JAX Random Fields with Different Coefficient Distributions

This example demonstrates the new JAX random field classes:
- JaxRandomFieldBase: Generic field with arbitrary coefficient distributions
- JaxGaussianRandomField: Gaussian field with kernel-based covariance
- JaxConstantField: Single constant parameter (unchanged)
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pdmp.random_field import get_jax_field


def example_1_custom_gaussian_covariance():
    """Example 1: Gaussian field with custom covariance matrix."""
    print("=" * 70)
    print("Example 1: Gaussian Field with Custom Covariance")
    print("=" * 70)

    # Custom covariance with strong correlation between adjacent coefficients
    dim = 10
    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cov[i, j] = 0.9**abs(i - j)

    config = {
        'name': 'JaxRandomField',
        'basis': {
            'type': 'PiecewiseConstant',
            'dim': dim,
            'interval': [0.0, 1.0]
        },
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': 5.0,
            'cov': cov
        }
    }

    field = get_jax_field(config)
    dist = field.coefficient_distribution

    # Sample and visualize
    x = np.linspace(0, 1, 200)

    plt.figure(figsize=(10, 6))
    for i in range(5):
        coeffs = dist.get_sample()
        field_values = field.evaluate(coeffs, x)
        plt.plot(x, field_values, alpha=0.7, label=f'Realization {i+1}')

    plt.xlabel('x')
    plt.ylabel('Field value')
    plt.title('Custom Covariance: Adjacent coefficients highly correlated')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/jax_field_custom_cov.png', dpi=150)
    print("  ✓ Figure saved: figures/jax_field_custom_cov.png")
    print()


def example_2_kernel_based_gaussian():
    """Example 2: Gaussian field from spatial covariance kernel."""
    print("=" * 70)
    print("Example 2: Kernel-Based Gaussian Field")
    print("=" * 70)

    config = {
        'name': 'JaxGaussianRandomField',
        'dim': 20,
        'mean': 0.0,
        'interval': [0.0, 1.0],
        'kernel_params': {
            'sigma': 1.0,
            'l': 0.2
        },  # Correlation length l=0.2
        'basis': 'PiecewiseConstant'
    }

    field = get_jax_field(config)
    dist = field.coefficient_distribution

    print(f"  Kernel parameters: {field.kernel_params}")
    print(f"  Field dimension: {field.dim}")

    # Sample and visualize
    x = np.linspace(0, 1, 200)

    plt.figure(figsize=(10, 6))
    for i in range(5):
        coeffs = dist.get_sample()
        field_values = field.evaluate(coeffs, x)
        plt.plot(x, field_values, alpha=0.7, label=f'Realization {i+1}')

    plt.xlabel('x')
    plt.ylabel('Field value')
    plt.title(
        f"Squared Exponential Kernel: σ={config['kernel_params']['sigma']}, "
        f"l={config['kernel_params']['l']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/jax_field_kernel_based.png', dpi=150)
    print("  ✓ Figure saved: figures/jax_field_kernel_based.png")
    print()


def example_3_compare_correlation_lengths():
    """Example 3: Compare different correlation lengths."""
    print("=" * 70)
    print("Example 3: Effect of Correlation Length")
    print("=" * 70)

    correlation_lengths = [0.05, 0.15, 0.3]
    x = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, l in zip(axes, correlation_lengths):
        config = {
            'name': 'JaxGaussianRandomField',
            'dim': 20,
            'mean': 0.0,
            'interval': [0.0, 1.0],
            'kernel_params': {
                'sigma': 1.0,
                'l': l
            },
            'basis': 'PiecewiseConstant'
        }

        field = get_jax_field(config)
        dist = field.coefficient_distribution

        # Sample multiple realizations
        for i in range(3):
            coeffs = dist.get_sample()
            field_values = field.evaluate(coeffs, x)
            ax.plot(x, field_values, alpha=0.7)

        ax.set_xlabel('x')
        ax.set_ylabel('Field value')
        ax.set_title(f'l = {l}')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Effect of Correlation Length on Field Smoothness')
    plt.tight_layout()
    plt.savefig('figures/jax_field_correlation_comparison.png', dpi=150)
    print("  ✓ Figure saved: figures/jax_field_correlation_comparison.png")
    print()


def example_4_jax_differentiability():
    """Example 4: JAX automatic differentiation."""
    print("=" * 70)
    print("Example 4: JAX Automatic Differentiation")
    print("=" * 70)

    import jax

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

    # Define a loss function
    def loss(coeffs):
        x = jnp.linspace(0, 1, 50)
        field_vals = field.evaluate(coeffs, x)
        return jnp.sum(field_vals**2)

    # Sample coefficients
    coeffs = jnp.ones(10) * 0.1

    # Compute gradient
    grad = jax.grad(loss)(coeffs)

    print(f"  Coefficients: {coeffs}")
    print(f"  Loss value: {loss(coeffs):.4f}")
    print(f"  Gradient: {grad}")
    print(f"  ✓ JAX differentiation works!")
    print()


def example_5_constant_field_backward_compatible():
    """Example 5: Constant field (backward compatibility)."""
    print("=" * 70)
    print("Example 5: Constant Field (Backward Compatible)")
    print("=" * 70)

    config = {'name': 'JaxConstantField', 'mean': 100.0, 'std': 20.0}

    field = get_jax_field(config)
    dist = field.coefficient_distribution

    # Sample and evaluate
    coeff = dist.get_sample()
    x = np.linspace(0, 1, 10)
    field_values = field.evaluate(coeff, x)

    print(f"  Coefficient: {coeff[0]:.2f}")
    print(f"  Field values (constant): {field_values[:3]}...")
    print(f"  ✓ Constant field works as before!")
    print()


if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)

    print("\n" + "=" * 70)
    print("JAX Random Field Examples")
    print("=" * 70 + "\n")

    example_1_custom_gaussian_covariance()
    example_2_kernel_based_gaussian()
    example_3_compare_correlation_lengths()
    example_4_jax_differentiability()
    example_5_constant_field_backward_compatible()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
