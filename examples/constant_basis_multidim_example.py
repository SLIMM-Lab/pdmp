"""Examples demonstrating multi-dimensional ConstantBasis functionality.

This example shows how to create constant random fields in 1D, 2D, and 3D
using the ConstantBasis class with JaxRandomFieldBase.
"""

import numpy as np
import jax.numpy as jnp
import jax

from pdmp.project_field import ConstantBasis
from pdmp.random_field import JaxRandomFieldBase
from pdmp.distributions import MultivariateNormal


def example_1d_constant_field():
    """Example 1: 1D constant field (backward compatible)."""
    print("=" * 70)
    print("Example 1: 1D Constant Field")
    print("=" * 70)

    # Create 1D constant basis (original tuple interface)
    basis = ConstantBasis((0.0, 1.0))

    print(f"Spatial dimension: {basis.spatial_dim}")
    print(f"Domain: {basis.domain}")
    print(
        f"Volume (length): {np.prod(basis.domain[:, 1] - basis.domain[:, 0])}")
    print(f"Norm: {basis.get_norms()[0, 0]:.4f}")

    # Create field with uncertain coefficient
    mean = np.array([2.0])
    cov = np.array([[0.3**2]])
    dist = MultivariateNormal(mean, cov)
    field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

    # Evaluate at 1D points
    x = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    coeffs = jnp.array([2.5])
    values = field.evaluate(coeffs, x)

    print(f"\nEvaluation at x = {x}")
    print(f"Field values (all constant): {values}")
    print()


def example_2d_constant_field():
    """Example 2: 2D constant field for material property."""
    print("=" * 70)
    print("Example 2: 2D Constant Field - Material Property")
    print("=" * 70)

    # Create 2D rectangular domain [0, 10] × [0, 5] (e.g., 10mm x 5mm plate)
    domain = [[0.0, 10.0], [0.0, 5.0]]
    basis = ConstantBasis(domain)

    print(f"Spatial dimension: {basis.spatial_dim}")
    print(f"Domain: {basis.domain}")
    print(f"Area: {np.prod(basis.domain[:, 1] - basis.domain[:, 0])}")
    print(f"Norm: {basis.get_norms()[0, 0]:.4f}")

    # Uncertain Young's modulus (GPa) - spatially uniform but uncertain
    mean = np.array([200.0])  # 200 GPa
    cov = np.array([[15.0**2]])  # std = 15 GPa
    dist = MultivariateNormal(mean, cov)
    field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

    # Evaluate at 2D points (e.g., mesh nodes or quadrature points)
    x = jnp.array([
        [0.0, 0.0],  # Corner
        [5.0, 2.5],  # Center
        [10.0, 5.0],  # Opposite corner
        [2.5, 1.25],  # Random point
    ])

    # Use a specific coefficient value
    coeffs = jnp.array([210.0])  # E = 210 GPa
    values = field.evaluate(coeffs, x)

    print(f"\nEvaluation at {x.shape[0]} 2D points:")
    print(f"All values equal to {coeffs[0]:.1f} GPa:")
    for i, (pt, val) in enumerate(zip(x, values)):
        print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f}) -> {val:.1f} GPa")
    print()


def example_3d_constant_field():
    """Example 3: 3D constant field for permeability."""
    print("=" * 70)
    print("Example 3: 3D Constant Field - Permeability")
    print("=" * 70)

    # Create 3D cubic domain [0, 1]³ (e.g., 1m x 1m x 1m porous medium)
    domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    basis = ConstantBasis(domain)

    print(f"Spatial dimension: {basis.spatial_dim}")
    print(f"Domain: {basis.domain}")
    print(f"Volume: {np.prod(basis.domain[:, 1] - basis.domain[:, 0])}")
    print(f"Norm: {basis.get_norms()[0, 0]:.4f}")

    # Uncertain permeability (m²) - spatially uniform but uncertain
    mean = np.array([1e-5])  # 10 microdarcy
    cov = np.array([[2e-6**2]])  # std = 2 microdarcy
    dist = MultivariateNormal(mean, cov)
    field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

    # Evaluate at 3D points
    x = jnp.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [0.25, 0.75, 0.5],
    ])

    coeffs = jnp.array([1.2e-5])
    values = field.evaluate(coeffs, x)

    print(f"\nEvaluation at {x.shape[0]} 3D points:")
    print(f"All values equal to {coeffs[0]:.2e} m²:")
    for i, (pt, val) in enumerate(zip(x, values)):
        print(
            f"  Point {i}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}) -> {val:.2e} m²"
        )
    print()


def example_2d_gradient_computation():
    """Example 4: Gradient computation in 2D."""
    print("=" * 70)
    print("Example 4: 2D Gradient Computation")
    print("=" * 70)

    # 2D domain
    domain = [[0.0, 1.0], [0.0, 1.0]]
    basis = ConstantBasis(domain)

    mean = np.array([1.0])
    cov = np.array([[0.1]])
    dist = MultivariateNormal(mean, cov)
    field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

    # Define a loss function that depends on the field
    def loss_fn(coeffs):
        # Evaluate at a grid of points
        x = jnp.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ])
        values = field.evaluate(coeffs, x)
        # Simple loss: sum of squared values
        return jnp.sum(values**2)

    # Compute gradient
    coeffs = jnp.array([2.0])
    loss = loss_fn(coeffs)
    grad = jax.grad(loss_fn)(coeffs)

    print(f"Coefficient: {coeffs[0]:.2f}")
    print(f"Loss (sum of 9 points): {loss:.4f}")
    print(f"Gradient w.r.t. coefficient: {grad[0]:.4f}")
    print(
        f"Analytical gradient (2 * coeff * n_points): {2 * coeffs[0] * 9:.4f}")
    print(f"Match: {jnp.allclose(grad[0], 2 * coeffs[0] * 9)}")
    print()


def example_3d_fem_simulation():
    """Example 5: Realistic 3D FEM simulation scenario."""
    print("=" * 70)
    print("Example 5: 3D FEM Simulation with Uncertain Material Property")
    print("=" * 70)

    # 3D domain representing a structural component
    domain = [[0.0, 0.1], [0.0, 0.05], [0.0, 0.02]]  # 100mm x 50mm x 20mm
    basis = ConstantBasis(domain)

    print(f"Component dimensions: {basis.domain[:, 1] * 1000} mm")
    print(
        f"Volume: {np.prod(basis.domain[:, 1] - basis.domain[:, 0]) * 1e9:.2f} mm³"
    )

    # Uncertain material density (kg/m³)
    mean = np.array([7850.0])  # Steel density
    cov = np.array([[50.0**2]])  # std = 50 kg/m³
    dist = MultivariateNormal(mean, cov)
    field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

    # Simulate FEM mesh with cells and quadrature points
    n_cells = 100
    n_quads_per_cell = 8  # 8 Gauss points per hexahedral element

    # Generate random quadrature points within the domain
    np.random.seed(42)
    quad_points = np.random.uniform(low=basis.domain[:, 0],
                                    high=basis.domain[:, 1],
                                    size=(n_cells * n_quads_per_cell, 3))

    # Sample material property
    sampled_density = field.coefficient_distribution.get_sample()
    print(f"\nSampled density: {sampled_density[0]:.2f} kg/m³")

    # Evaluate field at all quadrature points
    density_field = field.evaluate(jnp.array(sampled_density),
                                   jnp.array(quad_points))

    print(f"Evaluated at {n_cells * n_quads_per_cell} quadrature points")
    print(
        f"All values constant: {jnp.allclose(density_field, sampled_density[0])}"
    )
    print(f"Min value: {jnp.min(density_field):.2f} kg/m³")
    print(f"Max value: {jnp.max(density_field):.2f} kg/m³")
    print(f"Mean value: {jnp.mean(density_field):.2f} kg/m³")

    # Compute gradient of total mass w.r.t. density
    def total_mass(density_coeff):
        density_values = field.evaluate(density_coeff, jnp.array(quad_points))
        # Mass = density * volume (simplified, assumes uniform quadrature weights)
        volume = np.prod(basis.domain[:, 1] - basis.domain[:, 0])
        return jnp.sum(density_values) * volume / (n_cells * n_quads_per_cell)

    mass = total_mass(jnp.array(sampled_density))
    grad_mass = jax.grad(total_mass)(jnp.array(sampled_density))

    print(f"\nTotal mass: {mass:.6f} kg")
    print(f"∂(mass)/∂(density): {grad_mass[0]:.6e}")
    print()


def example_sampling_uncertainty():
    """Example 6: Sampling from uncertain constant fields in different dimensions."""
    print("=" * 70)
    print("Example 6: Sampling from Uncertain Constant Fields")
    print("=" * 70)

    # 1D case
    basis_1d = ConstantBasis((0.0, 1.0))
    dist_1d = MultivariateNormal(np.array([5.0]), np.array([[1.0]]))
    field_1d = JaxRandomFieldBase(basis=basis_1d, coefficient_dist=dist_1d)

    # 2D case
    basis_2d = ConstantBasis([[0.0, 1.0], [0.0, 1.0]])
    dist_2d = MultivariateNormal(np.array([10.0]), np.array([[2.0]]))
    field_2d = JaxRandomFieldBase(basis=basis_2d, coefficient_dist=dist_2d)

    # 3D case
    basis_3d = ConstantBasis([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    dist_3d = MultivariateNormal(np.array([20.0]), np.array([[3.0]]))
    field_3d = JaxRandomFieldBase(basis=basis_3d, coefficient_dist=dist_3d)

    print("Sampling 5 realizations from each field:\n")

    print("1D field (mean=5.0, std=1.0):")
    for i in range(5):
        sample = field_1d.coefficient_distribution.get_sample()
        print(f"  Sample {i+1}: {sample[0]:.3f}")

    print("\n2D field (mean=10.0, std=√2.0):")
    for i in range(5):
        sample = field_2d.coefficient_distribution.get_sample()
        print(f"  Sample {i+1}: {sample[0]:.3f}")

    print("\n3D field (mean=20.0, std=√3.0):")
    for i in range(5):
        sample = field_3d.coefficient_distribution.get_sample()
        print(f"  Sample {i+1}: {sample[0]:.3f}")
    print()


if __name__ == "__main__":
    example_1d_constant_field()
    example_2d_constant_field()
    example_3d_constant_field()
    example_2d_gradient_computation()
    example_3d_fem_simulation()
    example_sampling_uncertainty()

    print("=" * 70)
    print("All multi-dimensional examples completed successfully!")
    print("=" * 70)
    print("\nKey features:")
    print("  ✓ 1D, 2D, and 3D constant fields")
    print("  ✓ Backward compatible with 1D tuple interface")
    print("  ✓ Proper volume-based norm calculation")
    print("  ✓ JAX gradient computation in all dimensions")
    print("  ✓ Integration with JaxRandomFieldBase")
    print("  ✓ Realistic FEM simulation scenarios")
