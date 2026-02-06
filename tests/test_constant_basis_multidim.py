"""Unit tests for multi-dimensional ConstantBasis class."""

import numpy as np
import jax.numpy as jnp
import jax
import pytest

from pdmp.project_field import ConstantBasis
from pdmp.random_field import JaxRandomFieldBase
from pdmp.distributions import MultivariateNormal


class TestConstantBasisMultiDim:
    """Test suite for multi-dimensional ConstantBasis class."""

    def test_1d_tuple_initialization(self):
        """Test 1D initialization with tuple (backward compatibility)."""
        basis = ConstantBasis((0.0, 1.0))

        assert basis.get_n() == 1
        assert basis.spatial_dim == 1
        np.testing.assert_array_equal(basis.domain, [[0.0, 1.0]])
        np.testing.assert_almost_equal(basis.get_norms()[0, 0], 1.0)

    def test_1d_array_initialization(self):
        """Test 1D initialization with array."""
        basis = ConstantBasis([[0.0, 2.0]])

        assert basis.get_n() == 1
        assert basis.spatial_dim == 1
        np.testing.assert_array_equal(basis.domain, [[0.0, 2.0]])
        np.testing.assert_almost_equal(basis.get_norms()[0, 0], np.sqrt(2.0))

    def test_2d_initialization(self):
        """Test 2D initialization."""
        domain = [[0.0, 1.0], [0.0, 2.0]]
        basis = ConstantBasis(domain)

        assert basis.get_n() == 1
        assert basis.spatial_dim == 2
        np.testing.assert_array_equal(basis.domain, domain)
        # Volume = 1 * 2 = 2, norm = sqrt(2)
        np.testing.assert_almost_equal(basis.get_norms()[0, 0], np.sqrt(2.0))

    def test_3d_initialization(self):
        """Test 3D initialization."""
        domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        basis = ConstantBasis(domain)

        assert basis.get_n() == 1
        assert basis.spatial_dim == 3
        np.testing.assert_array_equal(basis.domain, domain)
        # Volume = 1 * 1 * 1 = 1, norm = 1
        np.testing.assert_almost_equal(basis.get_norms()[0, 0], 1.0)

    def test_3d_initialization_non_unit(self):
        """Test 3D initialization with non-unit dimensions."""
        domain = [[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]
        basis = ConstantBasis(domain)

        assert basis.spatial_dim == 3
        # Volume = 2 * 3 * 4 = 24, norm = sqrt(24)
        np.testing.assert_almost_equal(basis.get_norms()[0, 0], np.sqrt(24.0))

    def test_invalid_domain_shape(self):
        """Test that invalid domain shape raises error."""
        with pytest.raises(ValueError, match="Domain must have shape"):
            ConstantBasis([[0.0, 1.0, 2.0]])  # Wrong: 3 columns instead of 2

    def test_1d_evaluation(self):
        """Test 1D evaluation."""
        basis = ConstantBasis((0.0, 1.0))
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        phi = basis(x)
        assert phi.shape == (5, 1)
        np.testing.assert_array_equal(phi, np.ones((5, 1)))

    def test_2d_evaluation(self):
        """Test 2D evaluation."""
        basis = ConstantBasis([[0.0, 1.0], [0.0, 2.0]])

        # Create 2D points
        x = np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 2.0],
            [0.25, 0.5]
        ])

        phi = basis(x)
        assert phi.shape == (4, 1)
        np.testing.assert_array_equal(phi, np.ones((4, 1)))

    def test_3d_evaluation(self):
        """Test 3D evaluation."""
        basis = ConstantBasis([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        # Create 3D points
        x = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [0.25, 0.75, 0.5]
        ])

        phi = basis(x)
        assert phi.shape == (4, 1)
        np.testing.assert_array_equal(phi, np.ones((4, 1)))

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        basis = ConstantBasis([[0.0, 1.0], [0.0, 2.0]])  # 2D basis

        # Try to evaluate with 3D points
        x = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        with pytest.raises(ValueError, match="Expected points with spatial_dim=2"):
            basis(x)

    def test_2d_with_jax_random_field(self):
        """Test 2D ConstantBasis with JaxRandomFieldBase."""
        domain = [[0.0, 1.0], [0.0, 1.0]]
        basis = ConstantBasis(domain)

        mean = np.array([5.0])
        cov = np.array([[1.0]])
        dist = MultivariateNormal(mean, cov)

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        # Evaluate at 2D points
        x = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        coeffs = jnp.array([3.0])

        values = field.evaluate(coeffs, x)

        assert values.shape == (3,)
        np.testing.assert_allclose(values, 3.0)

    def test_3d_with_jax_random_field(self):
        """Test 3D ConstantBasis with JaxRandomFieldBase."""
        domain = [[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]
        basis = ConstantBasis(domain)

        mean = np.array([10.0])
        cov = np.array([[0.5**2]])
        dist = MultivariateNormal(mean, cov)

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        # Evaluate at 3D points
        x = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.5, 2.0],
            [2.0, 3.0, 4.0]
        ])
        coeffs = jnp.array([7.5])

        values = field.evaluate(coeffs, x)

        assert values.shape == (3,)
        np.testing.assert_allclose(values, 7.5)

    def test_2d_gradient_computation(self):
        """Test gradient computation with 2D ConstantBasis."""
        domain = [[0.0, 1.0], [0.0, 1.0]]
        basis = ConstantBasis(domain)

        mean = np.array([1.0])
        cov = np.array([[0.1]])
        dist = MultivariateNormal(mean, cov)
        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        def loss_fn(coeffs):
            x = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            values = field.evaluate(coeffs, x)
            return jnp.sum(values**2)

        coeffs = jnp.array([2.0])
        grad = jax.grad(loss_fn)(coeffs)

        # Analytical: d/dc sum(c^2) = 2*c*n_points = 2*2*3 = 12
        np.testing.assert_allclose(grad[0], 12.0)

    def test_3d_gradient_computation(self):
        """Test gradient computation with 3D ConstantBasis."""
        domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        basis = ConstantBasis(domain)

        mean = np.array([0.0])
        cov = np.array([[1.0]])
        dist = MultivariateNormal(mean, cov)
        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        def loss_fn(coeffs):
            x = jnp.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [0.25, 0.25, 0.25]
            ])
            values = field.evaluate(coeffs, x)
            return jnp.sum(values**2)

        coeffs = jnp.array([3.0])
        grad = jax.grad(loss_fn)(coeffs)

        # Analytical: d/dc sum(c^2) = 2*c*n_points = 2*3*4 = 24
        np.testing.assert_allclose(grad[0], 24.0)

    def test_backward_compatibility_1d(self):
        """Test that 1D tuple interface still works as before."""
        # Old style
        basis_old = ConstantBasis((0.0, 5.0))

        # New style
        basis_new = ConstantBasis([[0.0, 5.0]])

        x = np.array([0.0, 1.0, 2.5, 5.0])

        phi_old = basis_old(x)
        phi_new = basis_new(x)

        np.testing.assert_array_equal(phi_old, phi_new)
        np.testing.assert_almost_equal(
            basis_old.get_norms()[0, 0],
            basis_new.get_norms()[0, 0]
        )

    def test_properties(self):
        """Test spatial_dim and domain properties."""
        domain_2d = [[0.0, 1.0], [-1.0, 1.0]]
        basis_2d = ConstantBasis(domain_2d)

        assert basis_2d.spatial_dim == 2
        np.testing.assert_array_equal(basis_2d.domain, domain_2d)

        domain_3d = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
        basis_3d = ConstantBasis(domain_3d)

        assert basis_3d.spatial_dim == 3
        np.testing.assert_array_equal(basis_3d.domain, domain_3d)

    def test_realistic_2d_scenario(self):
        """Test realistic 2D scenario like FEM with spatial field."""
        # 2D domain representing a rectangular material sample
        domain = [[0.0, 10.0], [0.0, 5.0]]  # 10mm x 5mm
        basis = ConstantBasis(domain)

        # Uncertain material property (e.g., Young's modulus)
        mean = np.array([200.0])  # GPa
        cov = np.array([[10.0**2]])  # std = 10 GPa
        dist = MultivariateNormal(mean, cov)

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        # Sample material property
        sample_coeff = field.coefficient_distribution.get_sample()

        # Evaluate at quadrature points (simulating FEM)
        n_cells = 5
        n_quads_per_cell = 4
        quad_points = np.random.uniform([0.0, 0.0], [10.0, 5.0], (n_cells * n_quads_per_cell, 2))

        values = field.evaluate(jnp.array(sample_coeff), jnp.array(quad_points))

        # All values should be constant
        assert values.shape == (n_cells * n_quads_per_cell,)
        np.testing.assert_allclose(values, sample_coeff[0])

    def test_realistic_3d_scenario(self):
        """Test realistic 3D scenario."""
        # 3D domain representing a cubic material sample
        domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        basis = ConstantBasis(domain)

        # Uncertain permeability field
        mean = np.array([1e-5])  # m^2
        cov = np.array([[1e-6**2]])
        dist = MultivariateNormal(mean, cov)

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        # Evaluate at 3D mesh points
        x = np.random.uniform([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], (100, 3))
        coeffs = jnp.array([1.2e-5])

        values = field.evaluate(coeffs, jnp.array(x))

        # All values should be constant
        assert values.shape == (100,)
        np.testing.assert_allclose(values, 1.2e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
