"""Unit tests for ConstantBasis class."""

import numpy as np
import jax.numpy as jnp
import jax
import pytest

from pdmp.project_field import ConstantBasis
from pdmp.random_field import JaxRandomFieldBase
from pdmp.distributions import MultivariateNormal


class TestConstantBasis:
    """Test suite for ConstantBasis class."""

    def test_initialization(self):
        """Test ConstantBasis initialization."""
        basis = ConstantBasis((0.0, 1.0))

        assert basis.get_n() == 1
        assert basis.spatial_dim == 1
        np.testing.assert_array_equal(basis.domain, [[0.0, 1.0]])
        assert basis.support_.shape == (1, 2)
        np.testing.assert_array_equal(basis.support_[0], [0.0, 1.0])

    def test_norms(self):
        """Test norm calculation."""
        # Norm should be sqrt(interval length)
        basis1 = ConstantBasis((0.0, 1.0))
        assert basis1.get_norms().shape == (1, 1)
        np.testing.assert_almost_equal(basis1.get_norms()[0, 0], 1.0)

        basis2 = ConstantBasis((0.0, 4.0))
        np.testing.assert_almost_equal(basis2.get_norms()[0, 0], 2.0)

    def test_evaluation_all_basis(self):
        """Test evaluation of all basis functions."""
        basis = ConstantBasis((0.0, 1.0))
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        phi = basis(x)

        # Should return shape (n_points, 1)
        assert phi.shape == (5, 1)
        # All values should be 1
        np.testing.assert_array_equal(phi, np.ones((5, 1)))

    def test_evaluation_single_basis(self):
        """Test evaluation of single basis function."""
        basis = ConstantBasis((0.0, 1.0))
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        phi = basis(x, i=0)

        # Should return shape (n_points,)
        assert phi.shape == (5, )
        # All values should be 1
        np.testing.assert_array_equal(phi, np.ones(5))

    def test_evaluation_invalid_index(self):
        """Test that invalid basis index raises error."""
        basis = ConstantBasis((0.0, 1.0))
        x = np.array([0.5])

        with pytest.raises(ValueError, match="only has 1 basis function"):
            basis(x, i=1)

    def test_get_support_all(self):
        """Test getting support for all basis functions."""
        basis = ConstantBasis((0.0, 2.0))
        support = basis.get_support()

        assert support.shape == (1, 2)
        np.testing.assert_array_equal(support[0], [0.0, 2.0])

    def test_get_support_single(self):
        """Test getting support for single basis function."""
        basis = ConstantBasis((0.0, 2.0))
        support = basis.get_support(i=0)

        np.testing.assert_array_equal(support, [0.0, 2.0])

    def test_get_support_invalid_index(self):
        """Test that invalid support index raises error."""
        basis = ConstantBasis((0.0, 1.0))

        with pytest.raises(ValueError, match="only has 1 basis function"):
            basis.get_support(i=1)

    def test_with_jax_random_field_base(self):
        """Test ConstantBasis with JaxRandomFieldBase."""
        basis = ConstantBasis((0.0, 1.0))
        mean = np.array([2.0])
        cov = np.array([[0.5**2]])
        dist = MultivariateNormal(mean, cov)

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        assert field.dim == 1

        # Test evaluation
        x = jnp.array([0.0, 0.5, 1.0])
        coeffs = jnp.array([3.0])
        values = field.evaluate(coeffs, x)

        # All values should be 3.0 (constant)
        assert values.shape == (3, )
        np.testing.assert_allclose(values, 3.0)

    def test_gradient_computation(self):
        """Test that gradients work correctly with ConstantBasis."""
        basis = ConstantBasis((0.0, 1.0))
        mean = np.array([1.0])
        cov = np.array([[0.2**2]])
        dist = MultivariateNormal(mean, cov)
        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        def loss_fn(coeffs):
            x = jnp.array([0.0, 0.5, 1.0])
            values = field.evaluate(coeffs, x)
            return jnp.sum(values**2)

        coeffs = jnp.array([2.0])
        grad = jax.grad(loss_fn)(coeffs)

        # Analytical gradient: d/dc sum(c^2) = 2*c*n_points
        expected_grad = 2 * 2.0 * 3  # 12.0
        np.testing.assert_allclose(grad[0], expected_grad)

    def test_different_intervals(self):
        """Test ConstantBasis with different intervals."""
        intervals = [(0.0, 1.0), (-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]

        for interval in intervals:
            basis = ConstantBasis(interval)
            x = np.linspace(interval[0], interval[1], 10)
            phi = basis(x)

            # All values should be 1
            np.testing.assert_array_equal(phi, np.ones((10, 1)))

            # Norm should be sqrt(interval length)
            expected_norm = np.sqrt(interval[1] - interval[0])
            np.testing.assert_almost_equal(basis.get_norms()[0, 0],
                                           expected_norm)

    def test_scalar_input(self):
        """Test evaluation with scalar input."""
        basis = ConstantBasis((0.0, 1.0))

        # Single scalar point
        x = np.array([0.5])
        phi = basis(x)

        assert phi.shape == (1, 1)
        assert phi[0, 0] == 1.0

    def test_distribution_sampling(self):
        """Test that sampling works correctly with constant fields."""
        basis = ConstantBasis((0.0, 1.0))
        mean = np.array([5.0])
        cov = np.array([[1.0**2]])
        dist = MultivariateNormal(mean, cov, rng=np.random.default_rng(42))

        field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

        # Sample multiple times
        samples = [
            field.coefficient_distribution.get_sample()[0]
            for _ in range(10000)
        ]

        # Check that samples have approximately correct mean and std
        # Using looser tolerance for std since it has higher variance
        np.testing.assert_allclose(np.mean(samples), 5.0, rtol=0.1)
        np.testing.assert_allclose(np.std(samples), 1.0, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
