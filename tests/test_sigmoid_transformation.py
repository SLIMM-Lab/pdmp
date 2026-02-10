"""Tests for the Sigmoid transformation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdmp.distributions import SigmoidTransformation


class TestSigmoidTransformation:
    """Test suite for SigmoidTransformation."""

    def test_transform_bounds(self):
        """Test that transformation maps to correct bounds."""
        a, b = 0.0, 1.0
        transform = SigmoidTransformation(a, b)

        # Test extreme values
        xi_large = np.array([10.0, 20.0, 30.0])
        xi_small = np.array([-10.0, -20.0, -30.0])

        x_large = transform.transform(xi_large)
        x_small = transform.transform(xi_small)

        # Should be close to upper bound
        assert np.all(x_large < b)
        assert np.all(x_large > b - 0.01)

        # Should be close to lower bound
        assert np.all(x_small > a)
        assert np.all(x_small < a + 0.01)

    def test_transform_inverse_transform(self):
        """Test that inverse transformation is correct."""
        a, b = -2.0, 5.0
        transform = SigmoidTransformation(a, b)

        xi_original = np.array([0.0, 1.0, -1.0, 2.5])
        x = transform.transform(xi_original)
        xi_reconstructed = transform.inverse_transform(x)

        np.testing.assert_allclose(xi_original, xi_reconstructed, rtol=1e-10)

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        transform = SigmoidTransformation(0.0, 1.0)
        xi = np.array([0.0, 1.0, -1.0])

        J = transform.jacobian(xi)
        assert J.shape == (3, 3)

        # Should be diagonal
        assert np.allclose(J - np.diag(np.diag(J)), 0.0)

    def test_log_det_jacobian(self):
        """Test log determinant of Jacobian."""
        transform = SigmoidTransformation(0.0, 1.0)
        xi = np.array([0.0, 1.0, -1.0])

        # Compute via Jacobian matrix
        J = transform.jacobian(xi)
        log_det_expected = np.log(np.abs(np.linalg.det(J)))

        # Compute via direct method
        log_det = transform.log_det_jacobian(xi)

        np.testing.assert_allclose(log_det, log_det_expected, rtol=1e-10)

    def test_grad_log_det_jacobian_finite_diff(self):
        """Test gradient of log det Jacobian via finite differences."""
        transform = SigmoidTransformation(0.0, 1.0)
        xi = np.array([0.5, -0.3, 1.2])

        # Analytical gradient
        grad_analytical = transform.grad_log_det_jacobian(xi)

        # Numerical gradient via finite differences
        eps = 1e-7
        grad_numerical = np.zeros_like(xi)
        for i in range(len(xi)):
            xi_plus = xi.copy()
            xi_minus = xi.copy()
            xi_plus[i] += eps
            xi_minus[i] -= eps

            grad_numerical[i] = (
                transform.log_det_jacobian(xi_plus) -
                transform.log_det_jacobian(xi_minus)
            ) / (2 * eps)

        np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-8)

    def test_hessian_log_det_jacobian_finite_diff(self):
        """Test Hessian of log det Jacobian via finite differences."""
        transform = SigmoidTransformation(0.0, 1.0)
        xi = np.array([0.5, -0.3])

        # Analytical Hessian
        H_analytical = transform.hessian_log_det_jacobian(xi)

        # Numerical Hessian via finite differences (only check diagonal)
        eps = 1e-6
        H_numerical = np.zeros((len(xi), len(xi)))
        for i in range(len(xi)):
            xi_plus = xi.copy()
            xi_minus = xi.copy()
            xi_plus[i] += eps
            xi_minus[i] -= eps

            grad_plus = transform.grad_log_det_jacobian(xi_plus)
            grad_minus = transform.grad_log_det_jacobian(xi_minus)

            H_numerical[i, i] = (grad_plus[i] - grad_minus[i]) / (2 * eps)

        # Only check diagonal elements (element-wise transformation)
        np.testing.assert_allclose(np.diag(H_analytical), np.diag(H_numerical), rtol=1e-4, atol=1e-7)
        # Off-diagonal should be zero
        assert np.allclose(H_analytical - np.diag(np.diag(H_analytical)), 0.0)

    def test_hessian_tensor(self):
        """Test Hessian tensor of transformation."""
        transform = SigmoidTransformation(0.0, 1.0)
        xi = np.array([0.5, -0.3])

        # Analytical Hessian
        H = transform.hessian(xi)

        # Test shape
        assert H.shape == (2, 2, 2)

        # For element-wise transformation, only diagonal elements should be non-zero
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if i == j == k:
                        # Diagonal element should be non-zero
                        assert H[i, j, k] != 0.0
                    else:
                        # Off-diagonal should be zero
                        assert H[i, j, k] == 0.0

    def test_vector_bounds(self):
        """Test sigmoid transformation with vector bounds."""
        a = np.array([0.0, -1.0, 2.0])
        b = np.array([1.0, 1.0, 5.0])
        transform = SigmoidTransformation(a, b)

        xi = np.array([0.0, 0.0, 0.0])
        x = transform.transform(xi)

        # Check bounds
        assert np.all(x >= a)
        assert np.all(x <= b)

        # At xi=0, sigmoid(0) = 0.5, so x should be at midpoint
        expected = a + 0.5 * (b - a)
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_invalid_bounds(self):
        """Test that invalid bounds raise an error."""
        with pytest.raises(ValueError, match="Upper bound b must be greater than lower bound a"):
            SigmoidTransformation(1.0, 0.0)

        with pytest.raises(ValueError, match="Upper bound b must be greater than lower bound a"):
            SigmoidTransformation(np.array([0.0, 1.0]), np.array([1.0, 0.5]))

    def test_numerical_stability(self):
        """Test numerical stability for extreme values."""
        transform = SigmoidTransformation(0.0, 1.0)

        # Very large positive values
        xi_large = np.array([100.0, 500.0, 1000.0])
        x_large = transform.transform(xi_large)
        assert np.all(np.isfinite(x_large))
        # Due to clipping, very large values map to exactly 1.0
        assert np.all(x_large <= 1.0)
        assert np.all(x_large >= 0.999)  # Very close to 1.0

        # Very large negative values
        xi_small = np.array([-100.0, -500.0, -1000.0])
        x_small = transform.transform(xi_small)
        assert np.all(np.isfinite(x_small))
        assert np.all(x_small >= 0.0)
        assert np.all(x_small <= 0.001)  # Very close to 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
