"""Tests for CompositeTransformation."""
import numpy as np
import pytest
from pdmp.distributions import (
    CompositeTransformation,
    ExponentialTransformation,
    SigmoidTransformation,
    AffineTransformation,
    MultivariateNormal,
    JointDistribution,
    TransformedDistribution,
    COMPOSITE,
    EXPONENTIAL,
    SIGMOID,
)


class TestCompositeTransformation:
    """Test composite transformation functionality."""

    def test_basic_composite(self):
        """Test basic composite transformation with explicit indices."""
        # Create a composite: sigmoid on [0,1], exponential on [2,3,4]
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3, 4])])

        xi = np.array([0.0, 1.0, 0.0, 0.5, -0.5])

        # Transform
        x = trans.transform(xi)

        # Check sigmoid part (first 2 dims)
        assert 0 <= x[0] <= 1
        assert 0 <= x[1] <= 1

        # Check exponential part (last 3 dims)
        assert x[2] > 0
        assert x[3] > 0
        assert x[4] > 0
        assert np.allclose(x[2], np.exp(xi[2]))
        assert np.allclose(x[3], np.exp(xi[3]))
        assert np.allclose(x[4], np.exp(xi[4]))

    def test_inverse_transform(self):
        """Test that inverse transform is correct."""
        trans = CompositeTransformation(
            [SigmoidTransformation(-1, 2),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3])])

        xi = np.array([0.5, -0.5, 1.0, -1.0])
        x = trans.transform(xi)
        xi_recovered = trans.inverse_transform(x)

        assert np.allclose(xi, xi_recovered, rtol=1e-10)

    def test_jacobian_block_diagonal(self):
        """Test that Jacobian has correct block-diagonal structure."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3, 4])])

        xi = np.array([0.0, 0.5, 0.0, 0.5, -0.5])
        J = trans.jacobian(xi)

        # Check shape
        assert J.shape == (5, 5)

        # Check block-diagonal structure (off-diagonal blocks should be zero)
        assert np.allclose(J[0:2, 2:5], 0)
        assert np.allclose(J[2:5, 0:2], 0)

        # Check diagonal blocks are non-zero
        assert not np.allclose(J[0:2, 0:2], 0)
        assert not np.allclose(J[2:5, 2:5], 0)

    def test_log_det_jacobian(self):
        """Test log determinant calculation."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2])])

        xi = np.array([0.0, 0.5, 1.0])

        # Compute via composite
        log_det = trans.log_det_jacobian(xi)

        # Compute manually
        sigmoid_trans = SigmoidTransformation(0, 1)
        exp_trans = ExponentialTransformation()
        log_det_manual = (sigmoid_trans.log_det_jacobian(xi[0:2]) +
                          exp_trans.log_det_jacobian(xi[2:3]))

        assert np.allclose(log_det, log_det_manual)

    def test_grad_log_det_jacobian(self):
        """Test gradient of log determinant."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3])])

        xi = np.array([0.0, 0.5, 1.0, -0.5])
        grad = trans.grad_log_det_jacobian(xi)

        # Check shape
        assert grad.shape == xi.shape

        # Compute numerical gradient for verification
        eps = 1e-6
        grad_numerical = np.zeros_like(xi)
        for i in range(len(xi)):
            xi_plus = xi.copy()
            xi_plus[i] += eps
            xi_minus = xi.copy()
            xi_minus[i] -= eps
            grad_numerical[i] = (trans.log_det_jacobian(xi_plus) -
                                 trans.log_det_jacobian(xi_minus)) / (2 * eps)

        assert np.allclose(grad, grad_numerical, atol=1e-5)

    def test_hessian_log_det_jacobian(self):
        """Test Hessian of log determinant has block-diagonal structure."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3])])

        xi = np.array([0.0, 0.5, 1.0, -0.5])
        H = trans.hessian_log_det_jacobian(xi)

        # Check shape
        assert H.shape == (4, 4)

        # Check block-diagonal structure
        assert np.allclose(H[0:2, 2:4], 0)
        assert np.allclose(H[2:4, 0:2], 0)

    def test_with_slices(self):
        """Test that slice notation works for indices."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [slice(0, 2), slice(2, 5)])

        xi = np.array([0.0, 1.0, 0.0, 0.5, -0.5])
        x = trans.transform(xi)

        # Should work the same as array indices
        trans2 = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([2, 3, 4])])
        x2 = trans2.transform(xi)

        assert np.allclose(x, x2)

    def test_validation_overlapping_indices(self):
        """Test that overlapping indices raise an error."""
        with pytest.raises(ValueError, match="disjoint"):
            CompositeTransformation(
                [SigmoidTransformation(0, 1),
                 ExponentialTransformation()],
                [np.array([0, 1, 2]), np.array([2, 3])]  # 2 appears twice
            )

    def test_validation_incomplete_indices(self):
        """Test that incomplete indices raise an error."""
        trans = CompositeTransformation(
            [SigmoidTransformation(0, 1),
             ExponentialTransformation()],
            [np.array([0, 1]), np.array([3, 4])]  # Missing index 2
        )

        xi = np.array([0.0, 1.0, 0.0, 0.5, -0.5])

        # Should raise error about not covering all dimensions or not being a partition
        with pytest.raises(ValueError, match="(cover all|partition)"):
            trans.transform(xi)

    def test_three_way_composite(self):
        """Test composite with three different transformations."""
        # Sigmoid on [0,1], exponential on [2], affine on [3,4]
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, -1.0])

        trans = CompositeTransformation([
            SigmoidTransformation(0, 1),
            ExponentialTransformation(),
            AffineTransformation(M, b)
        ], [np.array([0, 1]),
            np.array([2]), np.array([3, 4])])

        xi = np.array([0.0, 0.5, 1.0, 0.0, 0.0])
        x = trans.transform(xi)

        # Check dimensions
        assert x.shape == xi.shape

        # Check invertibility
        xi_recovered = trans.inverse_transform(x)
        assert np.allclose(xi, xi_recovered)


class TestCompositeWithDistributions:
    """Test composite transformations with distributions."""

    def test_transformed_distribution_composite(self):
        """Test TransformedDistribution with composite transformation."""
        # Create a 5D Gaussian - this serves as the prior in the TRANSFORMED space
        # For exponential transform: x > 0, for sigmoid: x in [0,1]
        # So we need a base distribution with appropriate support
        # Instead of sampling (which would require compatible base dist),
        # we directly test with a manually created xi in unconstrained space
        mean = np.zeros(5)
        cov = np.eye(5)
        base_dist = MultivariateNormal(mean, cov)

        # Apply composite transformation
        params = {
            'transformation': COMPOSITE,
            'transformations': [EXPONENTIAL, SIGMOID],
            'indices': [np.array([0, 1, 2]),
                        np.array([3, 4])]
        }

        trans_dist = TransformedDistribution(base_dist, params)

        # Create a test point in unconstrained space (xi)
        xi = np.array([0.0, 0.5, -0.5, 0.0, 1.0])

        # Test log density
        log_p = trans_dist.log_density(xi)
        assert np.isfinite(log_p)

        # Test gradient
        grad = trans_dist.grad_log_density(xi)
        assert grad.shape == (5, )
        assert np.all(np.isfinite(grad))

    def test_joint_distribution_automatic_indices(self):
        """Test that indices are automatically created for JointDistribution."""
        # Create joint of three distributions
        dist1 = MultivariateNormal(np.zeros(2), np.eye(2))  # dim=2
        dist2 = MultivariateNormal(np.zeros(3), np.eye(3))  # dim=3
        dist3 = MultivariateNormal(np.zeros(1), np.eye(1))  # dim=1

        joint = JointDistribution([dist1, dist2, dist3])

        # Apply composite transformation (indices should be auto-generated)
        params = {
            'transformation': COMPOSITE,
            'transformations': [EXPONENTIAL, SIGMOID, EXPONENTIAL]
        }

        trans_dist = TransformedDistribution(joint, params)

        # Use manually created xi in unconstrained space instead of sampling
        xi = np.array([0.0, 0.5, 0.0, 0.5, -0.5, 1.0])
        assert xi.shape == (6, )

        log_p = trans_dist.log_density(xi)
        assert np.isfinite(log_p)

    def test_composite_with_dict_specs(self):
        """Test composite transformation with dict specifications."""
        base_dist = MultivariateNormal(np.zeros(4), np.eye(4))

        params = {
            'transformation':
            COMPOSITE,
            'transformations': [{
                'type': SIGMOID,
                'a': -1,
                'b': 2
            }, {
                'type': EXPONENTIAL
            }],
            'indices': [np.array([0, 1]), np.array([2, 3])]
        }

        trans_dist = TransformedDistribution(base_dist, params)

        # Use manually created xi in unconstrained space
        xi = np.array([0.5, -0.5, 1.0, -1.0])
        assert xi.shape == (4, )

        # Check that transformation was applied correctly
        x = trans_dist._transformation.transform(xi)
        # First two should be in [-1, 2]
        assert np.all(x[0:2] >= -1)
        assert np.all(x[0:2] <= 2)
        # Last two should be positive (exponential)
        assert np.all(x[2:4] > 0)

    def test_composite_with_transformation_objects(self):
        """Test composite transformation with pre-built Transformation objects."""
        base_dist = MultivariateNormal(np.zeros(3), np.eye(3))

        # Pre-build transformation objects
        sigmoid_trans = SigmoidTransformation(0, 10)
        exp_trans = ExponentialTransformation()

        params = {
            'transformation': COMPOSITE,
            'transformations': [sigmoid_trans, exp_trans],
            'indices': [np.array([0]), np.array([1, 2])]
        }

        trans_dist = TransformedDistribution(base_dist, params)

        # Use manually created xi in unconstrained space
        xi = np.array([0.0, 1.0, -0.5])
        assert xi.shape == (3, )

        x = trans_dist._transformation.transform(xi)
        assert 0 <= x[0] <= 10
        assert x[1] > 0
        assert x[2] > 0

    def test_gradient_consistency(self):
        """Test that gradients are consistent with finite differences."""
        dist1 = MultivariateNormal(np.zeros(2), np.eye(2))
        dist2 = MultivariateNormal(np.zeros(2), np.eye(2))
        joint = JointDistribution([dist1, dist2])

        params = {
            'transformation': COMPOSITE,
            'transformations': [SIGMOID, EXPONENTIAL]
        }

        trans_dist = TransformedDistribution(joint, params)

        xi = np.array([0.5, -0.5, 1.0, -1.0])

        # Analytical gradient
        grad = trans_dist.grad_log_density(xi)

        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros_like(xi)
        for i in range(len(xi)):
            xi_plus = xi.copy()
            xi_plus[i] += eps
            xi_minus = xi.copy()
            xi_minus[i] -= eps
            grad_numerical[i] = (trans_dist.log_density(xi_plus) -
                                 trans_dist.log_density(xi_minus)) / (2 * eps)

        assert np.allclose(grad, grad_numerical, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
