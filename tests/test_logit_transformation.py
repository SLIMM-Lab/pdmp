"""Test LogitTransformation implementation."""
import numpy as np
from scipy.special import logit, expit  # expit is sigmoid
from pdmp.distributions import (LogitTransformation, SigmoidTransformation,
                                MultivariateNormal, TransformedDistribution,
                                LOGIT)


def test_logit_transform_basic():
    """Test basic logit transformation."""
    trans = LogitTransformation(a=0.0, b=1.0)

    # Test some values in (0, 1)
    x = np.array([0.1, 0.5, 0.9])
    xi = trans.transform(x)
    x_back = trans.inverse_transform(xi)

    # Should be able to go back and forth
    assert np.allclose(x, x_back, rtol=1e-10), \
        f"Round-trip failed: x={x}, x_back={x_back}"

    print("✓ Basic logit transform test passed")


def test_logit_scipy_comparison():
    """Compare with scipy's logit function."""
    trans = LogitTransformation(a=0.0, b=1.0)

    # Test against scipy for standard [0, 1] case
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    our_result = trans.transform(x)
    scipy_result = logit(x)

    assert np.allclose(our_result, scipy_result, rtol=1e-10), \
        f"Logit differs from scipy: ours={our_result}, scipy={scipy_result}"

    # Test inverse against scipy expit (sigmoid)
    xi = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    our_result = trans.inverse_transform(xi)
    scipy_result = expit(xi)

    assert np.allclose(our_result, scipy_result, rtol=1e-10), \
        f"Inverse logit differs from scipy expit: ours={our_result}, scipy={scipy_result}"

    print("✓ SciPy comparison test passed")


def test_logit_inverse_of_sigmoid():
    """Test that logit is truly the inverse of sigmoid."""
    a = np.array([0.0, -1.0])
    b = np.array([1.0, 2.0])

    logit_trans = LogitTransformation(a=a, b=b)
    sigmoid_trans = SigmoidTransformation(a=a, b=b)

    # Start with unbounded values
    xi = np.array([[-2.0, 0.5], [0.0, 1.0], [2.0, -0.5]])

    # Sigmoid: ξ -> x (unbounded to bounded)
    x_from_sigmoid = sigmoid_trans.transform(xi)

    # Logit: x -> ξ (bounded to unbounded)
    xi_from_logit = logit_trans.transform(x_from_sigmoid)

    # Should get back original ξ
    assert np.allclose(xi, xi_from_logit, rtol=1e-10), \
        f"Logit is not inverse of sigmoid: original={xi}, recovered={xi_from_logit}"

    # Test the other direction
    # Start with bounded values (make sure they're within [a, b])
    x = np.array([[0.2, 0.5], [0.5, 1.0], [0.8,
                                           0.0]])  # All in [0, 1] x [-1, 2]

    # Logit: x -> ξ
    xi_from_logit = logit_trans.transform(x)

    # Sigmoid's transform (not inverse_transform): ξ -> x
    x_from_sigmoid = sigmoid_trans.transform(xi_from_logit)

    # Should get back original x
    assert np.allclose(x, x_from_sigmoid, rtol=1e-10), \
        f"Compositions don't match: original={x}, recovered={x_from_sigmoid}"

    print("✓ Logit is inverse of sigmoid test passed")


def test_logit_jacobian():
    """Test Jacobian computation."""
    trans = LogitTransformation(a=0.0, b=1.0)

    x = np.array([0.3, 0.7])
    J = trans.jacobian(x)

    # Compute numerical Jacobian
    eps = 1e-7
    J_numerical = np.zeros((2, 2))
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        J_numerical[:, i] = (trans.transform(x_plus) -
                             trans.transform(x_minus)) / (2 * eps)

    assert np.allclose(J, J_numerical, rtol=1e-5, atol=1e-8), \
        f"Jacobian mismatch:\nAnalytical:\n{J}\nNumerical:\n{J_numerical}"

    print("✓ Jacobian test passed")


def test_logit_log_det_jacobian():
    """Test log determinant of Jacobian."""
    trans = LogitTransformation(a=0.0, b=1.0)

    x = np.array([0.2, 0.5, 0.8])
    log_det = trans.log_det_jacobian(x)

    # Compute from Jacobian
    J = trans.jacobian(x)
    log_det_from_J = np.log(np.abs(np.linalg.det(J)))

    assert np.isclose(log_det, log_det_from_J, rtol=1e-10), \
        f"log_det_jacobian mismatch: direct={log_det}, from_J={log_det_from_J}"

    print("✓ Log det Jacobian test passed")


def test_logit_grad_log_det_jacobian():
    """Test gradient of log determinant."""
    trans = LogitTransformation(a=0.0, b=1.0)

    x = np.array([0.3, 0.7])
    grad = trans.grad_log_det_jacobian(x)

    # Compute numerical gradient
    eps = 1e-7
    grad_numerical = np.zeros(2)
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad_numerical[i] = (trans.log_det_jacobian(x_plus) -
                             trans.log_det_jacobian(x_minus)) / (2 * eps)

    assert np.allclose(grad, grad_numerical, rtol=1e-5, atol=1e-8), \
        f"Gradient mismatch:\nAnalytical: {grad}\nNumerical: {grad_numerical}"

    print("✓ Gradient log det Jacobian test passed")


def test_logit_hessian_log_det_jacobian():
    """Test Hessian of log determinant."""
    trans = LogitTransformation(a=0.0, b=1.0)

    x = np.array([0.3, 0.7])
    H = trans.hessian_log_det_jacobian(x)

    # Compute numerical Hessian
    eps = 1e-6
    H_numerical = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps
            x_pp[j] += eps

            x_pm[i] += eps
            x_pm[j] -= eps

            x_mp[i] -= eps
            x_mp[j] += eps

            x_mm[i] -= eps
            x_mm[j] -= eps

            H_numerical[i, j] = (trans.log_det_jacobian(x_pp) -
                                 trans.log_det_jacobian(x_pm) -
                                 trans.log_det_jacobian(x_mp) +
                                 trans.log_det_jacobian(x_mm)) / (4 * eps**2)

    # Check diagonal elements (should match well)
    diag_analytical = np.diag(H)
    diag_numerical = np.diag(H_numerical)
    assert np.allclose(diag_analytical, diag_numerical, rtol=1e-4, atol=1e-6), \
        f"Diagonal mismatch: analytical={diag_analytical}, numerical={diag_numerical}"

    # Check off-diagonal elements (should be zero, but numerical errors appear)
    H_offdiag = H - np.diag(np.diag(H))
    H_numerical_offdiag = H_numerical - np.diag(np.diag(H_numerical))
    assert np.allclose(H_offdiag, H_numerical_offdiag, atol=1e-3), \
        f"Off-diagonal mismatch (but both should be near zero)"

    print("✓ Hessian log det Jacobian test passed")


def test_logit_transformed_distribution():
    """Test using LogitTransformation with TransformedDistribution."""
    # Create a Gaussian in unbounded space
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    base_dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # Transform to bounded [0, 1]²
    params = {'transformation': LOGIT, 'a': 0.0, 'b': 1.0}

    transformed_dist = TransformedDistribution(base_dist, params)

    # Sample from transformed distribution (should be in [0, 1]²)
    samples = transformed_dist.get_sample(n=1000)

    # Check that samples are in valid range
    assert np.all(samples >= -0.01) and np.all(samples <= 1.01), \
        f"Samples out of bounds: min={np.min(samples)}, max={np.max(samples)}"

    # Evaluate log density at a point in bounded space
    x = np.array([0.3, 0.7])
    log_p = transformed_dist.log_density(x)

    assert np.isfinite(log_p), f"Log density is not finite: {log_p}"

    print("✓ TransformedDistribution with logit test passed")


def test_logit_scaling():
    """Test logit with different bounds."""
    # Test with [a, b] = [-1, 2]
    trans = LogitTransformation(a=-1.0, b=2.0)

    x = np.array([0.0, 0.5, 1.0])  # Values in [-1, 2]
    xi = trans.transform(x)
    x_back = trans.inverse_transform(xi)

    assert np.allclose(x, x_back, rtol=1e-10), \
        f"Round-trip failed for scaled logit: x={x}, x_back={x_back}"

    # Check transform values make sense
    # x=0.0 should map to logit((0-(-1))/(2-(-1))) = logit(1/3)
    expected_xi_0 = np.log((1 / 3) / (2 / 3))  # = np.log(0.5)
    assert np.isclose(xi[0], expected_xi_0, rtol=1e-10), \
        f"Transform at midpoint incorrect: {xi[0]} vs {expected_xi_0}"

    print("✓ Logit scaling test passed")


def test_logit_vectorized():
    """Test that logit works with vector bounds."""
    a = np.array([0.0, -1.0, 0.5])
    b = np.array([1.0, 1.0, 2.0])

    trans = LogitTransformation(a=a, b=b)

    x = np.array([0.5, 0.0, 1.0])
    xi = trans.transform(x)
    x_back = trans.inverse_transform(xi)

    assert np.allclose(x, x_back, rtol=1e-10), \
        f"Vector bounds round-trip failed: x={x}, x_back={x_back}"

    # Test batch
    x_batch = np.array([[0.2, -0.5, 1.0], [0.8, 0.5, 1.5]])
    xi_batch = trans.transform(x_batch)
    x_back_batch = trans.inverse_transform(xi_batch)

    assert np.allclose(x_batch, x_back_batch, rtol=1e-10), \
        f"Batch round-trip failed"

    print("✓ Vectorized logit test passed")


if __name__ == "__main__":
    print("=== Basic Tests ===")
    test_logit_transform_basic()
    test_logit_scipy_comparison()
    test_logit_inverse_of_sigmoid()

    print("\n=== Derivative Tests ===")
    test_logit_jacobian()
    test_logit_log_det_jacobian()
    test_logit_grad_log_det_jacobian()
    test_logit_hessian_log_det_jacobian()

    print("\n=== Integration Tests ===")
    test_logit_transformed_distribution()
    test_logit_scaling()
    test_logit_vectorized()

    print("\n✅ All logit transformation tests passed!")
