"""Test batch handling for MultivariateNormal distribution."""
import numpy as np
from scipy.stats import multivariate_normal
from pdmp.distributions import MultivariateNormal


def test_multivariate_normal_batch_log_density():
    """Test that batch log_density returns consistent results."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # Test single point
    x_single = np.array([1.5, 2.5, 3.5])
    log_dens_single = dist.log_density(x_single)

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
    log_dens_batch = dist.log_density(x_batch)

    # First element of batch should match single point
    assert isinstance(log_dens_single,
                      float), "Single point should return float"
    assert isinstance(log_dens_batch, np.ndarray), "Batch should return array"
    assert log_dens_batch.shape == (
        3, ), f"Expected shape (3,), got {log_dens_batch.shape}"
    assert np.isclose(log_dens_batch[0], log_dens_single), \
        f"Batch[0] = {log_dens_batch[0]} != single = {log_dens_single}"

    # Check all batch elements individually
    for i in range(x_batch.shape[0]):
        single = dist.log_density(x_batch[i])
        batch = log_dens_batch[i]
        assert np.isclose(single, batch), \
            f"Point {i}: single = {single}, batch = {batch}"

    print("✓ log_density batch test passed")


def test_multivariate_normal_batch_grad_log_density():
    """Test that batch grad_log_density returns consistent results."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # Test single point
    x_single = np.array([1.5, 2.5, 3.5])
    grad_single = dist.grad_log_density(x_single)

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
    grad_batch = dist.grad_log_density(x_batch)

    # Check shapes
    assert grad_single.shape == (
        3, ), f"Single grad shape: expected (3,), got {grad_single.shape}"
    assert grad_batch.shape == (
        3, 3), f"Batch grad shape: expected (3, 3), got {grad_batch.shape}"

    # First row of batch should match single point
    assert np.allclose(grad_batch[0], grad_single), \
        f"Batch[0] = {grad_batch[0]} != single = {grad_single}"

    # Check all batch elements individually
    for i in range(x_batch.shape[0]):
        single = dist.grad_log_density(x_batch[i])
        batch = grad_batch[i]
        assert np.allclose(single, batch), \
            f"Point {i}: single = {single}, batch = {batch}"

    print("✓ grad_log_density batch test passed")


def test_multivariate_normal_batch_hessian_log_density():
    """Test that batch hessian_log_density returns consistent results."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # Test single point
    x_single = np.array([1.5, 2.5, 3.5])
    hess_single = dist.hessian_log_density(x_single)

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
    hess_batch = dist.hessian_log_density(x_batch)

    # Check shapes
    assert hess_single.shape == (
        3, 3), f"Single hess shape: expected (3, 3), got {hess_single.shape}"
    assert hess_batch.shape == (
        3, 3,
        3), f"Batch hess shape: expected (3, 3, 3), got {hess_batch.shape}"

    # First element of batch should match single point
    assert np.allclose(hess_batch[0], hess_single), \
        f"Batch[0] != single"

    # Check all batch elements individually (should all be the same since Hessian is constant)
    for i in range(x_batch.shape[0]):
        single = dist.hessian_log_density(x_batch[i])
        batch = hess_batch[i]
        assert np.allclose(single, batch), \
            f"Point {i}: matrices don't match"
        assert np.allclose(batch, hess_single), \
            f"Point {i}: Hessian should be constant"

    print("✓ hessian_log_density batch test passed")


def test_efficiency():
    """Compare efficiency of batch vs looped evaluation."""
    import time

    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cov = np.eye(5) + 0.3 * np.ones((5, 5))

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # Generate test points
    n_points = 1000
    x_batch = dist.get_sample(n=n_points)

    # Test log_density
    start = time.time()
    result_batch = dist.log_density(x_batch)
    time_batch = time.time() - start

    start = time.time()
    result_loop = np.array([dist.log_density(x) for x in x_batch])
    time_loop = time.time() - start

    assert np.allclose(result_batch, result_loop), "Results don't match"
    print(
        f"log_density: batch={time_batch:.4f}s, loop={time_loop:.4f}s, speedup={time_loop/time_batch:.2f}x"
    )

    # Test grad_log_density
    start = time.time()
    result_batch = dist.grad_log_density(x_batch)
    time_batch = time.time() - start

    start = time.time()
    result_loop = np.array([dist.grad_log_density(x) for x in x_batch])
    time_loop = time.time() - start

    assert np.allclose(result_batch, result_loop), "Results don't match"
    print(
        f"grad_log_density: batch={time_batch:.4f}s, loop={time_loop:.4f}s, speedup={time_loop/time_batch:.2f}x"
    )

    # Test hessian_log_density
    start = time.time()
    result_batch = dist.hessian_log_density(x_batch)
    time_batch = time.time() - start

    start = time.time()
    result_loop = np.array([dist.hessian_log_density(x) for x in x_batch])
    time_loop = time.time() - start

    assert np.allclose(result_batch, result_loop), "Results don't match"
    print(
        f"hessian_log_density: batch={time_batch:.4f}s, loop={time_loop:.4f}s, speedup={time_loop/time_batch:.2f}x"
    )


def test_scipy_comparison_log_density():
    """Compare log_density results with scipy.stats.multivariate_normal."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)
    scipy_dist = multivariate_normal(mean=mean, cov=cov)

    # Test single point
    x_single = np.array([1.5, 2.5, 3.5])
    our_result = dist.log_density(x_single)
    scipy_result = scipy_dist.logpdf(x_single)

    assert np.isclose(our_result, scipy_result, rtol=1e-10), \
        f"Single point: ours={our_result}, scipy={scipy_result}"

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5],
                        [-1.0, 0.0, 1.0]])
    our_results = dist.log_density(x_batch)
    scipy_results = scipy_dist.logpdf(x_batch)

    assert np.allclose(our_results, scipy_results, rtol=1e-10), \
        f"Batch: max diff = {np.max(np.abs(our_results - scipy_results))}"

    print("✓ scipy comparison log_density test passed")


def test_scipy_comparison_samples():
    """Verify that samples have correct statistics."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=12345)
    scipy_dist = multivariate_normal(mean=mean, cov=cov)

    # Generate many samples
    n_samples = 10000
    samples = dist.get_sample(n=n_samples)

    # Check sample statistics
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples.T)

    # Mean should be close (within 3 standard errors)
    se_mean = np.sqrt(np.diag(cov) / n_samples)
    assert np.allclose(sample_mean, mean, atol=3*se_mean), \
        f"Sample mean {sample_mean} differs from true mean {mean}"

    # Covariance should be close
    assert np.allclose(sample_cov, cov, atol=0.1), \
        f"Sample cov differs from true cov, max diff={np.max(np.abs(sample_cov - cov))}"

    # Check that scipy agrees on log densities of our samples
    our_log_dens = dist.log_density(samples[:100])
    scipy_log_dens = scipy_dist.logpdf(samples[:100])
    assert np.allclose(our_log_dens, scipy_log_dens, rtol=1e-10), \
        f"Log densities differ, max diff={np.max(np.abs(our_log_dens - scipy_log_dens))}"

    print("✓ scipy comparison samples test passed")


def test_scipy_comparison_gradient():
    """Compare gradient with numerical gradient of scipy logpdf."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)
    scipy_dist = multivariate_normal(mean=mean, cov=cov)

    # Test single point
    x = np.array([1.5, 2.5, 3.5])
    our_grad = dist.grad_log_density(x)

    # Compute numerical gradient using scipy
    eps = 1e-7
    numerical_grad = np.zeros(3)
    for i in range(3):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        numerical_grad[i] = (scipy_dist.logpdf(x_plus) -
                             scipy_dist.logpdf(x_minus)) / (2 * eps)

    assert np.allclose(our_grad, numerical_grad, rtol=1e-5, atol=1e-8), \
        f"Gradient: ours={our_grad}, numerical={numerical_grad}, diff={our_grad - numerical_grad}"

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
    our_grads = dist.grad_log_density(x_batch)

    # Compute numerical gradients for batch
    numerical_grads = np.zeros((3, 3))
    for j, x_point in enumerate(x_batch):
        for i in range(3):
            x_plus = x_point.copy()
            x_minus = x_point.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            numerical_grads[j, i] = (scipy_dist.logpdf(x_plus) -
                                     scipy_dist.logpdf(x_minus)) / (2 * eps)

    assert np.allclose(our_grads, numerical_grads, rtol=1e-5, atol=1e-8), \
        f"Batch gradient max diff: {np.max(np.abs(our_grads - numerical_grads))}"

    print("✓ scipy comparison gradient test passed")


def test_scipy_comparison_hessian():
    """Compare Hessian with analytical result (should be -inv(cov))."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    dist = MultivariateNormal(mean=mean, cov=cov, seed=42)

    # For Gaussian, Hessian should be -inv(cov) everywhere
    expected_hessian = -np.linalg.inv(cov)

    # Test single point
    x = np.array([1.5, 2.5, 3.5])
    our_hessian = dist.hessian_log_density(x)

    assert np.allclose(our_hessian, expected_hessian, rtol=1e-10), \
        f"Hessian differs from -inv(cov)"

    # Test batch
    x_batch = np.array([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
    our_hessians = dist.hessian_log_density(x_batch)

    for i in range(len(x_batch)):
        assert np.allclose(our_hessians[i], expected_hessian, rtol=1e-10), \
            f"Hessian {i} differs from -inv(cov)"

    print("✓ scipy comparison Hessian test passed")


if __name__ == "__main__":
    print("=== Basic Batch Tests ===")
    test_multivariate_normal_batch_log_density()
    test_multivariate_normal_batch_grad_log_density()
    test_multivariate_normal_batch_hessian_log_density()

    print("\n=== SciPy Comparison Tests ===")
    test_scipy_comparison_log_density()
    test_scipy_comparison_samples()
    test_scipy_comparison_gradient()
    test_scipy_comparison_hessian()

    print("\n=== Performance Tests ===")
    test_efficiency()
    print("\n✅ All tests passed!")
