import numpy as np
from scipy.stats import multivariate_normal, wishart

from pdmp.distributions import MultivariateNormal, CubicDistribution
from pdmp.utils import grad_fd, hessian_fd

SMALL = -1e-6
LARGE = 1e6


def test_multivariate_normal():

    # get random mean and covariance and initialize the multivariate normal distribution
    rng = np.random.default_rng(0)
    d = rng.integers(2, 10)
    mean = rng.random(d)
    cov = wishart(df=d, scale=np.eye(d), seed=rng).rvs()
    cov = cov / np.linalg.norm(cov)
    mvn = MultivariateNormal(mean, cov)

    # test pdf
    sample = mvn.get_sample()
    assert len(sample) == d
    assert np.isclose(mvn.log_density(sample), multivariate_normal.logpdf(sample, mean, cov))

    # test gradient
    assert np.allclose(mvn.grad_log_density(sample), grad_fd(mvn.log_density, sample), atol=1e-5)

    # test hessian
    assert np.allclose(mvn.hessian_log_density(sample), hessian_fd(mvn.log_density, sample), atol=1e-5)

    # test sampling
    n_samples = 100000
    samples = np.zeros((n_samples, d))
    for i in range(n_samples):
        samples[i] = mvn.get_sample()

    mean_mvn = np.mean(samples, axis=0)
    cov_mvn = np.cov(samples, rowvar=False)
    assert np.allclose(mean_mvn, mean, atol=1e-2)
    assert np.allclose(cov_mvn, cov, atol=1e-2)


def test_cubic():

    # get random mean and covariance and initialize the multivariate normal distribution
    rng = np.random.default_rng(0)
    d = rng.integers(2, 10)
    mean = rng.random(d)
    cov = wishart(df=d, scale=np.eye(d), seed=rng).rvs()
    cov = cov / np.linalg.norm(cov)
    a = .5 * (rng.random() + 1)
    cubic_diag = rng.random(d, dtype=float)
    cubic_diag = cubic_diag / np.sum(cubic_diag)
    cubic = CubicDistribution(mean, cov, a, cubic_diag=cubic_diag)
    mvn = MultivariateNormal(mean, cov)

    # test gradient
    sample = mvn.get_sample()
    assert np.allclose(cubic.grad_log_density(sample), grad_fd(cubic.log_density, sample), atol=1e-5)

    # test hessian
    assert np.allclose(cubic.hessian_log_density(sample), hessian_fd(cubic.log_density, sample), atol=1e-5)

    # test limits
    assert np.isclose(np.exp(cubic.log_density(np.ones(d)*SMALL)), 0.0)
    assert np.isclose(np.exp(cubic.log_density(np.ones(d)*LARGE)), 0.0)

    # test symmetry around the mean
    delta = rng.random(d)
    assert np.isclose(cubic.log_density(mean + delta), cubic.log_density(mean - delta))
    assert np.allclose(cubic.grad_log_density(mean), np.zeros(d))
    hess = cubic.hessian_log_density(mean)
    assert np.allclose(hess, hess.T)
    assert np.linalg.det(hess) > 0
