import numpy as np
from scipy.stats import multivariate_normal, wishart

from pdmp.distributions import MultivariateNormal
from pdmp.utils import grad_fd, hessian_fd



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

if __name__ == '__main__':
    test_multivariate_normal()