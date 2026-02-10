import numpy as np
from scipy.stats import multivariate_normal, wishart, lognorm
from scipy.optimize import minimize

from pdmp.distributions import (MultivariateNormal, CubicDistribution,
                                GaussianLikelihood, TemperedLikelihood,
                                FlatLikelihood, Posterior, ExponentialTransformation,
                                TransformedDistribution)
from pdmp.utils import grad_fd, hessian_fd
from pdmp.forward_model import LinearModel

SMALL = 1e-6
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
    assert np.isclose(mvn.log_density(sample),
                      multivariate_normal.logpdf(sample, mean, cov))

    # test gradient
    assert np.allclose(mvn.grad_log_density(sample),
                       grad_fd(mvn.log_density, sample),
                       atol=1e-5)

    # test hessian
    assert np.allclose(mvn.hessian_log_density(sample),
                       hessian_fd(mvn.log_density, sample),
                       atol=1e-5)

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
    assert np.allclose(cubic.grad_log_density(sample),
                       grad_fd(cubic.log_density, sample),
                       atol=1e-5)

    # test hessian
    assert np.allclose(cubic.hessian_log_density(sample),
                       hessian_fd(cubic.log_density, sample),
                       atol=1e-5)

    # test limits
    assert np.isclose(np.exp(cubic.log_density(np.ones(d) * (-LARGE))), 0.0)
    assert np.isclose(np.exp(cubic.log_density(np.ones(d) * LARGE)), 0.0)

    # test symmetry around the mean
    delta = rng.random(d)
    assert np.isclose(cubic.log_density(mean + delta),
                      cubic.log_density(mean - delta))
    assert np.allclose(cubic.grad_log_density(mean), np.zeros(d))
    hess = cubic.hessian_log_density(mean)
    assert np.allclose(hess, hess.T)
    assert np.linalg.det(hess) > 0


def test_likelihood():

    rng = np.random.default_rng(0)

    n, m, n_obs = rng.integers(2, 10, size=3)
    A = np.array(rng.integers(2, 5, size=(n, m)), dtype=float)
    b = np.array(rng.integers(0, 5, size=n), dtype=float)
    model = LinearModel(A, b)
    x_true = 2 * np.array(rng.random(size=m), dtype=float) + 1
    y_true = model.eval(x_true)
    sig_obs = 0.5
    noise = rng.normal(0, sig_obs, size=(n_obs, n))
    y_obs = np.vstack(n_obs * (y_true,)) + noise

    # prior
    prior_mean = rng.random(m)
    prior_cov = wishart(df=m, scale=np.eye(m), seed=rng).rvs()
    prior_cov = prior_cov / np.linalg.norm(prior_cov)
    prior = MultivariateNormal(prior_mean, prior_cov, rng=rng)

    # --------------------------- Test GaussianLikelihood ---------------------------
    # gaussian likelihood
    gaussian_likelihood = GaussianLikelihood(model, y_obs, sig_obs, rng=rng)
    posterior = Posterior(prior, gaussian_likelihood)

    # get MAP estimate
    n_log_post = lambda x: -posterior.log_density(x)
    grad_n_log_post = lambda x: -posterior.grad_log_density(x)
    post_mean_num = minimize(n_log_post,
                             prior_mean,
                             jac=grad_n_log_post,
                             method='BFGS').x
    cov_num = np.linalg.inv(-posterior.hessian_log_density(post_mean_num))

    # get analytical solution
    S_y = np.eye(n) * sig_obs**2
    S_y_inv = np.linalg.inv(S_y)
    S_x = prior_cov
    S_x_inv = np.linalg.inv(S_x)
    v = np.sum(y_obs - b, axis=0)
    post_cov = np.linalg.inv(n_obs * A.T @ S_y_inv @ A + S_x_inv)
    post_mean = post_cov @ (A.T @ S_y_inv @ v.T + S_x_inv @ prior_mean)

    assert np.allclose(post_mean_num, post_mean, atol=1e-5)
    assert np.allclose(cov_num, post_cov, atol=1e-5)

    # --------------------------- Test TemperedLikelihood ---------------------------
    # test beta = 0
    tempered_likelihood = TemperedLikelihood(likelihood=gaussian_likelihood,
                                             beta=0.0)
    posterior = Posterior(prior, tempered_likelihood)
    sample = prior.get_sample()
    assert np.isclose(posterior.log_density(sample),
                      prior.log_density(sample),
                      atol=1e-5)
    assert np.allclose(posterior.grad_log_density(sample),
                       prior.grad_log_density(sample),
                       atol=1e-5)
    assert np.allclose(posterior.hessian_log_density(sample),
                       prior.hessian_log_density(sample),
                       atol=1e-5)

    # test beta = 1
    tempered_likelihood = TemperedLikelihood(likelihood=gaussian_likelihood,
                                             beta=1.0)
    tempered_posterior = Posterior(prior, tempered_likelihood)
    posterior = Posterior(prior, gaussian_likelihood)
    assert np.isclose(tempered_posterior.log_density(sample),
                      posterior.log_density(sample),
                      atol=1e-5)
    assert np.allclose(tempered_posterior.grad_log_density(sample),
                       posterior.grad_log_density(sample),
                       atol=1e-5)
    assert np.allclose(tempered_posterior.hessian_log_density(sample),
                       posterior.hessian_log_density(sample),
                       atol=1e-5)

    # test beta = random
    beta = 0.1 + 0.8 * rng.random()
    tempered_likelihood = TemperedLikelihood(likelihood=gaussian_likelihood,
                                             beta=beta)
    posterior = Posterior(prior, tempered_likelihood)

    # get MAP estimate
    n_log_post = lambda x: -posterior.log_density(x)
    grad_n_log_post = lambda x: -posterior.grad_log_density(x)
    post_mean_num = minimize(n_log_post,
                             prior_mean,
                             jac=grad_n_log_post,
                             method='BFGS').x
    cov_num = np.linalg.inv(-posterior.hessian_log_density(post_mean_num))

    # get analytical solution
    S_y = np.eye(n) * sig_obs**2
    S_y_inv = np.linalg.inv(S_y)
    S_x = prior_cov
    S_x_inv = np.linalg.inv(S_x)
    v = np.sum(y_obs - b, axis=0)
    post_cov = np.linalg.inv(beta * n_obs * A.T @ S_y_inv @ A + S_x_inv)
    post_mean = post_cov @ (beta * A.T @ S_y_inv @ v.T + S_x_inv @ prior_mean)

    assert np.allclose(post_mean_num, post_mean, atol=1e-5)
    assert np.allclose(cov_num, post_cov, atol=1e-5)

    # --------------------------- Test FlatLikelihood ---------------------------
    flat_likelihood = FlatLikelihood(dim=m)
    posterior = Posterior(prior, flat_likelihood)
    sample = prior.get_sample()
    assert np.isclose(posterior.log_density(sample),
                      prior.log_density(sample),
                      atol=1e-5)
    assert np.allclose(posterior.grad_log_density(sample),
                       prior.grad_log_density(sample),
                       atol=1e-5)
    assert np.allclose(posterior.hessian_log_density(sample),
                       prior.hessian_log_density(sample),
                       atol=1e-5)


def test_transformation():
    # --------------------------- Test AffineTransformation ---------------------------
    rng = np.random.default_rng(0)

    # get random mean and covariance and initialize the multivariate normal distribution
    d = rng.integers(2, 10)
    mean = rng.random(d)
    cov = wishart(df=d, scale=np.eye(d), seed=rng).rvs()
    cov = cov / np.linalg.norm(cov)
    mvn_1 = MultivariateNormal(np.zeros(d), np.eye(d))
    mvn_2 = MultivariateNormal(mean, cov)

    # get transformed distribution
    mvn_tr = TransformedDistribution(base_distribution=mvn_2,
                                     params={
                                         'transformation': 'Affine',
                                         'M': cov,
                                         'b': mean
                                     })

    x = rng.random(d)
    assert np.isclose(mvn_1.log_density(x), mvn_tr.log_density(x), atol=1e-5)
    assert np.allclose(mvn_1.grad_log_density(x),
                       mvn_tr.grad_log_density(x),
                       atol=1e-5)
    assert np.allclose(mvn_1.hessian_log_density(x),
                       mvn_tr.hessian_log_density(x),
                       atol=1e-5)

    # --------------------------- Test ExponentialTransformation ---------------------------
    cov = np.array([[1.]])
    mean = np.array([0.])
    normal = MultivariateNormal(mean, cov)
    t = ExponentialTransformation()

    x = np.linspace(SMALL, 10, 200)
    y = np.zeros_like(x)
    y_sp = np.zeros_like(x)

    for i in range(len(x)):
        xi = t.inverse_transform(x[i])
        y[i] = normal.log_density(xi) - np.log(t.log_det_jacobian(x[i]))
        y_sp[i] = lognorm.logpdf(x[i],
                                 s=np.sqrt(cov[0, 0]),
                                 scale=np.exp(mean[0]))

    assert np.allclose(y, y_sp, atol=1e-5)

    d = rng.integers(2, 10)
    x = rng.random(d)

    jac = t.jacobian(x)
    jac_fd = grad_fd(lambda x: t.transform(x), x)

    assert np.allclose(jac, jac_fd, atol=1e-5)

    hess = t.hessian(x)
    hess_fd = np.zeros((d, d, d))

    for i in range(d):
        hess_fd[:, :, i] = grad_fd(lambda x: t.jacobian(x)[i], x)

    assert np.allclose(hess, hess_fd, atol=1e-5)
