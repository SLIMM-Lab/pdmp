import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize

from datetime import datetime
from timeit import timeit
from typing import Union, Any, cast, override

from pdmp import logger
from pdmp.forward_model import Model
from pdmp.project_field import get_gaussian_random_field_projection_from_dict

small = 1e-12
large = 1e20


class Distribution:
    """Base class for probability distributions."""

    def __init__(self, rng: np.random.Generator = None, seed: int = None):
        if rng is None and seed is None:
            self.rng = np.random.default_rng(0)
        elif rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    @property
    def dim(self) -> int:
        """Get the dimension of the distribution.

        Returns:
            int: The dimension of the distribution.
        """
        raise NotImplementedError

    @property
    def mean(self) -> np.ndarray:
        """Get the mean of the distribution.

        Returns:
            np.ndarray: The mean of the distribution.
        """
        raise NotImplementedError

    @property
    def cov(self) -> np.ndarray:
        """Get the covariance of the distribution.

        Returns:
            np.ndarray: The covariance of the distribution.
        """
        raise NotImplementedError

    def get_sample(self, n: int = 1) -> np.ndarray:
        """Get a sample from the distribution.

        Args:
            n: The number of samples to draw. Default is 1.

        Returns:
            np.ndarray: A sample from the distribution.
        """
        raise NotImplementedError(
            f"Cannot sample directly from {self.__class__.__name__}. Use MCMC instead"
        )

    def log_density(self, x: np.ndarray) -> np.ndarray:
        """Get the log density of the distribution at a point.

        Args:
            x: The point at which to evaluate the log density.

        Returns:
            np.ndarray: The log density of the distribution at the point.
        """
        raise NotImplementedError

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Get the gradient of the log density of the distribution at a point.

        Args:
            x: The point at which to evaluate the gradient of the log density.

        Returns:
            np.ndarray: The gradient of the log density of the distribution at the point.
        """
        raise NotImplementedError

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        """Get the Hessian of the log density of the distribution at a point.

        Args:
            x: The point at which to evaluate the Hessian of the log density.

        Returns:
            np.ndarray: The Hessian of the log density of the distribution at the point.
        """
        raise NotImplementedError

    def get_n_obs(self) -> int:
        """Get the number of observations.

        Returns:
            int: The number of observations.
        """
        return 0


class MultivariateNormal(Distribution):
    """Multivariate normal distribution."""

    cov_L: np.ndarray
    """The Cholesky decomposition of the covariance matrix."""

    inv_C: np.ndarray
    """The inverse of the covariance matrix."""

    log_det: float
    """The log determinant of the covariance matrix."""

    constant: float
    """The constant term in the log density."""

    def __init__(self,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self._dim = mean.shape[0]
        self._mean = mean
        self._cov = cov
        self.cov_L = np.linalg.cholesky(cov)
        self.inv_C = sp.linalg.cho_solve((self.cov_L, True), np.eye(self._dim))
        self.log_det = np.log(self.cov_L.diagonal()).sum()
        self.constant = -0.5 * np.log(2.0 * np.pi) * self._dim

    @classmethod
    def from_dict(cls,
                  params: dict[str, np.ndarray],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")
        return cls(mean=params['mean'], cov=params['cov'], rng=rng, seed=seed)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self):
        return self._cov

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        if n == 1:
            z = self.rng.standard_normal(size=self._dim)
            return self.cov_L @ z + self._mean
        else:
            z = self.rng.standard_normal(size=(n, self._dim))
            return z @ self.cov_L + self._mean

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self._mean
        if diff.ndim == 1:
            if self._dim == 1:
                return self.constant - self.log_det - 0.5 * np.abs(
                    diff / self.cov_L[0, 0])**2
            else:
                return self.constant - self.log_det - 0.5 * np.linalg.norm(
                    np.linalg.solve(self.cov_L, diff))**2
        else:
            return (self.constant - self.log_det - 0.5 * np.linalg.norm(
                np.linalg.solve(self.cov_L, diff.T), axis=0) ** 2).T

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self._mean
        return -sp.linalg.solve_triangular(
            self.cov_L.transpose(),
            sp.linalg.solve_triangular(
                self.cov_L, diff, lower=True, check_finite=False),
            lower=False,
            check_finite=False)

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        return -self.inv_C


class GaussianMixture(Distribution):
    """Gaussian mixture distribution."""

    means: np.ndarray
    """The means of the components."""

    covs: np.ndarray
    """The covariances of the components."""

    weights: np.ndarray
    """The weights of the components."""

    n_components: int
    """The number of components."""

    dists: list[MultivariateNormal]
    """The component distributions."""

    def __init__(self,
                 means: np.ndarray,
                 covs: np.ndarray,
                 weights: np.ndarray,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.n_components = means.shape[0]
        self.dim_ = means.shape[1]
        self.means = means
        self.covs = covs
        self.weights = weights / np.sum(weights)
        assert len(means) == len(covs) == len(weights)
        self.dists = []
        for i in range(self.n_components):
            self.dists.append(MultivariateNormal(means[i], covs[i], rng=rng))
        # self.constant = - 0.5 * np.log(2.0 * np.pi) * self.dim

    @classmethod
    def from_dict(cls,
                  params: dict[str, np.ndarray],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'means' not in params or 'covs' not in params or 'weights' not in params:
            raise ValueError(
                "Parameters must include 'means', 'covs', and 'weights'.")

        return cls(means=params['means'],
                   covs=params['covs'],
                   weights=params['weights'],
                   rng=rng,
                   seed=seed)

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        x = np.zeros((n, self.dim_))
        for i in range(n):
            idx = self.rng.choice(self.n_components, p=self.weights)
            x[i] = self.dists[idx].get_sample()
        return x

    @property
    def mean(self) -> np.ndarray:
        mean = np.zeros(self.dim_)
        for i in range(self.n_components):
            mean += self.weights[i] * self.dists[i].mean
        return mean

    @property
    def cov(self) -> np.ndarray:
        cov = np.zeros((self.dim_, self.dim_))
        mean = self.mean
        for i in range(self.n_components):
            diff = self.dists[i].mean - mean
            cov += self.weights[i] * (self.dists[i].cov +
                                      np.outer(diff, diff))
        return cov

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        log_p = 0.
        for i in range(self.n_components):
            log_p += self.weights[i] * np.exp(self.dists[i].log_density(x))
        return np.log(log_p)

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.dim_)
        for i in range(self.n_components):
            gamma = self.weights[i] * np.exp(self.dists[i].log_density(x) -
                                             self.log_density(x))
            grad += gamma * self.dists[i].grad_log_density(x)
        return grad

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        hess = np.zeros((self.dim_, self.dim_))
        grad = self.grad_log_density(x)
        for i in range(self.n_components):
            gamma = self.weights[i] * np.exp(self.dists[i].log_density(x) -
                                             self.log_density(x))
            diff_grad = self.dists[i].grad_log_density(x) - grad
            grad = self.grad_log_density(x)
            hess += gamma * self.dists[i].hessian_log_density(x)
            hess += gamma * np.outer(diff_grad, diff_grad)
        return hess


class BananaDistribution(Distribution):
    """Banana distribution."""

    def __init__(self,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 a: float = 2.0,
                 b: float = 0.2,
                 rng: np.random.Generator = None,
                 seed: int = None):
        # TODO: docstring
        super().__init__(rng=rng, seed=seed)
        self._a = a
        self._b = b
        self._gaussian = MultivariateNormal(mean, cov, rng=rng, seed=seed)

    @classmethod
    def from_dict(cls,
                  params: dict[str, Union[np.ndarray, float]],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")

        return cls(mean=params['mean'],
                   cov=params['cov'],
                   a=params.get('a', 2.0),
                   b=params.get('b', 0.2),
                   rng=rng,
                   seed=seed)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the Gaussian to the banana distribution.

        Args:
            x: The point to transform.

        Returns:
            np.ndarray: The transformed point.
        """
        return np.array([
            x[0] / self._a,
            x[1] * self._a + self._a * self._b * (x[0]**2 + self._a**2)
        ])

    @property
    def dim(self) -> int:
        return self._gaussian.dim

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self._gaussian.log_density(self.transform(x))

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        nGrad = -self._gaussian.grad_log_density(self.transform(x))
        return -np.array([
            nGrad[0] / self._a + nGrad[1] * self._a * self._b * 2 * x[0],
            nGrad[1] * self._a
        ])


class MultivariateLogNormal(Distribution):
    """Multivariate log-normal distribution."""

    mean_normal_: np.ndarray
    """The mean of the underlying normal distribution."""

    cov_normal_: np.ndarray
    """The covariance of the underlying normal distribution."""

    covL_: np.ndarray
    """The Cholesky decomposition of the covariance matrix."""

    logDet_: float
    """The log determinant of the covariance matrix."""

    mean: np.ndarray
    """The mean of the log-normal distribution."""

    def __init__(self,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.mean_normal_ = mean
        self.cov_normal_ = cov
        self.covL_ = np.linalg.cholesky(self.cov_normal_)
        self.logDet_ = np.log(self.covL_.diagonal()).sum()

        self.mean = np.exp(mean + np.diagonal(cov) / 2)
        mu_i = np.repeat(mean[:, None], 2, axis=1)
        sig_ii = np.repeat(np.diag(cov)[:, None], 2, axis=1)
        # self.cov = np.exp(mu_i + mu_i.transpose() + 0.5 * (sig_ii + sig_ii.transpose())) @ (np.exp(cov) - 1)
        # TODO: finish

        self._dim = mean.shape[0]
        self.constant_ = -0.5 * np.log(2.0 * np.pi) * self._dim

    @classmethod
    def from_dict(cls,
                  params: dict[str, np.ndarray],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")
        return cls(mean=params['mean'], cov=params['cov'], rng=rng, seed=seed)

    @property
    def dim(self) -> int:
        return self._dim

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        return self.constant_ - self.logDet_ - np.sum(np.log(
            x)) - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff))**2

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        grad = -np.diag(1 / x)
        return -(1 + sp.linalg.solve_triangular(
            self.covL_.transpose(),
            sp.linalg.solve_triangular(self.covL_, diff, lower=True))) / x


class CubicDistribution(Distribution):
    """Distribution with a cubic term in the log-density."""

    _normal: MultivariateNormal
    """The underlying normal distribution."""

    _a: float
    """The shape parameter a."""

    cubic_diag: np.ndarray
    """The diagonal of the cubic matrix."""

    def __init__(self,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 a: float,
                 *,
                 cubic_diag: np.ndarray = None,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self._dim = mean.shape[0]
        self._normal = MultivariateNormal(mean, cov, rng=rng, seed=seed)
        self._a = a
        if cubic_diag is not None:
            self.cubic_diag = cubic_diag
        else:
            self.cubic_diag = np.ones(self._dim)

    @classmethod
    def from_dict(cls,
                  params: dict[str, Union[np.ndarray, float]],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'mean' not in params or 'cov' not in params or 'a' not in params:
            raise ValueError("Parameters must include 'mean', 'cov', and 'a'.")

        return cls(mean=np.array(params['mean']),
                   cov=np.array(params['cov']),
                   a=params['a'],
                   cubic_diag=params.get('cubic_diag', None),
                   rng=rng,
                   seed=seed)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        return self._normal.mean

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self._normal.mean
        return (np.sum(
            (1 - 2 * (d > 0)) * self._a / 3 * self.cubic_diag * d ** 3) +
                self._normal.log_density(x))

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self._normal.mean
        return ((1 - 2 * (d > 0)) * (self._a * self.cubic_diag * d ** 2) +
                self._normal.grad_log_density(x))

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self._normal.mean
        return ((1 - 2 *
                 (d > 0)) * (2 * self._a * np.diag(self.cubic_diag * d)) +
                self._normal.hessian_log_density(x))


class Likelihood(Distribution):
    """Base class for likelihoods."""

    n_obs: int
    """The number of observations."""

    def __init__(self, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        """Get the log density of the likelihood of the observation given by idx.

        Args:
            params: The parameters to evaluate the log density.
            idx: The index of the observation. Default is None.

        Returns:
            np.ndarray: The log density of the likelihood of the observation given by idx.
        """
        raise NotImplementedError

    @override
    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        """Get the gradient of the log density of the likelihood of the observation given by idx.

        Args:
            params: The parameters to evaluate the gradient of the log density.
            idx: The index of the observation. Default is None.

        Returns:
            np.ndarray: The gradient of the log density of the likelihood of the observation given by idx.
        """
        raise NotImplementedError

    @override
    def hessian_log_density(self,
                            params: np.ndarray,
                            idx: int = None) -> np.ndarray:
        """Get the Hessian of the log density of the likelihood of the observation given by idx.

        Args:
            params: The parameters to evaluate the Hessian of the log density.
            idx: The index of the observation. Default is None.

        Returns:
            np.ndarray: The Hessian of the log density of the likelihood of the observation given by idx.
        """
        raise NotImplementedError


class TemperedLikelihood(Likelihood):
    """Base class for tempered likelihoods."""

    def __init__(self,
                 likelihood: Likelihood,
                 *,
                 beta: float = 1.0,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self._likelihood = likelihood
        self._beta = beta

    @property
    def n_obs(self) -> int:
        return self._likelihood.n_obs

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self._beta * self._likelihood.log_density(params, idx=idx)

    @override
    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        return self._beta * self._likelihood.grad_log_density(params, idx=idx)

    @override
    def hessian_log_density(self,
                            params: np.ndarray,
                            idx: int = None) -> np.ndarray:
        return self._beta * self._likelihood.hessian_log_density(params,
                                                                 idx=idx)


class GaussianLikelihood(Likelihood):
    """Base class for Gaussian likelihoods."""

    def __init__(self,
                 model: Model,
                 u_obs: np.ndarray,
                 sigma: float,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self._model = model
        self._n_params_ = model.get_dim_in()
        self._u_obs = u_obs
        self.n_obs = self._u_obs.shape[0]
        self._dim = self._u_obs.shape[1]
        self._sigma = sigma
        self._dists = []
        for i in range(self.n_obs):
            self._dists.append(
                MultivariateNormal(self._u_obs[i],
                                   sigma ** 2 * np.eye(self._dim)))

    @property
    def dim(self) -> int:
        return self._dim

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            log_p = 0.
            for i in range(self.n_obs):
                log_p += self._dists[i].log_density(
                    self._model.eval(params, idx=i))
            return log_p
        else:
            return self._dists[idx].log_density(
                self._model.eval(params, idx=idx))

    @override
    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        if idx is None:
            grad = np.zeros(self._n_params_)
            for i in range(self.n_obs):
                m = self._model.eval(params, idx=i)
                grad_m = self._model.eval_grad(params, idx=i)
                grad += self._dists[i].grad_log_density(m) @ grad_m
            return grad
        else:
            m = self._model.eval(params, idx=idx)
            grad_m = self._model.eval_grad(params, idx=idx)
            return self._dists[idx].grad_log_density(m) @ grad_m

    @override
    def hessian_log_density(self,
                            params: np.ndarray,
                            idx: int = None) -> np.ndarray:

        def hess_comp(hess: np.ndarray, i: int):
            m = self._model.eval(params, idx=i)
            grad_m = self._model.eval_grad(params, idx=i)
            hess_m = self._model.eval_hessian(params, idx=i)
            hess += np.einsum('ij,jk,il', self._dists[i].hessian_log_density(m),
                              grad_m, grad_m)
            hess += np.einsum('i,ijk->jk', self._dists[i].grad_log_density(m),
                              hess_m)

        hess = np.zeros((self._n_params_, self._n_params_))

        if idx is None:
            for i in range(self.n_obs):
                hess_comp(hess, i)
        else:
            hess_comp(hess, idx)

        return hess


class FlatLikelihood(Likelihood):
    """Flat likelihood that assigns equal likelihood to all observations."""

    def __init__(self,
                 dim: int,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self._dim = dim
        self.n_obs = 0

    @property
    def dim(self) -> int:
        return self._dim

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return np.array([0.0])

    @override
    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        if idx is None:
            return np.zeros(self._dim)
        else:
            return np.zeros(1)

    @override
    def hessian_log_density(self,
                            params: np.ndarray,
                            idx: int = None) -> np.ndarray:
        return np.zeros((self._dim, self._dim))


def get_prior(
    config: dict[str, Union[str, np.ndarray, float]],
    rng: np.random.Generator = None,
) -> Distribution:
    """Get a prior distribution from a dictionary.

    Args:
        config: The configuration of the prior.
        rng: The random number generator. Default is None.

    Returns:
        Distribution: The prior distribution.

    Raises:
        ValueError: If the prior name is not recognized or the configuration is invalid.
    """

    if 'name' not in config:
        raise ValueError("Prior config must include 'name'.")
    if config['name'] == 'MultivariateNormal':
        return MultivariateNormal.from_dict(config, rng=rng)
    elif config['name'] == 'GaussianMixture':
        return GaussianMixture.from_dict(config, rng=rng)
    elif config['name'] == 'Banana':
        return BananaDistribution.from_dict(config, rng=rng)
    elif config['name'] == 'Cubic':
        return CubicDistribution.from_dict(config, rng=rng)
    elif config['name'] == 'MultivariateLogNormal':
        return MultivariateLogNormal.from_dict(config, rng=rng)
    elif config['name'] == 'GaussianRandomField':
        mean, cov = get_gaussian_random_field_projection_from_dict(config)
        return MultivariateNormal(mean, cov, rng=rng)
    else:
        raise ValueError(f"Prior {config['name']} not recognized.")


class Posterior(Distribution):
    """Base class for posterior distributions."""

    prior: Distribution
    """The prior distribution."""

    def __init__(self,
                 prior: Distribution,
                 likelihood: Likelihood,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.prior = prior
        self._likelihood = likelihood

    @property
    def dim(self):
        return self.prior.dim

    @property
    def n_obs(self) -> int:
        return self._likelihood.n_obs

    @override
    def log_density(self, params: np.ndarray) -> np.ndarray:
        return self._likelihood.log_density(params) + self.prior.log_density(
            params)

    @override
    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None,
                         sub_sampling: bool = False) -> np.ndarray:
        if sub_sampling:
            approx_llh = self._likelihood.get_n_obs(
            ) * self._likelihood.grad_log_density(params)
            return approx_llh + self.prior.grad_log_density(params)
        else:
            return self._likelihood.grad_log_density(
                params) + self.prior.grad_log_density(params)

    @override
    def hessian_log_density(self, params: np.ndarray) -> np.ndarray:
        return self._likelihood.hessian_log_density(
            params) + self.prior.hessian_log_density(params)

    def get_prior_sample(self) -> np.ndarray:
        return self.prior.get_sample()


class Transformation:
    """Base class for transformations."""

    def transform(self, xi: np.ndarray) -> np.ndarray:
        """Forward transformation from xi to x.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The transformed input.
        """
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Inverse transformation from x to xi.

        Args:
            x: The input to the transformation.

        Returns:
            np.ndarray: The inverse transformed input xi.
        """
        raise NotImplementedError

    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        """Return the Jacobian of the transformation.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The Jacobian of the transformation.
        """
        raise NotImplementedError

    def inv_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """Return the inverse Jacobian of the transformation.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The Jacobian of the inverse transformation.
        """
        raise NotImplementedError

    def log_det_jacobian(self, xi: np.ndarray) -> float:
        """Return the log determinant of the Jacobian.

        Args:
            xi: The input to the transformation.

        Returns:
            float: The log determinant of the Jacobian.
        """
        raise NotImplementedError

    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """Return the gradient of the log determinant of the Jacobian.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The gradient of the log determinant of the Jacobian.
        """
        raise NotImplementedError

    def hessian(self, xi: np.ndarray) -> np.ndarray:
        """Return the Hessian of the transformation.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The Hessian of the transformation.
        """
        raise NotImplementedError

    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """Return the Hessian of the log determinant of the Jacobian.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The Hessian of the log determinant of the Jacobian.
        """
        raise NotImplementedError


class ExponentialTransformation(Transformation):
    """Implements exponential transformation x = exp(ξ)."""

    @override
    def transform(self, xi: np.ndarray) -> np.ndarray:
        return np.exp(xi)

    @override
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    @override
    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.diag(np.exp(xi))

    @override
    def inv_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.diag(np.exp(-xi))  # Directly return inverse

    @override
    def log_det_jacobian(self, xi: np.ndarray) -> float:
        """log |det(J)| = sum(xi) since J is diagonal."""
        return np.sum(xi, axis=-1)  # log |det(J)| = sum(xi) for diagonal matrix

    @override
    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """Gradient of log determinant is always 1 (since log(exp(x)) = x and J is diagonal)."""
        return np.ones_like(xi)

    @override
    def hessian(self, xi: np.ndarray) -> np.ndarray:
        """Hessian is a 3-tensor with diagonal elements being exp(ξ)."""
        d = xi.shape[0]
        H = np.zeros((d, d, d))
        idx = np.arange(d)
        H[idx, idx, idx] = np.exp(xi)
        return H

    @override
    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros((xi.shape[0], xi.shape[0]))


class AffineTransformtion(Transformation):
    """Implements affine transformation x = A @ xi + b."""

    def __init__(self, M: np.ndarray, b: np.ndarray):
        self._M = M
        self._b = b
        self._M_inv = np.linalg.inv(M)
        self._log_abs_det_M = np.log(np.abs(np.linalg.det(M)))

    @override
    def transform(self, xi: np.ndarray) -> np.ndarray:
        return xi @ self._M.T + self._b

    @override
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self._M_inv @ (x - self._b)

    @override
    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        return self._M  # Jacobian is constant

    @override
    def inv_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return self._M_inv  # Directly return precomputed inverse

    @override
    def log_det_jacobian(self, xi: np.ndarray) -> float:
        return self._log_abs_det_M  # log |det(M)|

    @override
    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros_like(xi)  # Zero since M is constant

    @override
    def hessian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros(
            (xi.shape[0], xi.shape[0], xi.shape[0]))  # Hessian is zero

    @override
    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros((xi.shape[0], xi.shape[0]))


EXPONENTIAL = 'Exponential'
AFFINE = 'Affine'
TRANSFORMATIONS = [EXPONENTIAL, AFFINE]


class TransformedDistribution(Distribution):
    """Base class for transformed distributions."""

    def __init__(self, base_distribution: Distribution, params: dict[str, Any]):
        """Initialize a transformed distribution.

        Args:
            base_distribution: The base distribution to transform.
            params: The parameters of the transformation.

        Raises:
            NotImplementedError: If the transformation is not recognized.
        """

        super().__init__(rng=base_distribution.rng)
        self._base_distribution = base_distribution

        if params['transformation'] == EXPONENTIAL:
            self._transformation = ExponentialTransformation()
        elif params['transformation'] == AFFINE:
            M = params.get('M', None)
            b = params.get('b', None)
            x_0 = params.get('x_0', None)

            if b is None:
                b = find_mean(base_distribution, x_0=x_0)
                params['b'] = b

            if M is None:
                M = find_curvature(base_distribution, mean=b)
                params['M'] = M

            C = np.linalg.cholesky(M)

            self._transformation = AffineTransformtion(C, b)
        else:
            raise NotImplementedError(
                f"Transformation {params['transformation']} not recognized.\n"
                f"pick any of {TRANSFORMATIONS}")

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        """Generates a sample from the underlying distribution and transforms it.

        Returns:
            np.ndarray: A sample from the transformed distribution.
        """
        x_sample = self._base_distribution.get_sample(n=n)
        return self._transformation.inverse_transform(x_sample)

    @property
    def dim(self) -> int:
        """Get dimension of the transformed distribution.
        Returns:
            int: The dimension of the transformed distribution.
        """
        return self._base_distribution.dim

    @override
    def log_density(self, xi: np.ndarray) -> np.ndarray:
        """Computes log p(xi).

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The log density of the transformed distribution.
        """
        x = self._transformation.transform(xi)
        return self._base_distribution.log_density(
            x) + self._transformation.log_det_jacobian(xi)

    @override
    def grad_log_density(self, xi: np.ndarray) -> np.ndarray:
        """Computes the gradient of the log density with respect to xi.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The gradient of the log density with respect to xi.
        """
        x = self._transformation.transform(xi)

        # Precompute gradient of log density of base distribution
        grad_log_p_xi = self._transformation.jacobian(
            xi).T @ self._base_distribution.grad_log_density(x)

        return grad_log_p_xi + self._transformation.grad_log_det_jacobian(xi)

    @override
    def hessian_log_density(self, xi: np.ndarray) -> np.ndarray:
        """Computes the Hessian of the log density with respect to xi.

        Args:
            xi: The input to the transformation.

        Returns:
            np.ndarray: The Hessian of the log density with respect to xi.
        """

        x = self._transformation.transform(xi)
        H_x = self._base_distribution.hessian_log_density(x)
        J = self._transformation.jacobian(xi)

        # Compute Hessian of transformation
        grad_x = self._base_distribution.grad_log_density(x)
        H_f = self._transformation.hessian(xi)
        H_f_grad = np.einsum('ijk,j->ik', H_f, grad_x)

        # Compute Hessian of log density of base distribution
        jac_correction = self._transformation.hessian_log_det_jacobian(xi)

        return J.T @ H_x @ J + H_f_grad + jac_correction

    def get_prior_sample(self) -> np.ndarray:
        """Get a prior sample from the transformed distribution.

        Returns:
            np.ndarray: A prior sample from the transformed distribution.
        """
        assert hasattr(self._base_distribution, 'get_prior_sample')
        child = cast(Posterior, self._base_distribution)
        return self._transformation.inverse_transform(child.get_prior_sample())


class TransformedLikelihood(Likelihood):
    """Applies a transformation to a base likelihood function."""

    def __init__(self, likelihood: Likelihood, params: dict[str, Any]):
        """Initialize a transformed likelihood.

        Args:
            likelihood: The base likelihood to transform.
            params: The parameters of the transformation.

        Raises:
            NotImplementedError: If the transformation is not recognized.
        """
        super().__init__(rng=likelihood.rng)
        self._likelihood = likelihood

        if params['transformation'] == EXPONENTIAL:
            self._transformation = ExponentialTransformation()
        elif params['transformation'] == AFFINE:
            M = params.get('M', None)
            b = params.get('b', None)
            x_0 = params.get('x_0', None)

            if b is None:
                b = find_mean(likelihood, x_0=x_0)
                params['b'] = b

            if M is None:
                M = find_curvature(likelihood, mean=b)
                params['M'] = M

            C = np.linalg.cholesky(M)

            self._transformation = AffineTransformtion(C, b)
        else:
            raise NotImplementedError(
                f"Transformation {params['transformation']} not recognized.\n"
                f"pick any of {TRANSFORMATIONS}")

    @property
    def dim(self) -> int:
        """Get dimension of the transformed distribution.

        Returns:
            int: The dimension of the transformed distribution.
        """
        return self._likelihood.dim

    @override
    def log_density(self, xi: np.ndarray, idx: int = None) -> np.ndarray:
        """Computes log p(xi).

        Args:
            xi: The input to the transformation.
            idx: The index of the observation to evaluate.

        Returns:
            np.ndarray: The log density of the transformed distribution.
        """
        x = self._transformation.transform(xi)
        return self._likelihood.log_density(x, idx=idx)

    @override
    def grad_log_density(self, xi: np.ndarray, idx: int = None) -> np.ndarray:
        """Computes the gradient of the log density with respect to xi.

        Args:
            xi: The input to the transformation.
            idx: The index of the observation to evaluate.

        Returns:
            np.ndarray: The gradient of the log density with respect to xi.
        """
        x = self._transformation.transform(xi)

        # Precompute gradient of log density of base distribution
        grad_log_p_xi = self._transformation.jacobian(
            xi).T @ self._likelihood.grad_log_density(x, idx=idx)

        # return grad_log_p_xi + self.transformation.grad_log_det_jacobian(xi)
        return grad_log_p_xi

    @override
    def hessian_log_density(self,
                            xi: np.ndarray,
                            idx: int = None) -> np.ndarray:
        """Computes the Hessian of the log density with respect to xi.

        Args:
            xi: The input to the transformation.
            idx: The index of the observation to evaluate.

        Returns:
            np.ndarray: The Hessian of the log density with respect to xi.
        """
        x = self._transformation.transform(xi)
        H_x = self._likelihood.hessian_log_density(x, idx=idx)
        J = self._transformation.jacobian(xi)

        # Compute Hessian of transformation
        grad_x = self._likelihood.grad_log_density(x, idx=idx)
        H_f = self._transformation.hessian(xi)
        H_f_grad = np.einsum('ijk,j->ik', H_f, grad_x)

        return J.T @ H_x @ J + H_f_grad


def get_likelihood(
    config: dict[str, Any],
    model: Model,
    rng: np.random.Generator = None,
) -> Likelihood:
    """Get a likelihood distribution from a configuration.

    Args:
        config: The configuration of the likelihood.
        model: The model to evaluate the likelihood.
        rng: The random number generator.

    Returns:
        Likelihood: The likelihood distribution.
    """

    obs = None

    if 'name' not in config:
        raise ValueError("Likelihood config must include 'name'.")
    if ((config['name'] != 'FlatLikelihood') and
        (config['name'] != 'TransformedLikelihood') and
        ('observation_file' not in config)):
        raise ValueError("Likelihood config must include 'observation_file'.")

    if (config['name'] != 'FlatLikelihood') and (config['name']
                                                 != 'TransformedLikelihood'):
        obs = np.genfromtxt(config['observation_file'])
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

    # TODO: go from elif to if only
    if config['name'] == 'GaussianLikelihood':
        sigma = config['sigma']
        return GaussianLikelihood(model=model, u_obs=obs, sigma=sigma, rng=rng)
    elif config['name'] == 'TemperedLikelihood':
        likelihood = get_likelihood(config['likelihood'], model=model, rng=rng)
        beta = config['beta']
        return TemperedLikelihood(likelihood, beta=beta, rng=rng)
    elif config['name'] == 'TransformedLikelihood':
        likelihood = get_likelihood(config['likelihood'], model=model, rng=rng)
        return TransformedLikelihood(likelihood, params=config)
    elif config['name'] == 'FlatLikelihood':
        return FlatLikelihood(dim=model.get_dim_in(), rng=rng)
    else:
        raise ValueError(f"Likelihood {config['name']} not recognized.")


def find_mean(target: Distribution, x_0: np.ndarray = None) -> np.ndarray:
    """Find the mean of a distribution using the BFGS method

    Args:
        target: The target distribution
        x_0: The initial guess for the mean (optional). Default is None.

    Returns:
        np.ndarray: The mean of the distribution
    """

    n_log_post = lambda x: -target.log_density(x)
    n_grad_log_post = lambda x: -target.grad_log_density(x)

    if x_0 is None:
        logger.warning(
            "No initial point provided ... attempting to get sample from target."
        )
        success = False

        if hasattr(target, 'get_sample'):
            try:
                x_0 = target.get_sample()
                success = True
            except NotImplementedError as e:
                logger.warning(
                    "  Method get_sample not implemented for target.")

        if hasattr(target, 'mean'):
            try:
                x_0 = target.mean
                success = True
            except NotImplementedError as e:
                logger.warning("  Method get_mean not implemented for target.")

        if not success and hasattr(target, 'prior') and hasattr(
                target.prior, 'get_sample'):
            try:
                x_0 = target.prior.get_sample()
                success = True
            except NotImplementedError as e:
                logger.warning("  Method get_sample not implemented for prior.")

        if not success and hasattr(target, 'prior') and hasattr(
                target.prior, 'mean'):
            try:
                x_0 = target.prior.mean
            except NotImplementedError as e:
                logger.warning("  Method get_mean not implemented for prior.")

        if not success:
            x_0 = np.zeros(target.dim)

    return minimize(n_log_post, x_0, jac=n_grad_log_post, method='BFGS').x


def find_curvature(
    target: Distribution,
    mean: np.ndarray = None,
) -> np.ndarray:
    """Find the curvature of a distribution at the MAP point

    Args:
        target: The target distribution
        mean: The mean of the distribution (optional)

    Returns:
        np.ndarray: The covariance of the distribution
    """

    if mean is None:
        mean = find_mean(target)

    return -np.linalg.inv(target.hessian_log_density(mean))


# if __name__ == '__main__':
#
#     rng = np.random.default_rng(0)
#
#     x = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
#     x = np.stack((x[0].flatten(), x[1].flatten()), axis=1)
#
#     # Define distributions
#     mean = np.array([0., 0.])
#     cov = np.array([[1., 0.5], [0.5, 1.]])
#
#
#     old = MultivariateNormal(mean, cov, rng=rng)
#
#     y_old = old.log_density(x).reshape(100, 100)
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].contourf(x[:,0].reshape(100, 100), x[:,1].reshape(100, 100), y_old)
#     ax[0].set_title('Old')
#     plt.show()
#
#     # get a random mean vector and covariance matrix with d dimensions
#     d = 100
#     mean = rng.uniform(-5, 5, d)
#     cov = rng.uniform(-5, 5, (d, d))
#     cov = cov @ cov.T
#
#     # get n test inputs
#     n = 1000
#     x = rng.uniform(-5, 5, (n, d))
#
#     old = MultivariateNormal(mean, cov, rng=rng)
#
#     exec_old = timeit.timeit('old.log_density(x)', globals=globals(), number=1000)
#     print(f"Old: {exec_old}")
#
#     # get a random mean vector and covariance matrix with d dimensions
#     from scipy.linalg import cho_factor, cho_solve
#     import timeit
#
#     # Generate a random positive-definite matrix A
#     np.random.seed(42)
#     n = 1000  # Size of the matrix
#     A = np.random.randn(n, n)
#     A = np.dot(A, A.T)  # Make A symmetric positive definite
#
#     # Generate a random vector x
#     x = np.random.randn(n)
#
#     # Precompute the inverse of A
#     A_inv = np.linalg.inv(A)
#
#     # Precompute the Cholesky decomposition of A
#     L = np.linalg.cholesky(A)
#
#     # Method 1: Using np.matmul and A_inv
#     def matmul_method():
#         return np.matmul(x.T, np.matmul(A_inv, x))
#
#     # Method 2: Solving L y = x and computing the square norm
#     def cholesky_method():
#         y = np.linalg.solve(L, x)
#         return np.dot(y, y)
#
#     # Method 2 (alternative): Using scipy's cho_solve
#     def cho_solve_method():
#         y = cho_solve((L, True), x)
#         return np.dot(y, y)
#
#     # Timing the two methods
#     matmul_time = timeit.timeit(matmul_method, number=100)
#     cholesky_time = timeit.timeit(cholesky_method, number=100)
#     cho_solve_time = timeit.timeit(cho_solve_method, number=100)
#
#     # Results
#     print(f"Method 1 (Matmul with A_inv): {matmul_time:.6f} seconds")
#     print(f"Method 2 (Solving L y = x): {cholesky_time:.6f} seconds")
#     print(f"Method 2 (cho_solve from scipy): {cho_solve_time:.6f} seconds")
