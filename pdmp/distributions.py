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
    """
    Base class for probability distributions.

    ...
    Attributes
    ----------
    rng_ : np.random.Generator
        Random number generator.

    Methods
    -------
    get_mean()
        Get the mean of the distribution.
    get_cov()
        Get the covariance of the distribution.
    get_sample()
        Get a sample from the distribution.
    get_dim()
        Get the dimension of the distribution.
    log_density(x)
        Get the log density of the distribution at a point.
    grad_log_density(x)
        Get the gradient of the log density of the distribution at a point.
    hessian_log_density(x)
        Get the Hessian of the log density of the distribution at a point.
    get_n_obs()
        Get the number of observations.
    """

    def __init__(
            self,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng


    def get_mean(self) -> np.ndarray:
        """
        Get the mean of the distribution.
        Returns:
            np.ndarray: The mean of the distribution.
        """
        raise NotImplementedError

    def get_cov(self) -> np.ndarray:
        """
        Get the covariance of the distribution.
        Returns:
            np.ndarray: The covariance of the distribution.
        """
        raise NotImplementedError

    def get_sample(self, n: int = 1) -> np.ndarray:
        """
        Get a sample from the distribution.

        Parameters:
            n (int): The number of samples to draw. Default is 1.

        Returns:
            np.ndarray: A sample from the distribution.
        """
        raise NotImplementedError(f"Cannot sample directly from {self.__class__.__name__}. Use MCMC instead")

    def get_dim(self) -> int:
        """
        Get the dimension of the distribution.
        Returns:
            int: The dimension of the distribution.
        """
        raise NotImplementedError

    def log_density(self, x: np.ndarray) -> np.ndarray:
        """
        Get the log density of the distribution at a point.

        Parameters:
            x (np.ndarray): The point at which to evaluate the log density.

        Returns:
            np.ndarray: The log density of the distribution at the point.
        """
        raise NotImplementedError

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """
        Get the gradient of the log density of the distribution at a point.

        Parameters:
            x (np.ndarray): The point at which to evaluate the gradient of the log density.

        Returns:
            np.ndarray: The gradient of the log density of the distribution at the point.
        """
        raise NotImplementedError

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        """
        Get the Hessian of the log density of the distribution at a point.

        Parameters:
            x (np.ndarray): The point at which to evaluate the Hessian of the log density.

        Returns:
            np.ndarray: The Hessian of the log density of the distribution at the point.
        """
        raise NotImplementedError

    def get_n_obs(self) -> int:
        """
        Get the number of observations.

        Returns:
            int: The number of observations.
        """
        return 0


class MultivariateNormal(Distribution):
    """
    Multivariate normal distribution.

    ...
    Attributes
    ----------
    mean_ : np.ndarray
        The mean of the distribution.
    dim_ : int
        The dimension of the distribution.
    cov_ : np.ndarray
        The covariance matrix of the distribution.
    covL_ : np.ndarray
        The Cholesky decomposition of the covariance matrix.
    invC_ : np.ndarray
        The inverse of the covariance matrix.
    logDet_ : float
        The log determinant of the covariance matrix.
    constant_ : float
        The constant term in the log density.

    Methods
    -------
    get_inv_cov()
        Get the inverse of the covariance matrix.
    from_dict(params, rng, seed)
        Create a multivariate normal distribution from a dictionary.
    """

    def __init__(
            self,
            mean: np.ndarray,
            cov: np.ndarray,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.mean_ = mean
        self.dim_ = mean.shape[0]
        self.cov_ = cov
        self.covL_ = np.linalg.cholesky(cov)
        self.invC_ = sp.linalg.cho_solve((self.covL_, True), np.eye(self.dim_))
        self.logDet_ = np.log(self.covL_.diagonal()).sum()
        self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

    @classmethod
    def from_dict(
            cls,
            params: dict[str, np.ndarray],
            rng: np.random.Generator = None,
            seed: int = None
    ):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")
        return cls(
            mean=params['mean'],
            cov=params['cov'],
            rng=rng,
            seed=seed
        )

    def get_inv_cov(self) -> np.ndarray:
        """
        Get the inverse of the covariance matrix.
        Returns:
            np.ndarray: The inverse of the covariance matrix
        """
        return self.invC_

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        if n == 1:
            z = self.rng_.standard_normal(size=self.dim_)
            return self.covL_ @ z + self.mean_
        else:
            z = self.rng_.standard_normal(size=(n, self.dim_))
            return z @ self.covL_ + self.mean_

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_mean(self) -> np.ndarray:
        return self.mean_

    @override
    def get_cov(self) -> np.ndarray:
        return self.cov_

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.mean_
        if diff.ndim == 1:
            if self.dim_ == 1:
                return self.constant_ - self.logDet_ - 0.5 * np.abs(diff / self.covL_[0,0]) ** 2
            else:
                return self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff)) ** 2
        else:
            return (
                    self.constant_
                    - self.logDet_
                    - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff.T), axis=0) ** 2
            ).T

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        diff =  x - self.mean_
        return - sp.linalg.solve_triangular(
            self.covL_.transpose(),
            sp.linalg.solve_triangular(self.covL_, diff, lower=True, check_finite=False),
            lower=False,
            check_finite = False
        )

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        return - self.invC_


class GaussianMixture(Distribution):
    """
    Gaussian mixture distribution.

    ...
    Attributes
    ----------
    means_ : np.ndarray
        The means of the components.
    covs_ : np.ndarray
        The covariances of the components.
    weights_ : np.ndarray
        The weights of the components.
    n_components_ : int
        The number of components.
    dim_ : int
        The dimension of the distribution.
    dists_ : list[MultivariateNormal]
        The component distributions.

    Methods
    -------
    from_dict(params, rng, seed)
        Create a Gaussian mixture distribution from a dictionary.
    """

    def __init__(
            self,
            means: np.ndarray,
            covs: np.ndarray,
            weights: np.ndarray,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.n_components_ = means.shape[0]
        self.dim_ = means.shape[1]
        self.means_ = means
        self.covs_ = covs
        self.weights_ = weights / np.sum(weights)
        assert len(means) == len(covs) == len(weights)
        self.dists_ = []
        for i in range(self.n_components_):
            self.dists_.append(MultivariateNormal(means[i], covs[i], rng=rng))
        # self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

    @classmethod
    def from_dict(
            cls,
            params: dict[str, np.ndarray],
            rng: np.random.Generator = None,
            seed: int = None
    ):
        if 'means' not in params or 'covs' not in params or 'weights' not in params:
            raise ValueError("Parameters must include 'means', 'covs', and 'weights'.")

        return cls(
            means=params['means'],
            covs=params['covs'],
            weights=params['weights'],
            rng=rng,
            seed=seed
        )

    @override
    def get_sample(self, n: int=1) -> np.ndarray:
        x = np.zeros((n, self.dim_))
        for i in range(n):
            idx = self.rng_.choice(self.n_components_, p=self.weights_)
            x[i] = self.dists_[idx].get_sample()
        return x

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_mean(self) -> np.ndarray:
        mean = np.zeros(self.dim_)
        for i in range(self.n_components_):
            mean += self.weights_[i] * self.dists_[i].get_mean()
        return mean

    @override
    def get_cov(self) -> np.ndarray:
        cov = np.zeros((self.dim_, self.dim_))
        mean = self.get_mean()
        for i in range(self.n_components_):
            diff = self.dists_[i].get_mean() - mean
            cov += self.weights_[i] * ( self.dists_[i].get_cov() + np.outer(diff, diff))
        return cov

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        log_p = 0.
        for i in range(self.n_components_):
            log_p += self.weights_[i] * np.exp(self.dists_[i].log_density(x))
        return np.log(log_p)

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.dim_)
        for i in range(self.n_components_):
            gamma = self.weights_[i] * np.exp(self.dists_[i].log_density(x) - self.log_density(x))
            grad += gamma * self.dists_[i].grad_log_density(x)
        return grad

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        hess = np.zeros((self.dim_, self.dim_))
        grad = self.grad_log_density(x)
        for i in range(self.n_components_):
            gamma = self.weights_[i] * np.exp(self.dists_[i].log_density(x) - self.log_density(x))
            diff_grad = self.dists_[i].grad_log_density(x) - grad
            grad = self.grad_log_density(x)
            hess += gamma * self.dists_[i].hessian_log_density(x)
            hess += gamma * np.outer(diff_grad, diff_grad)
        return hess


class BananaDistribution(Distribution):
    """
    Banana distribution.

    ...
    Attributes
    ----------
    a_ : float
        The shape parameter a.
    b_ : float
        The shape parameter b.
    gaussian_ : MultivariateNormal
        The underlying Gaussian distribution.

    Methods
    -------
    from_dict(params, rng, seed)
        Create a banana distribution from a dictionary.
    transform(x)
        Transform a point from the Gaussian to the banana distribution.
    """

    def __init__(
            self, mean: np.ndarray,
            cov: np.ndarray,
            a: float = 2.0,
            b: float = 0.2,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.a_ = a
        self.b_ = b
        self.gaussian_ = MultivariateNormal(mean, cov, rng=rng, seed=seed)

    @classmethod
    def from_dict(
            cls,
            params: dict[str, Union[np.ndarray, float]],
            rng: np.random.Generator = None,
            seed: int = None
    ):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")

        return cls(
            mean=params['mean'],
            cov=params['cov'],
            a=params.get('a', 2.0),
            b=params.get('b', 0.2),
            rng=rng,
            seed=seed
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the Gaussian to the banana distribution.

        Parameters:
            x (np.ndarray): The point to transform.

        Returns:
            np.ndarray: The transformed point.
        """
        return np.array([x[0] / self.a_,
                         x[1] * self.a_ + self.a_ * self.b_ * (x[0] ** 2 + self.a_ ** 2)])

    @override
    def get_dim(self) -> int:
        return self.gaussian_.get_dim()

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self.gaussian_.log_density(self.transform(x))

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        nGrad = - self.gaussian_.grad_log_density(self.transform(x))
        return - np.array([nGrad[0] / self.a_ + nGrad[1] * self.a_ * self.b_ * 2 * x[0],
                           nGrad[1] * self.a_])


class MultivariateLogNormal(Distribution):
    """
    Multivariate log-normal distribution.

    ...
    Attributes
    ----------
    mean_normal_ : np.ndarray
        The mean of the underlying normal distribution.
    cov_normal_ : np.ndarray
        The covariance of the underlying normal distribution.
    covL_ : np.ndarray
        The Cholesky decomposition of the covariance matrix.
    logDet_ : float
        The log determinant of the covariance matrix.
    mean_ : np.ndarray

    Methods
    -------
    from_dict(params, rng, seed)
        Create a multivariate log-normal distribution from a dictionary.
    """

    def __init__(
            self,
            mean: np.ndarray,
            cov: np.ndarray,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.mean_normal_ = mean
        self.cov_normal_ = cov
        self.covL_ = np.linalg.cholesky(self.cov_normal_)
        self.logDet_ = np.log(self.covL_.diagonal()).sum()

        self.mean_ = np.exp(mean + np.diagonal(cov) / 2)
        mu_i = np.repeat(mean[:, None], 2, axis=1)
        sig_ii = np.repeat(np.diag(cov)[:, None], 2, axis=1)
        # self.cov_ = np.exp(mu_i + mu_i.transpose() + 0.5 * (sig_ii + sig_ii.transpose())) @ (np.exp(cov) - 1)
        # TODO: finish

        self.dim_ = mean.shape[0]
        self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

    @classmethod
    def from_dict(cls, params: dict[str, np.ndarray], rng: np.random.Generator = None, seed: int = None):
        if 'mean' not in params or 'cov' not in params:
            raise ValueError("Parameters must include 'mean' and 'cov'.")
        return cls(
            mean=params['mean'],
            cov=params['cov'],
            rng=rng,
            seed=seed
        )

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_cov(self) -> np.ndarray:
        return self.cov_

    @override
    def get_mean(self) -> np.ndarray:
        return self.mean_

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        return self.constant_ - self.logDet_ - np.sum(np.log(x)) - 0.5 * np.linalg.norm(
            np.linalg.solve(self.covL_, diff)) ** 2

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        grad = -np.diag(1/x)
        return -(1 + sp.linalg.solve_triangular(self.covL_.transpose(),
                                                sp.linalg.solve_triangular(self.covL_, diff, lower=True)))/x


class CubicDistribution(Distribution):
    """
    Distribution with a cubic term in the log-density.

    ...
    Attributes
    ----------
    normal_ : MultivariateNormal
        The underlying normal distribution.
    a_ : float
        The shape parameter a.
    cubic_diag : np.ndarray
        The diagonal of the cubic matrix.

    Methods
    -------
    from_dict(params, rng, seed)
        Create a cubic distribution from a dictionary.
    """

    def __init__(
            self,
            mean: np.ndarray,
            cov: np.ndarray,
            a: float,
            *,
            cubic_diag: np.ndarray=None,
            rng: np.random.Generator=None,
            seed: int=None
    ):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = mean.shape[0]
        self.normal_ = MultivariateNormal(mean, cov, rng=rng, seed=seed)
        self.a_ = a
        if cubic_diag is not None:
            self.cubic_diag = cubic_diag
        else:
            self.cubic_diag = np.ones(self.dim_)

    @classmethod
    def from_dict(
            cls,
            params: dict[str, Union[np.ndarray, float]],
            rng: np.random.Generator = None,
            seed: int = None
    ):
        if 'mean' not in params or 'cov' not in params or 'a' not in params:
            raise ValueError("Parameters must include 'mean', 'cov', and 'a'.")

        return cls(
            mean=np.array(params['mean']),
            cov=np.array(params['cov']),
            a=params['a'],
            cubic_diag=params.get('cubic_diag', None),
            rng=rng,
            seed=seed
        )

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_mean(self) -> np.ndarray:
        return self.normal_.get_mean()

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return (np.sum((1 - 2*(d>0)) * self.a_/3 * self.cubic_diag * d**3)
                + self.normal_.log_density(x))

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return ((1 - 2 * (d > 0)) * (self.a_ * self.cubic_diag * d**2)
                + self.normal_.grad_log_density(x))

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return ((1 - 2 * (d > 0)) * (2 * self.a_ * np.diag(self.cubic_diag * d))
                + self.normal_.hessian_log_density(x))


class Likelihood(Distribution):
    """
    Base class for likelihoods.

    ...
    Methods
    -------
    log_density(params, idx)
        Get the log density of the likelihood of the observation given by idx.
    grad_log_density(params, idx)
        Get the gradient of the log density of the likelihood of the observation given by idx.
    hessian_log_density(params, idx)
        Get the Hessian of the log density of the likelihood of the observation given by idx.
    """

    def __init__(
            self,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)

    @override
    def get_n_obs(self) -> int:
        raise NotImplementedError

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Get the log density of the likelihood of the observation given by idx.

        Parameters:
            params (np.ndarray): The parameters to evaluate the log density.
            idx (int): The index of the observation. Default is None.

        Returns:
            np.ndarray: The log density of the likelihood of the observation given by idx.
        """
        raise NotImplementedError

    @override
    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Get the gradient of the log density of the likelihood of the observation given by idx.

        Parameters:
            params (np.ndarray): The parameters to evaluate the gradient of the log density.
            idx (int): The index of the observation. Default is None.

        Returns:
            np.ndarray: The gradient of the log density of the likelihood of the observation given by idx.
        """
        raise NotImplementedError

    @override
    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Get the Hessian of the log density of the likelihood of the observation given by idx.

        Parameters:
            params (np.ndarray): The parameters to evaluate the Hessian of the log density.
            idx (int): The index of the observation. Default is None.

        Returns:
            np.ndarray: The Hessian of the log density of the likelihood of the observation given by idx.

        """
        raise NotImplementedError


class TemperedLikelihood(Likelihood):
    """
    Tempered likelihood.

    ...
    Attributes
    ----------
    likelihood_ : Likelihood
        The underlying likelihood.
    beta_ : float
        The tempering parameter.
    """
    def __init__(
            self,
            likelihood: Likelihood,
            *,
            beta: float = 1.0,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.likelihood_ = likelihood
        self.beta_ = beta

    @override
    def get_n_obs(self, n: int=1) -> int:
        return self.likelihood_.get_n_obs()

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.log_density(params, idx=idx)

    @override
    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.grad_log_density(params, idx=idx)

    @override
    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.hessian_log_density(params, idx=idx)


class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood.

    ...
    Attributes
    ----------
    model_ : Model
        The model used to evaluate the likelihood.
    n_params_ : int
        The number of parameters.
    u_obs_ : np.ndarray
        The observations.
    n_obs_ : int
        The number of observations.
    dim_ : int
        The dimension of the observations.
    sigma_ : float
        The standard deviation of the Gaussian noise.
    dists_ : list[MultivariateNormal]
        The component distributions.
    """
    def __init__(
            self,
            model: Model,
            u_obs: np.ndarray,
            sigma: float,
            rng: np.random.Generator = None,
            seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.model_ = model
        self.n_params_ = model.get_dim_in()
        self.u_obs_ = u_obs
        self.n_obs_ = self.u_obs_.shape[0]
        self.dim_ = self.u_obs_.shape[1]
        self.sigma_ = sigma
        self.dists_ = []
        for i in range(self.n_obs_):
            self.dists_.append(MultivariateNormal(self.u_obs_[i],
                                                  sigma ** 2 * np.eye(self.dim_)))

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_n_obs(self) -> int:
        return self.n_obs_

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            log_p = 0.
            for i in range(self.n_obs_):
                log_p += self.dists_[i].log_density(self.model_.eval(params, idx=i))
            return log_p
        else:
            return self.dists_[idx].log_density(self.model_.eval(params, idx=idx))

    @override
    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            grad = np.zeros(self.n_params_)
            for i in range(self.n_obs_):
                m = self.model_.eval(params, idx=i)
                grad_m = self.model_.eval_grad(params, idx=i)
                grad += self.dists_[i].grad_log_density(m) @ grad_m
            return grad
        else:
            m = self.model_.eval(params, idx=idx)
            grad_m = self.model_.eval_grad(params, idx=idx)
            return self.dists_[idx].grad_log_density(m) @ grad_m

    @override
    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:

        def hess_comp(hess: np.ndarray, i: int):
            m = self.model_.eval(params, idx=i)
            grad_m = self.model_.eval_grad(params, idx=i)
            hess_m = self.model_.eval_hessian(params, idx=i)
            hess += np.einsum('ij,jk,il', self.dists_[i].hessian_log_density(m), grad_m, grad_m)
            hess += np.einsum('i,ijk->jk', self.dists_[i].grad_log_density(m), hess_m)

        hess = np.zeros((self.n_params_, self.n_params_))

        if idx is None:
            for i in range(self.n_obs_):
                hess_comp(hess, i)
        else:
            hess_comp(hess, idx)

        return hess


class FlatLikelihood(Likelihood):
    """
    Flat likelihood that assigns equal likelihood to all observations.

    ...
    Attributes
    ----------
    dim_ : int
        The dimension of the observations.
    """

    def __init__(self, dim: int, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = dim

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_n_obs(self) -> int:
        return 0

    @override
    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return np.array([0.0])

    @override
    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            return np.zeros(self.dim_)
        else:
            return np.zeros(1)

    @override
    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return np.zeros((self.dim_, self.dim_))


def get_prior(
        config: dict[str, Union[str, np.ndarray, float]],
        rng: np.random.Generator = None,
    ) -> Distribution:
    """
    Get a prior distribution from a dictionary.

    Parameters:
        config (dict[str, Union[str, np.ndarray, float]]): The configuration of the prior.
        rng (np.random.Generator): The random number generator. Default is None.

    Returns:
        Distribution: The prior distribution
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
    """
    Posterior distribution given a prior and a likelihood.

    ...
    Attributes
    ----------
    dim_ : int
        The dimension of the distribution.
    prior_ : Distribution
        The prior distribution.
    likelihood_ : Likelihood
        The likelihood distribution.

    Methods
    -------
    get_prior_sample()
        Get a sample from the prior distribution.
    """
    def __init__(self, prior: Distribution, likelihood: Likelihood, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = prior.get_dim()
        self.prior_ = prior
        self.likelihood_ = likelihood

    @override
    def get_dim(self) -> int:
        return self.dim_

    @override
    def get_n_obs(self) -> int:
        return self.likelihood_.get_n_obs()

    @override
    def log_density(self, params: np.ndarray) -> np.ndarray:
        return self.likelihood_.log_density(params) + self.prior_.log_density(params)

    @override
    def grad_log_density(self, params: np.ndarray, idx: int = None, sub_sampling: bool = False) -> np.ndarray:
        if sub_sampling:
            approx_llh = self.likelihood_.get_n_obs() * self.likelihood_.grad_log_density(params)
            return approx_llh + self.prior_.grad_log_density(params)
        else:
            return self.likelihood_.grad_log_density(params) + self.prior_.grad_log_density(params)

    @override
    def hessian_log_density(self, params: np.ndarray) -> np.ndarray:
        return self.likelihood_.hessian_log_density(params) + self.prior_.hessian_log_density(params)

    def get_prior_sample(self) -> np.ndarray:
        return self.prior_.get_sample()


class Transformation:
    """
    Base class for transformations.

    A transformation is a bijective mapping between a latent space ξ and a target space x.

    ...
    Methods
    -------
    transform(xi)
        Forward transformation from xi to x.
    inverse_transform(x)
        Inverse transformation from x to xi.
    jacobian(xi)
        Return the Jacobian of the inverse transformation.
    inv_jacobian(xi)
        Return the inverse Jacobian of the transformation.
    log_det_jacobian(xi)
        Return the log determinant of the Jacobian.
    grad_log_det_jacobian(xi)
        Return the gradient of the log determinant of the Jacobian.
    hessian(xi)
        Return the Hessian of the transformation.
    hessian_log_det_jacobian(xi)
        Return the Hessian of the log determinant of the Jacobian.
    """

    def transform(self, xi: np.ndarray) -> np.ndarray:
        """
        Forward transformation from xi to x.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The transformed input
        """
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transformation from x to xi.

        Parameters:
            x: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The inverse transformed input xi
        """
        raise NotImplementedError

    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian of the transformation.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The Jacobian of the transformation
        """
        raise NotImplementedError

    def inv_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """
        Efficiently return the Jacobian of the inverse transformation.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The Jacobian of the inverse transformation
        """
        raise NotImplementedError

    def log_det_jacobian(self, xi: np.ndarray) -> float:
        """
        Computes the log determinant of the Jacobian.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            float: The log determinant of the Jacobian
        """
        raise NotImplementedError

    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """
        Efficiently return the gradient of log-determinant of Jacobian.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The gradient of the log determinant of the Jacobian
        """
        raise NotImplementedError

    def hessian(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian of the transformation.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The Hessian of the transformation
        """
        raise NotImplementedError

    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian of the log determinant of the Jacobian.

        Parameters:
            xi: np.ndarray: The input to the transformation

        Returns:
            np.ndarray: The Hessian of the log determinant of the Jacobian
        """
        raise NotImplementedError

class ExponentialTransformation(Transformation):
    """
    Implements x = exp(ξ).
    """

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
        """
        log |det(J)| = sum(xi) since J is diagonal.
        """
        return np.sum(xi, axis=-1)  # log |det(J)| = sum(xi) for diagonal matrix

    @override
    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        """
        Gradient of log determinant is always 1 (since log(exp(x)) = x and J is diagonal).
        """
        return np.ones_like(xi)

    @override
    def hessian(self, xi: np.ndarray) -> np.ndarray:
        """
        Hessian is a 3-tensor with diagonal elements being exp(ξ).
        """
        d = xi.shape[0]
        H = np.zeros((d, d, d))
        idx = np.arange(d)
        H[idx, idx, idx] = np.exp(xi)
        return H

    @override
    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros((xi.shape[0], xi.shape[0]))


class AffineTransformtion(Transformation):
    """
    Implements x = M * ξ + b.

    ...
    Attributes
    ----------
    M : np.ndarray
        The linear transformation matrix.
    b : np.ndarray
        The translation vector.
    M_inv : np.ndarray
        The inverse of the linear transformation matrix.
    log_abs_det_M : float
        The log determinant of the linear transformation matrix.
    """

    def __init__(self, M: np.ndarray, b: np.ndarray):
        self.M = M
        self.b = b
        self.M_inv = np.linalg.inv(M)
        self.log_abs_det_M = np.log(np.abs(np.linalg.det(M)))

    @override
    def transform(self, xi: np.ndarray) -> np.ndarray:
        return xi @ self.M.T + self.b

    @override
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.M_inv @ (x - self.b)

    @override
    def jacobian(self, xi: np.ndarray) -> np.ndarray:
        return self.M  # Jacobian is constant

    @override
    def inv_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return self.M_inv  # Directly return precomputed inverse

    @override
    def log_det_jacobian(self, xi: np.ndarray) -> float:
        return self.log_abs_det_M  # log |det(M)|

    @override
    def grad_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros_like(xi)  # Zero since M is constant

    @override
    def hessian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros((xi.shape[0], xi.shape[0], xi.shape[0]))  # Hessian is zero

    @override
    def hessian_log_det_jacobian(self, xi: np.ndarray) -> np.ndarray:
        return np.zeros((xi.shape[0], xi.shape[0]))

EXPONENTIAL = 'Exponential'
AFFINE = 'Affine'
TRANSFORMATIONS = [EXPONENTIAL, AFFINE]

class TransformedDistribution(Distribution):
    """
    Applies a transformation to a base distribution.

    ...
    Attributes
    ----------
    base_distribution : Distribution
        The base distribution to transform.
    transformation : Transformation
        The transformation to apply.

    Methods
    -------
    get_prior_sample()
        Get a prior sample from the transformed distribution in case the base distribution is a posterior.
    """

    def __init__(self, base_distribution: Distribution, params: dict[str, Any]):
        """
        Parameters:
            base_distribution (Distribution): The base distribution to transform.
            params (dict[str, Any]): The parameters of the transformation.

        Raises:
            NotImplementedError: If the transformation is not recognized.
        """
        super().__init__(rng=base_distribution.rng_)
        self.base_distribution = base_distribution

        if params['transformation'] == EXPONENTIAL:
            self.transformation = ExponentialTransformation()
        elif params['transformation'] == AFFINE:
            M = params.get('M', None)
            b = params.get('b', None)
            x_0 = params.get('x_0', None)

            if b is None:
                b = find_mean(base_distribution, x_0=x_0)

            if M is None:
                M = find_curvature(base_distribution, mean=b)

            C = np.linalg.cholesky(M)

            self.transformation = AffineTransformtion(C, b)
        else:
            raise NotImplementedError(f"Transformation {params['transformation']} not recognized.\n"
                                      f"pick any of {TRANSFORMATIONS}")

    @override
    def get_sample(self, n: int=1) -> np.ndarray:
        """
        Generates a sample from the underlying distribution and transforms it.

        Returns:
            np.ndarray: A sample from the transformed distribution.
        """
        x_sample = self.base_distribution.get_sample(n=n)
        return self.transformation.inverse_transform(x_sample)

    @override
    def get_dim(self) -> int:
        """
        Returns:
            int: The dimension of the transformed distribution.
        """
        return self.base_distribution.get_dim()

    @override
    def log_density(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes log p(xi).

        Parameters:
            xi (np.ndarray): The input to the transformation.

        Returns:
            np.ndarray: The log density of the transformed distribution.
        """
        x = self.transformation.transform(xi)
        return self.base_distribution.log_density(x) + self.transformation.log_det_jacobian(xi)

    @override
    def grad_log_density(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the log density with respect to xi.

        Parameters:
            xi (np.ndarray): The input to the transformation.

        Returns:
            np.ndarray: The gradient of the log density with respect to xi.
        """
        x = self.transformation.transform(xi)

        # Precompute gradient of log density of base distribution
        grad_log_p_xi = self.transformation.jacobian(xi).T @ self.base_distribution.grad_log_density(x)

        return grad_log_p_xi + self.transformation.grad_log_det_jacobian(xi)

    @override
    def hessian_log_density(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian of the log density with respect to xi.

        Parameters:
            xi (np.ndarray): The input to the transformation.

        Returns:
            np.ndarray: The Hessian of the log density with respect to xi.
        """

        x = self.transformation.transform(xi)
        H_x = self.base_distribution.hessian_log_density(x)
        J = self.transformation.jacobian(xi)

        # Compute Hessian of transformation
        grad_x = self.base_distribution.grad_log_density(x)
        H_f = self.transformation.hessian(xi)
        H_f_grad = np.einsum('ijk,j->ik', H_f, grad_x)

        # Compute Hessian of log density of base distribution
        jac_correction = self.transformation.hessian_log_det_jacobian(xi)

        return J.T @ H_x @ J + H_f_grad + jac_correction

    def get_prior_sample(self) -> np.ndarray:
        """
        Get a prior sample from the transformed distribution.

        Returns:
            np.ndarray: A prior sample from the transformed distribution.
        """
        assert hasattr(self.base_distribution, 'get_prior_sample')
        child = cast(Posterior, self.base_distribution)
        return self.transformation.inverse_transform(child.get_prior_sample())


class TransformedLikelihood(Likelihood):
    """
    Applies a transformation to a base distribution.

    ...
    Attributes
    ----------
    likelihood : Likelihood
        The likelihood distribution.
    transformation : Transformation
        The transformation to apply.
    """

    def __init__(self, likelihood: Likelihood, params: dict[str, Any]):
        super().__init__(rng=likelihood.rng_)
        self.likelihood = likelihood

        if params['transformation'] == EXPONENTIAL:
            self.transformation = ExponentialTransformation()
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

            self.transformation = AffineTransformtion(C, b)
        else:
            raise NotImplementedError(f"Transformation {params['transformation']} not recognized.\n"
                                      f"pick any of {TRANSFORMATIONS}")

    @override
    def get_dim(self) -> int:
        """
        Returns:
            int: The dimension of the transformed distribution.
        """
        return self.likelihood.get_dim()

    @override
    def log_density(self, xi: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Computes log p(xi).

        Parameters:
            xi (np.ndarray): The input to the transformation.
            idx (int): The index of the observation to evaluate.

        Returns:
            np.ndarray: The log density of the transformed distribution.
        """
        x = self.transformation.transform(xi)
        return self.likelihood.log_density(x, idx=idx)

    @override
    def grad_log_density(self, xi: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Computes the gradient of the log density with respect to xi.

        Parameters:
            xi (np.ndarray): The input to the transformation.
            idx (int): The index of the observation to evaluate.

        Returns:
            np.ndarray: The gradient of the log density with respect to xi.
        """
        x = self.transformation.transform(xi)

        # Precompute gradient of log density of base distribution
        grad_log_p_xi = self.transformation.jacobian(xi).T @ self.likelihood.grad_log_density(x, idx=idx)

        # return grad_log_p_xi + self.transformation.grad_log_det_jacobian(xi)
        return grad_log_p_xi

    @override
    def hessian_log_density(self, xi: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Computes the Hessian of the log density with respect to xi.

        Parameters:
            xi (np.ndarray): The input to the transformation.
            idx (int): The index of the observation to evaluate.

        Returns:
            np.ndarray: The Hessian of the log density with respect to xi.
        """
        x = self.transformation.transform(xi)
        H_x = self.likelihood.hessian_log_density(x, idx=idx)
        J = self.transformation.jacobian(xi)

        # Compute Hessian of transformation
        grad_x = self.likelihood.grad_log_density(x, idx=idx)
        H_f = self.transformation.hessian(xi)
        H_f_grad = np.einsum('ijk,j->ik', H_f, grad_x)

        return J.T @ H_x @ J + H_f_grad

def get_likelihood(
        config: dict[str, Any],
        model: Model,
        rng: np.random.Generator = None,
) -> Likelihood:
    """
    Get a likelihood distribution from a configuration.

    Parameters:
        config (dict[str, Any]): The configuration of the likelihood.
        model (Model): The model to evaluate the likelihood.
        rng (np.random.Generator): The random number generator.

    Returns:
        Likelihood: The likelihood distribution.
    """

    obs = None

    if 'name' not in config:
        raise ValueError("Likelihood config must include 'name'.")
    if (
            (config['name'] != 'FlatLikelihood')
        and (config['name'] != 'TransformedLikelihood')
        and ('observation_file' not in config)
    ):
        raise ValueError("Likelihood config must include 'observation_file'.")

    if (config['name'] != 'FlatLikelihood') and (config['name'] != 'TransformedLikelihood'):
        obs = np.genfromtxt(config['observation_file'])
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

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

def find_mean(
        target: Distribution,
        x_0: np.ndarray = None
) -> np.ndarray:
    """
    Find the mean of a distribution using the BFGS method

    Parameters:
        target (Distribution): The target distribution
        x_0 (np.ndarray): The initial guess for the mean (optional). Default is None.
    Returns:
        np.ndarray: The mean of the distribution
    """

    n_log_post = lambda x: - target.log_density(x)
    n_grad_log_post = lambda x: - target.grad_log_density(x)

    if x_0 is None:
        logger.warning("No initial point provided ... attempting to get sample from target.")
        success = False

        if hasattr(target, 'get_sample'):
            try:
                x_0 = target.get_sample()
                success = True
            except NotImplementedError as e:
                logger.warning("  Method get_sample not implemented for target.")

        if hasattr(target, 'get_mean'):
            try:
                x_0 = target.get_mean()
                success = True
            except NotImplementedError as e:
                logger.warning("  Method get_mean not implemented for target.")

        if not success and hasattr(target, 'prior_') and hasattr(target.prior_, 'get_sample'):
            try:
                x_0 = target.prior_.get_sample()
                success = True
            except NotImplementedError as e:
                logger.warning("  Method get_sample not implemented for prior.")

        if not success and hasattr(target, 'prior_') and hasattr(target.prior_, 'get_mean'):
            try:
                x_0 = target.prior_.get_mean()
            except NotImplementedError as e:
                logger.warning("  Method get_mean not implemented for prior.")

        if not success:
            x_0 = np.zeros(target.get_dim())

    return minimize(n_log_post, x_0, jac=n_grad_log_post, method='BFGS').x

def find_curvature(
        target: Distribution,
        mean: np.ndarray = None,
) -> np.ndarray:
    """
    Find the curvature of a distribution at the MAP point

    Parameters:
        target (Distribution): The target distribution
        mean (np.ndarray): The mean of the distribution (optional)
    Returns:
        np.ndarray: The covariance of the distribution
    """

    if mean is None:
        mean = find_mean(target)

    return - np.linalg.inv(target.hessian_log_density(mean))


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
