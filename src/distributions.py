import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from datetime import datetime
from typing import Any
from src.forward_model import ForwardModel

small = 1e-12
large = 1e20


class Distribution:

    def __init__(self, rng: np.random.Generator = None, seed: int = None):
        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

    def getSample(self) -> np.ndarray:
        pass

    def getDim(self) -> int:
        pass

    def logDensity(self, x: np.ndarray) -> float:
        pass

    def gradLogDensity(self, x: np.ndarray) -> np.ndarray:
        pass

    def hessianLogDensity(self, x: np.ndarray) -> np.ndarray:
        pass


class MultivariateNormal(Distribution):

    def __init__(self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.mean_ = mean
        self.dim_ = mean.shape[0]
        self.setCovariance(cov)
        self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

    def setCovariance(self, cov: np.ndarray) -> None:
        self.cov_ = cov
        self.covL_ = np.linalg.cholesky(cov)
        self.invC_ = sp.linalg.cho_solve((self.covL_, True), np.eye(self.dim_))
        self.logDet_ = np.log(self.covL_.diagonal()).sum()

    def getSample(self) -> np.ndarray:
        z = self.rng_.standard_normal(self.dim_)
        return self.covL_ @ z + self.mean_

    def getDim(self) -> int:
        return self.dim_

    def getCov(self) -> np.ndarray:
        return self.cov_

    def get_n_obs(self) -> int:
        return 0

    def logDensity(self, x: np.ndarray) -> float:
        diff = x - self.mean_
        if diff.ndim == 1:
            if self.dim_ == 1:
                return self.constant_ - self.logDet_ - 0.5 * np.abs(diff / self.covL_[0,0]) ** 2
            else:
                return self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff)) ** 2
        else:
            return (self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff.T), axis=0) ** 2).T

    def gradLogDensity(self, x: np.ndarray) -> np.ndarray:
        diff =  x - self.mean_
        return - sp.linalg.solve_triangular(self.covL_.transpose(),
                                            sp.linalg.solve_triangular(self.covL_, diff, lower=True))

    def hessianLogDensity(self, x: np.ndarray) -> np.ndarray:
        return - self.invC_


class BananaDistribution(MultivariateNormal):

    def __init__(self, mean: np.ndarray, cov: np.ndarray, a: float = 2.0, b: float = 0.2, rng: np.random.Generator = None, seed: int = None):
        super().__init__(mean, cov, rng=rng, seed=seed)
        self.a_ = a
        self.b_ = b

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] / self.a_,
                         x[1] * self.a_ + self.a_ * self.b_ * (x[0] ** 2 + self.a_ ** 2)])

    def getSample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Banana. Use MCMC instead")

    def getDim(self) -> int:
        return self.dim_

    def logDensity(self, x: np.ndarray) -> float:
        return super().logDensity(self.transform(x))

    def gradLogDensity(self, x: np.ndarray) -> np.ndarray:
        nGrad = - super().gradLogDensity(self.transform(x))
        return - np.array([nGrad[0] / self.a_ + nGrad[1] * self.a_ * self.b_ * 2 * x[0],
                           nGrad[1] * self.a_])


class MultivariateLogNormal(Distribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator = None, seed: int = None):
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

    def getDim(self) -> int:
        return self.dim_

    def getCov(self) -> np.ndarray:
        return self.cov_

    def getMean(self) -> np.ndarray:
        return self.mean_

    def logDensity(self, x: np.ndarray) -> float:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        return self.constant_ - self.logDet_ - np.sum(np.log(x)) - 0.5 * np.linalg.norm(
            np.linalg.solve(self.covL_, diff)) ** 2

    def gradLogDensity(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        grad = -np.diag(1/x)
        return -(1 + sp.linalg.solve_triangular(self.covL_.transpose(),
                                                sp.linalg.solve_triangular(self.covL_, diff, lower=True)))/x

class Likelihood(Distribution):
    def __init__(self, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)

    def get_n_obs(self) -> int:
        pass

    def gradLogDensity(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        pass

    def hessianLogDensity(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        pass

class GaussianLikelihood(Likelihood):
    def __init__(
            self,
            model: ForwardModel,
            x_obs: np.ndarray,
            u_obs: np.ndarray,
            sigma_obs: float,
            rng: np.random.Generator = None, seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.model_ = model
        self.n_params_ = model.getDim()
        self.x_obs_ = x_obs
        self.u_obs_ = u_obs
        self.n_obs_ = self.u_obs_.shape[0]
        self.dim_ = self.u_obs_.shape[1]
        self.sigma_obs_ = sigma_obs
        self.dists_ = []
        for i in range(self.n_obs_):
            self.dists_.append(MultivariateNormal(self.u_obs_[i],
                                                  sigma_obs**2 * np.eye(self.dim_)))

    def getDim(self) -> int:
        return self.dim_

    def getSample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from GaussianLikelihood. Use MCMC instead")

    def get_n_obs(self) -> int:
        return self.n_obs_

    def logDensity(self, params: np.ndarray, idx: int = None) -> float:
        if idx is None:
            log_p = 0.
            for i in range(self.n_obs_):
                log_p += self.dists_[i].logDensity(self.model_.eval(self.x_obs_, params))
            return log_p
        else:
            return self.dists_[idx].logDensity(self.model_.eval(self.x_obs_, params))

    def gradLogDensity(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            grad = np.zeros(self.n_params_)
            for i in range(self.n_obs_):
                m = self.model_.eval(self.x_obs_, params, idx=i)
                grad_m = self.model_.eval_grad(self.x_obs_, params, idx=i)
                grad += self.dists_[i].gradLogDensity(m) @ grad_m
            return grad
        else:
            m = self.model_.eval(self.x_obs_, params, idx=idx)
            grad_m = self.model_.eval_grad(self.x_obs_, params, idx=idx)
            return self.dists_[idx].gradLogDensity(m) @ grad_m

    def hessianLogDensity(self, params: np.ndarray, idx: int = None) -> np.ndarray:

        def hess_comp(hess, i):
            m = self.model_.eval(self.x_obs_, params, idx=i)
            grad_m = self.model_.eval_grad(self.x_obs_, params, idx=i)
            hess_m = self.model_.eval_hessian(self.x_obs_, params, idx=i)
            hess += np.einsum('ij,jk,il', self.dists_[i].hessianLogDensity(m), grad_m, grad_m)
            hess += np.einsum('i,ijk->jk', self.dists_[i].gradLogDensity(m), hess_m)

        hess = np.zeros((self.n_params_, self.n_params_))

        if idx is None:
            for i in range(self.n_obs_):
                hess_comp(hess, i)
        else:
            hess_comp(hess, idx)

        return hess


class FlatLikelihood(Likelihood):

    def __init__(self, dim: int, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = dim

    def getDim(self) -> int:
        return self.dim_

    def getSample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Flat GaussianLikelihood. Use MCMC instead")

    def get_n_obs(self) -> int:
        return 0

    def logDensity(self, params: np.ndarray, idx: int = None) -> float:
        return 1.0

    def gradLogDensity(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            return np.zeros(self.dim_)
        else:
            return np.zeros(1)

class Posterior(Distribution):
    def __init__(self, prior: Distribution, likelihood: Likelihood, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = prior.getDim()
        self.prior_ = prior
        self.likelihood_ = likelihood

    def getDim(self) -> int:
        return self.dim_

    def getSample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Posterior. Use MCMC instead")

    def get_n_obs(self) -> int:
        return self.likelihood_.get_n_obs()

    def logDensity(self, params: np.ndarray) -> float:
        return self.likelihood_.logDensity(params) + self.prior_.logDensity(params)

    def gradLogDensity(self, params: np.ndarray, idx: int = None, sub_sampling: bool = False) -> np.ndarray:
        if sub_sampling:
            approx_llh = self.likelihood_.get_n_obs() * self.likelihood_.gradLogDensity(params, idx)
            return approx_llh + self.prior_.gradLogDensity(params)
        else:
            return self.likelihood_.gradLogDensity(params, idx) + self.prior_.gradLogDensity(params)

    def getPriorSample(self) -> np.ndarray:
        return self.prior_.getSample()

    def hessianLogDensity(self, params: np.ndarray) -> np.ndarray:
        return self.likelihood_.hessianLogDensity(params) + self.prior_.hessianLogDensity(params)

def getSample(dim, rng=None):
    if rng is None:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), seed=datetime.now().microsecond)
    else:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), rng=rng)
    return dist.getSample()
