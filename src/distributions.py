import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from datetime import datetime

small = 1e-12
large = 1e20

class MultivariateNormal:

    def __init__(self, mean, cov, rng=None, seed=None):
        self.mean_ = mean
        self.dim_ = mean.shape[0]
        self.setCovariance(cov)
        self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

    def setCovariance(self, cov):
        self.cov_ = cov
        self.covL_ = np.linalg.cholesky(cov)
        self.invC_ = sp.linalg.cho_solve((self.covL_, True), np.eye(self.dim_))
        self.logDet_ = np.log(self.covL_.diagonal()).sum()

    def getSample(self):
        z = self.rng_.standard_normal(self.dim_)
        return self.covL_ @ z + self.mean_

    def getDim(self):
        return self.dim_

    def getCov(self):
        return self.cov_

    def logDensity(self, x):
        diff = x - self.mean_
        return self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff)) ** 2

    def gradLogDensity(self, x):
        diff = x - self.mean_
        return - sp.linalg.solve_triangular(self.covL_.transpose(),
                                            sp.linalg.solve_triangular(self.covL_, diff, lower=True))

    def hessianLogDensity(self, x):
        return - self.invC_


class BananaDistribution(MultivariateNormal):

    def __init__(self, mean, cov, a=2., b=0.2, rng=None, seed=None):
        super().__init__(mean, cov, rng=rng, seed=seed)
        self.a_ = a
        self.b_ = b

    def transform(self, x):
        return np.array([x[0] / self.a_,
                         x[1] * self.a_ + self.a_ * self.b_ * (x[0] ** 2 + self.a_ ** 2)])

    def getSample(self):
        raise Exception("Cannot sample directly from Banana. Use MCMC instead")

    def getDim(self):
        return self.dim_

    def logDensity(self, x):
        return super().logDensity(self.transform(x))

    def gradLogDensity(self, x):
        nGrad = - super().gradLogDensity(self.transform(x))
        return - np.array([nGrad[0] / self.a_ + nGrad[1] * self.a_ * self.b_ * 2 * x[0],
                           nGrad[1] * self.a_])


class MultivariateLogNormal:
    def __init__(self, mean, cov, rng=None, seed=None):
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

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

    def getDim(self):
        return self.dim_

    def getCov(self):
        return self.cov_

    def getMean(self):
        return self.mean_

    def logDensity(self, x):
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        return self.constant_ - self.logDet_ - np.sum(np.log(x)) - 0.5 * np.linalg.norm(
            np.linalg.solve(self.covL_, diff)) ** 2

    def gradLogDensity(self, x):
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        grad = -np.diag(1/x)
        return -(1 + sp.linalg.solve_triangular(self.covL_.transpose(),
                                                  sp.linalg.solve_triangular(self.covL_, diff, lower=True)))/x


class Likelihood:
    def __init__(self, model, x_obs, u_obs, sigma_obs):
        self.model_ = model
        self.x_obs_ = x_obs
        self.u_obs_ = u_obs
        self.n_obs_ = self.u_obs_.shape[0]
        self.dim_ = self.u_obs_.shape[1]
        self.sigma_obs_ = sigma_obs
        self.dists_ = []
        for i in range(self.n_obs_):
            self.dists_.append(MultivariateNormal(self.u_obs_[i],
                                                  sigma_obs**2 * np.eye(self.dim_)))

    def getDim(self):
        return self.dim_

    def get_n_obs(self):
        return self.n_obs_

    def logDensity(self, params, idx=None):

        if idx is None:
            log_p = 0.
            for i in range(self.n_obs_):
                log_p += self.dists_[i].logDensity(self.model_.eval(self.x_obs_, params))
            return log_p
        else:
            return self.dists_[idx].logDensity(self.model_.eval(self.x_obs_, params))

    def gradLogDensity(self, params, idx=None):

        if idx is None:
            grad = np.zeros(self.dim_)
            for i in range(self.n_obs_):
                m = self.model_.eval(self.x_obs_, params, idx=i)
                grad_m = self.model_.eval_grad(self.x_obs_, params, idx=i)
                # grad += self.dists_[i].gradLogDensity(self.model_.eval(self.x_obs_, params)) @ grad_m
                grad += self.dists_[i].gradLogDensity(m) @ grad_m
            return grad
        else:
            m = self.model_.eval(self.x_obs_, params, idx=idx)
            grad_m = self.model_.eval_grad(self.x_obs_, params, idx=idx)
            return self.dists_[idx].gradLogDensity(m) @ grad_m

    # todo: fix this implementation
    def hessianLogDensity(self, params):

        m = self.model_.eval(self.x_obs_, params)
        grad_m = self.model_.eval_grad(self.x_obs_, params)
        hess_m = self.model_.eval_hessian(self.x_obs_, params)
        hess = np.zeros((self.dim_, self.dim_))

        for i in range(self.n_obs_):
            hess += self.dists_[i].gradLogDensity(m) @ hess_m + \
                    self.dists_[i].hessianLogDensity(m) @ grad_m @ grad_m.T

        return hess


class Posterior:
    def __init__(self, prior, likelihood):
        self.dim_ = prior.getDim()
        self.prior_ = prior
        self.likelihood_ = likelihood

    def getDim(self):
        return self.dim_

    def get_n_obs(self):
        return self.likelihood_.get_n_obs()

    def logDensity(self, params):
        return self.likelihood_.logDensity(params) + self.prior_.logDensity(params)

    def gradLogDensity(self, params, idx=None, sub_sampling=False):
        if sub_sampling:
            approx_llh = self.likelihood_.get_n_obs() * self.likelihood_.gradLogDensity(params, idx)
            return approx_llh + self.prior_.gradLogDensity(params)
        else:
            return self.likelihood_.gradLogDensity(params, idx) + self.prior_.gradLogDensity(params)
        # return self.likelihood_.gradLogDensity(params, idx) + self.prior_.gradLogDensity(params)

    def getPriorSample(self):
        return self.prior_.getSample()

    def hessianLogDensity(self, params):
        return self.likelihood_.hessianLogDensity(params) + self.prior_.hessianLogDensity(params)


def getSample(dim, rng=None):
    if rng is None:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), seed=datetime.now().microsecond)
    else:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), rng=rng)
    return dist.getSample()


if __name__ == '__main__':
    mean = np.array([0., 1.])
    # cov = np.array([[1., ]])
    cov = np.eye(2)

    rng = np.random.default_rng(0)

    dist = MultivariateLogNormal(mean, cov, rng=rng)

    x = np.array([-0.01, 1.])

    dist.logDensity(x)

    # x = np.linspace(0.01, 3, 100)
    # y = np.zeros_like(x)
    # y_np = np.zeros_like(x)
    #
    # for i in range(len(x)):
    #     y[i] = np.exp(dist.logDensity(x[i]))
    #     # y[i] = dist.logDensity(x[i])
    #     y_np[i] = pdf(x[i], mean[0], cov[0, 0])
    #
    # x0 = x[50]
    # y0 = y[50]
    # dydx = y0 * dist.gradLogDensity(np.array([x0]))[0]
    # # dydx = dist.gradLogDensity(np.array([x0]))[0]
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.plot(x, y0 + (x - x0) * dydx)
    # ax.scatter(x0, y0)
    # fig.show()
