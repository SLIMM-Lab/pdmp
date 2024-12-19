import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns

from datetime import datetime
from timeit import timeit

from pdmp.forward_model import Model
from pdmp.utils import get_2d_despined_figure, plot_samples

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

    def get_sample(self) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self) -> int:
        raise NotImplementedError

    def log_density(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_n_obs(self) -> int:
        return 0


class MultivariateNormal(Distribution):

    def __init__(self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.mean_ = mean
        self.dim_ = mean.shape[0]
        self.cov_ = cov
        self.covL_ = np.linalg.cholesky(cov)
        self.invC_ = sp.linalg.cho_solve((self.covL_, True), np.eye(self.dim_))
        self.logDet_ = np.log(self.covL_.diagonal()).sum()
        self.constant_ = - 0.5 * np.log(2.0 * np.pi) * self.dim_

    def get_sample(self, n: int = 1) -> np.ndarray:
        if n == 1:
            z = self.rng_.standard_normal(size=self.dim_)
            return self.covL_ @ z + self.mean_
        else:
            z = self.rng_.standard_normal(size=(n, self.dim_))
            return z @ self.covL_ + self.mean_

    def get_dim(self) -> int:
        return self.dim_

    def get_mean(self) -> np.ndarray:
        return self.mean_

    def get_cov(self) -> np.ndarray:
        return self.cov_

    def get_inv_cov(self) -> np.ndarray:
        return self.invC_

    def log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.mean_
        if diff.ndim == 1:
            if self.dim_ == 1:
                return self.constant_ - self.logDet_ - 0.5 * np.abs(diff / self.covL_[0,0]) ** 2
            else:
                return self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff)) ** 2
        else:
            return (self.constant_ - self.logDet_ - 0.5 * np.linalg.norm(np.linalg.solve(self.covL_, diff.T), axis=0) ** 2).T

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        diff =  x - self.mean_
        return - sp.linalg.solve_triangular(self.covL_.transpose(),
                                            sp.linalg.solve_triangular(self.covL_, diff, lower=True))

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        return - self.invC_


class GaussianMixture(Distribution):

    def __init__(self, means: np.ndarray, covs: np.ndarray, weights: np.ndarray, rng: np.random.Generator = None, seed: int = None):
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

    def get_sample(self) -> np.ndarray:
        idx = self.rng_.choice(self.n_components_, p=self.weights_)
        return self.dists_[idx].get_sample()

    def get_dim(self) -> int:
        return self.dim_

    def get_mean(self) -> np.ndarray:
        mean = np.zeros(self.dim_)
        for i in range(self.n_components_):
            mean += self.weights_[i] * self.dists_[i].get_mean()
        return mean

    def get_cov(self) -> np.ndarray:
        cov = np.zeros((self.dim_, self.dim_))
        mean = self.get_mean()
        for i in range(self.n_components_):
            diff = self.dists_[i].get_mean() - mean
            cov += self.weights_[i] * ( self.dists_[i].get_cov() + np.outer(diff, diff))
        return cov

    def log_density(self, x: np.ndarray) -> np.ndarray:
        log_p = 0.
        for i in range(self.n_components_):
            log_p += self.weights_[i] * np.exp(self.dists_[i].log_density(x))
        return np.log(log_p)

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.dim_)
        for i in range(self.n_components_):
            gamma = self.weights_[i] * np.exp(self.dists_[i].log_density(x) - self.log_density(x))
            grad += gamma * self.dists_[i].grad_log_density(x)
        return grad

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

    def __init__(self, mean: np.ndarray, cov: np.ndarray, a: float = 2.0, b: float = 0.2, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.a_ = a
        self.b_ = b
        self.gaussian_ = MultivariateNormal(mean, cov, rng=rng, seed=seed)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] / self.a_,
                         x[1] * self.a_ + self.a_ * self.b_ * (x[0] ** 2 + self.a_ ** 2)])

    def get_sample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Banana. Use MCMC instead")

    def get_dim(self) -> int:
        return self.gaussian_.get_dim()

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self.gaussian_.log_density(self.transform(x))

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        nGrad = - self.gaussian_.grad_log_density(self.transform(x))
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

    def get_dim(self) -> int:
        return self.dim_

    def get_cov(self) -> np.ndarray:
        return self.cov_

    def get_mean(self) -> np.ndarray:
        return self.mean_

    def log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        return self.constant_ - self.logDet_ - np.sum(np.log(x)) - 0.5 * np.linalg.norm(
            np.linalg.solve(self.covL_, diff)) ** 2

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        x[np.where(x < 0)] = small
        diff = np.log(x) - self.mean_normal_
        grad = -np.diag(1/x)
        return -(1 + sp.linalg.solve_triangular(self.covL_.transpose(),
                                                sp.linalg.solve_triangular(self.covL_, diff, lower=True)))/x


class CubicDistribution(Distribution):

    def __init__(self, mean: np.ndarray, cov: np.ndarray, a: float, *, cubic_diag: np.ndarray=None,
                 rng: np.random.Generator=None, seed: int=None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = mean.shape[0]
        self.normal_ = MultivariateNormal(mean, cov, rng=rng, seed=seed)
        self.a_ = a
        if cubic_diag is not None:
            self.cubic_diag = cubic_diag
        else:
            self.cubic_diag = np.ones(self.dim_)

    def get_dim(self) -> int:
        return self.dim_

    def get_sample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Cubic. Use MCMC instead")

    def log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return (np.sum((1 - 2*(d>0)) * self.a_/3 * self.cubic_diag * d**3)
                + self.normal_.log_density(x))

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return ((1 - 2 * (d > 0)) * (self.a_ * self.cubic_diag * d**2)
                + self.normal_.grad_log_density(x))

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        d = x - self.normal_.get_mean()
        return ((1 - 2 * (d > 0)) * (2 * self.a_ * np.diag(self.cubic_diag * d))
                + self.normal_.hessian_log_density(x))


class Likelihood(Distribution):
    def __init__(self, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)

    def get_n_obs(self) -> int:
        raise NotImplementedError

    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        raise NotImplementedError

    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        raise NotImplementedError

    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        raise NotImplementedError


class TemperedLikelihood(Likelihood):
    def __init__(self,
                 likelihood: Likelihood,
                 *,
                 beta: float = 1.0,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.likelihood_ = likelihood
        self.beta_ = beta

    def get_n_obs(self) -> int:
        return self.likelihood_.get_n_obs()

    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.log_density(params, idx=idx)

    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.grad_log_density(params, idx=idx)

    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return self.beta_ * self.likelihood_.hessian_log_density(params, idx=idx)


class GaussianLikelihood(Likelihood):
    def __init__(self, model: Model, u_obs: np.ndarray, sigma_obs: float, rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.model_ = model
        self.n_params_ = model.get_dim()
        self.u_obs_ = u_obs
        self.n_obs_ = self.u_obs_.shape[0]
        self.dim_ = self.u_obs_.shape[1]
        self.sigma_obs_ = sigma_obs
        self.dists_ = []
        for i in range(self.n_obs_):
            self.dists_.append(MultivariateNormal(self.u_obs_[i],
                                                  sigma_obs**2 * np.eye(self.dim_)))

    def get_dim(self) -> int:
        return self.dim_

    def get_sample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from GaussianLikelihood. Use MCMC instead")

    def get_n_obs(self) -> int:
        return self.n_obs_

    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            log_p = 0.
            for i in range(self.n_obs_):
                log_p += self.dists_[i].log_density(self.model_.eval(params, idx=i))
            return log_p
        else:
            return self.dists_[idx].log_density(self.model_.eval(params, idx=idx))

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

    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:

        def hess_comp(hess, i):
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

    def __init__(self, dim: int, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = dim

    def get_dim(self) -> int:
        return self.dim_

    def get_sample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Flat GaussianLikelihood. Use MCMC instead")

    def get_n_obs(self) -> int:
        return 0

    def log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return np.array([0.0])

    def grad_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            return np.zeros(self.dim_)
        else:
            return np.zeros(1)

    def hessian_log_density(self, params: np.ndarray, idx: int = None) -> np.ndarray:
        return np.zeros((self.dim_, self.dim_))


class Posterior(Distribution):
    def __init__(self, prior: Distribution, likelihood: Likelihood, rng: np.random.Generator = None, seed: int = None):
        super().__init__(rng=rng, seed=seed)
        self.dim_ = prior.get_dim()
        self.prior_ = prior
        self.likelihood_ = likelihood

    def get_dim(self) -> int:
        return self.dim_

    def get_sample(self) -> np.ndarray:
        raise Exception("Cannot sample directly from Posterior. Use MCMC instead")

    def get_n_obs(self) -> int:
        return self.likelihood_.get_n_obs()

    def log_density(self, params: np.ndarray) -> np.ndarray:
        return self.likelihood_.log_density(params) + self.prior_.log_density(params)

    def grad_log_density(self, params: np.ndarray, idx: int = None, sub_sampling: bool = False) -> np.ndarray:
        if sub_sampling:
            approx_llh = self.likelihood_.get_n_obs() * self.likelihood_.grad_log_density(params)
            return approx_llh + self.prior_.grad_log_density(params)
        else:
            return self.likelihood_.grad_log_density(params) + self.prior_.grad_log_density(params)

    def get_prior_sample(self) -> np.ndarray:
        return self.prior_.get_sample()

    def hessian_log_density(self, params: np.ndarray) -> np.ndarray:
        return self.likelihood_.hessian_log_density(params) + self.prior_.hessian_log_density(params)


def plot_pdf_contours(
        distribution: Distribution,
        ax: plt.Axes,
        plot_limits: tuple[list[float], list[float]],
        n_grid: int = 100,
        alpha: float = 0.6,
        n_levels: int = 20,
        cmap: matplotlib.colors.Colormap = sns.color_palette('rocket', as_cmap=True)
) -> plt.Axes:
    """
    Plot the probability density function (PDF) contours of a distribution.

    Parameters:
    distribution (Distribution): The distribution to plot.
    ax (plt.Axes): The matplotlib axes object to plot on.
    plot_limits (tuple): A tuple containing two lists, each specifying the x and y axis limits respectively.
    n_grid (int, optional): Number of grid points for the x and y axes. Default is 100.
    alpha (float, optional): Transparency level of the contour plot. Default is 0.6.
    n_levels (int, optional): Number of contour levels to plot. Default is 20.
    cmap (sns.palettes._ColorPalette, optional): Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
    plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = np.exp(distribution.log_density(np.array([X[i, j], Y[i, j]])))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax

def plot_pfd_contour_conditional(
        distribution: Distribution,
        ax: plt.Axes,
        plot_limits: tuple[list[float], list[float]],
        slice: np.ndarray,
        idcs_plane: tuple[int, int] = (0, 1),
        n_grid: int = 100,
        alpha: float = 0.6,
        n_levels: int = 20,
        cmap: matplotlib.colors.ListedColormap = sns.color_palette('rocket', as_cmap=True)
) -> plt.Axes:
    """
    Plot the conditional probability density function (PDF) contours of a distribution.

    Parameters:
    distribution (Distribution): The distribution to plot.
    ax (plt.Axes): The matplotlib axes object to plot on.
    plot_limits (tuple): A tuple containing two lists, each specifying the x and y axis limits respectively.
    slice_loc (np.ndarray): The coordinates to condition on.
    idcs (tuple, optional): The indices of the plane to condition on. Default is (0, 1).
    n_grid (int, optional): Number of grid points for the x and y axes. Default is 100.
    alpha (float, optional): Transparency level of the contour plot. Default is 0.6.
    n_levels (int, optional): Number of contour levels to plot. Default is 20.
    cmap (sns.palettes._ColorPalette, optional): Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
    plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            point = slice.copy()
            point[idcs_plane[0]] = X[i, j]
            point[idcs_plane[1]] = Y[i, j]
            Z[i, j] = np.exp(distribution.log_density(point))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax

def plot_pfd_contour_marginal(
        samples: np.ndarray,
        ax: plt.Axes,
        idcs: tuple[int, int] = (0, 1),
        alpha: float = 0.6,
        n_levels: int = 15,
        cmap: matplotlib.colors.ListedColormap = sns.color_palette('rocket', as_cmap=True),
        **kde_kwargs
) -> plt.Axes:
    """
    Plot the probability density function (PDF) contours of a multivariate normal distribution.

    Parameters:
    distribution (MultivariateNormal): The multivariate normal distribution to plot.
    ax (plt.Axes): The matplotlib axes object to plot on.
    idcs (tuple, optional): The indices of the dimensions to plot. Default is (0, 1).
    alpha (float, optional): Transparency level of the contour plot. Default is 0.6.
    n_levels (int, optional): Number of contour levels to plot. Default is 20.
    cmap (matplotlib.colors.ListedColormap, optional): Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
    plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    sns.kdeplot(x=samples[:, idcs[0]], y=samples[:, idcs[1]],
                ax=ax, cmap=cmap, levels=n_levels, alpha=alpha, zorder=1, **kde_kwargs)

    return ax

def get_sample(dim, rng=None):
    if rng is None:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), seed=datetime.now().microsecond)
    else:
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim), rng=rng)
    return dist.get_sample()


if __name__ == '__main__':

    rng = np.random.default_rng(0)

    x = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    x = np.stack((x[0].flatten(), x[1].flatten()), axis=1)

    # Define distributions
    mean = np.array([0., 0.])
    cov = np.array([[1., 0.5], [0.5, 1.]])


    old = MultivariateNormal(mean, cov, rng=rng)

    y_old = old.log_density(x).reshape(100, 100)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].contourf(x[:,0].reshape(100, 100), x[:,1].reshape(100, 100), y_old)
    ax[0].set_title('Old')
    plt.show()

    # get a random mean vector and covariance matrix with d dimensions
    d = 100
    mean = rng.uniform(-5, 5, d)
    cov = rng.uniform(-5, 5, (d, d))
    cov = cov @ cov.T

    # get n test inputs
    n = 1000
    x = rng.uniform(-5, 5, (n, d))

    old = MultivariateNormal(mean, cov, rng=rng)

    exec_old = timeit.timeit('old.log_density(x)', globals=globals(), number=1000)
    print(f"Old: {exec_old}")


    # from scipy.linalg import cho_factor, cho_solve
    # import timeit
    #
    # # Generate a random positive-definite matrix A
    # np.random.seed(42)
    # n = 1000  # Size of the matrix
    # A = np.random.randn(n, n)
    # A = np.dot(A, A.T)  # Make A symmetric positive definite
    #
    # # Generate a random vector x
    # x = np.random.randn(n)
    #
    # # Precompute the inverse of A
    # A_inv = np.linalg.inv(A)
    #
    # # Precompute the Cholesky decomposition of A
    # L = np.linalg.cholesky(A)
    #
    # # Method 1: Using np.matmul and A_inv
    # def matmul_method():
    #     return np.matmul(x.T, np.matmul(A_inv, x))
    #
    # # Method 2: Solving L y = x and computing the square norm
    # def cholesky_method():
    #     y = np.linalg.solve(L, x)
    #     return np.dot(y, y)
    #
    # # Method 2 (alternative): Using scipy's cho_solve
    # def cho_solve_method():
    #     y = cho_solve((L, True), x)
    #     return np.dot(y, y)
    #
    # # Timing the two methods
    # matmul_time = timeit.timeit(matmul_method, number=100)
    # cholesky_time = timeit.timeit(cholesky_method, number=100)
    # cho_solve_time = timeit.timeit(cho_solve_method, number=100)
    #
    # # Results
    # print(f"Method 1 (Matmul with A_inv): {matmul_time:.6f} seconds")
    # print(f"Method 2 (Solving L y = x): {cholesky_time:.6f} seconds")
    # print(f"Method 2 (cho_solve from scipy): {cho_solve_time:.6f} seconds")

    # ---------------------------- test visualization ----------------------------
    # normal 2d
    rng = np.random.default_rng(0)
    mean, cov = np.array([0, 0]), np.array([[1, 0.3], [0.3, 1.]])
    posterior = MultivariateNormal(mean, cov, rng=rng)
    plot_limits = ([-3, 3], [-3, 3])

    fig, ax = get_2d_despined_figure(plot_limits, figsize=(5, 3.5))
    plot_pdf_contours(posterior, ax, plot_limits)

    n_samples = 5000

    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        samples[i] = posterior.get_sample()

    plot_samples(samples, ax, color_code=True)

    plt.show()
