import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize

from datetime import datetime
from timeit import timeit
from typing import Union, Any, cast
from typing_extensions import override

from pdmp import logger
from pdmp.forward_model import Model
from pdmp.project_field import get_gaussian_random_field_projection_from_dict, get_gaussian_random_field_projection_norm_from_dict


small = 1e-12
large = 1e20


def _safe_cholesky(
    A: np.ndarray,
    *,
    jitter0: float = 1e-12,
    max_tries: int = 8,
    symmetrize: bool = True,
    check_finite: bool = False,
    name: str = "matrix",
) -> tuple[np.ndarray, float]:
    """Compute a numerically robust Cholesky factor.

    This is mainly to stabilize Cholesky factorizations that arise from
    discretized random-field covariances / curvature matrices that can become
    nearly singular for fine meshes.

    Strategy:
      1) Optional symmetrization: (A + A.T)/2
      2) Try Cholesky(A + jitter * I) with increasing jitter
      3) Fallback: eigenvalue clipping (project to SPD-ish)

    Returns:
        L: lower-triangular Cholesky factor
        jitter: final diagonal jitter added (may be 0)
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"_safe_cholesky expects a square 2D array, got shape {A.shape}")

    if symmetrize:
        A = 0.5 * (A + A.T)

    n = A.shape[0]
    I = np.eye(n)

    # scale jitter by typical diagonal magnitude to make it dimensionless/robust
    diag = np.diag(A)
    diag_scale = float(np.mean(diag)) if diag.size else 1.0
    diag_scale = max(abs(diag_scale), 1.0)

    last_err: Exception | None = None
    for k in range(max_tries):
        jitter = (jitter0 * (10.0 ** k)) * diag_scale
        try:
            L = sp.linalg.cholesky(
                A + jitter * I,
                lower=True,
                check_finite=check_finite,
                overwrite_a=False,
            )
            if jitter > 0:
                logger.debug(f"Cholesky stabilized for {name} with jitter={jitter:.3e}")
            return L, float(jitter)
        except Exception as e:  # LinAlgError (and friends)
            last_err = e

    # Fallback: eigenvalue clipping
    w, V = np.linalg.eigh(A)
    floor = float(jitter0 * diag_scale)
    w_clipped = np.maximum(w, floor)
    A_spd = (V * w_clipped) @ V.T
    A_spd = 0.5 * (A_spd + A_spd.T)

    try:
        L = sp.linalg.cholesky(A_spd, lower=True, check_finite=check_finite)
        logger.warning(
            f"Cholesky failed for {name} after jitter attempts; used eigenvalue clipping "
            f"with floor={floor:.3e}. Original error: {last_err}"
        )
        return L, float(floor)
    except Exception as e:
        raise np.linalg.LinAlgError(
            f"Cholesky failed for {name} even after jitter and eigenvalue clipping. "
            f"Last errors: {last_err} / {e}"
        )


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


class JointDistribution(Distribution):
    """Independent joint of multiple distributions."""
    def __init__(self, distributions: list[Distribution], *, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)
        self.distributions = distributions
        self._dims = [dist.dim for dist in distributions]
        self._dim = sum(self._dims)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        return np.concatenate([dist.mean for dist in self.distributions])

    @property
    def cov(self) -> np.ndarray:
        return sp.linalg.block_diag(*[dist.cov for dist in self.distributions])

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        samples = [dist.get_sample(n) for dist in self.distributions]
        if n == 1:
            parts = [s if s.ndim == 1 else s.reshape(-1) for s in samples]
            return np.concatenate(parts)
        else:
            rows = [(s if s.ndim == 2 else s.reshape(n, -1)) for s in samples]
            return np.hstack(rows)

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            total = np.array(0.0)
            idx = 0
            for dist, d in zip(self.distributions, self._dims):
                total += dist.log_density(x[idx:idx + d])
                idx += d
            return total
        # batch of points
        out = np.zeros(x.shape[0])
        starts = np.cumsum([0] + self._dims[:-1])
        for dist, d, start in zip(self.distributions, self._dims, starts):
            out += dist.log_density(x[:, start:start + d])
        return out

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            grads = []
            idx = 0
            for dist, d in zip(self.distributions, self._dims):
                grads.append(dist.grad_log_density(x[idx:idx + d]))
                idx += d
            return np.concatenate(grads)
        # batch of points
        return np.vstack([self.grad_log_density(xi) for xi in x])

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            blocks = []
            idx = 0
            for dist, d in zip(self.distributions, self._dims):
                blocks.append(dist.hessian_log_density(x[idx:idx + d]))
                idx += d
            return sp.linalg.block_diag(*blocks)
        # batch of points
        return np.array([self.hessian_log_density(xi) for xi in x])

def stable_cholesky_and_inv(cov: np.ndarray, *, jitter0: float = 1e-12, max_tries: int = 8):
    """
    Compute a numerically stable Cholesky factor and inverse using diagonal jitter.

    - Tries Cholesky(cov + jitter * I) with increasing jitter.
    - If that still fails, falls back to eigenvalue clipping (SPD projection).

    Returns:
        L: lower-triangular Cholesky factor
        invC: (regularized) inverse computed via cho_solve
        jitter: final jitter added (0 if none)
    """
    cov = np.asarray(cov, dtype=float)

    # Symmetrize to reduce numerical asymmetry from assembly/roundoff
    cov = 0.5 * (cov + cov.T)

    n = cov.shape[0]
    I = np.eye(n)

    # Scale jitter by typical diagonal magnitude (dimensionless robustness)
    diag_scale = float(np.mean(np.diag(cov))) if n > 0 else 1.0
    diag_scale = max(diag_scale, 1.0)

    jitter = 0.0
    for k in range(max_tries):
        jitter = (jitter0 * (10.0 ** k)) * diag_scale
        try:
            L = np.linalg.cholesky(cov + jitter * I)
            invC = sp.linalg.cho_solve((L, True), I, check_finite=False)
            return L, invC, jitter
        except np.linalg.LinAlgError:
            pass

    # Fallback: eigenvalue clipping (nearest SPD-ish)
    w, V = np.linalg.eigh(cov)
    # Floor eigenvalues relative to scale
    floor = jitter0 * diag_scale
    w_clipped = np.maximum(w, floor)
    cov_spd = (V * w_clipped) @ V.T
    cov_spd = 0.5 * (cov_spd + cov_spd.T)

    L = np.linalg.cholesky(cov_spd)
    invC = sp.linalg.cho_solve((L, True), I, check_finite=False)
    return L, invC, float(max(0.0, floor))


# Usage in your class:
# self.cov_L, self.inv_C, self._jitter = stable_cholesky_and_inv(self._cov)

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
                 cov: np.ndarray = None,
                 prec: np.ndarray = None,
                 rng: np.random.Generator = None,
                 seed: int = None):
        super().__init__(rng=rng, seed=seed)

        self._mean = np.atleast_1d(mean)
        self._dim = len(self._mean)

        if (cov is None) == (prec is None):
            raise ValueError("Exactly one of 'cov' or 'prec' must be provided.")

        if cov is not None:
            # Covariance given: use numerically stable Cholesky + inverse.
            self._cov = np.atleast_2d(cov)
            if self._cov.shape != (self._dim, self._dim):
                raise ValueError("Shape mismatch between mean and cov.")
            self.cov_L, self.inv_C, self._jitter = stable_cholesky_and_inv(self._cov)
        else:
            # Precision given: avoid forming inv(prec) explicitly.
            prec = np.atleast_2d(prec)
            if prec.shape != (self._dim, self._dim):
                raise ValueError("Shape mismatch between mean and prec.")
            # Symmetrize for numerical robustness
            prec = 0.5 * (prec + prec.T)
            # Cholesky factor of precision: P = R^T R  (upper-triangular R)
            R = np.linalg.cholesky(prec)
            # Covariance C = P^{-1} = R^{-1} (R^{-1})^T
            self.cov_L = sp.linalg.solve_triangular(
                R, np.eye(self._dim), lower=False, check_finite=False
            )
            # Inverse covariance is just the precision itself
            self.inv_C = prec
            self._jitter = 0.0
            # Lazily construct covariance matrix for interface compatibility
            self._cov = self.cov_L @ self.cov_L.T

        # log|C| where C = cov; since cov_L is Cholesky factor, |C| = (prod diag(L))^2
        self.log_det = 2.0 * np.log(self.cov_L.diagonal()).sum()
        self.constant = -0.5 * np.log(2.0 * np.pi) * self._dim

    @classmethod
    def from_dict(cls,
                  params: dict[str, np.ndarray],
                  rng: np.random.Generator = None,
                  seed: int = None):
        if 'mean' not in params:
            raise ValueError("Parameters must include 'mean'.")
        if ('cov' in params) == ('prec' in params):
            raise ValueError("Parameters must include exactly one of 'cov' or 'prec'.")
        return cls(
            mean=params['mean'],
            cov=params.get('cov', None),
            prec=params.get('prec', None),
            rng=rng,
            seed=seed,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        # Always return a full covariance matrix; kept for backwards compatibility.
        return self._cov

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        if n == 1:
            z = self.rng.standard_normal(size=self._dim)
            return self.cov_L @ z + self._mean
        else:
            z = self.rng.standard_normal(size=(n, self._dim))
            return z @ self.cov_L.T + self._mean

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self._mean
        if diff.ndim == 1:
            if self._dim == 1:
                return self.constant - 0.5 * self.log_det - 0.5 * np.abs(
                    diff / self.cov_L[0, 0]
                ) ** 2
            else:
                y = sp.linalg.solve_triangular(
                    self.cov_L, diff, lower=True, check_finite=False
                )
                return self.constant - 0.5 * self.log_det - 0.5 * np.dot(y, y)
        else:
            # batch of points
            y = sp.linalg.solve_triangular(
                self.cov_L, diff.T, lower=True, check_finite=False
            )
            quad = np.sum(y * y, axis=0)
            return (self.constant - 0.5 * self.log_det - 0.5 * quad).T

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self._mean
        return -sp.linalg.solve_triangular(
            self.cov_L.T,
            sp.linalg.solve_triangular(
                self.cov_L, diff, lower=True, check_finite=False
            ),
            lower=False,
            check_finite=False,
        )

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


class GammaDistribution(Distribution):
    """Gamma distribution with shape alpha and scale beta."""
    def __init__(
        self,
        alpha: float,
        beta: float,
        rng: np.random.Generator = None,
        seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.alpha = alpha
        self.beta = beta
        self._dim = 1
        self._mean = alpha * beta
        self._cov = np.array([[alpha * beta**2]])
        self._log_norm = alpha * np.log(beta) + sp.special.gammaln(alpha)

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Union[float, np.ndarray]],
        rng: np.random.Generator = None,
        seed: int = None
    ):
        if 'alpha' not in params or 'beta' not in params:
            raise ValueError("Parameters must include 'alpha' and 'beta'.")
        return cls(alpha=params['alpha'], beta=params['beta'], rng=rng, seed=seed)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        return np.array([self._mean])

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        samples = self.rng.gamma(self.alpha, self.beta, size=n)
        if n == 1:
            return np.array([samples])  # keep shape (1,)
        return samples.reshape(n, 1)

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        xv = np.atleast_1d(x).astype(float)
        # clamp to small positive
        xv[xv <= 0] = small
        return (self.alpha - 1) * np.log(xv) - xv / self.beta - self._log_norm

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        xv = float(x) if x.ndim == 0 else x.flatten()[0]
        # derivative w.r.t. x
        return np.array([(self.alpha - 1) / xv - 1 / self.beta])

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        xv = float(x) if x.ndim == 0 else x.flatten()[0]
        # second derivative
        return np.array([[-(self.alpha - 1) / (xv**2)]])


class BetaDistribution(Distribution):
    """Beta distribution with shape parameters alpha and beta."""
    def __init__(
        self,
        alpha: float,
        beta: float,
        rng: np.random.Generator = None,
        seed: int = None
    ):
        super().__init__(rng=rng, seed=seed)
        self.alpha = alpha
        self.beta = beta
        self._dim = 1
        # E[X] and Var[X]
        self._mean = alpha / (alpha + beta)
        self._cov = np.array([[alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))]])
        # log normalization constant = ln B(alpha,beta)
        self._log_norm = sp.special.betaln(alpha, beta)

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Union[float, np.ndarray]],
        rng: np.random.Generator = None,
        seed: int = None
    ):
        if 'alpha' not in params or 'beta' not in params:
            raise ValueError("Parameters must include 'alpha' and 'beta'.")
        return cls(alpha=params['alpha'], beta=params['beta'], rng=rng, seed=seed)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        return np.array([self._mean])

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @override
    def get_sample(self, n: int = 1) -> np.ndarray:
        samples = self.rng.beta(self.alpha, self.beta, size=n)
        if n == 1:
            return np.array([samples])
        return samples.reshape(n, 1)

    @override
    def log_density(self, x: np.ndarray) -> np.ndarray:
        xv = np.atleast_1d(x).astype(float)
        xv[xv <= 0] = small
        xv[xv >= 1] = 1 - small
        return ((self.alpha - 1) * np.log(xv)
                + (self.beta - 1) * np.log(1 - xv)
                - self._log_norm)

    @override
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        xv = float(x) if x.ndim == 0 else x.flatten()[0]
        xv = max(min(xv, 1 - small), small)
        return np.array([
            (self.alpha - 1) / xv - (self.beta - 1) / (1 - xv)
        ])

    @override
    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        xv = float(x) if x.ndim == 0 else x.flatten()[0]
        xv = max(min(xv, 1 - small), small)
        return np.array([[
            -(self.alpha - 1) / (xv**2)
            - (self.beta - 1) / ((1 - xv)**2)
        ]])


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
        """Gradient of log-likelihood w.r.t. parameters.

        Enhancement: if the underlying model exposes a `linearize(params, idx)` method
        returning (outputs, vjp_fun) where vjp_fun maps a vector of shape (d,) to the
        parameter gradient (p,), we use a single reverse-mode pass instead of forming
        the full Jacobian. If only an `eval_vjp(params, idx, v)` method is provided,
        we call that (may re-run the forward internally). Otherwise we fall back to
        forming the Jacobian via `eval_grad` and doing a matrix product.
        """

        def _grad_single(i: int) -> np.ndarray:
            # Prefer single forward + VJP closure
            if hasattr(self._model, 'linearize'):
                m, vjp_fun = self._model.linearize(params, idx=i)
                v = self._dists[i].grad_log_density(m)  # shape (d,)
                g = vjp_fun(v)  # shape (p,)
                return np.asarray(g, dtype=float)
            # Compute forward output once
            m = self._model.eval(params, idx=i)
            v = self._dists[i].grad_log_density(m)
            # If model supplies a direct J^T v path
            if hasattr(self._model, 'eval_vjp'):
                g = self._model.eval_vjp(params, idx=i, v=v)
                return np.asarray(g, dtype=float)
            # Fallback: form full Jacobian
            J = self._model.eval_grad(params, idx=i)  # shape (d, p)
            return np.asarray(v @ J, dtype=float)

        if idx is None:
            grad = np.zeros(self._n_params_, dtype=float)
            for i in range(self.n_obs):
                grad += _grad_single(i)
            return grad
        else:
            return _grad_single(idx)

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
    field=None,
) -> Distribution:
    """Get a prior distribution from a dictionary.

    Args:
        config: The configuration of the prior.
        rng: The random number generator. Default is None.
        field: Optional GaussianRandomField instance (used when name == 'FromField').

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
    elif config['name'] == 'Gamma':
        return GammaDistribution.from_dict(config, rng=rng)
    elif config['name'] == 'Beta':
        return BetaDistribution.from_dict(config, rng=rng)
    elif config['name'] == 'FromField':
        if field is None:
            raise ValueError("Prior 'FromField' requires a field instance.")
        return field.coefficient_distribution
    # this one is here for legacy reasons
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
        return (x - self._b) @ self._M_inv.T

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

            C, _ = _safe_cholesky(M, name="AffineTransformtion(M) for TransformedDistribution")

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

            C, _ = _safe_cholesky(M, name="AffineTransformtion(M) for TransformedLikelihood")

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

        if 'mean' in type(target).__dict__ and not isinstance(type(target).__dict__['mean'], property):
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
