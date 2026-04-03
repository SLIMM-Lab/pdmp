"""Kennedy–O'Hagan model discrepancy: kernel functions and likelihood.

Implements the marginalized K&O likelihood where the observation model is

    y = eta(x, theta) + delta(x) + epsilon

with delta ~ GP(0, sigma2_delta * C_delta(rho)) and epsilon ~ N(0, sigma2_eps * I).
After marginalizing delta, the likelihood covariance becomes

    Sigma = sigma2_delta * C_delta(rho) + sigma2_eps * I

The sampler operates on the extended parameter vector
    [theta, log_sigma2_delta, log_sigma2_eps, log_rho_1, ..., log_rho_d]
so that all hyperparameters are unconstrained.
"""

from typing import Union

import numpy as np
import scipy.linalg as sla

from pdmp import logger
from pdmp.distributions import Likelihood, _safe_cholesky
from pdmp.forward_model import Model


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def rbf_kernel_matrix(x_locs: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ARD squared-exponential (RBF) kernel matrix.

    C_ij = exp(-sum_k rho_k (x_ik - x_jk)^2)

    Args:
        x_locs: (m, d_x) sensor locations.
        rho: (d_x,) inverse squared length-scale per dimension.

    Returns:
        (m, m) positive-definite kernel matrix.
    """
    x_locs = np.atleast_2d(x_locs)
    rho = np.atleast_1d(rho).astype(float)
    # Weighted squared distances: sum_k rho_k (x_ik - x_jk)^2
    # Compute per-dimension squared differences scaled by rho
    diff = x_locs[:, np.newaxis, :] - x_locs[np.newaxis, :, :]  # (m, m, d_x)
    sq_dist = np.sum(rho[np.newaxis, np.newaxis, :] * diff**2, axis=2)  # (m, m)
    return np.exp(-sq_dist)


def _squared_diff_per_dim(x_locs: np.ndarray) -> np.ndarray:
    """Per-dimension squared difference matrices.

    Args:
        x_locs: (m, d_x) sensor locations.

    Returns:
        (d_x, m, m) array where [k, i, j] = (x_ik - x_jk)^2.
    """
    x_locs = np.atleast_2d(x_locs)
    diff = x_locs[:, np.newaxis, :] - x_locs[np.newaxis, :, :]  # (m, m, d_x)
    return np.transpose(diff**2, (2, 0, 1))  # (d_x, m, m)


def rbf_kernel_matrix_drho(x_locs: np.ndarray, rho: np.ndarray,
                           k: int) -> np.ndarray:
    """Derivative of the RBF kernel matrix w.r.t. rho_k.

    dC/d(rho_k) = -D_k * C   (element-wise)

    where D_k[i,j] = (x_ik - x_jk)^2.

    Args:
        x_locs: (m, d_x) sensor locations.
        rho: (d_x,) inverse squared length-scales.
        k: dimension index.

    Returns:
        (m, m) derivative matrix.
    """
    C = rbf_kernel_matrix(x_locs, rho)
    D_k = _squared_diff_per_dim(x_locs)[k]
    return -D_k * C


def build_noise_covariance(x_locs: np.ndarray, rho: np.ndarray,
                           sigma2_delta: float,
                           sigma2_eps: float) -> np.ndarray:
    """Build the full observation covariance.

    Sigma = sigma2_delta * C_delta(rho) + sigma2_eps * I

    Args:
        x_locs: (m, d_x) sensor locations.
        rho: (d_x,) inverse squared length-scales.
        sigma2_delta: discrepancy variance.
        sigma2_eps: noise variance.

    Returns:
        (m, m) positive-definite covariance matrix.
    """
    C = rbf_kernel_matrix(x_locs, rho)
    m = C.shape[0]
    return sigma2_delta * C + sigma2_eps * np.eye(m)


# ---------------------------------------------------------------------------
# K&O Gaussian likelihood
# ---------------------------------------------------------------------------

class KOGaussianLikelihood(Likelihood):
    """Gaussian likelihood with Kennedy–O'Hagan model discrepancy.

    The observation model (per setting i) is:
        y_i | theta, psi ~ N(eta_i(theta), Sigma(psi))

    where Sigma = sigma2_delta * C_delta(rho) + sigma2_eps * I and the
    discrepancy GP has been marginalized analytically.

    The class operates on the extended parameter vector
        params = [theta, log_sigma2_delta, log_sigma2_eps, log_rho_1, ...]
    """

    def __init__(self,
                 model: Model,
                 u_obs: np.ndarray,
                 x_locs: np.ndarray,
                 psi_prior,
                 rng: np.random.Generator = None,
                 seed: int = None):
        """
        Args:
            model: Forward model providing eval/eval_grad.
            u_obs: (n_settings, m) observation matrix.
            x_locs: (m, d_x) or (m,) sensor locations.
            psi_prior: Distribution over psi = [log_s2d, log_s2e, log_rho...].
                       Stored as attribute for prior augmentation in get_target.
            rng: Random number generator.
            seed: RNG seed.
        """
        super().__init__(rng=rng, seed=seed)
        self._model = model
        self._n_theta = model.get_dim_in()
        self._u_obs = np.atleast_2d(u_obs)
        self.n_obs = self._u_obs.shape[0]
        self._m = self._u_obs.shape[1]  # measurement dimension per setting

        # Sensor locations — ensure 2D
        self._x_locs = np.atleast_2d(x_locs)
        if self._x_locs.shape[0] == 1 and self._x_locs.shape[1] > 1:
            # Row vector → treat as column (m locations in 1D)
            self._x_locs = self._x_locs.T
        self._d_x = self._x_locs.shape[1]

        if self._x_locs.shape[0] != self._m:
            raise ValueError(
                f"Number of sensor locations ({self._x_locs.shape[0]}) must "
                f"match measurement dimension ({self._m}).")

        self._n_psi = 2 + self._d_x  # log_s2d, log_s2e, log_rho_1..d
        self._constant = -0.5 * self._m * np.log(2.0 * np.pi)

        # Pre-compute per-dimension squared differences (fixed geometry)
        self._sq_diff = _squared_diff_per_dim(self._x_locs)  # (d_x, m, m)

        # Store psi prior for use by get_target()
        self.psi_prior = psi_prior

    @property
    def dim(self) -> int:
        return self._m

    # ----- helpers -----

    def _split_params(self, params):
        """Split [theta, log_psi] and exponentiate psi."""
        theta = params[:self._n_theta]
        log_psi = params[self._n_theta:]
        sigma2_delta = np.exp(log_psi[0])
        sigma2_eps = np.exp(log_psi[1])
        rho = np.exp(log_psi[2:])
        return theta, sigma2_delta, sigma2_eps, rho

    def _build_cholesky(self, sigma2_delta, sigma2_eps, rho):
        """Build Sigma and return (L, Sigma_inv, C_delta)."""
        C_delta = rbf_kernel_matrix(self._x_locs, rho)
        Sigma = sigma2_delta * C_delta + sigma2_eps * np.eye(self._m)
        L, _ = _safe_cholesky(Sigma, name="KO_Sigma")
        # Compute Sigma_inv via Cholesky solve
        Sigma_inv = sla.cho_solve((L, True), np.eye(self._m))
        return L, Sigma_inv, C_delta

    # ----- log_density -----

    def log_density(self, params: np.ndarray,
                    idx: int = None) -> Union[float, np.ndarray]:
        theta, sigma2_delta, sigma2_eps, rho = self._split_params(params)
        L, _, _ = self._build_cholesky(sigma2_delta, sigma2_eps, rho)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        settings = range(self.n_obs) if idx is None else [idx]
        ll = 0.0
        for i in settings:
            r = self._u_obs[i] - self._model.eval(theta, idx=i)
            alpha = sla.cho_solve((L, True), r)
            ll += self._constant - 0.5 * log_det - 0.5 * r @ alpha
        return ll

    # ----- grad_log_density -----

    def _theta_grad_single(self, theta, alpha_i, i):
        """Gradient of log L_i w.r.t. theta: J_i^T alpha_i."""
        if hasattr(self._model, 'linearize'):
            _, vjp_fun = self._model.linearize(theta, idx=i)
            return np.asarray(vjp_fun(alpha_i), dtype=float)
        if hasattr(self._model, 'eval_vjp'):
            return np.asarray(self._model.eval_vjp(theta, idx=i, v=alpha_i),
                              dtype=float)
        J = self._model.eval_grad(theta, idx=i)  # (m, n_theta)
        return J.T @ alpha_i

    def grad_log_density(self, params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        theta, sigma2_delta, sigma2_eps, rho = self._split_params(params)
        L, Sigma_inv, C_delta = self._build_cholesky(sigma2_delta, sigma2_eps,
                                                     rho)

        settings = range(self.n_obs) if idx is None else [idx]

        grad_theta = np.zeros(self._n_theta)
        grad_log_s2d = 0.0
        grad_log_s2e = 0.0
        grad_log_rho = np.zeros(self._d_x)

        # Pre-compute traces (shared across settings)
        # tr(Sigma_inv @ C_delta) via Frobenius inner product
        tr_SinvC = np.sum(Sigma_inv * C_delta)
        tr_Sinv = np.trace(Sigma_inv)

        # Per-dimension kernel derivatives and their traces
        dC_drho = []  # list of (m, m)
        tr_Sinv_dC = np.zeros(self._d_x)
        for k in range(self._d_x):
            dCk = -self._sq_diff[k] * C_delta  # dC/d(rho_k)
            dC_drho.append(dCk)
            tr_Sinv_dC[k] = np.sum(Sigma_inv * dCk)

        for i in settings:
            r = self._u_obs[i] - self._model.eval(theta, idx=i)
            alpha = Sigma_inv @ r

            # theta gradient
            grad_theta += self._theta_grad_single(theta, alpha, i)

            # psi gradients (in log-space, chain rule: d/d(log x) = x * d/dx)
            aCa = alpha @ C_delta @ alpha
            grad_log_s2d += sigma2_delta * 0.5 * (-tr_SinvC + aCa)
            grad_log_s2e += sigma2_eps * 0.5 * (-tr_Sinv + alpha @ alpha)

            for k in range(self._d_x):
                adCa = alpha @ dC_drho[k] @ alpha
                grad_log_rho[k] += (rho[k] * sigma2_delta * 0.5 *
                                    (-tr_Sinv_dC[k] + adCa))

        grad_psi = np.concatenate(
            [[grad_log_s2d, grad_log_s2e], grad_log_rho])
        return np.concatenate([grad_theta, grad_psi])

    # ----- hessian_log_density -----

    def hessian_log_density(self, params: np.ndarray,
                            idx: int = None,
                            h: float = 1e-5) -> np.ndarray:
        """Hessian via central finite differences of grad_log_density."""
        n = self._n_theta + self._n_psi
        hess = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            g_fwd = self.grad_log_density(params + h * e_j, idx=idx)
            g_bwd = self.grad_log_density(params - h * e_j, idx=idx)
            hess[:, j] = (g_fwd - g_bwd) / (2.0 * h)
        return 0.5 * (hess + hess.T)
