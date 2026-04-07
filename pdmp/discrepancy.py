"""Kennedy–O'Hagan model discrepancy: kernel functions and likelihood.

Implements the marginalized K&O likelihood where the observation model is

    y = eta(x, theta) + delta(x) + epsilon

with delta ~ GP(0, sigma2_delta * C_delta(rho)) and epsilon ~ N(0, sigma2_eps * I).
After marginalizing delta, the likelihood covariance becomes

    Sigma = sigma2_delta * C_delta(rho) + sigma2_eps * I

The sampler operates on the extended parameter vector
    ARD:       [theta, log_sigma2_delta, log_sigma2_eps, log_rho_1, ..., log_rho_d]
    isotropic: [theta, log_sigma2_delta, log_sigma2_eps, log_rho]
so that all hyperparameters are unconstrained.
"""

from typing import Union

import numpy as np
import scipy.linalg as sla

from pdmp import logger
from pdmp.distributions import Likelihood, _safe_cholesky
from pdmp.forward_model import Model
from pdmp.kernels import (  # noqa: F401 — re-exported for backward compat
    rbf_kernel_matrix,
    rbf_kernel_matrix_drho,
    _squared_diff_per_dim,
)


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
                 seed: int = None,
                 n_components: int = 1,
                 kernel: str = "ard"):
        """
        Args:
            model: Forward model providing eval/eval_grad.
            u_obs: (n_settings, m) observation matrix.
            x_locs: (m, d_x) or (m,) sensor locations. When n_components > 1,
                    x_locs contains only the *base* locations (m_base, d_x)
                    where m_base = m // n_components.
            psi_prior: Distribution over psi = [log_s2d, log_s2e, log_rho...].
                       Stored as attribute for prior augmentation in get_target.
            rng: Random number generator.
            seed: RNG seed.
            n_components: Number of independent output components sharing the
                          same spatial kernel. When > 1 the full covariance is
                          block_diag(C_z, ..., C_z) with n_components blocks,
                          so different components at the same sensor are exactly
                          independent while spatial correlation is preserved.
            kernel: ``"ard"`` (default) — one length-scale per spatial dimension;
                    ``"isotropic"`` — a single shared length-scale.
        """
        super().__init__(rng=rng, seed=seed)
        self._model = model
        self._n_theta = model.get_dim_in()
        self._u_obs = np.atleast_2d(u_obs)
        self.n_obs = self._u_obs.shape[0]
        self._m = self._u_obs.shape[1]  # measurement dimension per setting
        self._n_components = n_components

        if kernel not in ("ard", "isotropic"):
            raise ValueError(f"kernel must be 'ard' or 'isotropic', got '{kernel}'.")
        self._isotropic = kernel == "isotropic"

        # Sensor locations — ensure 2D
        self._x_locs = np.atleast_2d(x_locs)
        if self._x_locs.shape[0] == 1 and self._x_locs.shape[1] > 1:
            # Row vector → treat as column (m locations in 1D)
            self._x_locs = self._x_locs.T
        self._d_x = self._x_locs.shape[1]

        if n_components > 1:
            m_base = self._x_locs.shape[0]
            if self._m % n_components != 0:
                raise ValueError(
                    f"m={self._m} must be divisible by n_components={n_components}.")
            if m_base != self._m // n_components:
                raise ValueError(
                    f"With n_components={n_components}, x_locs must have "
                    f"{self._m // n_components} rows (got {m_base}).")
        else:
            if self._x_locs.shape[0] != self._m:
                raise ValueError(
                    f"Number of sensor locations ({self._x_locs.shape[0]}) must "
                    f"match measurement dimension ({self._m}).")

        self._n_rho = 1 if self._isotropic else self._d_x
        self._n_psi = 2 + self._n_rho  # log_s2d, log_s2e, log_rho...
        self._constant = -0.5 * self._m * np.log(2.0 * np.pi)

        # Pre-compute per-dimension squared differences on the base locations.
        # Shape (d_x, m_base, m_base) where m_base = m // n_components (or m).
        self._sq_diff = _squared_diff_per_dim(self._x_locs)

        # Store psi prior for use by get_target()
        self.psi_prior = psi_prior

    @property
    def dim(self) -> int:
        return self._m

    # ----- helpers -----

    def _split_params(self, params):
        """Split [theta, log_psi] and exponentiate psi.

        Returns rho with shape (n_rho,): (1,) for isotropic, (d_x,) for ARD.
        """
        theta = params[:self._n_theta]
        log_psi = params[self._n_theta:]
        sigma2_delta = np.exp(log_psi[0])
        sigma2_eps = np.exp(log_psi[1])
        rho = np.exp(log_psi[2:])
        return theta, sigma2_delta, sigma2_eps, rho

    def _expand_rho(self, rho: np.ndarray) -> np.ndarray:
        """Expand rho to shape (d_x,) for kernel evaluation.

        For isotropic kernels rho has shape (1,) and is broadcast to all
        spatial dimensions.  For ARD kernels rho is returned unchanged.
        """
        if self._isotropic:
            return np.repeat(rho, self._d_x)
        return rho

    def _build_cholesky(self, sigma2_delta, sigma2_eps, rho):
        """Build Sigma and return (L, Sigma_inv, C_delta, C_z).

        rho: (n_rho,) — (1,) for isotropic, (d_x,) for ARD.
        C_z is the base (m_base × m_base) kernel matrix.  When n_components=1
        C_delta == C_z; otherwise C_delta = block_diag(C_z, ..., C_z).
        """
        C_z = rbf_kernel_matrix(self._x_locs, self._expand_rho(rho))
        if self._n_components > 1:
            C_delta = sla.block_diag(*[C_z] * self._n_components)
        else:
            C_delta = C_z
        Sigma = sigma2_delta * C_delta + sigma2_eps * np.eye(self._m)
        L, _ = _safe_cholesky(Sigma, name="KO_Sigma")
        Sigma_inv = sla.cho_solve((L, True), np.eye(self._m))
        return L, Sigma_inv, C_delta, C_z

    # ----- log_density -----

    def log_density(self, params: np.ndarray,
                    idx: int = None) -> Union[float, np.ndarray]:
        theta, sigma2_delta, sigma2_eps, rho = self._split_params(params)
        L, _, _, _ = self._build_cholesky(sigma2_delta, sigma2_eps, rho)
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
        L, Sigma_inv, C_delta, C_z = self._build_cholesky(sigma2_delta,
                                                           sigma2_eps, rho)

        settings = range(self.n_obs) if idx is None else [idx]

        grad_theta = np.zeros(self._n_theta)
        grad_log_s2d = 0.0
        grad_log_s2e = 0.0
        grad_log_rho = np.zeros(self._n_rho)

        # Pre-compute traces (shared across settings)
        # tr(Sigma_inv @ C_delta) via Frobenius inner product
        tr_SinvC = np.sum(Sigma_inv * C_delta)
        tr_Sinv = np.trace(Sigma_inv)

        # Per-dimension kernel derivatives and their traces.
        # For the block-diagonal case dC/d(rho_k) is also block-diagonal with
        # blocks equal to the base-kernel derivative.
        # For isotropic kernels, aggregate all d_x per-dimension matrices into
        # one (dC/d(rho) = sum_k dC/d(rho_k)) so the gradient loop is unified.
        dC_per_dim = []
        for k in range(self._d_x):
            if self._n_components > 1:
                dCk_z = -self._sq_diff[k] * C_z  # (m_base, m_base)
                dCk = sla.block_diag(*[dCk_z] * self._n_components)
            else:
                dCk = -self._sq_diff[k] * C_delta  # dC/d(rho_k)
            dC_per_dim.append(dCk)

        if self._isotropic:
            dC_drho = [sum(dC_per_dim)]          # single (m, m) aggregate
        else:
            dC_drho = dC_per_dim                 # d_x (m, m) matrices

        tr_Sinv_dC = np.array([np.sum(Sigma_inv * dCk) for dCk in dC_drho])
        rho_raw = rho  # (n_rho,)

        for i in settings:
            r = self._u_obs[i] - self._model.eval(theta, idx=i)
            alpha = Sigma_inv @ r

            # theta gradient
            grad_theta += self._theta_grad_single(theta, alpha, i)

            # psi gradients (in log-space, chain rule: d/d(log x) = x * d/dx)
            aCa = alpha @ C_delta @ alpha
            grad_log_s2d += sigma2_delta * 0.5 * (-tr_SinvC + aCa)
            grad_log_s2e += sigma2_eps * 0.5 * (-tr_Sinv + alpha @ alpha)

            for ki, dCk in enumerate(dC_drho):
                adCa = alpha @ dCk @ alpha
                grad_log_rho[ki] += (rho_raw[ki] * sigma2_delta * 0.5 *
                                     (-tr_Sinv_dC[ki] + adCa))

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
