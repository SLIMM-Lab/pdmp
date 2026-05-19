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
    rbf_kernel_matrix, rbf_kernel_matrix_drho, _squared_diff_per_dim,
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
                 psi_prior=None,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 n_components: int = 1,
                 n_groups: int = 1,
                 kernel: str = "ard",
                 fixed_psi: dict = None):
        """
        Args:
            model: Forward model providing eval/eval_grad.
            u_obs: (n_settings, m) observation matrix.
            x_locs: (n_groups * P, d_x) sensor locations.  When n_groups == 1
                    this is just (P, d_x) — the *base* locations shared by
                    all n_components blocks.  When n_groups > 1 the rows are
                    expected geom-major: rows ``[g*P : (g+1)*P]`` give the
                    P sensor coordinates for group g.
            psi_prior: Distribution over the *free* psi parameters (i.e. those
                       not listed in ``fixed_psi``).  May be ``None`` when all
                       psi parameters are fixed.  Stored as attribute for prior
                       augmentation in ``get_target``.
            rng: Random number generator.
            seed: RNG seed.
            n_components: Number of independent output components sharing the
                          same spatial kernel. When > 1 the full covariance is
                          block_diag(C_z, ..., C_z) with n_components blocks,
                          so different components at the same sensor are exactly
                          independent while spatial correlation is preserved.
            n_groups: Number of independent measurement groups (e.g. distinct
                      microstructure realizations) sharing the same KO
                      hyperparameters but with independent discrepancy GP
                      realizations.  When > 1 each ``n_components`` block is
                      itself split into ``n_groups`` sub-blocks of size
                      ``P = m / (n_components * n_groups)``.  ``x_locs`` is
                      expected with shape ``(n_groups * P, d_x)`` in geom-major
                      order; per-group kernels are evaluated on each
                      ``(P, d_x)`` slice.
            kernel: ``"ard"`` (default) — one length-scale per spatial dimension;
                    ``"isotropic"`` — a single shared length-scale.
            fixed_psi: Optional dict of ``{name: log_value}`` pairs that fix psi
                       parameters to deterministic values (in log-space, matching
                       the sampler's parameter vector).  Valid names are
                       ``"log_s2d"``, ``"log_s2e"``, and ``"log_rho"``
                       (isotropic / 1-D ARD) or ``"log_rho_0"``, ``"log_rho_1"``,
                       … (ARD with d_x > 1).  Parameters not listed are learned.
        """
        super().__init__(rng=rng, seed=seed)
        self._model = model
        self._n_theta = model.get_dim_in()
        self._u_obs = np.atleast_2d(u_obs)
        self.n_obs = self._u_obs.shape[0]
        self._m = self._u_obs.shape[1]  # measurement dimension per setting
        self._n_components = n_components
        self._n_groups = n_groups

        if kernel not in ("ard", "isotropic"):
            raise ValueError(
                f"kernel must be 'ard' or 'isotropic', got '{kernel}'.")
        self._isotropic = kernel == "isotropic"

        # Sensor locations — ensure 2D
        self._x_locs = np.atleast_2d(x_locs)
        if self._x_locs.shape[0] == 1 and self._x_locs.shape[1] > 1:
            # Row vector → treat as column (m locations in 1D)
            self._x_locs = self._x_locs.T
        self._d_x = self._x_locs.shape[1]

        # Total measurement count factors as m = n_components * n_groups * P
        denom = n_components * n_groups
        if self._m % denom != 0:
            raise ValueError(
                f"m={self._m} must be divisible by n_components*n_groups={denom}."
            )
        self._P = self._m // denom  # measurements per (component, group) block
        expected_x_rows = n_groups * self._P
        if self._x_locs.shape[0] != expected_x_rows:
            raise ValueError(
                f"With n_components={n_components}, n_groups={n_groups}, "
                f"x_locs must have {expected_x_rows} rows "
                f"(got {self._x_locs.shape[0]}).")

        self._n_rho = 1 if self._isotropic else self._d_x
        self._n_psi = 2 + self._n_rho  # total psi count (fixed + free)

        # Psi parameter names (used for fixed_psi look-up)
        if self._n_rho == 1:
            rho_names = ["log_rho"]
        else:
            rho_names = [f"log_rho_{k}" for k in range(self._n_rho)]
        self._psi_names = ["log_s2d", "log_s2e"] + rho_names

        # Build fixed-psi mask: NaN entries are free, finite entries are fixed.
        fixed_psi_dict = fixed_psi or {}
        unknown = set(fixed_psi_dict) - set(self._psi_names)
        if unknown:
            raise ValueError(f"Unknown psi parameters: {unknown}. "
                             f"Valid names: {self._psi_names}")
        self._fixed_log_psi = np.array([
            fixed_psi_dict[n] if n in fixed_psi_dict else np.nan
            for n in self._psi_names
        ])
        self._free_psi_mask = np.isnan(self._fixed_log_psi)
        self._n_psi_free = int(self._free_psi_mask.sum())

        self._constant = -0.5 * self._m * np.log(2.0 * np.pi)

        # Reshape sensor locations into per-group views and cache per-group
        # per-dim squared differences. Shapes:
        #   self._x_locs_g    : (n_groups, P, d_x)
        #   self._sq_diff_g   : (n_groups, d_x, P, P)
        self._x_locs_g = self._x_locs.reshape(n_groups, self._P, self._d_x)
        self._sq_diff_g = np.stack(
            [
                _squared_diff_per_dim(self._x_locs_g[g])
                for g in range(n_groups)
            ],
            axis=0,
        )

        # Store psi prior for use by get_target()
        self.psi_prior = psi_prior

    @property
    def dim(self) -> int:
        return self._m

    # ----- helpers -----

    def _fill_psi(self, free_log_psi: np.ndarray) -> np.ndarray:
        """Reconstruct full log_psi (length n_psi) from free entries only."""
        log_psi = self._fixed_log_psi.copy()
        log_psi[self._free_psi_mask] = free_log_psi
        return log_psi

    def _split_params(self, params):
        """Split [theta, free_log_psi] and exponentiate psi.

        Returns rho with shape (n_rho,): (1,) for isotropic, (d_x,) for ARD.
        """
        theta = params[:self._n_theta]
        log_psi = self._fill_psi(params[self._n_theta:])
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

    def _build_per_group(self, sigma2_delta, sigma2_eps, rho):
        """Build per-group kernel/Cholesky/inverse arrays.

        Returns three arrays each of shape (n_groups, P, P):
            L_g    : lower Cholesky factor of Σ_g = σ²_δ C_g(ρ) + σ²_ε I_P
            Sinv_g : Σ_g^{-1}
            C_g    : kernel matrix at group g's sensor locations

        Every (component d, group g) block of the full covariance shares
        these factors (only the realization of δ differs across blocks,
        not the hyperparameters).
        """
        rho_full = self._expand_rho(rho)
        n_groups = self._n_groups
        P = self._P
        eye_P = np.eye(P)
        C_g = np.empty((n_groups, P, P))
        L_g = np.empty((n_groups, P, P))
        Sinv_g = np.empty((n_groups, P, P))
        for g in range(n_groups):
            C_g[g] = rbf_kernel_matrix(self._x_locs_g[g], rho_full)
            Sigma_g = sigma2_delta * C_g[g] + sigma2_eps * eye_P
            L, _ = _safe_cholesky(Sigma_g, name="KO_Sigma_g")
            L_g[g] = L
            Sinv_g[g] = sla.cho_solve((L, True), eye_P)
        return L_g, Sinv_g, C_g

    def _per_group_dC_drho(self, C_g: np.ndarray) -> np.ndarray:
        """Per-group, per-ρ derivatives of the kernel.

        Returns shape (n_groups, n_rho, P, P) where entry [g, k] is
        dC_g/dρ_k (ARD) or sum over spatial dims of dC_g/dρ_full_k (isotropic).
        """
        n_groups = self._n_groups
        out = np.empty((n_groups, self._n_rho, self._P, self._P))
        for g in range(n_groups):
            per_dim = -self._sq_diff_g[g] * C_g[g]  # (d_x, P, P)
            if self._isotropic:
                out[g, 0] = per_dim.sum(axis=0)
            else:
                out[g] = per_dim
        return out

    # ----- log_density -----

    def log_density(self,
                    params: np.ndarray,
                    idx: int = None) -> Union[float, np.ndarray]:
        theta, sigma2_delta, sigma2_eps, rho = self._split_params(params)
        L_g, _, _ = self._build_per_group(sigma2_delta, sigma2_eps, rho)

        # log|Σ| = n_components · Σ_g 2·log diag(L_g).sum()
        log_det = self._n_components * 2.0 * np.sum(
            np.log(np.array([np.diag(L_g[g]) for g in range(self._n_groups)])))

        settings = range(self.n_obs) if idx is None else [idx]
        ll = 0.0
        for i in settings:
            r = self._u_obs[i] - self._model.eval(theta, idx=i)
            r_dgp = r.reshape(self._n_components, self._n_groups, self._P)
            quad = 0.0
            for g in range(self._n_groups):
                R = r_dgp[:, g, :].T  # (P, n_components)
                alpha = sla.cho_solve((L_g[g], True), R)  # (P, n_components)
                quad += float(np.sum(R * alpha))
            ll += self._constant - 0.5 * log_det - 0.5 * quad
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

    def grad_log_density(self,
                         params: np.ndarray,
                         idx: int = None) -> np.ndarray:
        theta, sigma2_delta, sigma2_eps, rho = self._split_params(params)
        L_g, Sinv_g, C_g = self._build_per_group(sigma2_delta, sigma2_eps, rho)
        dC_g = self._per_group_dC_drho(C_g)  # (n_groups, n_rho, P, P)

        n_components = self._n_components
        n_groups = self._n_groups
        P = self._P

        # Traces shared across components and settings.  For the full Σ
        # (block-diagonal in n_components × n_groups blocks of size P) we have
        #   tr(Σ^{-1} dΣ/dψ) = n_components · Σ_g tr(Σ_g^{-1} dΣ_g/dψ)
        tr_SinvC_g = np.array(
            [np.sum(Sinv_g[g] * C_g[g]) for g in range(n_groups)])
        tr_Sinv_g = np.array([np.trace(Sinv_g[g]) for g in range(n_groups)])
        tr_Sinv_dC_g = np.einsum('gpq,gkqp->gk', Sinv_g,
                                 dC_g)  # (n_groups, n_rho)

        sum_tr_SinvC = n_components * float(tr_SinvC_g.sum())
        sum_tr_Sinv = n_components * float(tr_Sinv_g.sum())
        sum_tr_Sinv_dC = n_components * tr_Sinv_dC_g.sum(axis=0)  # (n_rho,)

        settings = range(self.n_obs) if idx is None else [idx]

        grad_theta = np.zeros(self._n_theta)
        grad_log_s2d = 0.0
        grad_log_s2e = 0.0
        grad_log_rho = np.zeros(self._n_rho)

        for i in settings:
            r = self._u_obs[i] - self._model.eval(theta, idx=i)
            r_dgp = r.reshape(n_components, n_groups, P)

            alpha_full = np.empty(self._m)
            alpha_dgp = alpha_full.reshape(n_components, n_groups, P)

            aCa_sum = 0.0
            aa_sum = 0.0
            a_dC_a_sum = np.zeros(self._n_rho)

            for g in range(n_groups):
                R = r_dgp[:, g, :].T  # (P, n_components)
                Ag = Sinv_g[g] @ R  # (P, n_components)
                alpha_dgp[:, g, :] = Ag.T

                # α^T C_g α and α^T α, summed across n_components
                aCa_sum += float(np.sum(Ag * (C_g[g] @ Ag)))
                aa_sum += float(np.sum(Ag * Ag))
                # α^T dC_g/dρ_k α per ρ-index, summed across n_components
                a_dC_a_sum += np.einsum('pc,kpq,qc->k', Ag, dC_g[g], Ag)

            grad_theta += self._theta_grad_single(theta, alpha_full, i)

            grad_log_s2d += sigma2_delta * 0.5 * (-sum_tr_SinvC + aCa_sum)
            grad_log_s2e += sigma2_eps * 0.5 * (-sum_tr_Sinv + aa_sum)
            for k in range(self._n_rho):
                grad_log_rho[k] += (rho[k] * sigma2_delta * 0.5 *
                                    (-sum_tr_Sinv_dC[k] + a_dC_a_sum[k]))

        grad_psi_full = np.concatenate([[grad_log_s2d, grad_log_s2e],
                                        grad_log_rho])
        return np.concatenate([grad_theta, grad_psi_full[self._free_psi_mask]])

    # ----- hessian_log_density -----

    def hessian_log_density(self,
                            params: np.ndarray,
                            idx: int = None,
                            h: float = 1e-5) -> np.ndarray:
        """Hessian via central finite differences of grad_log_density."""
        n = self._n_theta + self._n_psi_free
        hess = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            g_fwd = self.grad_log_density(params + h * e_j, idx=idx)
            g_bwd = self.grad_log_density(params - h * e_j, idx=idx)
            hess[:, j] = (g_fwd - g_bwd) / (2.0 * h)
        return 0.5 * (hess + hess.T)
