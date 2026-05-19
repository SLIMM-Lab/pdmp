"""Synthetic recovery for KOGaussianLikelihood with n_groups > 1.

Draws one observation from the exact per-geom block-diagonal KO model with
known (θ, σ²_δ, σ²_ε, ρ), then runs MAP optimization on (θ, log_s2d,
log_s2e, log_rho) and checks the recovered values are close to truth.

Run as a script (`python tests/test_grouped_ko_recovery.py`) for a verbose
summary, or via pytest for the pass/fail check.
"""
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from pdmp.discrepancy import KOGaussianLikelihood
from pdmp.distributions import (
    JointDistribution,
    MultivariateNormal,
    Posterior,
)
from pdmp.forward_model import PiecewiseConstantModel
from pdmp.kernels import rbf_kernel_matrix


def _build_setup(seed=2026, psi_prior_mean=None, psi_prior_cov=None):
    """Build joint posterior + truth.

    Args:
        seed: RNG seed for the synthetic draw.
        psi_prior_mean: ψ prior mean.  Default = truth (used by the unit
            test).  Pass a shifted vector to stress-test the recovery (used
            by the visualization script).
        psi_prior_cov: ψ prior covariance.  Default = 4·I_3.
    """
    rng = np.random.default_rng(seed)

    G = 10
    P = 10
    n_components = 3
    m = n_components * G * P  # 300

    # Forward model: piecewise-constant 1D, 2 params, m output points.
    F = np.array([1.0])
    x_obs_full = np.linspace(0.02, 1.0, m)
    model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs_full)

    # Per-group sensor coordinates in 1D, geom-major flat layout.
    x_locs_g = [rng.uniform(0.0, 1.0, (P, 1)) for _ in range(G)]
    x_locs_flat = np.vstack(x_locs_g)  # (G*P, 1)

    # Truth.
    theta_true = np.array([2.0, 3.0])
    log_s2d_true = -5.0  # σ²_δ ≈ 6.7e-3
    log_s2e_true = -9.2  # σ²_ε ≈ 1e-4  → σ_ε ≈ 0.01 (mirrors itz NOISE_STD)
    log_rho_true = 1.5  # ρ ≈ 4.48  → correlation length ~0.33 in [0,1]
    psi_true = np.array([log_s2d_true, log_s2e_true, log_rho_true])

    sigma2_delta = np.exp(log_s2d_true)
    sigma2_eps = np.exp(log_s2e_true)
    rho = np.array([np.exp(log_rho_true)])

    # Hand-build the full Σ as block_diag over (d, g).
    blocks = []
    for _d in range(n_components):
        for g in range(G):
            C_g = rbf_kernel_matrix(x_locs_g[g], rho)
            blocks.append(sigma2_delta * C_g + sigma2_eps * np.eye(P))
    Sigma = sla.block_diag(*blocks)  # (m, m)
    L = np.linalg.cholesky(Sigma)

    # Draw one observation.
    eta = model.eval(theta_true, idx=0)  # (m,)
    z = rng.standard_normal(m)
    u_obs = (eta + L @ z).reshape(1, -1)

    if psi_prior_mean is None:
        psi_prior_mean = psi_true.copy()
    if psi_prior_cov is None:
        psi_prior_cov = 4.0 * np.eye(3)
    psi_prior = MultivariateNormal(
        mean=np.asarray(psi_prior_mean, dtype=float),
        cov=np.asarray(psi_prior_cov, dtype=float),
        rng=rng,
    )
    theta_prior = MultivariateNormal(
        mean=np.array([0.0, 0.0]),
        cov=100.0 * np.eye(2),
        rng=rng,
    )

    lik = KOGaussianLikelihood(
        model=model,
        u_obs=u_obs,
        x_locs=x_locs_flat,
        psi_prior=psi_prior,
        rng=rng,
        n_components=n_components,
        n_groups=G,
        kernel='isotropic',
    )
    joint_prior = JointDistribution([theta_prior, psi_prior], rng=rng)
    posterior = Posterior(prior=joint_prior, likelihood=lik, rng=rng)

    return posterior, theta_true, psi_true


def _run_map(posterior, theta_true, psi_true):
    """L-BFGS-B on -log posterior, starting from truth perturbed by noise."""
    rng = np.random.default_rng(0)
    x0 = np.concatenate([theta_true, psi_true]) + 0.5 * rng.standard_normal(5)

    def neg_log_post(x):
        return -posterior.log_density(x)

    def neg_grad(x):
        return -posterior.grad_log_density(x)

    res = minimize(neg_log_post,
                   x0,
                   jac=neg_grad,
                   method='L-BFGS-B',
                   options=dict(maxiter=500, ftol=1e-10, gtol=1e-7))
    return res


def _check_recovery(res, theta_true, psi_true, psi_atol=0.5, theta_atol=2.0):
    """Per-coordinate absolute tolerances against truth.  Returns (ok, table)."""
    rec = res.x
    truth = np.concatenate([theta_true, psi_true])
    names = ['theta_0', 'theta_1', 'log_s2d', 'log_s2e', 'log_rho']
    tols = [theta_atol, theta_atol, psi_atol, psi_atol, psi_atol]
    errs = rec - truth
    ok = bool(np.all(np.abs(errs) < np.array(tols)))
    table = list(zip(names, truth, rec, errs, tols))
    return ok, table


def test_grouped_ko_recovery_psi():
    """MAP recovers each ψ entry within ±0.5 of truth on synthetic data.

    The decisive check: when data is generated from the per-geom
    block-diagonal KO model, the new likelihood's MAP should pin all three
    hyperparameters near truth.  θ has a looser tolerance because the
    1D PiecewiseConstantModel is partially insensitive to the second
    parameter under non-trivial KO noise — that is unrelated to the
    discrepancy module under test.
    """
    posterior, theta_true, psi_true = _build_setup()
    res = _run_map(posterior, theta_true, psi_true)
    assert res.success, f"optimizer did not converge: {res.message}"
    ok, table = _check_recovery(res, theta_true, psi_true)
    if not ok:
        msg = "\n".join(
            f"  {n:<8} truth={t:+.4f}  map={m:+.4f}  err={e:+.4f}  (tol ±{tol})"
            for n, t, m, e, tol in table)
        raise AssertionError("MAP recovery exceeded tolerance:\n" + msg)


if __name__ == '__main__':
    posterior, theta_true, psi_true = _build_setup()
    res = _run_map(posterior, theta_true, psi_true)
    print(f"converged: {res.success}  ({res.message})")
    print(f"final -log_posterior: {res.fun:.4f}")
    print(f"{'name':<8} {'truth':>10} {'map':>10} {'err':>10}  tol")
    ok, table = _check_recovery(res, theta_true, psi_true)
    for n, t, m, e, tol in table:
        print(f"{n:<8} {t:+10.4f} {m:+10.4f} {e:+10.4f}  ±{tol}")
    print(f"recovery within tolerance: {ok}")
