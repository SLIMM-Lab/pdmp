#!/usr/bin/env python
"""Forward UQ via moment matching (unscented transform + linearization).

Takes a Laplace approximation of an inverse-problem posterior (produced by
`run_laplace.py`) and pushes its moments through a forward model
(e.g. the 2D RVE) using two complementary moment-matching schemes:

    1. Unscented transform (UT)  — 2d+1 sigma points, deterministic.
    2. Linearization              — central finite-difference Jacobian
                                    around the posterior mean, then
                                    Σ_y = J Σ_x Jᵀ.

The two are run together so the user can see where they agree (smooth
output quantities) and where they diverge (max-type quantities like
`max_von_mises` / `max_strain` whose argmax element switches under finite
parameter perturbations).

The RVE model breaks the JAX trace at the dict-output boundary (it casts
to plain numpy / Python floats), so `jax.jacfwd` cannot be used directly.
Central finite differences are used instead — at the same per-evaluation
cost as UT (6 model solves vs. 7) and following the precedent set by
`JaxFemModel.eval_hessian` in this codebase.

Usage:
    python forward_uq_moment.py path/to/config.yaml

Required config keys:

    model:
      name: RVE                   # any forward model registered in get_model
      ...
    forward_uq_moment:
      posterior_dir:    /…/laplace               # contains mean.dat, cov.dat
      inference_config: /…/inference/config.yaml # source of latent→physical T
      param_indices: [0, 1, 2]                   # which physical-space dims
                                                 # feed the forward model
      ut:
        alpha: 1.0      # sigma-point spread; alpha=1 puts points at ±√d·σ.
        beta:  2.0      # optimal for Gaussian inputs.
        kappa: 0.0
      fd_step: 1.0e-3
      n_synthetic_samples: 1000
      seed: 0
    output:
      dir: results_moment
"""

import os
import sys
import argparse

import numpy as np

# Local utilities reused from forward_uq.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_uq import flatten_output  # noqa: E402

from pdmp import logger  # noqa: E402
from pdmp.loader import get_config  # noqa: E402
from pdmp.forward_model import get_model  # noqa: E402
from pdmp.distributions import MultivariateNormal, get_transformation  # noqa: E402
from pdmp.logger_setup import suppress_external_loggers  # noqa: E402


def build_latent_to_physical(inference_cfg):
    """Reconstruct the latent → physical transformation from an inference config.

    Walks past any outer ``Transformed`` (Affine-whitening) wrappers, then
    inspects the inner ``BayesianInverse`` problem's likelihood. If the
    likelihood is a ``TransformedLikelihood`` (the K&O setup), the Composite
    transformation it carries IS the latent → physical map.

    Returns a callable ``T(xi) -> x_physical`` or ``None`` (identity).
    The Laplace mean/cov are assumed to live in the same space the inner
    BayesianInverse evaluates its log-density on — i.e. natural [θ, ψ]
    coordinates, not the outer-Affine whitened ones.
    """
    problem = inference_cfg['problem']
    while problem.get('name') == 'Transformed':
        problem = problem['distribution']

    if problem.get('name') != 'BayesianInverse':
        raise ValueError(
            f"Expected inner 'BayesianInverse' problem after stripping outer "
            f"Transformed wrappers; got '{problem.get('name')}'.")

    likelihood_cfg = problem.get('likelihood', {})
    if likelihood_cfg.get('name') == 'TransformedLikelihood':
        T = get_transformation(likelihood_cfg)
        return lambda xi: T.transform(np.asarray(xi, dtype=float))
    return None


def unscented_sigma_points(mu, cov, alpha=1.0, beta=2.0, kappa=0.0):
    """Scaled unscented transform sigma points (Wan/van der Merwe).

    With ``alpha=1, beta=2, kappa=0`` the sigma points sit at ``±√d σ``
    along the principal axes of ``cov`` — wide enough to probe genuine
    nonlinearity in the forward model. Smaller ``alpha`` collapses UT
    onto a finite-difference Jacobian, which is fine for nearly linear
    maps but uninformative when comparing against the linearization
    variant in this driver.

    Returns ``(points, wm, wc)`` where ``points`` has shape ``(2d+1, d)``
    and ``wm``, ``wc`` are mean and covariance weights.
    """
    d = mu.shape[0]
    lam = alpha * alpha * (d + kappa) - d
    c = np.sqrt(d + lam)

    L = np.linalg.cholesky(cov)

    points = np.empty((2 * d + 1, d))
    points[0] = mu
    for i in range(d):
        points[1 + i] = mu + c * L[:, i]
        points[1 + d + i] = mu - c * L[:, i]

    wm = np.full(2 * d + 1, 1.0 / (2.0 * (d + lam)))
    wc = wm.copy()
    wm[0] = lam / (d + lam)
    wc[0] = lam / (d + lam) + (1.0 - alpha * alpha + beta)

    return points, wm, wc


def fd_jacobian(f, x, h=1e-3):
    """Central finite-difference Jacobian of ``f`` at ``x``.

    Performs ``2 * d`` evaluations. Returns ``(J, f0)`` where ``J`` has
    shape ``(m, d)`` with ``m = len(f(x))``, and ``f0 = f(x)``.
    """
    d = x.size
    f0 = f(x)
    m = f0.size
    J = np.empty((m, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        f_plus = f(x + e)
        f_minus = f(x - e)
        J[:, i] = (f_plus - f_minus) / (2.0 * h)
    return J, f0


def main():
    parser = argparse.ArgumentParser(
        description='Forward UQ via moment matching (UT + linearization).')
    parser.add_argument('config',
                        type=str,
                        help='Path to the moment-matching config YAML.')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(args.config)))

    config = get_config(args.config)
    mm_cfg = config['forward_uq_moment']

    # ---- Inputs ----------------------------------------------------------
    posterior_dir = mm_cfg['posterior_dir']
    inference_cfg_path = mm_cfg['inference_config']
    param_indices = np.asarray(mm_cfg.get('param_indices', [0, 1, 2]),
                               dtype=int)

    mean_full = np.loadtxt(os.path.join(posterior_dir, 'mean.dat'))
    cov_full = np.loadtxt(os.path.join(posterior_dir, 'cov.dat'))

    inference_cfg = get_config(inference_cfg_path)
    T = build_latent_to_physical(inference_cfg)

    # ---- Forward model ---------------------------------------------------
    model = get_model(config['model'])
    suppress_external_loggers()

    def latent_to_input(xi_full):
        x_phys = T(xi_full) if T is not None else np.asarray(xi_full,
                                                             dtype=float)
        return x_phys[param_indices]

    legend = [None]  # captured on first call

    def f_eval(xi_full):
        x_in = latent_to_input(xi_full)
        raw = model.eval(x_in)
        flat, leg = flatten_output(raw)
        if legend[0] is None:
            legend[0] = leg
        return flat

    # ---- Output dir ------------------------------------------------------
    out_dir = config.get('output', {}).get('dir', './results_moment')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Unscented transform --------------------------------------------
    ut_cfg = mm_cfg.get('ut', {})
    alpha = float(ut_cfg.get('alpha', 1.0))
    beta = float(ut_cfg.get('beta', 2.0))
    kappa = float(ut_cfg.get('kappa', 0.0))

    print('--- Unscented transform ---', flush=True)
    points, wm, wc = unscented_sigma_points(mean_full,
                                            cov_full,
                                            alpha=alpha,
                                            beta=beta,
                                            kappa=kappa)
    sigma_outputs = []
    for i, p in enumerate(points):
        print(f'  sigma point {i + 1}/{len(points)}', flush=True)
        sigma_outputs.append(f_eval(p))
    sigma_outputs = np.asarray(sigma_outputs)

    mu_ut = (wm[:, None] * sigma_outputs).sum(axis=0)
    diffs = sigma_outputs - mu_ut
    cov_ut = (wc[:, None, None] * diffs[:, :, None] *
              diffs[:, None, :]).sum(axis=0)

    # ---- Linearization (central finite differences) ---------------------
    print('--- Linearization (central finite differences) ---', flush=True)
    h = float(mm_cfg.get('fd_step', 1e-3))
    J, f0 = fd_jacobian(f_eval, mean_full, h=h)
    mu_lin = f0
    cov_lin = J @ cov_full @ J.T

    # ---- Save moments ---------------------------------------------------
    np.savetxt(os.path.join(out_dir, 'mean_ut.dat'), mu_ut)
    np.savetxt(os.path.join(out_dir, 'cov_ut.dat'), cov_ut)
    np.savetxt(os.path.join(out_dir, 'mean_lin.dat'), mu_lin)
    np.savetxt(os.path.join(out_dir, 'cov_lin.dat'), cov_lin)
    np.savetxt(os.path.join(out_dir, 'sigma_points.dat'), points)
    np.savetxt(os.path.join(out_dir, 'sigma_outputs.dat'), sigma_outputs)
    np.savetxt(os.path.join(out_dir, 'jacobian.dat'), J)

    legend = legend[0]
    if legend is not None:
        with open(os.path.join(out_dir, 'outputs_legend.txt'), 'w') as f:
            for i, name in enumerate(legend):
                f.write(f'{i}\t{name}\n')

    # ---- Synthetic samples for compatibility with forward_uq.py plotting -
    n_syn = int(mm_cfg.get('n_synthetic_samples', 1000))
    rng = np.random.default_rng(int(mm_cfg.get('seed', 0)))

    def _draw(mu, cov):
        return MultivariateNormal(mu, cov, rng=rng).get_sample(n_syn)

    np.savetxt(os.path.join(out_dir, 'samples_ut.dat'), _draw(mu_ut, cov_ut))
    np.savetxt(os.path.join(out_dir, 'samples_lin.dat'),
               _draw(mu_lin, cov_lin))

    # ---- Summary --------------------------------------------------------
    print()
    print(f'Saved moment-matched results to {out_dir}/')
    names = legend if legend is not None else [
        f'out[{j}]' for j in range(mu_ut.size)
    ]
    for i, name in enumerate(names):
        print(f'  {name}: '
              f'UT  μ={mu_ut[i]:.4g} σ={np.sqrt(max(cov_ut[i, i], 0)):.4g} | '
              f'LIN μ={mu_lin[i]:.4g} σ={np.sqrt(max(cov_lin[i, i], 0)):.4g}')


if __name__ == '__main__':
    main()
