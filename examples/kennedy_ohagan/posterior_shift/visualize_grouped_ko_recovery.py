"""Visualize KOGaussianLikelihood recovery: joint vs separate, displaced prior.

Generates one synthetic dataset under the per-geom block-diagonal KO model
(`tests/test_grouped_ko_recovery.py::_build_setup`).  The prior on ψ is
deliberately shifted away from truth so the recovery is not trivially
pre-anchored — only the data can pull the posterior toward truth.

Two fits are run on the same data:
  - JOINT     — one MAP over all G groups with `n_groups=G`.
  - SEPARATE  — G independent MAPs, one per group, each with `n_groups=1`,
                seeing only that group's 30 measurements.

For each parameter the plot shows: truth, prior mean, the joint MAP with
1σ bar (Laplace), and the G separate MAPs with their own 1σ bars.  This
visualises the trade-off the user observed on the real ITZ data — joint
fits benefit from G× more data and override a (mis-anchored) prior, while
separate fits with 30 measurements each are dragged toward whatever the
prior says.

Run from the repo root:

    python examples/kennedy_ohagan/posterior_shift/visualize_grouped_ko_recovery.py [--prior-shift 2.0] [--out fig.png]
"""
import argparse
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
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
    rng = np.random.default_rng(seed)

    G = 10
    P = 10
    n_components = 3
    m = n_components * G * P  # 300

    F = np.array([1.0])
    x_obs_full = np.linspace(0.02, 1.0, m)
    model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs_full)

    x_locs_g = [rng.uniform(0.0, 1.0, (P, 1)) for _ in range(G)]
    x_locs_flat = np.vstack(x_locs_g)

    theta_true = np.array([2.0, 3.0])
    psi_true = np.array([-5.0, -9.2, 1.5])
    sigma2_delta = np.exp(psi_true[0])
    sigma2_eps = np.exp(psi_true[1])
    rho = np.array([np.exp(psi_true[2])])

    blocks = []
    for _d in range(n_components):
        for g in range(G):
            C_g = rbf_kernel_matrix(x_locs_g[g], rho)
            blocks.append(sigma2_delta * C_g + sigma2_eps * np.eye(P))
    Sigma = sla.block_diag(*blocks)
    L = np.linalg.cholesky(Sigma)

    eta = model.eval(theta_true, idx=0)
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
    rng = np.random.default_rng(0)
    x0 = np.concatenate([theta_true, psi_true]) + 0.5 * rng.standard_normal(5)

    res = minimize(
        lambda x: -posterior.log_density(x),
        x0,
        jac=lambda x: -posterior.grad_log_density(x),
        method='L-BFGS-B',
        options=dict(maxiter=500, ftol=1e-10, gtol=1e-7),
    )
    return res


PARAM_NAMES = ['theta_0', 'theta_1', 'log_s2d', 'log_s2e', 'log_rho']


def _indices_for_geom(g, n_components, n_groups, P):
    """Flat indices of (any DOF, group g, any P) in the joint layout."""
    return np.concatenate([
        np.arange(d * n_groups * P + g * P, d * n_groups * P + g * P + P)
        for d in range(n_components)
    ])


def _build_separate_for_geom(joint_posterior, g, psi_prior_mean, psi_prior_cov,
                             rng):
    """Build a Posterior fit to only group g's 30 observations."""
    lik_joint = joint_posterior._likelihood
    G = lik_joint._n_groups
    P = lik_joint._P
    n_c = lik_joint._n_components

    idx = _indices_for_geom(g, n_c, G, P)
    u_obs_g = lik_joint._u_obs[0][idx].reshape(1, -1)
    x_obs_g = lik_joint._model.x_obs_[idx]
    x_locs_g = lik_joint._x_locs_g[g]  # (P, d_x)

    model_g = PiecewiseConstantModel(
        F=lik_joint._model.F_vals,
        n_params=lik_joint._n_theta,
        x_obs=x_obs_g,
    )

    psi_prior = MultivariateNormal(
        mean=psi_prior_mean.copy(),
        cov=psi_prior_cov.copy(),
        rng=rng,
    )
    theta_prior = MultivariateNormal(
        mean=np.array([0.0, 0.0]),
        cov=100.0 * np.eye(2),
        rng=rng,
    )
    lik_g = KOGaussianLikelihood(
        model=model_g,
        u_obs=u_obs_g,
        x_locs=x_locs_g,
        psi_prior=psi_prior,
        rng=rng,
        n_components=n_c,
        n_groups=1,
        kernel='isotropic',
    )
    prior_g = JointDistribution([theta_prior, psi_prior], rng=rng)
    return Posterior(prior=prior_g, likelihood=lik_g, rng=rng)


def _hessian_via_fd(posterior, x_map, h=1e-4):
    n = len(x_map)
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        H[:, j] = (posterior.grad_log_density(x_map + e) -
                   posterior.grad_log_density(x_map - e)) / (2.0 * h)
    return 0.5 * (H + H.T)


def _laplace_std(posterior, x_map):
    try:
        cov = np.linalg.inv(-_hessian_via_fd(posterior, x_map))
        return np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        return np.full(len(x_map), np.nan)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--prior-shift',
                    type=float,
                    default=2.0,
                    help='offset added to each truth ψ to form the prior mean '
                    '(default: 2.0)')
    ap.add_argument('--prior-std',
                    type=float,
                    default=2.0,
                    help='1-σ width of the ψ prior on each coordinate '
                    '(default: 2.0)')
    ap.add_argument('--seed',
                    type=int,
                    default=2026,
                    help='RNG seed for the synthetic draw (default: 2026)')
    ap.add_argument(
        '--out',
        type=str,
        default=None,
        help='output PNG path; if omitted, opens an interactive window')
    args = ap.parse_args()

    # Truth values (re-stated for the prior shift).
    psi_true = np.array([-5.0, -9.2, 1.5])
    psi_prior_mean = psi_true + args.prior_shift
    psi_prior_cov = (args.prior_std**2) * np.eye(3)

    # JOINT fit.
    rng = np.random.default_rng(args.seed)
    posterior, theta_true, psi_true = _build_setup(
        seed=args.seed,
        psi_prior_mean=psi_prior_mean,
        psi_prior_cov=psi_prior_cov,
    )
    truth = np.concatenate([theta_true, psi_true])
    res_joint = _run_map(posterior, theta_true, psi_true)
    joint_map = res_joint.x
    joint_std = _laplace_std(posterior, joint_map)

    # SEPARATE fits — one per group.
    lik_joint = posterior._likelihood
    n_g = lik_joint._n_groups
    sep_maps, sep_stds = [], []
    for g in range(n_g):
        post_g = _build_separate_for_geom(
            posterior,
            g,
            psi_prior_mean,
            psi_prior_cov,
            rng,
        )
        res_g = _run_map(post_g, theta_true, psi_true)
        sep_maps.append(res_g.x)
        sep_stds.append(_laplace_std(post_g, res_g.x))
    sep_maps = np.asarray(sep_maps)
    sep_stds = np.asarray(sep_stds)

    # Plot: one panel per parameter; horizontal axis is the parameter value;
    # rows are joint (top) and separate fits (below).
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.5), constrained_layout=True)

    for j, name in enumerate(PARAM_NAMES):
        ax = axes[j]
        # Truth and prior reference lines.
        ax.axvline(truth[j],
                   color='k',
                   linewidth=1.5,
                   linestyle='--',
                   label='truth')
        if j >= 2:
            ax.axvline(psi_prior_mean[j - 2],
                       color='C2',
                       linewidth=1.0,
                       linestyle=':',
                       label='prior mean')
        # Joint MAP at y=0 (top).
        ax.errorbar([joint_map[j]], [0],
                    xerr=[joint_std[j]],
                    fmt='o',
                    color='C0',
                    markersize=9,
                    capsize=4,
                    elinewidth=1.5,
                    label='joint MAP ± 1σ')
        # Separate MAPs at y = -1 .. -n_g.
        ys = -(1 + np.arange(n_g))
        ax.errorbar(sep_maps[:, j],
                    ys,
                    xerr=sep_stds[:, j],
                    fmt='.',
                    color='C1',
                    markersize=7,
                    elinewidth=0.8,
                    capsize=2,
                    label='separate MAPs ± 1σ')

        ax.set_yticks([0] + list(ys))
        ax.set_yticklabels(['joint'] + [f'sep {g}' for g in range(n_g)],
                           fontsize=7)
        ax.set_xlabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3, axis='x')
        ax.set_ylim(-n_g - 0.8, 1.0)
        if j == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        f'KO recovery: joint vs {n_g} separate fits — '
        f'prior shifted by +{args.prior_shift:.1f} from truth (std={args.prior_std:.1f})',
        fontsize=12,
    )

    if args.out:
        fig.savefig(args.out, dpi=140)
        print(f'wrote {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
