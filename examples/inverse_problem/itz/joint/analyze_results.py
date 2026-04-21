#!/usr/bin/env python3
"""Analyse RWM + K&O results for the multi-geometry ITZ inference.

The sampler operates in whitened ξ-space (Affine transformation applied by the
Transformed wrapper).  Samples are back-transformed to physical space:

    ξ  →  (affine)  →  θ  →  (composite)  →  physical

    θ[0]  →  sigmoid  →  ρ  ∈ (0, 1)
    θ[1]  →  exp      →  l  > 0  (µm)
    θ[2]  →  exp      →  f_∞ > 0
    θ[3]  →  identity →  log σ²_δ
    θ[4]  →  identity →  log σ²_ε
    θ[5]  →  identity →  log ρ_KO

Produces (saved in rwm/figures/):
  traces.pdf            — full chain traces, all 6 params
  theta_marginals.pdf   — histogram/KDE for ρ, l, f_∞
  psi_marginals.pdf     — histogram/KDE for the 3 KO hyper-parameters
  theta_pairplot.pdf    — pairwise scatter/diagonal for ρ, l, f_∞

Run from itz_all_geom/:
    python analyze_results.py [--rwm-dir ./rwm] [--burnin-frac 0.25]
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pdmp.loader import get_config

HERE = os.path.dirname(os.path.abspath(__file__))

THETA_NAMES = [r'$\rho$', r'$l$ (µm)', r'$f_\infty$']
PSI_NAMES = [
    r'$\log\,\sigma^2_\delta$', r'$\log\,\sigma^2_\varepsilon$',
    r'$\log\,\rho_\mathrm{KO}$'
]
ALL_NAMES = THETA_NAMES + PSI_NAMES
THETA_LATENT_NAMES = [
    r'$\theta_0 = \mathrm{logit}(\rho)$',
    r'$\theta_1 = \log\,l$',
    r'$\theta_2 = \log\,f_\infty$',
]

# ── Loading ──────────────────────────────────────────────────────────────────


def _load_samples(rwm_dir):
    path = os.path.join(rwm_dir, 'samples.dat')
    if not os.path.exists(path):
        raise FileNotFoundError(f"samples.dat not found in {rwm_dir}")
    samples = np.loadtxt(path)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    print(f"Loaded {samples.shape[0]} samples (dim={samples.shape[1]})")
    return samples


def _print_acceptance_rate(rwm_dir):
    path = os.path.join(rwm_dir, 'other.pkl')
    if not os.path.exists(path):
        print("other.pkl not found — acceptance rate unavailable")
        return
    with open(path, 'rb') as f:
        data = pickle.load(f)
    rate = data.get('acceptance_rate')
    if rate is not None:
        print(f"Acceptance rate: {rate:.3f}")


def _to_physical(xi_chain, M, b, f_inf):
    """ξ-chain → physical parameters  (n_samples × 6).

    Affine:    θ = ξ @ M.T + b
    Composite: ρ = sigmoid(θ[0]),  l = exp(θ[1]),  f_∞ = exp(θ[2]),  psi = θ[3:]
    """
    theta_chain = xi_chain @ M.T + b
    physical = np.empty_like(theta_chain)
    physical[:, 0] = 1.0 / (1.0 + np.exp(-theta_chain[:, 0]))  # sigmoid → ρ
    physical[:, 1] = np.exp(theta_chain[:, 1])  # exp     → l
    # physical[:, 2] = np.exp(theta_chain[:, 2])  # exp     → f_∞
    physical[:, 2] = f_inf / (1.0 + np.exp(-theta_chain[:, 2]))  # sigmoid → ρ
    physical[:, 3:] = theta_chain[:, 3:]  # identity → log-psi
    return theta_chain, physical


def _sample_physical_prior(prior_mean, prior_cov, f_inf=1, n=200_000, seed=0):
    """Draw samples from the Gaussian prior in θ-space and map to physical space."""
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(prior_mean, prior_cov, size=n)
    physical = np.empty_like(theta)
    physical[:, 0] = 1.0 / (1.0 + np.exp(-theta[:, 0]))  # sigmoid → ρ
    physical[:, 1] = np.exp(theta[:, 1])  # exp     → l
    # physical[:, 2] = np.exp(theta[:, 2])  # exp     → f_∞
    physical[:, 2] = f_inf / (1.0 + np.exp(-theta[:, 2]))  # sigmoid → ρ
    physical[:, 3:] = theta[:, 3:]
    return physical


# ── Summaries ────────────────────────────────────────────────────────────────


def _print_summary(post):
    n = post.shape[0]
    print(f"\n{'─'*56}")
    print(f"{'Parameter':<24}  {'Median':>8}  {'5%':>8}  {'95%':>8}")
    print(f"{'─'*56}")
    short = ['rho', 'l', 'f_inf', 'log_s2d', 'log_s2e', 'log_rho_ko']
    for k, name in enumerate(short):
        lo, med, hi = np.percentile(post[:, k], [5, 50, 95])
        print(f"  {name:<22}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}")
    print(f"{'─'*56}")
    print(f"Post burn-in samples: {n}\n")


# ── Plots ────────────────────────────────────────────────────────────────────


def _plot_traces(chain_physical, burnin, fig_dir):
    n, d = chain_physical.shape
    fig, axes = plt.subplots(d, 1, figsize=(10, 2.0 * d), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(chain_physical[:, k], lw=0.35, color='steelblue', alpha=0.8)
        if burnin > 0:
            ax.axvspan(0,
                       burnin,
                       color='lightcoral',
                       alpha=0.25,
                       label='burn-in' if k == 0 else None)
        ax.set_ylabel(ALL_NAMES[k], fontsize=9)
        ax.tick_params(labelsize=8)
    if burnin > 0:
        axes[0].legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Sample index', fontsize=9)
    fig.suptitle('Trace plots — multi-geometry ITZ (physical space)',
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'traces.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _theta_xlims(prior_mean, prior_std, n_sigma=4):
    """Return [(lo, hi), ...] bounds for θ-space parameters (mean ± n_sigma*std)."""
    return [(prior_mean[k] - n_sigma * prior_std[k],
             prior_mean[k] + n_sigma * prior_std[k])
            for k in range(len(prior_mean))]


def _filter_outliers(vals, pct):
    """Return a boolean mask dropping outliers based on pct.

    pct scalar  → drop bottom and top pct %
    pct (lo, hi) → drop bottom lo % and top hi % independently (0 = no clipping)
    """
    if np.ndim(pct) == 0:
        pct_lo, pct_hi = float(pct), float(pct)
    else:
        pct_lo, pct_hi = float(pct[0]), float(pct[1])
    lo = np.percentile(vals, pct_lo) if pct_lo > 0 else -np.inf
    hi = np.percentile(vals, 100 - pct_hi) if pct_hi > 0 else np.inf
    return (vals >= lo) & (vals <= hi)


def _plot_marginals(post,
                    indices,
                    names,
                    title,
                    filename,
                    fig_dir,
                    xlims=None,
                    outlier_pct=None,
                    prior_mean=None,
                    prior_std=None,
                    prior_samples=None):
    from scipy.stats import gaussian_kde, norm
    xlims_arr = np.asarray(xlims) if xlims is not None else None
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for k, (ax, i) in enumerate(zip(axes, indices)):
        vals = post[:, i]
        if outlier_pct is not None:
            vals = vals[_filter_outliers(vals, outlier_pct)]

        # Clip vals to xlims so histogram and KDE reflect only what is plotted
        if xlims_arr is not None:
            if xlims_arr.ndim == 1:  # upper bound only
                vals = vals[vals <= xlims_arr[k]]
            else:  # (lo, hi) pair
                vals = vals[(vals >= xlims_arr[k, 0])
                            & (vals <= xlims_arr[k, 1])]

        xs = np.linspace(float(vals.min()), float(vals.max()), 400)

        ax.hist(vals,
                bins=60,
                density=True,
                alpha=0.45,
                color='steelblue',
                edgecolor='none',
                label='posterior')
        kde = gaussian_kde(vals)
        ax.plot(xs, kde(xs), color='steelblue', lw=1.5)
        if prior_mean is not None and prior_std is not None:
            ax.plot(xs,
                    norm.pdf(xs, prior_mean[k], prior_std[k]),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')
        if prior_samples is not None:

            p_vals = prior_samples[:, i]

            # Clip vals to xlims so histogram and KDE reflect only what is plotted
            xs = np.linspace(float(p_vals.min()), float(p_vals.max()), 400)

            prior_kde = gaussian_kde(prior_samples[:, i])
            ax.plot(xs,
                    prior_kde(xs),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')
        ax.axvline(np.median(vals),
                   color='C1',
                   lw=1.2,
                   ls='--',
                   label=f'median={np.median(vals):.3f}')
        if xlims_arr is not None:
            if xlims_arr.ndim == 1:  # upper bound only
                ax.set_xlim(left=0.0, right=xlims_arr[k])
            else:  # (lo, hi) pair
                ax.set_xlim(left=xlims_arr[k, 0], right=xlims_arr[k, 1])
        ax.set_xlabel(names[i - indices[0]], fontsize=9)
        ax.set_ylabel('density', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_pairplot(post, fig_dir, theta_xlims=None, outlier_pct=None):
    """Pairwise scatter + diagonal histograms for ρ, l, f_∞."""
    params = post[:, :3]
    if outlier_pct is not None:
        mask = np.ones(len(params), dtype=bool)
        for k in range(3):
            mask &= _filter_outliers(params[:, k], outlier_pct)
        params = params[mask]
    names = THETA_NAMES
    d = 3
    fig, axes = plt.subplots(d, d, figsize=(8, 8))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(params[:, i],
                        bins=40,
                        density=True,
                        color='steelblue',
                        alpha=0.5,
                        edgecolor='none')
                ax.set_xlabel(names[i], fontsize=8)
                if theta_xlims is not None and theta_xlims[i] is not None:
                    ax.set_xlim(right=theta_xlims[i])
            elif i > j:
                ax.scatter(params[:, j],
                           params[:, i],
                           s=2,
                           alpha=0.25,
                           color='steelblue',
                           linewidths=0)
                ax.set_xlabel(names[j], fontsize=8)
                ax.set_ylabel(names[i], fontsize=8)
                if theta_xlims is not None and theta_xlims[j] is not None:
                    ax.set_xlim(right=theta_xlims[j])
                if theta_xlims is not None and theta_xlims[i] is not None:
                    ax.set_ylim(top=theta_xlims[i])
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    fig.suptitle(r'Pairwise marginals — $\rho$, $l$, $f_\infty$', fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'theta_pairplot.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rwm-dir', default=os.path.join(HERE, 'rwm'))
    parser.add_argument('--burnin-frac',
                        type=float,
                        default=0.25,
                        help='Fraction of samples to discard as burn-in')
    args = parser.parse_args()

    rwm_dir = args.rwm_dir
    fig_dir = os.path.join(rwm_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    _print_acceptance_rate(rwm_dir)
    xi_chain = _load_samples(rwm_dir)

    # M and b are written to config_used.yml by the sampler — no FEM calls needed
    config_used = get_config(os.path.join(rwm_dir, 'config_used.yml'))
    M = config_used['problem']['M']
    b = config_used['problem']['b']

    f_inf = config_used['problem']['distribution']['likelihood'][
        'transformations'][2]['b']

    print("Transforming to physical space...")
    theta_chain, chain_physical = _to_physical(xi_chain, M, b, f_inf)

    burnin = int(args.burnin_frac * len(chain_physical))
    post = chain_physical[burnin:]
    post_theta = theta_chain[burnin:]
    _print_summary(post)

    # Upper x-limits for θ in physical space: prior_mean_phys + 4 * prior_std_phys
    #
    # ρ = sigmoid(θ[0])  — bounded [0,1], use 4σ in θ-space mapped through sigmoid
    # l, f_∞ = exp(θ[k]) — log-normal moments:
    #   E[X]   = exp(μ + σ²/2)
    #   Std[X] = exp(μ + σ²/2) * sqrt(exp(σ²) - 1)
    coeff_dist = (config_used['problem']['distribution']['model']['field']
                  ['coefficient_distribution'])
    prior_mean = np.asarray(coeff_dist['mean'])
    prior_cov = np.asarray(coeff_dist['cov'])
    prior_var = np.diag(prior_cov)
    prior_std = np.sqrt(prior_var)
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)

    psi_prior = (config_used['problem']['distribution']['likelihood']
                 ['likelihood']['psi_prior'])
    psi_prior_mean = np.asarray(psi_prior['mean'])
    psi_prior_std = np.sqrt(np.diag(np.asarray(psi_prior['cov'])))

    def _lognormal_upper(mu, var, n_sigma=2):
        phys_mean = np.exp(mu + var / 2)
        phys_std = phys_mean * np.sqrt(np.exp(var) - 1)
        return phys_mean + n_sigma * phys_std

    theta_xlims = [
        1.0 /
        (1.0 + np.exp(-(prior_mean[0] + 4 * prior_std[0]))),  # sigmoid → ρ
        _lognormal_upper(prior_mean[1], prior_var[1]),  # log-normal → l
        _lognormal_upper(prior_mean[2], prior_var[2]),  # log-normal → f_∞
    ]

    theta_xlims = [1.0, 300., 130.]

    _plot_traces(chain_physical, burnin, fig_dir)
    _plot_marginals(post, [0, 1, 2],
                    THETA_NAMES,
                    r'$\theta$ posterior marginals  ($\rho$, $l$, $f_\infty$)',
                    'theta_marginals.pdf',
                    fig_dir,
                    xlims=theta_xlims,
                    outlier_pct=(0.0, 0.0),
                    prior_samples=prior_physical)
    _plot_marginals(post, [3, 4, 5], PSI_NAMES,
                    r'$\psi$ posterior marginals (KO hyper-parameters)',
                    'psi_marginals.pdf',
                    fig_dir,
                    prior_mean=psi_prior_mean,
                    prior_std=psi_prior_std)
    theta_latent_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])
    _plot_marginals(post_theta, [0, 1, 2],
                    THETA_LATENT_NAMES,
                    r'$\theta$ posterior marginals (latent space)',
                    'theta_latent_marginals.pdf',
                    fig_dir,
                    xlims=theta_latent_xlims,
                    outlier_pct=0.0,
                    prior_mean=prior_mean[:3],
                    prior_std=prior_std[:3])
    _plot_pairplot(post, fig_dir, theta_xlims=theta_xlims, outlier_pct=1.0)

    out = os.path.join(rwm_dir, 'samples_physical.dat')
    np.savetxt(out,
               post,
               header='rho l f_inf log_s2d log_s2e log_rho_ko',
               fmt='%.6e')
    print(f"Saved physical samples: {out}")


if __name__ == '__main__':
    main()
