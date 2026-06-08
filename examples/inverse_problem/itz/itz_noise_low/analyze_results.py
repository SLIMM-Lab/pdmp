#!/usr/bin/env python3
"""Analyse RWM + K&O results for both separate and joint ITZ geometry inference.

Produces per-geometry figures (same as joint/analyze_results.py) for each
separate run, plus combined comparison figures overlaying all separate
posteriors against the joint posterior.

Usage:
    python analyze_results.py [--separate-dir ./separate] [--joint-dir ./joint]
                              [--burnin-frac 0.25]
                              [--skip-separate] [--skip-joint] [--skip-comparison]
                              [--skip-separate-standard]

Output:
    separate/NN/rwm/figures/   — traces, marginals, pairplot per geometry
    separate/NN/rwm/samples_physical.dat
    joint/rwm/figures/         — same plots for joint run
    joint/rwm/samples_physical.dat
    figures/                   — comparison plots (separate mixture vs joint)
"""

import argparse
import os
import json
from glob import glob

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

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
PSI_PHYSICAL_NAMES = [
    r'$\sigma^2_\delta$', r'$\sigma^2_\varepsilon$', r'$\rho_\mathrm{KO}$'
]
ALL_PHYSICAL_NAMES = THETA_NAMES + PSI_PHYSICAL_NAMES

# ── Shared helpers ────────────────────────────────────────────────────────────


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
    path = os.path.join(rwm_dir, 'other.json')
    if not os.path.exists(path):
        print("other.json not found — acceptance rate unavailable")
        return
    with open(path) as f:
        data = json.load(f)
    rate = data.get('acceptance_rate')
    if rate is not None:
        print(f"Acceptance rate: {rate:.3f}")


def _to_physical(xi_chain, M, b, f_inf):
    theta_chain = xi_chain @ M.T + b
    physical = np.empty_like(theta_chain)
    physical[:, 0] = 1.0 / (1.0 + np.exp(-theta_chain[:, 0]))
    physical[:, 1] = np.exp(theta_chain[:, 1])
    physical[:, 2] = f_inf / (1.0 + np.exp(-theta_chain[:, 2]))
    physical[:, 3:] = theta_chain[:, 3:]
    return theta_chain, physical


def _sample_physical_prior(prior_mean, prior_cov, f_inf=1, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(prior_mean, prior_cov, size=n)
    physical = np.empty_like(theta)
    physical[:, 0] = 1.0 / (1.0 + np.exp(-theta[:, 0]))
    physical[:, 1] = np.exp(theta[:, 1])
    physical[:, 2] = f_inf / (1.0 + np.exp(-theta[:, 2]))
    physical[:, 3:] = theta[:, 3:]
    return physical


def _print_summary(post, label=''):
    n = post.shape[0]
    header = f'  {label}' if label else ''
    print(f"\n{'─'*56}{header}")
    print(f"{'Parameter':<24}  {'Median':>8}  {'5%':>8}  {'95%':>8}")
    print(f"{'─'*56}")
    short = ['rho', 'l', 'f_inf', 'log_s2d', 'log_s2e',
             'log_rho_ko'][:post.shape[1]]
    for k, name in enumerate(short):
        lo, med, hi = np.percentile(post[:, k], [5, 50, 95])
        print(f"  {name:<22}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}")
    print(f"{'─'*56}")
    print(f"Post burn-in samples: {n}\n")


def _filter_outliers(vals, pct):
    if np.ndim(pct) == 0:
        pct_lo, pct_hi = float(pct), float(pct)
    else:
        pct_lo, pct_hi = float(pct[0]), float(pct[1])
    lo = np.percentile(vals, pct_lo) if pct_lo > 0 else -np.inf
    hi = np.percentile(vals, 100 - pct_hi) if pct_hi > 0 else np.inf
    return (vals >= lo) & (vals <= hi)


def _theta_xlims(prior_mean, prior_std, n_sigma=4):
    return [(prior_mean[k] - n_sigma * prior_std[k],
             prior_mean[k] + n_sigma * prior_std[k])
            for k in range(len(prior_mean))]


def _plot_traces(chain_physical, burnin, fig_dir, title_suffix=''):
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
    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(f'Trace plots — ITZ (physical space){suffix}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'traces.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


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
    from scipy.stats import norm
    xlims_arr = np.asarray(xlims) if xlims is not None else None
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for k, (ax, i) in enumerate(zip(axes, indices)):
        vals = post[:, i]
        if outlier_pct is not None:
            vals = vals[_filter_outliers(vals, outlier_pct)]
        if xlims_arr is not None:
            if xlims_arr.ndim == 1:
                vals = vals[vals <= xlims_arr[k]]
            else:
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
            xs_p = np.linspace(float(p_vals.min()), float(p_vals.max()), 400)
            prior_kde = gaussian_kde(p_vals)
            ax.plot(xs_p,
                    prior_kde(xs_p),
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
            if xlims_arr.ndim == 1:
                ax.set_xlim(left=0.0, right=xlims_arr[k])
            else:
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


def _plot_pairplot(post,
                   fig_dir,
                   theta_xlims=None,
                   outlier_pct=None,
                   max_scatter=1000):
    params = post[:, :3]
    if outlier_pct is not None:
        mask = np.ones(len(params), dtype=bool)
        for k in range(3):
            mask &= _filter_outliers(params[:, k], outlier_pct)
        params = params[mask]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(params),
                     size=min(max_scatter, len(params)),
                     replace=False)
    scatter_params = params[idx]
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
                if theta_xlims is not None:
                    ax.set_xlim(*theta_xlims[i])
            elif i > j:
                ax.scatter(scatter_params[:, j],
                           scatter_params[:, i],
                           s=2,
                           alpha=0.25,
                           color='steelblue',
                           linewidths=0)
                ax.set_xlabel(names[j], fontsize=8)
                ax.set_ylabel(names[i], fontsize=8)
                if theta_xlims is not None:
                    ax.set_xlim(*theta_xlims[j])
                    ax.set_ylim(*theta_xlims[i])
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    fig.suptitle(r'Pairwise marginals — $\rho$, $l$, $f_\infty$', fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'theta_pairplot.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_psi_pairplot(post,
                       fig_dir,
                       psi_prior_mean=None,
                       psi_prior_std=None,
                       title_suffix='',
                       filename='psi_pairplot.pdf',
                       max_scatter=1000):
    """3×3 corner plot of (log σ²_δ, log σ²_ε, log ρ_KO).

    The (log σ²_δ, log σ²_ε) panel reveals the ridge that arises because only
    σ²_δ + σ²_ε is tightly identified by the residual scale; the marginal mode
    of either axis is therefore sensitive to sensor placement.  See
    HYPERPARAMETER_DISCREPANCY.md at the itz/ level.
    """
    from scipy.stats import norm
    psi = post[:, 3:6]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(psi), size=min(max_scatter, len(psi)), replace=False)
    scatter = psi[idx]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                ax.hist(psi[:, i],
                        bins=40,
                        density=True,
                        color='steelblue',
                        alpha=0.5,
                        edgecolor='none',
                        label='posterior')
                if psi_prior_mean is not None and psi_prior_std is not None:
                    lo, hi = float(psi[:, i].min()), float(psi[:, i].max())
                    xs = np.linspace(lo, hi, 200)
                    ax.plot(xs,
                            norm.pdf(xs, psi_prior_mean[i], psi_prior_std[i]),
                            color='C2',
                            lw=1.2,
                            ls='--',
                            label='prior')
                ax.set_xlabel(PSI_NAMES[i], fontsize=8)
                if i == 0:
                    ax.legend(fontsize=7)
            elif i > j:
                ax.scatter(scatter[:, j],
                           scatter[:, i],
                           s=2,
                           alpha=0.3,
                           color='steelblue',
                           linewidths=0)
                ax.set_xlabel(PSI_NAMES[j], fontsize=8)
                ax.set_ylabel(PSI_NAMES[i], fontsize=8)
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(f'KO hyperparameter pairplot{suffix}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_xi_pairplot(post_xi,
                      fig_dir,
                      title_suffix='',
                      filename='xi_pairplot.pdf',
                      max_scatter=1000):
    """Corner plot of the affine (whitened) sampling space xi.

    These are the raw PDMP positions, before the linear map to theta and the
    non-linear physical transform. The prior is standard normal in this space,
    so an N(0,1) reference is overlaid on each diagonal to make prior/posterior
    contraction visible per latent direction.
    """
    from scipy.stats import norm
    d = post_xi.shape[1]
    names = [rf'$\xi_{{{k}}}$' for k in range(d)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(post_xi),
                     size=min(max_scatter, len(post_xi)),
                     replace=False)
    scatter = post_xi[idx]

    # Common limits per dimension (combine posterior spread with the N(0,1)
    # reference so the standard normal stays visible).
    lims = []
    for k in range(d):
        lo, hi = np.percentile(post_xi[:, k], [0.5, 99.5])
        lo = min(float(lo), -3.0)
        hi = max(float(hi), 3.0)
        pad = 0.05 * (hi - lo)
        lims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(d, d, figsize=(2.4 * d, 2.4 * d))
    if d == 1:
        axes = np.array([[axes]])
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                xs = np.linspace(lims[i][0], lims[i][1], 300)
                vals = post_xi[:, i]
                vals = vals[(vals >= lims[i][0]) & (vals <= lims[i][1])]
                ax.hist(vals,
                        bins=40,
                        density=True,
                        color='steelblue',
                        alpha=0.5,
                        edgecolor='none',
                        label='posterior')
                if len(vals) > 2:
                    ax.plot(xs, gaussian_kde(vals)(xs), color='steelblue',
                            lw=1.2)
                ax.plot(xs,
                        norm.pdf(xs, 0.0, 1.0),
                        color='C2',
                        lw=1.0,
                        ls='--',
                        label='N(0,1)')
                ax.set_xlabel(names[i], fontsize=8)
                ax.set_xlim(*lims[i])
                if i == 0:
                    ax.legend(fontsize=7)
            elif i > j:
                ax.scatter(scatter[:, j],
                           scatter[:, i],
                           s=2,
                           alpha=0.25,
                           color='steelblue',
                           linewidths=0)
                ax.set_xlabel(names[j], fontsize=8)
                ax.set_ylabel(names[i], fontsize=8)
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    suffix = f' — {title_suffix}' if title_suffix else ''
    fig.suptitle(f'Affine (whitened) xi-space pairplot{suffix}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_psi_sum_marginal(post,
                           fig_dir,
                           psi_prior_mean=None,
                           psi_prior_std=None,
                           title_suffix='',
                           filename='psi_sum_marginal.pdf'):
    """KDE of σ²_δ + σ²_ε (the diagonal of Σ_g).

    This sum is identified directly by the residual scale and is more robust
    than either component alone — the individual components sit on a
    confounding ridge.  See HYPERPARAMETER_DISCREPANCY.md at the itz/ level.
    """
    s2_sum = np.exp(post[:, 3]) + np.exp(post[:, 4])
    lo, hi = float(np.percentile(s2_sum, 1)), float(np.percentile(s2_sum, 99))
    s2_clip = s2_sum[(s2_sum >= lo) & (s2_sum <= hi)]
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.hist(s2_clip,
            bins=60,
            density=True,
            alpha=0.4,
            color='steelblue',
            edgecolor='none')
    if len(s2_clip) > 2:
        ax.plot(xs,
                gaussian_kde(s2_clip)(xs),
                color='steelblue',
                lw=1.5,
                label='posterior')
    if psi_prior_mean is not None and psi_prior_std is not None:
        rng = np.random.default_rng(0)
        log_psi = rng.multivariate_normal(psi_prior_mean,
                                          np.diag(psi_prior_std**2),
                                          size=100_000)
        prior_sum = np.exp(log_psi[:, 0]) + np.exp(log_psi[:, 1])
        prior_sum = prior_sum[(prior_sum >= lo) & (prior_sum <= hi)]
        if len(prior_sum) > 2:
            ax.plot(xs,
                    gaussian_kde(prior_sum)(xs),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')
    ax.axvline(float(np.median(s2_sum)),
               color='C1',
               lw=1.0,
               ls='--',
               label=f'median={np.median(s2_sum):.3e}')
    ax.set_xlabel(r'$\sigma^2_\delta + \sigma^2_\varepsilon$ (µm²)',
                  fontsize=9)
    ax.set_ylabel('density', fontsize=9)
    ax.legend(fontsize=8)
    suffix = f' — {title_suffix}' if title_suffix else ''
    ax.set_title(f'KO variance sum (better-identified than the split){suffix}',
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ── Config helpers ────────────────────────────────────────────────────────────


def _extract_config_params(config_used):
    """Pull M, b, f_inf, field prior, psi prior from a loaded config_used dict."""
    M = config_used['problem']['M']
    b = config_used['problem']['b']
    f_inf = config_used['problem']['distribution']['likelihood'][
        'transformations'][2]['b']
    coeff_dist = (config_used['problem']['distribution']['model']['field']
                  ['coefficient_distribution'])
    prior_mean = np.asarray(coeff_dist['mean'])
    prior_cov = np.asarray(coeff_dist['cov'])
    psi_prior = (config_used['problem']['distribution']['likelihood']
                 ['likelihood']['psi_prior'])
    psi_prior_mean = np.asarray(psi_prior['mean'])
    psi_prior_std = np.sqrt(np.diag(np.asarray(psi_prior['cov'])))
    return M, b, f_inf, prior_mean, prior_cov, psi_prior_mean, psi_prior_std


def _extract_config_params_standard(config_used):
    """Pull M, b, f_inf, field prior from a standard-likelihood config_used dict.

    Lighter version of _extract_config_params — omits psi_prior, which is
    absent from the GaussianLikelihood config.
    """
    M = config_used['problem']['M']
    b = config_used['problem']['b']
    f_inf = config_used['problem']['distribution']['likelihood'][
        'transformations'][2]['b']
    coeff_dist = (config_used['problem']['distribution']['model']['field']
                  ['coefficient_distribution'])
    prior_mean = np.asarray(coeff_dist['mean'])
    prior_cov = np.asarray(coeff_dist['cov'])
    return M, b, f_inf, prior_mean, prior_cov


def _standard_xlims(lower=None):
    upper = [1.0, 300., 130.]
    lo = lower if lower is not None else [0.0, 25.0, 0.0]
    return [[l, u] for l, u in zip(lo, upper)]


# ── Per-geometry separate analysis ───────────────────────────────────────────


def analyze_one_separate(geom_name, rwm_dir, burnin_frac):
    """Analyse a single separate geometry run; return post-burnin physical samples."""
    print(f"\n{'='*60}")
    print(f"Separate geometry: {geom_name}")
    print(f"{'='*60}")

    _print_acceptance_rate(rwm_dir)
    xi_chain = _load_samples(rwm_dir)

    config_used = get_config(os.path.join(rwm_dir, 'config_used.yaml'))
    M, b, f_inf, prior_mean, prior_cov, psi_prior_mean, psi_prior_std = \
        _extract_config_params(config_used)

    print("Transforming to physical space...")
    theta_chain, chain_physical = _to_physical(xi_chain, M, b, f_inf)

    burnin = int(burnin_frac * len(chain_physical))
    post = chain_physical[burnin:]
    post_theta = theta_chain[burnin:]
    _print_summary(post, label=f'geom {geom_name}')

    fig_dir = os.path.join(rwm_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    prior_var = np.diag(prior_cov)
    prior_std = np.sqrt(prior_var)
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    theta_xlims = _standard_xlims()
    theta_latent_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])

    _plot_traces(chain_physical,
                 burnin,
                 fig_dir,
                 title_suffix=f'geom {geom_name}')
    _plot_marginals(post, [0, 1, 2],
                    THETA_NAMES,
                    r'$\theta$ posterior marginals  ($\rho$, $l$, $f_\infty$)'
                    f' — geom {geom_name}',
                    'theta_marginals.pdf',
                    fig_dir,
                    xlims=theta_xlims,
                    outlier_pct=(0.0, 0.0),
                    prior_samples=prior_physical)
    _plot_marginals(post, [3, 4, 5],
                    PSI_NAMES,
                    r'$\psi$ posterior marginals (KO hyper-parameters)'
                    f' — geom {geom_name}',
                    'psi_marginals.pdf',
                    fig_dir,
                    prior_mean=psi_prior_mean,
                    prior_std=psi_prior_std)
    _plot_marginals(post_theta, [0, 1, 2],
                    THETA_LATENT_NAMES,
                    r'$\theta$ posterior marginals (latent space)'
                    f' — geom {geom_name}',
                    'theta_latent_marginals.pdf',
                    fig_dir,
                    xlims=theta_latent_xlims,
                    outlier_pct=0.0,
                    prior_mean=prior_mean[:3],
                    prior_std=prior_std[:3])
    _plot_pairplot(post, fig_dir, theta_xlims=theta_xlims, outlier_pct=1.0)
    _plot_xi_pairplot(xi_chain[burnin:], fig_dir, title_suffix=f'geom {geom_name}')
    _plot_psi_pairplot(post,
                       fig_dir,
                       psi_prior_mean=psi_prior_mean,
                       psi_prior_std=psi_prior_std,
                       title_suffix=f'geom {geom_name}')
    _plot_psi_sum_marginal(post,
                           fig_dir,
                           psi_prior_mean=psi_prior_mean,
                           psi_prior_std=psi_prior_std,
                           title_suffix=f'geom {geom_name}')

    out = os.path.join(rwm_dir, 'samples_physical.dat')
    np.savetxt(out,
               post,
               header='rho l f_inf log_s2d log_s2e log_rho_ko',
               fmt='%.6e')
    print(f"Saved physical samples: {out}")

    return post


def analyze_all_separate(separate_dir, burnin_frac):
    """Run per-geometry analysis for all completed separate runs.

    Returns a dict mapping geom_name → post-burnin physical samples array.
    """
    candidates = sorted(
        glob(os.path.join(separate_dir, '*', 'rwm', 'samples.dat')))
    if not candidates:
        print(f"No separate samples.dat found under {separate_dir}")
        return {}

    results = {}
    for samples_path in candidates:
        rwm_dir = os.path.dirname(samples_path)
        geom_name = os.path.basename(os.path.dirname(rwm_dir))
        try:
            post = analyze_one_separate(geom_name, rwm_dir, burnin_frac)
            results[geom_name] = post
        except Exception as exc:
            print(f"  WARNING: skipping geom {geom_name}: {exc}")

    return results


def load_separate_physical(separate_dir):
    """Load pre-computed samples_physical.dat files from all separate runs."""
    candidates = sorted(
        glob(os.path.join(separate_dir, '*', 'rwm', 'samples_physical.dat')))
    if not candidates:
        raise FileNotFoundError(
            f"No samples_physical.dat found under {separate_dir}. "
            "Run without --skip-separate first.")
    results = {}
    for path in candidates:
        geom_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        results[geom_name] = np.loadtxt(path, comments='#')
        print(f"Loaded separate physical samples: geom {geom_name} "
              f"({results[geom_name].shape[0]} samples)")
    return results


# ── Per-geometry separate analysis (standard likelihood) ─────────────────────


def analyze_one_separate_standard(geom_name, rwm_dir, burnin_frac):
    """Analyse a single separate geometry run (standard likelihood); return post-burnin physical samples."""
    print(f"\n{'='*60}")
    print(f"Separate geometry (standard): {geom_name}")
    print(f"{'='*60}")

    _print_acceptance_rate(rwm_dir)
    xi_chain = _load_samples(rwm_dir)

    config_used = get_config(os.path.join(rwm_dir, 'config_used.yaml'))
    M, b, f_inf, prior_mean, prior_cov = _extract_config_params_standard(
        config_used)

    print("Transforming to physical space...")
    theta_chain, chain_physical = _to_physical(xi_chain, M, b, f_inf)

    burnin = int(burnin_frac * len(chain_physical))
    post = chain_physical[burnin:]
    post_theta = theta_chain[burnin:]
    _print_summary(post, label=f'geom {geom_name} standard')

    fig_dir = os.path.join(rwm_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    prior_std = np.sqrt(np.diag(prior_cov))
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    theta_xlims = _standard_xlims()
    theta_latent_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])

    _plot_traces(chain_physical,
                 burnin,
                 fig_dir,
                 title_suffix=f'geom {geom_name} standard')
    _plot_marginals(post, [0, 1, 2],
                    THETA_NAMES,
                    r'$\theta$ posterior marginals  ($\rho$, $l$, $f_\infty$)'
                    f' — geom {geom_name} standard',
                    'theta_marginals.pdf',
                    fig_dir,
                    xlims=theta_xlims,
                    outlier_pct=(0.0, 0.0),
                    prior_samples=prior_physical)
    _plot_marginals(post_theta, [0, 1, 2],
                    THETA_LATENT_NAMES,
                    r'$\theta$ posterior marginals (latent space)'
                    f' — geom {geom_name} standard',
                    'theta_latent_marginals.pdf',
                    fig_dir,
                    xlims=theta_latent_xlims,
                    outlier_pct=0.0,
                    prior_mean=prior_mean[:3],
                    prior_std=prior_std[:3])
    _plot_pairplot(post, fig_dir, theta_xlims=theta_xlims, outlier_pct=1.0)
    _plot_xi_pairplot(xi_chain[burnin:], fig_dir, title_suffix=f'geom {geom_name} standard')

    out = os.path.join(rwm_dir, 'samples_physical.dat')
    np.savetxt(out, post, header='rho l f_inf', fmt='%.6e')
    print(f"Saved physical samples: {out}")

    return post


def analyze_all_separate_standard(separate_dir, burnin_frac):
    """Run per-geometry standard analysis for all completed separate standard runs.

    Returns a dict mapping geom_name → post-burnin physical samples array (3 columns).
    """
    candidates = sorted(
        glob(os.path.join(separate_dir, '*', 'rwm_standard', 'samples.dat')))
    if not candidates:
        print(f"No separate standard samples.dat found under {separate_dir}")
        return {}

    results = {}
    for samples_path in candidates:
        rwm_dir = os.path.dirname(samples_path)
        geom_name = os.path.basename(os.path.dirname(rwm_dir))
        try:
            post = analyze_one_separate_standard(geom_name, rwm_dir,
                                                 burnin_frac)
            results[geom_name] = post
        except Exception as exc:
            print(f"  WARNING: skipping geom {geom_name} standard: {exc}")

    return results


def load_separate_standard_physical(separate_dir):
    """Load pre-computed samples_physical.dat files from all separate standard runs."""
    candidates = sorted(
        glob(
            os.path.join(separate_dir, '*', 'rwm_standard',
                         'samples_physical.dat')))
    if not candidates:
        raise FileNotFoundError(
            f"No standard samples_physical.dat found under {separate_dir}. "
            "Run without --skip-separate-standard first.")
    results = {}
    for path in candidates:
        geom_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        results[geom_name] = np.loadtxt(path, comments='#')
        print(f"Loaded separate standard physical samples: geom {geom_name} "
              f"({results[geom_name].shape[0]} samples)")
    return results


# ── Joint analysis (mirrors joint/analyze_results.py) ────────────────────────


def analyze_joint(joint_dir, burnin_frac):
    """Analyse the joint run; return post-burnin physical samples."""
    rwm_dir = os.path.join(joint_dir, 'rwm')
    print(f"\n{'='*60}")
    print("Joint inference")
    print(f"{'='*60}")

    _print_acceptance_rate(rwm_dir)
    xi_chain = _load_samples(rwm_dir)

    config_used = get_config(os.path.join(rwm_dir, 'config_used.yaml'))
    M, b, f_inf, prior_mean, prior_cov, psi_prior_mean, psi_prior_std = \
        _extract_config_params(config_used)

    print("Transforming to physical space...")
    theta_chain, chain_physical = _to_physical(xi_chain, M, b, f_inf)

    burnin = int(burnin_frac * len(chain_physical))
    post = chain_physical[burnin:]
    post_theta = theta_chain[burnin:]
    _print_summary(post, label='joint')

    fig_dir = os.path.join(rwm_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    prior_var = np.diag(prior_cov)
    prior_std = np.sqrt(prior_var)
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    theta_xlims = _standard_xlims()
    theta_latent_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])

    _plot_traces(chain_physical, burnin, fig_dir, title_suffix='joint')
    _plot_marginals(
        post, [0, 1, 2],
        THETA_NAMES,
        r'$\theta$ posterior marginals  ($\rho$, $l$, $f_\infty$) — joint',
        'theta_marginals.pdf',
        fig_dir,
        xlims=theta_xlims,
        outlier_pct=(0.0, 0.0),
        prior_samples=prior_physical)
    _plot_marginals(
        post, [3, 4, 5],
        PSI_NAMES,
        r'$\psi$ posterior marginals (KO hyper-parameters) — joint',
        'psi_marginals.pdf',
        fig_dir,
        prior_mean=psi_prior_mean,
        prior_std=psi_prior_std)
    _plot_marginals(post_theta, [0, 1, 2],
                    THETA_LATENT_NAMES,
                    r'$\theta$ posterior marginals (latent space) — joint',
                    'theta_latent_marginals.pdf',
                    fig_dir,
                    xlims=theta_latent_xlims,
                    outlier_pct=0.0,
                    prior_mean=prior_mean[:3],
                    prior_std=prior_std[:3])
    _plot_pairplot(post, fig_dir, theta_xlims=theta_xlims, outlier_pct=1.0)
    _plot_xi_pairplot(xi_chain[burnin:], fig_dir, title_suffix='joint')
    _plot_psi_pairplot(post,
                       fig_dir,
                       psi_prior_mean=psi_prior_mean,
                       psi_prior_std=psi_prior_std,
                       title_suffix='joint')
    _plot_psi_sum_marginal(post,
                           fig_dir,
                           psi_prior_mean=psi_prior_mean,
                           psi_prior_std=psi_prior_std,
                           title_suffix='joint')

    out = os.path.join(rwm_dir, 'samples_physical.dat')
    np.savetxt(out,
               post,
               header='rho l f_inf log_s2d log_s2e log_rho_ko',
               fmt='%.6e')
    print(f"Saved physical samples: {out}")

    return post, prior_physical, prior_mean, prior_cov, f_inf, psi_prior_mean, psi_prior_std


def load_joint_physical(joint_dir):
    path = os.path.join(joint_dir, 'rwm', 'samples_physical.dat')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"samples_physical.dat not found in {joint_dir}/rwm. "
            "Run without --skip-joint first.")
    post = np.loadtxt(path, comments='#')
    print(f"Loaded joint physical samples: {post.shape[0]} samples")
    config_used = get_config(os.path.join(joint_dir, 'rwm', 'config_used.yml'))
    _, _, f_inf, prior_mean, prior_cov, psi_prior_mean, psi_prior_std = \
        _extract_config_params(config_used)
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    return post, prior_physical, prior_mean, prior_cov, f_inf, psi_prior_mean, psi_prior_std


def _load_joint_prior_params(joint_dir):
    """Load prior parameters from the joint config without requiring samples."""
    for ext in ('yaml', 'yml'):
        config_path = os.path.join(joint_dir, 'rwm', f'config_used.{ext}')
        if os.path.exists(config_path):
            config_used = get_config(config_path)
            _, _, f_inf, prior_mean, prior_cov, psi_prior_mean, psi_prior_std = \
                _extract_config_params(config_used)
            prior_physical = _sample_physical_prior(prior_mean,
                                                    prior_cov,
                                                    f_inf=f_inf)
            return prior_physical, prior_mean, prior_cov, f_inf, psi_prior_mean, psi_prior_std
    raise FileNotFoundError(
        f"No config_used.yaml/yml found in {joint_dir}/rwm/")


# ── Joint standard analysis ──────────────────────────────────────────────────


def analyze_joint_standard(joint_dir, burnin_frac):
    """Analyse joint/rwm_standard; return post-burnin physical samples (3D)."""
    rwm_dir = os.path.join(joint_dir, 'rwm_standard')
    print(f"\n{'='*60}")
    print("Joint inference (standard likelihood)")
    print(f"{'='*60}")

    _print_acceptance_rate(rwm_dir)
    xi_chain = _load_samples(rwm_dir)

    config_used = get_config(os.path.join(rwm_dir, 'config_used.yaml'))
    M, b, f_inf, prior_mean, prior_cov = _extract_config_params_standard(
        config_used)

    print("Transforming to physical space...")
    theta_chain, chain_physical = _to_physical(xi_chain, M, b, f_inf)

    burnin = int(burnin_frac * len(chain_physical))
    post = chain_physical[burnin:]
    post_theta = theta_chain[burnin:]
    _print_summary(post, label='joint standard')

    fig_dir = os.path.join(rwm_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    prior_std = np.sqrt(np.diag(prior_cov))
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    theta_xlims = _standard_xlims()
    theta_latent_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])

    _plot_traces(chain_physical,
                 burnin,
                 fig_dir,
                 title_suffix='joint standard')
    _plot_marginals(
        post, [0, 1, 2],
        THETA_NAMES,
        r'$\theta$ posterior marginals  ($\rho$, $l$, $f_\infty$) — joint standard',
        'theta_marginals.pdf',
        fig_dir,
        xlims=theta_xlims,
        outlier_pct=(0.0, 0.0),
        prior_samples=prior_physical)
    _plot_marginals(
        post_theta, [0, 1, 2],
        THETA_LATENT_NAMES,
        r'$\theta$ posterior marginals (latent space) — joint standard',
        'theta_latent_marginals.pdf',
        fig_dir,
        xlims=theta_latent_xlims,
        outlier_pct=0.0,
        prior_mean=prior_mean[:3],
        prior_std=prior_std[:3])
    _plot_pairplot(post, fig_dir, theta_xlims=theta_xlims, outlier_pct=1.0)
    _plot_xi_pairplot(xi_chain[burnin:], fig_dir, title_suffix='joint standard')

    out = os.path.join(rwm_dir, 'samples_physical.dat')
    np.savetxt(out, post, header='rho l f_inf', fmt='%.6e')
    print(f"Saved physical samples: {out}")

    return post, prior_physical, prior_mean, prior_cov, f_inf


def load_joint_standard_physical(joint_dir):
    path = os.path.join(joint_dir, 'rwm_standard', 'samples_physical.dat')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"samples_physical.dat not found in {joint_dir}/rwm_standard. "
            "Run without --skip-joint-standard first.")
    post = np.loadtxt(path, comments='#')
    print(f"Loaded joint standard physical samples: {post.shape[0]} samples")
    config_used = get_config(
        os.path.join(joint_dir, 'rwm_standard', 'config_used.yaml'))
    _, _, f_inf, prior_mean, prior_cov = _extract_config_params_standard(
        config_used)
    prior_physical = _sample_physical_prior(prior_mean, prior_cov, f_inf=f_inf)
    return post, prior_physical, prior_mean, prior_cov, f_inf


# ── Comparison plots ──────────────────────────────────────────────────────────


def _kde_on_grid(vals, xs):
    try:
        return gaussian_kde(vals)(xs)
    except Exception:
        return np.zeros_like(xs)


def _plot_marginals_comparison(separate_posts,
                               joint_post,
                               prior_physical,
                               psi_prior_mean,
                               psi_prior_std,
                               fig_dir,
                               theta_xlims=None,
                               joint_standard_post=None,
                               separate_standard_posts=None):
    """Two-panel figure: θ marginals and ψ marginals side by side, comparing
    each separate posterior, their mixture, the joint posterior, and the prior.
    """
    separate_list = list(separate_posts.values())
    has_separate = len(separate_list) > 0
    has_joint = joint_post is not None
    mixture = np.concatenate(separate_list, axis=0) if has_separate else None
    n_geoms = len(separate_list)

    for param_indices, names, filename, title, xlims, prior_kws in [
        ([0, 1, 2], THETA_NAMES, 'theta_marginals_comparison.pdf',
         r'$\theta$ marginals: separate vs joint  ($\rho$, $l$, $f_\infty$)',
         theta_xlims, {
             'prior_samples': prior_physical
         }),
        ([3, 4, 5], PSI_NAMES, 'psi_marginals_comparison.pdf',
         r'$\psi$ marginals: separate vs joint (KO hyper-parameters)', None, {
             'prior_mean': psi_prior_mean,
             'prior_std': psi_prior_std
         }),
    ]:
        n = len(param_indices)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for k, (ax, i) in enumerate(zip(axes, param_indices)):
            xlim = xlims[
                k] if xlims is not None else None  # [lo, hi] pair or None

            # Collect all values to set a common x-range
            parts = []
            if has_separate:
                parts.extend([post[:, i] for post in separate_list])
            if has_joint:
                parts.append(joint_post[:, i])
            if not parts:
                continue
            all_vals = np.concatenate(parts)
            if xlim is not None:
                xs = np.linspace(xlim[0], xlim[1], 400)
            else:
                xs = np.linspace(float(all_vals.min()), float(all_vals.max()),
                                 400)

            def _clip(vals):
                if xlim is None:
                    return vals
                return vals[(vals >= xlim[0]) & (vals <= xlim[1])]

            # Individual separate posteriors
            if has_separate:
                for idx, post in enumerate(separate_list):
                    vals = _clip(post[:, i])
                    if len(vals) < 2:
                        continue
                    kde_y = _kde_on_grid(vals, xs)
                    label = 'separate' if idx == 0 else None
                    ax.plot(xs,
                            kde_y,
                            color='steelblue',
                            lw=0.7,
                            alpha=0.35,
                            label=label)

                # Mixture of separate posteriors
                mix_vals = _clip(mixture[:, i])
                ax.plot(xs,
                        _kde_on_grid(mix_vals, xs),
                        color='steelblue',
                        lw=2.0,
                        ls='--',
                        label=f'mixture (n={n_geoms})')
                ax.axvline(np.median(mix_vals),
                           color='steelblue',
                           lw=1.0,
                           ls=':',
                           alpha=0.7)

            # Joint posterior
            if has_joint:
                j_vals = _clip(joint_post[:, i])
                ax.plot(xs,
                        _kde_on_grid(j_vals, xs),
                        color='C1',
                        lw=2.0,
                        label='joint')
                ax.axvline(np.median(j_vals),
                           color='C1',
                           lw=1.0,
                           ls=':',
                           alpha=0.7)

            # Joint standard posterior (θ panel only)
            if joint_standard_post is not None and param_indices[0] == 0:
                js_vals = _clip(joint_standard_post[:, i])
                ax.plot(xs,
                        _kde_on_grid(js_vals, xs),
                        color='C3',
                        lw=2.0,
                        label='joint standard')
                ax.axvline(np.median(js_vals),
                           color='C3',
                           lw=1.0,
                           ls=':',
                           alpha=0.7)

            # Mixture of separate standard posteriors (θ panel only)
            if separate_standard_posts and param_indices[0] == 0:
                std_sep_list = list(separate_standard_posts.values())
                n_std = len(std_sep_list)
                std_mix = np.concatenate(std_sep_list, axis=0)
                for idx, post in enumerate(std_sep_list):
                    vals = _clip(post[:, i])
                    if len(vals) < 2:
                        continue
                    ax.plot(xs,
                            _kde_on_grid(vals, xs),
                            color='C4',
                            lw=0.7,
                            alpha=0.35,
                            label='separate std' if idx == 0 else None)
                std_mix_vals = _clip(std_mix[:, i])
                if len(std_mix_vals) >= 2:
                    ax.plot(xs,
                            _kde_on_grid(std_mix_vals, xs),
                            color='C4',
                            lw=2.0,
                            ls='--',
                            label=f'mixture std (n={n_std})')
                    ax.axvline(np.median(std_mix_vals),
                               color='C4',
                               lw=1.0,
                               ls=':',
                               alpha=0.7)

            # Prior
            if 'prior_samples' in prior_kws:
                p_vals = prior_kws['prior_samples'][:, i]
                xs_p = np.linspace(float(p_vals.min()), float(p_vals.max()),
                                   400)
                ax.plot(xs_p,
                        _kde_on_grid(p_vals, xs_p),
                        color='C2',
                        lw=1.5,
                        ls='--',
                        label='prior')
            elif 'prior_mean' in prior_kws:
                from scipy.stats import norm
                ax.plot(xs,
                        norm.pdf(xs, prior_kws['prior_mean'][k],
                                 prior_kws['prior_std'][k]),
                        color='C2',
                        lw=1.5,
                        ls='--',
                        label='prior')

            if xlim is not None:
                ax.set_xlim(*xlim)

            ax.set_xlabel(names[k], fontsize=9)
            ax.set_ylabel('density', fontsize=9)
            ax.tick_params(labelsize=8)
            if k == 0:
                ax.legend(fontsize=7)

        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        path = os.path.join(fig_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")


def _physical_to_theta(post, f_inf):
    """Inverse of the composite transform for columns 0–2.

    Columns 3–5 (log-psi) are returned unchanged — they are already in θ-space.
    """
    out = post.copy()
    rho = np.clip(post[:, 0], 1e-8, 1 - 1e-8)
    out[:, 0] = np.log(rho / (1.0 - rho))  # logit(ρ)
    out[:, 1] = np.log(post[:, 1])  # log(l)
    f = np.clip(post[:, 2], 1e-8, f_inf - 1e-8)
    out[:, 2] = np.log(f / (f_inf - f))  # logit(f_∞ / f_inf)
    return out


def _plot_theta_latent_comparison(separate_posts,
                                  joint_post,
                                  f_inf,
                                  prior_mean,
                                  prior_cov,
                                  fig_dir,
                                  joint_standard_post=None,
                                  separate_standard_posts=None):
    """Overlaid KDE comparison for logit(ρ), log l, logit(f_∞/f_inf)."""
    if f_inf is None:
        print("WARNING: f_inf unavailable — skipping latent theta comparison.")
        return

    has_separate = bool(separate_posts)
    has_joint = joint_post is not None

    separate_thetas = [
        _physical_to_theta(p, f_inf) for p in separate_posts.values()
    ] if has_separate else []
    mixture_theta = np.concatenate(separate_thetas,
                                   axis=0) if has_separate else None
    joint_theta = _physical_to_theta(joint_post, f_inf) if has_joint else None
    n_geoms = len(separate_thetas)

    if prior_mean is not None and prior_cov is not None:
        prior_std = np.sqrt(np.diag(prior_cov))
        theta_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])
    else:
        prior_std = None
        theta_xlims = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for k, ax in enumerate(axes):
        parts = []
        if has_separate:
            parts.extend([t[:, k] for t in separate_thetas])
        if has_joint:
            parts.append(joint_theta[:, k])
        if not parts:
            continue
        all_vals = np.concatenate(parts)

        if theta_xlims is not None:
            lo, hi = theta_xlims[k]
        else:
            lo, hi = float(np.percentile(all_vals, 0.5)), float(
                np.percentile(all_vals, 99.5))
        xs = np.linspace(max(lo, float(np.percentile(all_vals, 0.5))),
                         min(hi, float(np.percentile(all_vals, 99.5))), 400)

        if has_separate:
            for idx, t in enumerate(separate_thetas):
                vals = t[:, k]
                vals = vals[(vals >= xs[0]) & (vals <= xs[-1])]
                if len(vals) < 2:
                    continue
                ax.plot(xs,
                        _kde_on_grid(vals, xs),
                        color='steelblue',
                        lw=0.7,
                        alpha=0.35,
                        label='separate' if idx == 0 else None)

            mix_vals = mixture_theta[:, k]
            mix_vals = mix_vals[(mix_vals >= xs[0]) & (mix_vals <= xs[-1])]
            ax.plot(xs,
                    _kde_on_grid(mix_vals, xs),
                    color='steelblue',
                    lw=2.0,
                    ls='--',
                    label=f'mixture (n={n_geoms})')
            ax.axvline(np.median(mix_vals),
                       color='steelblue',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)

        if has_joint:
            j_vals = joint_theta[:, k]
            j_vals = j_vals[(j_vals >= xs[0]) & (j_vals <= xs[-1])]
            ax.plot(xs,
                    _kde_on_grid(j_vals, xs),
                    color='C1',
                    lw=2.0,
                    label='joint')
            ax.axvline(np.median(j_vals),
                       color='C1',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)

        if joint_standard_post is not None:
            js_theta = _physical_to_theta(joint_standard_post, f_inf)
            js_vals = js_theta[:, k]
            js_vals = js_vals[(js_vals >= xs[0]) & (js_vals <= xs[-1])]
            ax.plot(xs,
                    _kde_on_grid(js_vals, xs),
                    color='C3',
                    lw=2.0,
                    label='joint standard')
            ax.axvline(np.median(js_vals),
                       color='C3',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)

        if separate_standard_posts:
            std_sep_thetas = [
                _physical_to_theta(p, f_inf)
                for p in separate_standard_posts.values()
            ]
            n_std = len(std_sep_thetas)
            std_mix_theta = np.concatenate(std_sep_thetas, axis=0)
            for idx, t in enumerate(std_sep_thetas):
                vals = t[:, k]
                vals = vals[(vals >= xs[0]) & (vals <= xs[-1])]
                if len(vals) < 2:
                    continue
                ax.plot(xs,
                        _kde_on_grid(vals, xs),
                        color='C4',
                        lw=0.7,
                        alpha=0.35,
                        label='separate std' if idx == 0 else None)
            std_mix_vals = std_mix_theta[:, k]
            std_mix_vals = std_mix_vals[(std_mix_vals >= xs[0])
                                        & (std_mix_vals <= xs[-1])]
            if len(std_mix_vals) >= 2:
                ax.plot(xs,
                        _kde_on_grid(std_mix_vals, xs),
                        color='C4',
                        lw=2.0,
                        ls='--',
                        label=f'mixture std (n={n_std})')
                ax.axvline(np.median(std_mix_vals),
                           color='C4',
                           lw=1.0,
                           ls=':',
                           alpha=0.7)

        if prior_mean is not None and prior_std is not None:
            from scipy.stats import norm
            ax.plot(xs,
                    norm.pdf(xs, prior_mean[k], prior_std[k]),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')

        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel(THETA_LATENT_NAMES[k], fontsize=9)
        ax.set_ylabel('density', fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        r'$\theta$ marginals (latent space): separate vs joint'
        r' — $\mathrm{logit}(\rho)$, $\log\,l$, $\mathrm{logit}(f_\infty/f_\infty^{\max})$',
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'theta_latent_marginals_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _psi_to_physical(post):
    """Exp-transform columns 3–5 (log-psi) to linear physical units."""
    out = post.copy()
    out[:, 3:] = np.exp(post[:, 3:])
    return out


def _plot_psi_comparison_physical(separate_posts, joint_post,
                                  prior_physical_psi, fig_dir):
    """KDE comparison for σ²_δ, σ²_ε, ρ_KO in linear (physical) units.

    prior_physical_psi: array (n × 3) of prior samples already exp-transformed.
    """
    separate_list = [_psi_to_physical(p) for p in separate_posts.values()]
    has_separate = len(separate_list) > 0
    has_joint = joint_post is not None
    mixture = np.concatenate(separate_list, axis=0) if has_separate else None
    joint_phys = _psi_to_physical(joint_post) if has_joint else None
    n_geoms = len(separate_list)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for k, ax in enumerate(axes):
        i = k + 3  # column index in the full array

        parts = []
        if has_separate:
            parts.extend([p[:, i] for p in separate_list])
        if has_joint:
            parts.append(joint_phys[:, i])
        if not parts:
            continue
        all_vals = np.concatenate(parts)
        xs = np.linspace(float(np.percentile(all_vals, 1)),
                         float(np.percentile(all_vals, 99)), 400)

        if has_separate:
            for idx, post in enumerate(separate_list):
                vals = post[:, i]
                vals = vals[(vals >= xs[0]) & (vals <= xs[-1])]
                if len(vals) < 2:
                    continue
                ax.plot(xs,
                        _kde_on_grid(vals, xs),
                        color='steelblue',
                        lw=0.7,
                        alpha=0.35,
                        label='separate' if idx == 0 else None)

            mix_vals = mixture[:, i]
            mix_vals = mix_vals[(mix_vals >= xs[0]) & (mix_vals <= xs[-1])]
            ax.plot(xs,
                    _kde_on_grid(mix_vals, xs),
                    color='steelblue',
                    lw=2.0,
                    ls='--',
                    label=f'mixture (n={n_geoms})')
            ax.axvline(np.median(mix_vals),
                       color='steelblue',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)

        if has_joint:
            j_vals = joint_phys[:, i]
            j_vals = j_vals[(j_vals >= xs[0]) & (j_vals <= xs[-1])]
            ax.plot(xs,
                    _kde_on_grid(j_vals, xs),
                    color='C1',
                    lw=2.0,
                    label='joint')
            ax.axvline(np.median(j_vals),
                       color='C1',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)

        if prior_physical_psi is not None:
            p_vals = prior_physical_psi[:, k]
            xs_p = np.linspace(float(np.percentile(p_vals, 1)),
                               float(np.percentile(p_vals, 99)), 400)
            ax.plot(xs_p,
                    _kde_on_grid(p_vals, xs_p),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')

        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel(PSI_PHYSICAL_NAMES[k], fontsize=9)
        ax.set_ylabel('density', fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        r'$\psi$ marginals (physical units): separate vs joint'
        r' — $\sigma^2_\delta$, $\sigma^2_\varepsilon$, $\rho_\mathrm{KO}$',
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'psi_marginals_comparison_physical.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_psi_sum_comparison(separate_posts,
                             joint_post,
                             fig_dir,
                             psi_prior_mean=None,
                             psi_prior_std=None,
                             filename='psi_sum_comparison.pdf'):
    """Overlay σ²_δ + σ²_ε for each separate run, the mixture, and the joint.

    The diagonal sum is well-identified by the residual scale even when σ²_δ
    and σ²_ε individually sit on a confounding ridge — checking that this sum
    agrees across separate/joint inference is the right consistency check.
    """
    separate_list = list(separate_posts.values())
    has_separate = len(separate_list) > 0
    has_joint = joint_post is not None

    def _sum(p):
        return np.exp(p[:, 3]) + np.exp(p[:, 4])

    parts = []
    if has_separate:
        parts.extend([_sum(p) for p in separate_list])
    if has_joint:
        parts.append(_sum(joint_post))
    if not parts:
        return
    all_vals = np.concatenate(parts)
    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    xs = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if has_separate:
        for idx, p in enumerate(separate_list):
            vals = _sum(p)
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) < 2:
                continue
            ax.plot(xs,
                    _kde_on_grid(vals, xs),
                    color='steelblue',
                    lw=0.7,
                    alpha=0.35,
                    label='separate' if idx == 0 else None)
        mix = np.concatenate([_sum(p) for p in separate_list])
        mix = mix[(mix >= lo) & (mix <= hi)]
        if len(mix) >= 2:
            ax.plot(xs,
                    _kde_on_grid(mix, xs),
                    color='steelblue',
                    lw=2.0,
                    ls='--',
                    label=f'mixture (n={len(separate_list)})')
            ax.axvline(float(np.median(mix)),
                       color='steelblue',
                       lw=1.0,
                       ls=':',
                       alpha=0.7)
    if has_joint:
        j_vals = _sum(joint_post)
        j_vals = j_vals[(j_vals >= lo) & (j_vals <= hi)]
        ax.plot(xs,
                _kde_on_grid(j_vals, xs),
                color='C1',
                lw=2.0,
                label='joint')
        ax.axvline(float(np.median(j_vals)),
                   color='C1',
                   lw=1.0,
                   ls=':',
                   alpha=0.7)
    if psi_prior_mean is not None and psi_prior_std is not None:
        rng = np.random.default_rng(0)
        log_psi = rng.multivariate_normal(psi_prior_mean,
                                          np.diag(psi_prior_std**2),
                                          size=200_000)
        prior_sum = np.exp(log_psi[:, 0]) + np.exp(log_psi[:, 1])
        prior_sum = prior_sum[(prior_sum >= lo) & (prior_sum <= hi)]
        if len(prior_sum) > 2:
            ax.plot(xs,
                    _kde_on_grid(prior_sum, xs),
                    color='C2',
                    lw=1.5,
                    ls='--',
                    label='prior')
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r'$\sigma^2_\delta + \sigma^2_\varepsilon$ (µm²)',
                  fontsize=9)
    ax.set_ylabel('density', fontsize=9)
    ax.set_title('KO variance sum: separate vs joint', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def _plot_pairplot_comparison(separate_posts,
                              joint_post,
                              fig_dir,
                              theta_xlims=None,
                              outlier_pct=1.0,
                              max_scatter=1000,
                              joint_standard_post=None,
                              separate_standard_posts=None):
    """3×3 pairwise scatter for ρ, l, f_∞: mixture of separate (blue) vs joint (orange)."""
    separate_list = list(separate_posts.values())
    has_separate = len(separate_list) > 0
    has_joint = joint_post is not None
    mixture = np.concatenate(separate_list,
                             axis=0)[:, :3] if has_separate else None
    joint_params = joint_post[:, :3] if has_joint else None

    # Optional outlier removal
    if outlier_pct is not None:
        if has_separate:
            mask_mix = np.ones(len(mixture), dtype=bool)
            for k in range(3):
                mask_mix &= _filter_outliers(mixture[:, k], outlier_pct)
            mixture = mixture[mask_mix]
        if has_joint:
            mask_jnt = np.ones(len(joint_params), dtype=bool)
            for k in range(3):
                mask_jnt &= _filter_outliers(joint_params[:, k], outlier_pct)
            joint_params = joint_params[mask_jnt]

    std_params = None
    scatter_std = None
    if joint_standard_post is not None:
        std_params = joint_standard_post[:, :3]
        if outlier_pct is not None:
            mask_std = np.ones(len(std_params), dtype=bool)
            for k in range(3):
                mask_std &= _filter_outliers(std_params[:, k], outlier_pct)
            std_params = std_params[mask_std]

    std_sep_mixture = None
    scatter_std_sep = None
    if separate_standard_posts:
        std_sep_mixture = np.concatenate(list(
            separate_standard_posts.values()),
                                         axis=0)[:, :3]
        if outlier_pct is not None:
            mask_ss = np.ones(len(std_sep_mixture), dtype=bool)
            for k in range(3):
                mask_ss &= _filter_outliers(std_sep_mixture[:, k], outlier_pct)
            std_sep_mixture = std_sep_mixture[mask_ss]

    rng = np.random.default_rng(0)
    scatter_mix = None
    scatter_jnt = None
    if has_separate:
        idx_mix = rng.choice(len(mixture),
                             size=min(max_scatter, len(mixture)),
                             replace=False)
        scatter_mix = mixture[idx_mix]
    if has_joint:
        idx_jnt = rng.choice(len(joint_params),
                             size=min(max_scatter, len(joint_params)),
                             replace=False)
        scatter_jnt = joint_params[idx_jnt]
    if std_params is not None:
        idx_std = rng.choice(len(std_params),
                             size=min(max_scatter, len(std_params)),
                             replace=False)
        scatter_std = std_params[idx_std]
    if std_sep_mixture is not None:
        idx_ss = rng.choice(len(std_sep_mixture),
                            size=min(max_scatter, len(std_sep_mixture)),
                            replace=False)
        scatter_std_sep = std_sep_mixture[idx_ss]

    names = THETA_NAMES
    d = 3
    fig, axes = plt.subplots(d, d, figsize=(8, 8))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            xlim = theta_xlims[j] if theta_xlims is not None else None
            ylim = theta_xlims[i] if theta_xlims is not None else None
            if i == j:
                if has_separate:
                    ax.hist(mixture[:, i],
                            bins=40,
                            density=True,
                            color='steelblue',
                            alpha=0.4,
                            edgecolor='none',
                            label='mixture')
                if has_joint:
                    ax.hist(joint_params[:, i],
                            bins=40,
                            density=True,
                            color='C1',
                            alpha=0.4,
                            edgecolor='none',
                            label='joint')
                if std_params is not None:
                    ax.hist(std_params[:, i],
                            bins=40,
                            density=True,
                            color='C3',
                            alpha=0.4,
                            edgecolor='none',
                            label='joint standard')
                if std_sep_mixture is not None:
                    ax.hist(std_sep_mixture[:, i],
                            bins=40,
                            density=True,
                            color='C4',
                            alpha=0.4,
                            edgecolor='none',
                            label='mixture std')
                ax.set_xlabel(names[i], fontsize=8)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                if i == 0:
                    ax.legend(fontsize=7)
            elif i > j:
                if scatter_mix is not None:
                    ax.scatter(scatter_mix[:, j],
                               scatter_mix[:, i],
                               s=2.5,
                               alpha=0.4,
                               color='steelblue',
                               linewidths=0)
                if scatter_jnt is not None:
                    ax.scatter(scatter_jnt[:, j],
                               scatter_jnt[:, i],
                               s=2.5,
                               alpha=0.4,
                               color='C1',
                               linewidths=0)
                if scatter_std is not None:
                    ax.scatter(scatter_std[:, j],
                               scatter_std[:, i],
                               s=2.5,
                               alpha=0.4,
                               color='C3',
                               linewidths=0)
                if scatter_std_sep is not None:
                    ax.scatter(scatter_std_sep[:, j],
                               scatter_std_sep[:, i],
                               s=2.5,
                               alpha=0.4,
                               color='C4',
                               linewidths=0)
                ax.set_xlabel(names[j], fontsize=8)
                ax.set_ylabel(names[i], fontsize=8)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                if ylim is not None:
                    ax.set_ylim(*ylim)
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        r'Pairwise marginals — $\rho$, $l$, $f_\infty$: mixture vs joint',
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'theta_pairplot_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Analyse separate and joint ITZ inference results.')
    parser.add_argument('--separate-dir',
                        default=os.path.join(HERE, 'separate'))
    parser.add_argument('--joint-dir', default=os.path.join(HERE, 'joint'))
    parser.add_argument('--burnin-frac',
                        type=float,
                        default=0.05,
                        help='Fraction of samples to discard as burn-in')
    parser.add_argument('--skip-separate',
                        action='store_true',
                        help='Skip separate analysis entirely (no processing, '
                        'no plotting)')
    parser.add_argument('--skip-joint',
                        action='store_true',
                        help='Skip joint analysis entirely (no processing, '
                        'no plotting); prior params are still loaded from '
                        'config for overlays')
    parser.add_argument('--skip-comparison',
                        action='store_true',
                        help='Skip comparison figures')
    parser.add_argument('--skip-joint-standard',
                        action='store_true',
                        help='Skip joint standard analysis entirely (no '
                        'processing, no plotting)')
    parser.add_argument('--skip-separate-standard',
                        action='store_true',
                        help='Skip separate standard analysis entirely (no '
                        'processing, no plotting)')
    args = parser.parse_args()

    # ── Separate analysis ────────────────────────────────────────────────────
    separate_posts = {}
    if not args.skip_separate:
        separate_posts = analyze_all_separate(args.separate_dir,
                                              args.burnin_frac)

    # ── Joint analysis ───────────────────────────────────────────────────────
    joint_post = None
    prior_physical = prior_mean = prior_cov = f_inf = None
    psi_prior_mean = psi_prior_std = None
    if not args.skip_joint:
        (joint_post, prior_physical, prior_mean, prior_cov, f_inf,
         psi_prior_mean,
         psi_prior_std) = analyze_joint(args.joint_dir, args.burnin_frac)
    else:
        try:
            (prior_physical, prior_mean, prior_cov, f_inf, psi_prior_mean,
             psi_prior_std) = _load_joint_prior_params(args.joint_dir)
        except (FileNotFoundError, KeyError) as exc:
            print(f"WARNING: could not load joint prior params: {exc}")

    # ── Separate standard analysis ───────────────────────────────────────────
    separate_standard_posts = {}
    if not args.skip_separate_standard:
        separate_standard_posts = analyze_all_separate_standard(
            args.separate_dir, args.burnin_frac)

    # ── Joint standard analysis ──────────────────────────────────────────────
    joint_standard_post = None
    if not args.skip_joint_standard:
        try:
            joint_standard_post, *_ = analyze_joint_standard(
                args.joint_dir, args.burnin_frac)
        except FileNotFoundError as exc:
            print(
                f"WARNING: joint standard run not available — skipping: {exc}")

    # ── Comparison figures ───────────────────────────────────────────────────
    if not args.skip_comparison:
        has_data = (separate_posts or joint_post is not None
                    or joint_standard_post is not None
                    or separate_standard_posts)
        if not has_data:
            print("Nothing to compare — all data sources skipped.")
        else:
            fig_dir = os.path.join(HERE, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print("Comparison figures")
            print(f"{'='*60}")
            theta_xlims = _standard_xlims()
            _plot_marginals_comparison(
                separate_posts,
                joint_post,
                prior_physical,
                psi_prior_mean,
                psi_prior_std,
                fig_dir,
                theta_xlims=theta_xlims,
                joint_standard_post=joint_standard_post,
                separate_standard_posts=separate_standard_posts)
            _plot_theta_latent_comparison(
                separate_posts,
                joint_post,
                f_inf,
                prior_mean,
                prior_cov,
                fig_dir,
                joint_standard_post=joint_standard_post,
                separate_standard_posts=separate_standard_posts)
            prior_psi_physical = None
            if psi_prior_mean is not None and psi_prior_std is not None:
                _rng = np.random.default_rng(0)
                _psi_log_samples = _rng.multivariate_normal(
                    psi_prior_mean, np.diag(psi_prior_std**2), size=200_000)
                prior_psi_physical = np.exp(_psi_log_samples)
            _plot_psi_comparison_physical(separate_posts, joint_post,
                                          prior_psi_physical, fig_dir)
            _plot_psi_sum_comparison(separate_posts,
                                     joint_post,
                                     fig_dir,
                                     psi_prior_mean=psi_prior_mean,
                                     psi_prior_std=psi_prior_std)
            _plot_pairplot_comparison(
                separate_posts,
                joint_post,
                fig_dir,
                theta_xlims=theta_xlims,
                outlier_pct=0.0,
                joint_standard_post=joint_standard_post,
                separate_standard_posts=separate_standard_posts)
            print(f"\nComparison figures written to {fig_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    main()
