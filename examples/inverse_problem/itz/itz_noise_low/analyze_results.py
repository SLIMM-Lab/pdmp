#!/usr/bin/env python3
"""Analyse RWM + K&O results for both separate and joint ITZ geometry inference.

Produces per-geometry figures (same as joint/analyze_results.py) for each
separate run, plus combined comparison figures overlaying all separate
posteriors against the joint posterior.

Usage:
    python analyze_results.py [--separate-dir ./separate] [--joint-dir ./joint]
                              [--burnin-frac 0.25]
                              [--skip-separate] [--skip-joint] [--skip-comparison]

Output:
    separate/NN/rwm/figures/   — traces, marginals, pairplot per geometry
    separate/NN/rwm/samples_physical.dat
    joint/rwm/figures/         — same plots for joint run
    joint/rwm/samples_physical.dat
    figures/                   — comparison plots (separate mixture vs joint)
"""

import argparse
import os
import pickle
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
                               joint_standard_post=None):
    """Two-panel figure: θ marginals and ψ marginals side by side, comparing
    each separate posterior, their mixture, the joint posterior, and the prior.
    """
    separate_list = list(separate_posts.values())
    mixture = np.concatenate(separate_list, axis=0)
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
            all_vals = np.concatenate([post[:, i] for post in separate_list] +
                                      [joint_post[:, i]])
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
                                  joint_standard_post=None):
    """Overlaid KDE comparison for logit(ρ), log l, logit(f_∞/f_inf)."""
    prior_std = np.sqrt(np.diag(prior_cov))
    theta_xlims = _theta_xlims(prior_mean[:3], prior_std[:3])

    separate_thetas = [
        _physical_to_theta(p, f_inf) for p in separate_posts.values()
    ]
    mixture_theta = np.concatenate(separate_thetas, axis=0)
    joint_theta = _physical_to_theta(joint_post, f_inf)
    n_geoms = len(separate_thetas)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for k, ax in enumerate(axes):
        lo, hi = theta_xlims[k]

        all_vals = np.concatenate([t[:, k] for t in separate_thetas] +
                                  [joint_theta[:, k]])
        xs = np.linspace(max(lo, float(np.percentile(all_vals, 0.5))),
                         min(hi, float(np.percentile(all_vals, 99.5))), 400)

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

        j_vals = joint_theta[:, k]
        j_vals = j_vals[(j_vals >= xs[0]) & (j_vals <= xs[-1])]
        ax.plot(xs,
                _kde_on_grid(j_vals, xs),
                color='C1',
                lw=2.0,
                label='joint')
        ax.axvline(np.median(j_vals), color='C1', lw=1.0, ls=':', alpha=0.7)

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
    mixture = np.concatenate(separate_list, axis=0)
    joint_phys = _psi_to_physical(joint_post)
    n_geoms = len(separate_list)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for k, ax in enumerate(axes):
        i = k + 3  # column index in the full array

        all_vals = np.concatenate([p[:, i] for p in separate_list] +
                                  [joint_phys[:, i]])
        xs = np.linspace(float(np.percentile(all_vals, 1)),
                         float(np.percentile(all_vals, 99)), 400)

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

        j_vals = joint_phys[:, i]
        j_vals = j_vals[(j_vals >= xs[0]) & (j_vals <= xs[-1])]
        ax.plot(xs,
                _kde_on_grid(j_vals, xs),
                color='C1',
                lw=2.0,
                label='joint')
        ax.axvline(np.median(j_vals), color='C1', lw=1.0, ls=':', alpha=0.7)

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


def _plot_pairplot_comparison(separate_posts,
                              joint_post,
                              fig_dir,
                              theta_xlims=None,
                              outlier_pct=1.0,
                              max_scatter=1000,
                              joint_standard_post=None):
    """3×3 pairwise scatter for ρ, l, f_∞: mixture of separate (blue) vs joint (orange)."""
    separate_list = list(separate_posts.values())
    mixture = np.concatenate(separate_list, axis=0)[:, :3]
    joint_params = joint_post[:, :3]

    # Optional outlier removal
    if outlier_pct is not None:
        mask_mix = np.ones(len(mixture), dtype=bool)
        mask_jnt = np.ones(len(joint_params), dtype=bool)
        for k in range(3):
            mask_mix &= _filter_outliers(mixture[:, k], outlier_pct)
            mask_jnt &= _filter_outliers(joint_params[:, k], outlier_pct)
        mixture = mixture[mask_mix]
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

    rng = np.random.default_rng(0)
    idx_mix = rng.choice(len(mixture),
                         size=min(max_scatter, len(mixture)),
                         replace=False)
    idx_jnt = rng.choice(len(joint_params),
                         size=min(max_scatter, len(joint_params)),
                         replace=False)
    scatter_mix = mixture[idx_mix]
    scatter_jnt = joint_params[idx_jnt]
    if std_params is not None:
        idx_std = rng.choice(len(std_params),
                             size=min(max_scatter, len(std_params)),
                             replace=False)
        scatter_std = std_params[idx_std]

    names = THETA_NAMES
    d = 3
    fig, axes = plt.subplots(d, d, figsize=(8, 8))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            xlim = theta_xlims[j] if theta_xlims is not None else None
            ylim = theta_xlims[i] if theta_xlims is not None else None
            if i == j:
                ax.hist(mixture[:, i],
                        bins=40,
                        density=True,
                        color='steelblue',
                        alpha=0.4,
                        edgecolor='none',
                        label='mixture')
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
                ax.set_xlabel(names[i], fontsize=8)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                if i == 0:
                    ax.legend(fontsize=7)
            elif i > j:
                ax.scatter(scatter_mix[:, j],
                           scatter_mix[:, i],
                           s=2.5,
                           alpha=0.4,
                           color='steelblue',
                           linewidths=0)
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
                        help='Skip per-geometry analysis; load existing '
                        'samples_physical.dat files instead')
    parser.add_argument('--skip-joint',
                        action='store_true',
                        help='Skip joint analysis; load existing '
                        'samples_physical.dat instead')
    parser.add_argument('--skip-comparison',
                        action='store_true',
                        help='Skip comparison figures')
    parser.add_argument('--skip-joint-standard',
                        action='store_true',
                        help='Skip joint standard analysis; load existing '
                        'samples_physical.dat instead')
    args = parser.parse_args()

    # ── Separate analysis ────────────────────────────────────────────────────
    if not args.skip_separate:
        separate_posts = analyze_all_separate(args.separate_dir,
                                              args.burnin_frac)
    else:
        separate_posts = load_separate_physical(args.separate_dir)

    if not separate_posts:
        print("No separate results available — skipping comparison.")
        return

    # ── Joint analysis ───────────────────────────────────────────────────────
    if not args.skip_joint:
        (joint_post, prior_physical, prior_mean, prior_cov, f_inf,
         psi_prior_mean,
         psi_prior_std) = analyze_joint(args.joint_dir, args.burnin_frac)
    else:
        (joint_post, prior_physical, prior_mean, prior_cov, f_inf,
         psi_prior_mean, psi_prior_std) = load_joint_physical(args.joint_dir)

    # ── Joint standard analysis ──────────────────────────────────────────────
    joint_standard_post = None
    try:
        if not args.skip_joint_standard:
            joint_standard_post, *_ = analyze_joint_standard(
                args.joint_dir, args.burnin_frac)
        else:
            joint_standard_post, *_ = load_joint_standard_physical(
                args.joint_dir)
    except FileNotFoundError as exc:
        print(f"WARNING: joint standard run not available — skipping: {exc}")

    # ── Comparison figures ───────────────────────────────────────────────────
    if not args.skip_comparison:
        fig_dir = os.path.join(HERE, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("Comparison figures")
        print(f"{'='*60}")
        theta_xlims = _standard_xlims()
        _plot_marginals_comparison(separate_posts,
                                   joint_post,
                                   prior_physical,
                                   psi_prior_mean,
                                   psi_prior_std,
                                   fig_dir,
                                   theta_xlims=theta_xlims,
                                   joint_standard_post=joint_standard_post)
        _plot_theta_latent_comparison(separate_posts,
                                      joint_post,
                                      f_inf,
                                      prior_mean,
                                      prior_cov,
                                      fig_dir,
                                      joint_standard_post=joint_standard_post)
        _rng = np.random.default_rng(0)
        _psi_log_samples = _rng.multivariate_normal(psi_prior_mean,
                                                    np.diag(psi_prior_std**2),
                                                    size=200_000)
        prior_psi_physical = np.exp(_psi_log_samples)
        _plot_psi_comparison_physical(separate_posts, joint_post,
                                      prior_psi_physical, fig_dir)
        _plot_pairplot_comparison(separate_posts,
                                  joint_post,
                                  fig_dir,
                                  theta_xlims=theta_xlims,
                                  outlier_pct=0.0,
                                  joint_standard_post=joint_standard_post)
        print(f"\nComparison figures written to {fig_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    main()
