#!/usr/bin/env python3
"""Visualise the prior in θ-space and the implied distribution in physical space.

Specify the prior as (mean, std) in physical space in the PRIOR DEFINITION block
below.  Moment matching automatically computes the corresponding θ-space Gaussian
parameters.  The transformations are:

    θ[0]  →  sigmoid              →  ρ       ∈ (0, 1)
    θ[1]  →  exp                  →  l       > 0  (µm)
    θ[2]  →  F_INF_MAX * sigmoid  →  f_∞     ∈ (0, F_INF_MAX)
    ψ[0]  →  exp                  →  σ²_δ    > 0
    ψ[1]  →  exp                  →  σ²_ε    > 0
    ψ[2]  →  exp                  →  ρ_KO    > 0

Output: prior_exploration.pdf  (saved next to this script)
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde, norm

# ── MOMENT-MATCHING HELPERS ───────────────────────────────────────────────────


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """θ-space (μ, σ) for X = exp(θ) ~ LogNormal matching target mean and std."""
    sigma2 = np.log(1.0 + (std / mean)**2)
    mu = np.log(mean) - 0.5 * sigma2
    return float(mu), float(np.sqrt(sigma2))


def _logitnormal_moments(mu: float,
                         sigma: float,
                         upper: float = 1.0,
                         n_gh: int = 64) -> tuple[float, float]:
    """Mean and variance of upper·sigmoid(N(μ, σ²)) via Gauss-Hermite quadrature."""
    z, w = np.polynomial.hermite.hermgauss(n_gh)
    s = upper / (1.0 + np.exp(-(mu + np.sqrt(2) * sigma * z)))
    w_norm = w / np.sqrt(np.pi)
    mean = float(np.dot(w_norm, s))
    var = float(np.dot(w_norm, s**2)) - mean**2
    return mean, var


def _logitnormal_params(mean: float,
                        std: float,
                        upper: float = 1.0) -> tuple[float, float]:
    """θ-space (μ, σ) for X = upper·sigmoid(θ) matching target mean and std."""
    target_var = std**2
    m_n = mean / upper
    mu0 = np.log(m_n / (1.0 - m_n))  # logit of normalised mean

    def residuals(params):
        m, v = _logitnormal_moments(params[0], np.exp(params[1]), upper)
        return [m - mean, v - target_var]

    sol, _, ier, msg = fsolve(residuals, [mu0, 0.0], full_output=True)
    if ier != 1:
        print(
            f"Warning: logit-normal moment matching did not converge — {msg}")
    return float(sol[0]), float(np.exp(sol[1]))


# ── PRIOR DEFINITION  (edit this block) ──────────────────────────────────────

# Hard upper bound on f_∞
F_INF_MAX = 130.0

# Specify the prior as [mean, std] in physical space.
#                           mean     std
THETA_PHYS = np.array([
    [0.6, 0.15],  # ρ    ∈ (0, 1)
    [150.0, 75.0],  # l    (µm)
    [80.0, 25.0],  # f_∞  ∈ (0, F_INF_MAX)
])

PSI_PHYS = np.array([
    [0.10, 0.20],  # σ²_δ
    [0.01, 0.02],  # σ²_ε
    [150.0, 200.0],  # ρ_KO
])

N_SAMPLES = 100_000
SEED = 0

# ── END OF PRIOR DEFINITION ───────────────────────────────────────────────────

# Compute θ-space parameters via moment matching
PRIOR = np.array([
    _logitnormal_params(THETA_PHYS[0, 0], THETA_PHYS[0, 1], upper=1.0),
    _lognormal_params(THETA_PHYS[1, 0], THETA_PHYS[1, 1]),
    _logitnormal_params(THETA_PHYS[2, 0], THETA_PHYS[2, 1], upper=F_INF_MAX),
])
PSI_PRIOR = np.array(
    [_lognormal_params(PSI_PHYS[k, 0], PSI_PHYS[k, 1]) for k in range(3)])

THETA_LABELS = [
    r'$\theta_0 = \mathrm{logit}(\rho)$',
    r'$\theta_1 = \ln\,l$',
    r'$\theta_2 = \mathrm{logit}(f_\infty\,/\,f_{\infty,\max})$',
]
PHYS_LABELS = [r'$\rho$', r'$l$ (µm)', r'$f_\infty$']
PHYS_TRANSFORMS = [
    lambda t: 1.0 / (1.0 + np.exp(-t)),  # sigmoid → ρ
    np.exp,  # exp     → l
    lambda t: F_INF_MAX / (1.0 + np.exp(-t)),  # scaled sigmoid → f_∞
]

PSI_THETA_LABELS = [
    r'$\psi_0 = \ln\,\sigma^2_\delta$',
    r'$\psi_1 = \ln\,\sigma^2_\varepsilon$',
    r'$\psi_2 = \ln\,\rho_\mathrm{KO}$',
]
PSI_PHYS_LABELS = [
    r'$\sigma^2_\delta$', r'$\sigma^2_\varepsilon$', r'$\rho_\mathrm{KO}$'
]
PSI_PHYS_TRANSFORMS = [np.exp, np.exp, np.exp]

XLIMS_PHYS = [(0.0, 1.0), (0.0, 500.0), (0.0, F_INF_MAX)]
XLIMS_PSI_PHYS = [(0.0, 0.2), (0.0, 0.2), (0.0, 500.0)]

# ── Sampling ──────────────────────────────────────────────────────────────────


def _sample_prior(prior: np.ndarray, transforms: list, n: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_samples, physical_samples), each shape (n, d)."""
    rng = np.random.default_rng(seed)
    d = prior.shape[0]
    theta = rng.normal(prior[:, 0], prior[:, 1], size=(n, d))
    physical = np.stack([f(theta[:, k]) for k, f in enumerate(transforms)],
                        axis=1)
    return theta, physical


# ── Plotting ──────────────────────────────────────────────────────────────────


def _xlims_theta(prior: np.ndarray, n_sigma: float = 4.0):
    return [(prior[k, 0] - n_sigma * prior[k, 1],
             prior[k, 0] + n_sigma * prior[k, 1])
            for k in range(prior.shape[0])]


def _plot_theta_row(axes, prior, theta_labels, color):
    """Plot Gaussian PDFs for one group of parameters in θ-space."""
    xlims = _xlims_theta(prior)
    for k, ax in enumerate(axes):
        lo, hi = xlims[k]
        xs = np.linspace(lo, hi, 400)
        mu, sigma = prior[k, 0], prior[k, 1]
        ax.plot(xs, norm.pdf(xs, mu, sigma), color=color, lw=2.0)
        ax.fill_between(xs, norm.pdf(xs, mu, sigma), alpha=0.25, color=color)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(theta_labels[k], fontsize=9)
        ax.set_ylabel('density', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(fr'$\mu_\theta={mu:.3g}$,  $\sigma_\theta={sigma:.3g}$',
                     fontsize=8,
                     loc='right')


def _plot_phys_row(axes, samples, xlims, labels, phys_spec, color):
    """Plot physical-space KDE + histogram, marking the target mean."""
    for k, ax in enumerate(axes):
        vals = samples[:, k]
        lo, hi = xlims[k]
        vals = vals[(vals >= lo) & (vals <= hi)]
        xs = np.linspace(lo, hi, 400)
        ax.hist(vals,
                bins=80,
                density=True,
                alpha=0.35,
                color=color,
                edgecolor='none')
        kde = gaussian_kde(vals)
        ax.plot(xs, kde(xs), color=color, lw=1.8)
        target_mean, target_std = phys_spec[k]
        ax.axvline(target_mean, color='k', lw=1.0, ls='--', alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(labels[k], fontsize=9)
        ax.set_ylabel('density', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(
            fr'target: $\bar{{x}}={target_mean:.3g}$,  $s={target_std:.3g}$',
            fontsize=8,
            loc='right')


# ── Config writing ────────────────────────────────────────────────────────────


def _fmt_list(vals: np.ndarray) -> str:
    return '[' + ', '.join(f'{v:.6g}' for v in vals) + ']'


def _fmt_diag_cov(stds: np.ndarray) -> str:
    n = len(stds)
    rows = [
        '[' + ', '.join(f'{stds[j]**2:.6g}' if i == j else '0.0'
                        for j in range(n)) + ']' for i in range(n)
    ]
    return '[' + ', '.join(rows) + ']'


def _replace_in_block(text: str, section_marker: str, mean_vals: np.ndarray,
                      std_vals: np.ndarray) -> str:
    """Replace mean/cov lines in the first occurrence of section_marker."""
    parts = text.split(section_marker, 1)
    if len(parts) != 2:
        raise ValueError(
            f"Section marker {section_marker!r} not found in config")
    before, after = parts
    after = re.sub(r'([ \t]+mean:[ \t]*)\[.*]',
                   lambda m: m.group(1) + _fmt_list(mean_vals),
                   after,
                   count=1)
    after = re.sub(r'([ \t]+cov:[ \t]*)\[.*]',
                   lambda m: m.group(1) + _fmt_diag_cov(std_vals),
                   after,
                   count=1)
    return before + section_marker + after


def _fmt_transformations(f_inf_max: float) -> str:
    return (f'[{{a: 0.0, b: 1.0, type: Sigmoid}}, Exponential, '
            f'{{a: 0.0, b: {f_inf_max:.6g}, type: Sigmoid}}, Identity]')


def write_config(config_path: str, prior: np.ndarray, psi_prior: np.ndarray,
                 f_inf_max: float) -> None:
    """Write θ-space parameters and f_∞ transformation into the YAML config."""
    with open(config_path) as f:
        text = f.read()
    text = _replace_in_block(text, 'coefficient_distribution:', prior[:, 0],
                             prior[:, 1])
    text = _replace_in_block(text, 'psi_prior:', psi_prior[:, 0], psi_prior[:,
                                                                            1])
    text = re.sub(r'([ \t]+transformations:[ \t]*)\[.*?]',
                  lambda m: m.group(1) + _fmt_transformations(f_inf_max),
                  text,
                  count=1,
                  flags=re.DOTALL)
    with open(config_path, 'w') as f:
        f.write(text)
    print(f"Updated {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--write-config',
        metavar='PATH',
        default=os.path.join(HERE, 'rwm', 'config.yaml'),
        help='YAML config to update (default: rwm/config.yaml)')
    parser.add_argument('--no-write',
                        action='store_true',
                        help='Skip writing the config file')
    args = parser.parse_args()

    print("Computing θ-space parameters via moment matching...")
    print(f"  PRIOR (θ-space μ, σ):")
    names = ['rho', 'l', 'f_inf']
    for k, name in enumerate(names):
        print(f"    {name:<8}  μ={PRIOR[k,0]:+.4f}  σ={PRIOR[k,1]:.4f}")
    print(f"  PSI_PRIOR (ψ-space μ, σ):")
    psi_names = ['s2_delta', 's2_eps', 'rho_KO']
    for k, name in enumerate(psi_names):
        print(
            f"    {name:<8}  μ={PSI_PRIOR[k,0]:+.4f}  σ={PSI_PRIOR[k,1]:.4f}")

    theta, physical = _sample_prior(PRIOR, PHYS_TRANSFORMS, N_SAMPLES, SEED)
    psi_theta, psi_physical = _sample_prior(PSI_PRIOR, PSI_PHYS_TRANSFORMS,
                                            N_SAMPLES, SEED)

    fig, axes = plt.subplots(4, 3, figsize=(11, 12))

    # Row 0 — θ-space for ρ, l, f_∞
    _plot_theta_row(axes[0], PRIOR, THETA_LABELS, 'steelblue')
    axes[0, 0].set_title(r'$\theta$-space prior  ($\rho$, $l$, $f_\infty$)',
                         fontsize=9,
                         loc='left',
                         fontweight='bold')

    # Row 1 — physical space for ρ, l, f_∞
    _plot_phys_row(axes[1], physical, XLIMS_PHYS, PHYS_LABELS, THETA_PHYS,
                   'C1')
    axes[1, 0].set_title(r'Physical-space prior  ($\rho$, $l$, $f_\infty$)',
                         fontsize=9,
                         loc='left',
                         fontweight='bold')

    # Row 2 — ψ-space for ψ
    _plot_theta_row(axes[2], PSI_PRIOR, PSI_THETA_LABELS, 'steelblue')
    axes[2, 0].set_title(
        r'$\psi$-space prior  ($\sigma^2_\delta$, $\sigma^2_\varepsilon$, $\rho_\mathrm{KO}$)',
        fontsize=9,
        loc='left',
        fontweight='bold')

    # Row 3 — physical space for ψ
    _plot_phys_row(axes[3], psi_physical, XLIMS_PSI_PHYS, PSI_PHYS_LABELS,
                   PSI_PHYS, 'C1')
    axes[3, 0].set_title(
        r'Physical-space prior  ($\sigma^2_\delta$, $\sigma^2_\varepsilon$, $\rho_\mathrm{KO}$)',
        fontsize=9,
        loc='left',
        fontweight='bold')

    # Summary statistics
    print(f"\n{'─'*58}")
    print(
        f"{'Parameter':<14} {'target mean':>12} {'MC mean':>10} {'5%':>10} {'95%':>10}"
    )
    print(f"{'─'*58}")
    for k, name in enumerate(names):
        med = np.mean(physical[:, k])
        lo, hi = np.percentile(physical[:, k], [5, 95])
        print(
            f"  {name:<12}  {THETA_PHYS[k,0]:>12.4f}  {med:>10.4f}  {lo:>10.4f}  {hi:>10.4f}"
        )
    for k, name in enumerate(psi_names):
        med = np.mean(psi_physical[:, k])
        lo, hi = np.percentile(psi_physical[:, k], [5, 95])
        print(
            f"  {name:<12}  {PSI_PHYS[k,0]:>12.4f}  {med:>10.4f}  {lo:>10.4f}  {hi:>10.4f}"
        )
    print(f"{'─'*58}\n")

    fig.tight_layout()
    out = os.path.join(HERE, 'prior_exploration.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")
    plt.show()

    if not args.no_write:
        write_config(args.write_config, PRIOR, PSI_PRIOR, F_INF_MAX)


if __name__ == '__main__':
    main()
