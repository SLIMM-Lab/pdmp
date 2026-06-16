#!/usr/bin/env python
"""Overlay an empirical forward-UQ output distribution against a moment-matched one.

Compares two forward-UQ result sets on the same figures, in two colors:

    1. Empirical   — per-sample outputs from ``forward_uq.py`` (``outputs.dat``
                     + ``outputs_meta.json``), e.g. the BPS posterior pushed
                     through the RVE. Shown as histogram + KDE.
    2. Moment-match — the moment-matched output Gaussian produced by
                     ``forward_uq_moment.py`` (``mean_lin.dat`` / ``cov_lin.dat``
                     or the ``_ut`` variant + ``outputs_legend.txt``). The
                     linearization / UT output distribution IS a Gaussian, so it
                     is drawn analytically from its mean and covariance — no
                     Monte-Carlo / KDE noise.

Both drivers flatten the same model output, so their columns share names; this
script aligns by name and plots the shared, non-location columns (the two runs
need not have identical column sets). ``*_cell`` argmax-location columns are
skipped — they are integer cell indices whose finite-difference Jacobian is
zero, so the moment-matched Gaussian assigns them no variance.

Produces the marginal and pairwise (corner) figures of ``plot_forward_uq.py``,
but with one color per approach so the agreement (smooth quantities) and the
divergence (max-type quantities) are visible at a glance.

Usage:
    python plot_forward_uq_compare.py EMPIRICAL_DIR MOMENT_DIR
    python plot_forward_uq_compare.py bps/results lap/results_moment --method ut
    python plot_forward_uq_compare.py EMP MOM --columns max_von_mises avg_stress_xx
    python plot_forward_uq_compare.py EMP MOM --labels BPS Linearization -o figures/cmp
"""

# The moment side is a Gaussian (mean + covariance), drawn analytically: its
# marginals are exact 1-D normals and its pairwise densities exact 2-D normals.

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Reuse helpers from the single-distribution plotting script.
from plot_forward_uq import (LOCATION_SUFFIX, _kde2d_grid, _apply_style,
                             hdr_levels)

# Empirical first, moment-matched second.
DEFAULT_COLORS = ('#4C72B0', '#C44E52')  # seaborn blue / red


def load_empirical(results_dir):
    """Load per-sample outputs, column names and labels from a forward_uq dir."""
    outputs = np.loadtxt(os.path.join(results_dir, 'outputs.dat'))
    if outputs.ndim == 1:
        outputs = outputs[:, None]

    meta_path = os.path.join(results_dir, 'outputs_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        names = [c['name'] for c in meta['columns']]
        labels = [c['label'] for c in meta['columns']]
    else:
        names = [f'out[{j}]' for j in range(outputs.shape[1])]
        labels = list(names)
    return outputs, names, labels


def load_moment(results_dir, method='lin'):
    """Load the moment-matched output Gaussian (mean, covariance) and names.

    ``method`` selects ``mean_lin.dat`` / ``cov_lin.dat`` (linearization) or
    the ``_ut`` variant (unscented transform). Column names come from
    ``outputs_legend.txt`` (``index<TAB>name`` per line).
    """
    mean = np.atleast_1d(np.loadtxt(os.path.join(results_dir,
                                                 f'mean_{method}.dat')))
    cov = np.atleast_2d(np.loadtxt(os.path.join(results_dir,
                                                f'cov_{method}.dat')))

    legend_path = os.path.join(results_dir, 'outputs_legend.txt')
    if os.path.exists(legend_path):
        names = []
        with open(legend_path) as f:
            for line in f:
                if line.strip():
                    names.append(line.rstrip('\n').split('\t', 1)[1])
    else:
        names = [f'out[{j}]' for j in range(mean.size)]
    return mean, cov, names


def normal_pdf(x, mu, var):
    """1-D Gaussian density."""
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


# Both overlays are drawn as the same credible regions: the empirical KDE as
# highest-density-region contours (see hdr_levels) and the moment-matched
# Gaussian as the analytic ellipses enclosing the same masses. For a 2-D
# Gaussian the mass inside Mahalanobis radius r is 1 - exp(-r²/2), so the radius
# enclosing mass m is √(-2 ln(1 - m)).
MASS_LEVELS = (0.5, 0.9, 0.99)
GAUSS_RADII = tuple(float(np.sqrt(-2.0 * np.log(1.0 - m))) for m in MASS_LEVELS)


def gauss_ellipse(mu, cov, radius, n=256):
    """Points on the Mahalanobis-``radius`` iso-density ellipse of a 2-D Gaussian.

    ``mu`` is ``[mu_x, mu_y]`` and ``cov`` the corresponding 2×2 block. The
    ellipse is generated parametrically from the covariance eigendecomposition
    (``mu + radius · V diag(√w) · unit_circle``), so it stays perfectly smooth
    at any anisotropy. This is what keeps the contours sharp: the linearization
    covariance ``J Σ Jᵀ`` has rank ≤ n_inputs, so a 2×2 block of strongly
    correlated outputs collapses onto a thin, tilted ridge that a Cartesian
    density grid cannot resolve without aliasing.

    Eigenvalues are floored to a tiny fraction of the trace so a numerically
    singular (or slightly indefinite) block still yields a well-defined — if
    very thin — ellipse, faithfully showing the near-perfect linear correlation.

    Returns ``(x, y)`` arrays of length ``n`` tracing the closed ellipse.
    """
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    w, V = np.linalg.eigh(cov)
    floor = 1e-9 * max(np.trace(cov), 1e-300)
    w = np.maximum(w, floor)

    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])      # (2, n) unit circle
    axes = V * (radius * np.sqrt(w))               # (2, 2) scaled principal axes
    pts = np.asarray(mu)[:, None] + axes @ circle  # (2, n)
    return pts[0], pts[1]


def resolve_shared_columns(names_a, names_b, include, exclude):
    """Names present in both datasets that should be plotted.

    Defaults to every shared, non-location (``*_cell``) column. ``include``
    overrides the default set; ``exclude`` removes names from it. Order follows
    the empirical (first) dataset.
    """
    shared = [n for n in names_a if n in set(names_b)]

    if include:
        missing = [n for n in include if n not in shared]
        if missing:
            raise SystemExit(f'Requested column(s) not in both datasets: '
                             f'{missing}\nShared columns: {shared}')
        selected = list(include)
    else:
        selected = [n for n in shared if not n.endswith(LOCATION_SUFFIX)]

    if exclude:
        selected = [n for n in selected if n not in exclude]

    if not selected:
        raise SystemExit('No shared columns selected to plot.')
    return selected


def combined_window(emp_col, mu, sigma, k=3.5, lo_pct=0.5, hi_pct=99.5):
    """Display window covering the empirical central ~99% and the Gaussian ±kσ.

    Spans the union so neither the empirical histogram nor the analytical
    Gaussian curve is clipped.
    """
    lo = min(float(np.percentile(emp_col, lo_pct)), mu - k * sigma)
    hi = max(float(np.percentile(emp_col, hi_pct)), mu + k * sigma)
    return lo, hi


def plot_marginals(emp, mom_mean, mom_cov, labels, fig_dir, ds_labels, colors,
                   overrides=None):
    """Per-dimension marginals: empirical histogram + KDE vs analytical Gaussian.

    ``overrides`` is an optional per-column list of ``(lo, hi)`` x-axis windows
    (``None`` entries fall back to the automatic :func:`combined_window`).
    """
    d = emp.shape[1]
    mom_std = np.sqrt(np.diag(mom_cov))
    n_cols = min(d, 2)
    n_rows = (d + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for j in range(d):
        ax = axes[j // n_cols, j % n_cols]
        y = emp[:, j]
        mu, sd = mom_mean[j], mom_std[j]
        if overrides and overrides[j] is not None:
            lo, hi = overrides[j]
        else:
            lo, hi = combined_window(y, mu, sd)
        grid = np.linspace(lo, hi, 300)

        # Empirical: histogram + KDE.
        ax.hist(y, bins=40, range=(lo, hi), density=True,
                alpha=0.35, color=colors[0])
        ax.plot(grid, gaussian_kde(y)(grid), color=colors[0], lw=2)
        ax.axvline(y.mean(), color=colors[0], lw=1.2, ls='--')

        # Moment-matched: analytical Gaussian.
        ax.plot(grid, normal_pdf(grid, mu, sd ** 2), color=colors[1], lw=2)
        ax.axvline(mu, color=colors[1], lw=1.2, ls='--')

        ax.set_xlim(lo, hi)
        ax.set_xlabel(labels[j])
        ax.set_ylabel('Density' if j % n_cols == 0 else '')

    handles = [Line2D([0], [0], color=colors[k], lw=2, label=ds_labels[k])
               for k in range(2)]
    axes[0, 0].legend(handles=handles, fontsize=8)

    for j in range(d, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.tight_layout()
    _apply_style(fig)
    path = os.path.join(fig_dir, 'compare_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def _subsample(x, max_samples, rng):
    if len(x) > max_samples:
        idx = rng.choice(len(x), size=max_samples, replace=False)
        return x[idx]
    return x


def plot_pairwise(emp, mom_mean, mom_cov, labels, fig_dir, ds_labels, colors,
                  max_samples=2000, overrides=None, bw_adjust=1.5):
    """Corner plot: empirical samples vs the analytical moment-matched Gaussian.

    Diagonal: empirical histogram + analytical 1-D Gaussian. Lower triangle:
    empirical scatter + analytical Gaussian contour lines. Upper triangle:
    empirical KDE contours + analytical Gaussian contours. Two colors throughout.

    ``overrides`` is an optional per-column list of ``(lo, hi)`` windows
    (``None`` entries fall back to the automatic :func:`combined_window`).
    ``bw_adjust`` widens (>1) the empirical KDE bandwidth so the mass-enclosing
    HDR contours stay smooth out in the low-density tail.
    """
    d = emp.shape[1]
    if d < 2:
        return

    rng = np.random.default_rng(0)
    emp_s = _subsample(emp, max_samples, rng)
    mom_std = np.sqrt(np.diag(mom_cov))

    # Shared per-column window over the empirical central 99% and Gaussian ±kσ,
    # unless pinned by hand via ``overrides``.
    lims = [overrides[k] if (overrides and overrides[k] is not None)
            else combined_window(emp[:, k], mom_mean[k], mom_std[k])
            for k in range(d)]

    def gauss_contour(ax, j, i):
        # Analytic iso-density ellipses. matplotlib clips the parts that fall
        # outside the panel window via the axis limits set by each caller.
        mu = [mom_mean[j], mom_mean[i]]
        cov2 = mom_cov[np.ix_([j, i], [j, i])]
        for radius in GAUSS_RADII:
            ex, ey = gauss_ellipse(mu, cov2, radius)
            ax.plot(ex, ey, color=colors[1], lw=0.8)

    fig, axes = plt.subplots(d, d, figsize=(2 * d, 2 * d), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(emp[:, i], bins=40, range=lims[i], density=True,
                        color=colors[0], alpha=0.45, edgecolor='none')
                grid = np.linspace(*lims[i], 300)
                ax.plot(grid, normal_pdf(grid, mom_mean[i], mom_std[i] ** 2),
                        color=colors[1], lw=1.5)
                ax.set_xlim(*lims[i])
                ax.set_xlabel(labels[i], fontsize=8)
            elif i > j:  # lower triangle — empirical scatter + Gaussian contour
                ax.scatter(emp_s[:, j], emp_s[:, i], s=2, alpha=0.25,
                           color=colors[0], linewidths=0)
                gauss_contour(ax, j, i)
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])
                ax.set_xlabel(labels[j], fontsize=8)
                ax.set_ylabel(labels[i], fontsize=8)
            else:  # upper triangle — empirical KDE + Gaussian contour
                xr, yr = lims[j], lims[i]
                g = _kde2d_grid(emp[:, j], emp[:, i], xr, yr,
                                bw_adjust=bw_adjust)
                if g is not None:
                    xx, yy, zz = g
                    levels = hdr_levels(zz, masses=MASS_LEVELS)
                    if levels:
                        ax.contour(xx, yy, zz, levels=levels,
                                   colors=colors[0], linewidths=0.8)
                gauss_contour(ax, j, i)
                ax.set_xlim(*xr)
                ax.set_ylim(*yr)
                ax.set_xlabel(labels[j], fontsize=8)
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [Line2D([0], [0], color=colors[k], lw=2, label=ds_labels[k])
               for k in range(2)]
    fig.legend(handles=handles, loc='upper right', fontsize=9)

    fig.tight_layout()
    _apply_style(fig)
    path = os.path.join(fig_dir, 'compare_pairwise.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Overlay an empirical forward-UQ distribution against a '
        'moment-matched one.')
    parser.add_argument('empirical_dir',
                        help='forward_uq.py results dir (outputs.dat + '
                        'outputs_meta.json)')
    parser.add_argument('moment_dir',
                        help='forward_uq_moment.py results dir '
                        '(samples_{lin,ut}.dat + outputs_legend.txt)')
    parser.add_argument('--method', choices=['lin', 'ut'], default='lin',
                        help='Which moment-matched draws to compare against '
                        '(default: lin)')
    parser.add_argument('--columns', nargs='+', default=None,
                        help='Column names to plot (default: all shared '
                        'non-location columns)')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Column names to exclude')
    parser.add_argument('--xlim', nargs=3, action='append', default=None,
                        metavar=('COLUMN', 'LO', 'HI'),
                        help='Pin the x-axis window for a column, overriding '
                        'the automatic one. Repeatable, e.g. '
                        '--xlim max_von_mises 0 100 --xlim avg_stress_xx -5 5')
    parser.add_argument('--labels', nargs=2, default=None,
                        metavar=('EMPIRICAL', 'MOMENT'),
                        help='Legend labels for the two datasets '
                        '(default: "Empirical" / method name)')
    parser.add_argument('--colors', nargs=2, default=list(DEFAULT_COLORS),
                        metavar=('EMPIRICAL', 'MOMENT'),
                        help='Colors for the two datasets')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max points per dataset in the pairwise scatter '
                        '(default: 2000)')
    parser.add_argument('--bw-adjust', type=float, default=1.5,
                        help='Multiplier on the empirical KDE bandwidth for the '
                        'pairwise contours; >1 smooths the mass-enclosing '
                        'contours, <1 sharpens them (default: 1.5)')
    parser.add_argument('-o', '--out-dir', default=None,
                        help='Figure output dir (default: '
                        'MOMENT_DIR/figures/compare)')
    args = parser.parse_args()

    emp, names_emp, labels_emp = load_empirical(args.empirical_dir)
    mom_mean, mom_cov, names_mom = load_moment(args.moment_dir,
                                               method=args.method)

    selected = resolve_shared_columns(names_emp, names_mom,
                                      args.columns, args.exclude)
    idx_emp = {n: i for i, n in enumerate(names_emp)}
    idx_mom = {n: i for i, n in enumerate(names_mom)}
    sel_emp = [idx_emp[n] for n in selected]
    sel_mom = [idx_mom[n] for n in selected]
    emp_sel = emp[:, sel_emp]
    mom_mean_sel = mom_mean[sel_mom]
    mom_cov_sel = mom_cov[np.ix_(sel_mom, sel_mom)]
    # LaTeX labels come from the empirical meta; fall back to the raw name.
    sel_labels = [labels_emp[idx_emp[n]] for n in selected]

    # Hand-fixed x-axis windows, aligned to the selected columns.
    xlim_map = {}
    for name, lo, hi in (args.xlim or []):
        if name not in selected:
            raise SystemExit(f'--xlim column {name!r} is not among the plotted '
                             f'columns: {selected}')
        xlim_map[name] = (float(lo), float(hi))
    overrides = [xlim_map.get(n) for n in selected]

    method_name = {'lin': 'Linearization', 'ut': 'Unscented'}[args.method]
    ds_labels = args.labels if args.labels else ['Monte Carlo', method_name]

    out_dir = args.out_dir or os.path.join(args.moment_dir, 'figures',
                                           'compare')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Comparing {len(selected)} columns: {selected}')
    print(f'  empirical n={len(emp_sel)}, {args.method}: analytical Gaussian')
    print(f'Saving figures to {out_dir}/')

    plot_marginals(emp_sel, mom_mean_sel, mom_cov_sel, sel_labels, out_dir,
                   ds_labels, args.colors, overrides=overrides)
    plot_pairwise(emp_sel, mom_mean_sel, mom_cov_sel, sel_labels, out_dir,
                  ds_labels, args.colors, max_samples=args.max_samples,
                  overrides=overrides, bw_adjust=args.bw_adjust)


if __name__ == '__main__':
    main()
