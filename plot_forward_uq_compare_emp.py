#!/usr/bin/env python
"""Overlay two empirical forward-UQ output distributions on the same figures.

The companion to ``plot_forward_uq_compare.py``, but for comparing **two
empirical** result sets rather than empirical-vs-moment-matched. Both inputs are
per-sample outputs from ``forward_uq.py`` (``outputs.dat`` + ``outputs_meta.json``),
e.g. two different posteriors / modeling choices pushed through the RVE such as

    examples/forward_uq/rve_joint/bps/results
    examples/forward_uq/rve_per_fiber/results

Because neither side is a Gaussian, both are drawn the same way — histogram + KDE
for the marginals, scatter + highest-density-region (HDR) KDE contours for the
pairwise panels — in two colors, so agreement and divergence are visible at a
glance.

The two runs need not share an identical column set: columns are aligned by name
and only the shared, non-location columns are plotted. ``*_cell`` argmax-location
columns are skipped (integer cell indices, not continuous quantities).

Usage:
    python plot_forward_uq_compare_emp.py DIR_A DIR_B
    python plot_forward_uq_compare_emp.py rve_joint/bps/results rve_per_fiber/results \
        --labels Joint "Per fiber" -o figures/cmp
    python plot_forward_uq_compare_emp.py DIR_A DIR_B --columns max_von_mises avg_stress_xx
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Reuse helpers from the single-distribution plotting script and the
# empirical-vs-moment comparison script.
from plot_forward_uq import (LOCATION_SUFFIX, _kde2d_grid, _apply_style,
                             hdr_levels)
from plot_forward_uq_compare import load_empirical, resolve_shared_columns

# One hue per dataset: seaborn "deep" orange (C1) then blue (C0). Hard-coded so
# the colors hold whether or not the seaborn palette is the active rc cycle.
# DEFAULT_COLORS = ('#DD8452', '#4C72B0')  # orange / blue
DEFAULT_COLORS = ('C1', 'C0')  # orange / blue

# Both overlays use the same mass-enclosing HDR contour levels.
MASS_LEVELS = (0.5, 0.9, 0.99)


def combined_window(col_a, col_b, lo_pct=0.5, hi_pct=99.5):
    """Display window covering the central ~99% of *both* empirical columns."""
    lo = min(float(np.percentile(col_a, lo_pct)),
             float(np.percentile(col_b, lo_pct)))
    hi = max(float(np.percentile(col_a, hi_pct)),
             float(np.percentile(col_b, hi_pct)))
    return lo, hi


def plot_marginals(a, b, labels, fig_dir, ds_labels, colors, overrides=None):
    """Per-dimension marginals: histogram + KDE for each dataset, two colors.

    ``overrides`` is an optional per-column list of ``(lo, hi)`` x-axis windows
    (``None`` entries fall back to the automatic :func:`combined_window`).
    """
    d = a.shape[1]
    n_cols = min(d, 3)
    n_rows = (d + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for j in range(d):
        ax = axes[j // n_cols, j % n_cols]
        ya, yb = a[:, j], b[:, j]
        if overrides and overrides[j] is not None:
            lo, hi = overrides[j]
        else:
            lo, hi = combined_window(ya, yb)
        grid = np.linspace(lo, hi, 300)

        for y, color in ((ya, colors[0]), (yb, colors[1])):
            ax.hist(y, bins=40, range=(lo, hi), density=True,
                    alpha=0.30, color=color)
            ax.plot(grid, gaussian_kde(y)(grid), color=color, lw=2)
            ax.axvline(y.mean(), color=color, lw=1.2, ls='--')

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
    path = os.path.join(fig_dir, 'compare_emp_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def _subsample(x, max_samples, rng):
    if len(x) > max_samples:
        idx = rng.choice(len(x), size=max_samples, replace=False)
        return x[idx]
    return x


def plot_pairwise(a, b, labels, fig_dir, ds_labels, colors,
                  max_samples=2000, overrides=None, bw_adjust=1.5):
    """Corner plot overlaying two empirical datasets in two colors.

    Diagonal: overlaid histograms + KDE curves. Lower triangle: scatter of both
    datasets. Upper triangle: mass-enclosing HDR KDE contours of both. ``A`` is
    drawn first (under) and ``B`` second so the second color stays readable.

    ``overrides`` is an optional per-column list of ``(lo, hi)`` windows.
    ``bw_adjust`` widens (>1) the KDE bandwidth so the HDR contours stay smooth
    in the low-density tails.
    """
    d = a.shape[1]
    if d < 2:
        return

    rng = np.random.default_rng(0)
    a_s = _subsample(a, max_samples, rng)
    b_s = _subsample(b, max_samples, rng)

    lims = [overrides[k] if (overrides and overrides[k] is not None)
            else combined_window(a[:, k], b[:, k]) for k in range(d)]

    def kde_contour(ax, data, j, i, color):
        xr, yr = lims[j], lims[i]
        g = _kde2d_grid(data[:, j], data[:, i], xr, yr, bw_adjust=bw_adjust)
        if g is not None:
            xx, yy, zz = g
            levels = hdr_levels(zz, masses=MASS_LEVELS)
            if levels:
                ax.contour(xx, yy, zz, levels=levels, colors=color,
                           linewidths=0.8)

    fig, axes = plt.subplots(d, d, figsize=(2 * d, 2 * d), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                grid = np.linspace(*lims[i], 300)
                for data, color in ((a, colors[0]), (b, colors[1])):
                    ax.hist(data[:, i], bins=40, range=lims[i], density=True,
                            color=color, alpha=0.35, edgecolor='none')
                    ax.plot(grid, gaussian_kde(data[:, i])(grid),
                            color=color, lw=1.5)
                ax.set_xlim(*lims[i])
                ax.set_xlabel(labels[i], fontsize=8)
            elif i > j:  # lower triangle — scatter of both datasets
                for data, color in ((a_s, colors[0]), (b_s, colors[1])):
                    ax.scatter(data[:, j], data[:, i], s=2, alpha=0.20,
                               color=color, linewidths=0)
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])
                ax.set_xlabel(labels[j], fontsize=8)
                ax.set_ylabel(labels[i], fontsize=8)
            else:  # upper triangle — HDR KDE contours of both
                kde_contour(ax, a, j, i, colors[0])
                kde_contour(ax, b, j, i, colors[1])
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])
                ax.set_xlabel(labels[j], fontsize=8)
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [Line2D([0], [0], color=colors[k], lw=2, label=ds_labels[k])
               for k in range(2)]
    fig.legend(handles=handles, loc='upper right', fontsize=9)

    fig.tight_layout()
    _apply_style(fig)
    path = os.path.join(fig_dir, 'compare_emp_pairwise.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Overlay two empirical forward-UQ output distributions.')
    parser.add_argument('dir_a',
                        help='first forward_uq.py results dir (outputs.dat + '
                        'outputs_meta.json)')
    parser.add_argument('dir_b',
                        help='second forward_uq.py results dir')
    parser.add_argument('--columns', nargs='+', default=None,
                        help='Column names to plot (default: all shared '
                        'non-location columns)')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Column names to exclude')
    parser.add_argument('--xlim', nargs=3, action='append', default=None,
                        metavar=('COLUMN', 'LO', 'HI'),
                        help='Pin the x-axis window for a column, overriding '
                        'the automatic one. Repeatable.')
    parser.add_argument('--labels', nargs=2, default=None,
                        metavar=('A', 'B'),
                        help='Legend labels for the two datasets '
                        '(default: directory basenames)')
    parser.add_argument('--colors', nargs=2, default=list(DEFAULT_COLORS),
                        metavar=('A', 'B'),
                        help='Colors for the two datasets')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max points per dataset in the pairwise scatter '
                        '(default: 2000)')
    parser.add_argument('--bw-adjust', type=float, default=1.5,
                        help='Multiplier on the KDE bandwidth for the pairwise '
                        'contours; >1 smooths, <1 sharpens (default: 1.5)')
    parser.add_argument('-o', '--out-dir', default=None,
                        help='Figure output dir (default: '
                        'DIR_A/figures/compare)')
    args = parser.parse_args()

    a, names_a, labels_a = load_empirical(args.dir_a)
    b, names_b, _ = load_empirical(args.dir_b)

    selected = resolve_shared_columns(names_a, names_b,
                                      args.columns, args.exclude)
    idx_a = {n: i for i, n in enumerate(names_a)}
    idx_b = {n: i for i, n in enumerate(names_b)}
    a_sel = a[:, [idx_a[n] for n in selected]]
    b_sel = b[:, [idx_b[n] for n in selected]]
    # LaTeX labels come from the first dataset's meta; fall back to raw name.
    sel_labels = [labels_a[idx_a[n]] for n in selected]

    # Hand-fixed x-axis windows, aligned to the selected columns.
    xlim_map = {}
    for name, lo, hi in (args.xlim or []):
        if name not in selected:
            raise SystemExit(f'--xlim column {name!r} is not among the plotted '
                             f'columns: {selected}')
        xlim_map[name] = (float(lo), float(hi))
    overrides = [xlim_map.get(n) for n in selected]

    ds_labels = args.labels if args.labels else [
        os.path.basename(os.path.normpath(args.dir_a)),
        os.path.basename(os.path.normpath(args.dir_b))]

    out_dir = args.out_dir or os.path.join(args.dir_a, 'figures', 'compare')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Comparing {len(selected)} columns: {selected}')
    print(f'  {ds_labels[0]} n={len(a_sel)}, {ds_labels[1]} n={len(b_sel)}')
    print(f'Saving figures to {out_dir}/')

    plot_marginals(a_sel, b_sel, sel_labels, out_dir, ds_labels, args.colors,
                   overrides=overrides)
    plot_pairwise(a_sel, b_sel, sel_labels, out_dir, ds_labels, args.colors,
                  max_samples=args.max_samples, overrides=overrides,
                  bw_adjust=args.bw_adjust)


if __name__ == '__main__':
    main()
