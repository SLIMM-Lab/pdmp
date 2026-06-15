#!/usr/bin/env python
"""Plot forward-UQ results produced by ``forward_uq.py``.

Reads ``samples.dat``, ``outputs.dat`` and ``outputs_meta.json`` from a results
directory and writes marginal, pairwise and input-output scatter figures. The
column labels come from ``outputs_meta.json`` so this script needs no access to
the forward model.

Run forward UQ once with all outputs, then call this script repeatedly with
different ``--columns`` / ``--exclude`` selections to make different plots.

If the run stored geometry (``cell_centroids.dat`` + ``geometry.json``, written
for RVE models), a domain heatmap of every ``*_cell`` location column is also
produced, showing where the max stress/strain occurs across the samples.

Usage:
    python plot_forward_uq.py path/to/results_dir
    python plot_forward_uq.py path/to/results_dir --columns max_von_mises avg_stress_xx
    python plot_forward_uq.py path/to/results_dir --exclude max_principal_strain_cell
    python plot_forward_uq.py path/to/results_dir --locations max_von_mises_strain_cell
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde
import seaborn as sns

from pdmp.plotting_utils import get_2d_despined_figure

LOCATION_SUFFIX = '_cell'


def load_results(results_dir):
    """Load samples, outputs, column names and labels from a results dir."""
    samples = np.loadtxt(os.path.join(results_dir, 'samples.dat'))
    if samples.ndim == 1:
        samples = samples[:, None]
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
        labels = [f'Output[{j}]' for j in range(outputs.shape[1])]
    return samples, outputs, names, labels


def resolve_selection(names, include, exclude):
    """Return indices of columns to plot in the distribution figures.

    Defaults to every column except ``*_cell`` location indices (those are
    shown in the domain heatmap instead). ``include`` overrides the default
    set; ``exclude`` removes names from it.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}

    def check(requested):
        missing = [n for n in requested if n not in name_to_idx]
        if missing:
            raise SystemExit(f'Unknown column(s): {missing}\n'
                             f'Available: {names}')

    if include:
        check(include)
        selected = list(include)
    else:
        selected = [n for n in names if not n.endswith(LOCATION_SUFFIX)]

    if exclude:
        check(exclude)
        selected = [n for n in selected if n not in exclude]

    return [name_to_idx[n] for n in selected]


def plot_output_marginals(outputs, fig_dir, labels):
    """Histogram + KDE for each output dimension."""
    dim_out = outputs.shape[1]
    n_cols = min(dim_out, 2)
    n_rows = (dim_out + n_cols - 1) // n_cols
    fig, axes = get_2d_despined_figure(nrows=n_rows,
                                       ncols=n_cols,
                                       figsize=(4 * n_cols, 3.5 * n_rows),
                                       keep_ticks=True,
                                       equal_axes=False)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for j in range(dim_out):
        ax = axes[j // n_cols, j % n_cols]
        y = outputs[:, j]
        # Display window: central 99% of the data (1% total excluded, 0.5% at
        # each end) so long tails don't squash the plotted range.
        lo, hi = (float(np.percentile(y, 0.5)), float(np.percentile(y, 99.5)))
        ax.hist(y,
                bins=40,
                range=(lo, hi),
                density=True,
                alpha=0.4,
                color='steelblue',
                label='Histogram')

        kde = gaussian_kde(y)
        grid = np.linspace(lo, hi, 300)
        ax.plot(grid, kde(grid), color='steelblue', lw=2, label='KDE')

        ax.axvline(y.mean(),
                   color='steelblue',
                   lw=1.2,
                   ls='--',
                   label='Empirical mean')

        ax.set_xlim(lo, hi)
        ax.set_xlabel(labels[j])
        ax.set_ylabel('Density' if j % n_cols == 0 else '')
        if j == 0:
            ax.legend(fontsize=8)

    for j in range(dim_out, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)
    path = os.path.join(fig_dir, 'output_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def _kde2d_grid(x, y, xr, yr, n=80, min_pts=10):
    """2D gaussian KDE evaluated on a grid over the window xr × yr.

    Fits only on samples inside the window so a long tail doesn't inflate the
    Scott's-rule bandwidth and oversmooth the density. Returns (xx, yy, zz) or
    None when too few samples fall inside the window.
    """
    m = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
    if m.sum() <= min_pts:
        return None
    kde = gaussian_kde(np.vstack([x[m], y[m]]))
    xs = np.linspace(xr[0], xr[1], n)
    ys = np.linspace(yr[0], yr[1], n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _apply_style(fig):
    """Apply the project despined figure style (cf. pdmp.plotting_utils).

    Drops the top/right spines, removes the grid, and normalises spine width
    across every axis. Ticks and labels are kept — these analysis plots are
    quantitative. Call after fig.tight_layout(), before fig.savefig().
    """
    for ax in fig.axes:
        ax.grid(False)
        for spine in ('top', 'bottom', 'left', 'right'):
            ax.spines[spine].set_linewidth(1.0)
    sns.despine(fig=fig)


def plot_output_pairwise(outputs, fig_dir, labels, max_samples=2000):
    """Corner plot of output dimensions (only if dim_out > 1).

    Lower-triangular layout: histograms on the diagonal, scatter below the
    diagonal, blank above. Mirrors the styling of the ITZ analyze_results
    ``_plot_pairplot`` corner plot. At most ``max_samples`` samples are used.
    """
    d = outputs.shape[1]
    if d < 2:
        return

    if len(outputs) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(outputs), size=max_samples, replace=False)
        outputs = outputs[idx]

    # Per-parameter display window: the central 99% of the data, so the long
    # tails (1% total, 0.5% at each end) don't squash the plotted range.
    lims = [(float(np.percentile(outputs[:, k], 0.5)),
             float(np.percentile(outputs[:, k], 99.5))) for k in range(d)]

    fig, axes = plt.subplots(d, d, figsize=(2*d, 2*d))
    axes = np.array(axes).reshape(d, d)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(outputs[:, i],
                        bins=40,
                        range=lims[i],
                        density=True,
                        color='steelblue',
                        alpha=0.5,
                        edgecolor='none')
                ax.set_xlim(*lims[i])
                ax.set_xlabel(labels[i], fontsize=8)
            elif i > j:
                ax.scatter(outputs[:, j],
                           outputs[:, i],
                           s=2,
                           alpha=0.25,
                           color='steelblue',
                           linewidths=0)
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])
                ax.set_xlabel(labels[j], fontsize=8)
                ax.set_ylabel(labels[i], fontsize=8)
            else:  # i < j : upper triangle — filled blue KDE contours
                # Shared display window: the central 99% of the samples for
                # this pair, so a long tail doesn't squash the contours.
                xr, yr = lims[j], lims[i]
                g = _kde2d_grid(outputs[:, j], outputs[:, i], xr, yr)
                if g is not None:
                    xx, yy, zz = g
                    levels = np.linspace(zz.max() * 0.05, zz.max(), 7)
                    ax.contourf(xx, yy, zz, levels=levels, cmap='Blues')
                    ax.set_xlim(*xr)
                    ax.set_ylim(*yr)
                    ax.set_xlabel(labels[j], fontsize=8)
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_visible(False)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    _apply_style(fig)
    path = os.path.join(fig_dir, 'output_pairwise.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_input_output_scatter(samples, outputs, fig_dir, labels):
    """Scatter plots: each input vs each output, coloured by output value."""
    dim_in = samples.shape[1]
    dim_out = outputs.shape[1]

    fig, axes = get_2d_despined_figure(nrows=dim_out,
                                       ncols=dim_in,
                                       figsize=(3.5 * dim_in, 3 * dim_out),
                                       keep_ticks=True,
                                       equal_axes=False)
    axes = np.array(axes).reshape(dim_out, dim_in)

    for i in range(dim_out):
        sc = None
        for j in range(dim_in):
            ax = axes[i, j]
            sc = ax.scatter(samples[:, j],
                            outputs[:, i],
                            s=2,
                            alpha=0.4,
                            c=outputs[:, i],
                            cmap='viridis')
            if i == dim_out - 1:
                ax.set_xlabel(f'Input[{j}]')
            if j == 0:
                ax.set_ylabel(labels[i])
        fig.colorbar(sc, ax=axes[i, :].tolist(), shrink=0.8, label=labels[i])
    path = os.path.join(fig_dir, 'input_output_scatter.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_location_heatmaps(results_dir, fig_dir, names, labels, outputs,
                           locations, bins, color):
    """Domain hex-density of the cells holding the max of each location column.

    For every ``*_cell`` column, each sample contributes the centroid of its
    argmax cell; a hexbin over the RVE domain shows where extrema concentrate,
    with fiber circles overlaid. The colormap ramps from white to *color*.
    """
    cc_path = os.path.join(results_dir, 'cell_centroids.dat')
    geom_path = os.path.join(results_dir, 'geometry.json')
    if not (os.path.exists(cc_path) and os.path.exists(geom_path)):
        print('  No geometry stored — skipping location heatmaps')
        return

    centroids = np.loadtxt(cc_path)
    with open(geom_path) as f:
        geom = json.load(f)
    L = geom['L']
    fibers = geom['fibers']

    loc_names = [n for n in names if n.endswith(LOCATION_SUFFIX)]
    if locations:
        loc_names = [n for n in loc_names if n in locations]
    if not loc_names:
        print('  No location (*_cell) columns to plot')
        return

    cmap = sns.light_palette(color, as_cmap=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in loc_names:
        col = name_to_idx[name]
        cells = outputs[:, col].astype(int)
        pts = centroids[cells]

        base = name[:-len(LOCATION_SUFFIX)]
        title = labels[name_to_idx[base]] if base in name_to_idx else name

        # Report the cells that most often hold the extremum for this quantity.
        uniq, counts = np.unique(cells, return_counts=True)
        order = np.argsort(counts)[::-1][:5]
        n_samples = len(cells)
        print(f'  Top 5 cells for {title}:')
        for cell, cnt in zip(uniq[order], counts[order]):
            cx, cy = centroids[cell]
            print(f'    cell {cell:6d}: {cnt:4d}/{n_samples} samples '
                  f'({100 * cnt / n_samples:5.1f}%), '
                  f'centroid ({cx:.3f}, {cy:.3f})')

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        hb = ax.hexbin(pts[:, 0],
                       pts[:, 1],
                       gridsize=bins,
                       extent=(0, L, 0, L),
                       cmap=cmap,
                       mincnt=1)
        fig.colorbar(hb, ax=ax, label='Sample count')

        for cx, cy, r in fibers:
            ax.add_patch(
                Circle((cx, cy),
                       r,
                       fill=False,
                       edgecolor='0.3',
                       lw=0.8,
                       alpha=0.7))

        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Location of {title}')
        path = os.path.join(fig_dir, f'location_{name}.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved {path}')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot forward-UQ results')
    parser.add_argument('results_dir',
                        type=str,
                        help='Directory containing samples.dat, outputs.dat '
                        'and outputs_meta.json')
    parser.add_argument('--columns',
                        nargs='+',
                        default=None,
                        help='Column names for the distribution plots '
                        '(default: all non-location columns)')
    parser.add_argument('--exclude',
                        nargs='+',
                        default=None,
                        help='Column names to exclude from the distribution '
                        'plots')
    parser.add_argument('--locations',
                        nargs='+',
                        default=None,
                        help='*_cell column names to plot as domain heatmaps '
                        '(default: all)')
    parser.add_argument('--max-samples',
                        type=int,
                        default=2000,
                        help='Maximum number of samples shown in the pairwise '
                        'plot (randomly subsampled if exceeded; default: 2000)')
    parser.add_argument('--bins',
                        type=int,
                        default=40,
                        help='Hexbin grid size for location heatmaps')
    parser.add_argument('--color',
                        type=str,
                        default='#4C72B0',
                        help='Base color for the location heatmaps; the '
                        'colormap ramps from white to this color '
                        '(default: seaborn blue)')
    args = parser.parse_args()

    samples, outputs, names, labels = load_results(args.results_dir)

    fig_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f'Saving figures to {fig_dir}/')

    sel = resolve_selection(names, args.columns, args.exclude)
    if sel:
        sel_outputs = outputs[:, sel]
        sel_labels = [labels[i] for i in sel]
        plot_output_marginals(sel_outputs, fig_dir, sel_labels)
        plot_output_pairwise(sel_outputs, fig_dir, sel_labels,
                             max_samples=args.max_samples)
        # plot_input_output_scatter(samples, sel_outputs, fig_dir, sel_labels)
    else:
        print('  No columns selected for distribution plots')

    plot_location_heatmaps(args.results_dir, fig_dir, names, labels, outputs,
                           args.locations, args.bins, args.color)


if __name__ == '__main__':
    main()
