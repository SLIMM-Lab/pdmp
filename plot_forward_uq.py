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
    python plot_forward_uq.py path/to/results_dir --columns max_stress avg_stress_xx
    python plot_forward_uq.py path/to/results_dir --exclude max_strain_cell
    python plot_forward_uq.py path/to/results_dir --locations max_stress_cell
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
    n_cols = min(dim_out, 3)
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
        ax.hist(y,
                bins=40,
                density=True,
                alpha=0.4,
                color='steelblue',
                label='Histogram')

        kde = gaussian_kde(y)
        grid = np.linspace(y.min(), y.max(), 300)
        ax.plot(grid, kde(grid), color='steelblue', lw=2, label='KDE')

        ax.axvline(y.mean(),
                   color='steelblue',
                   lw=1.2,
                   ls='--',
                   label=f'Empirical mean {y.mean():.3f}')

        ax.set_xlabel(labels[j])
        ax.set_ylabel('Density' if j % n_cols == 0 else '')
        ax.legend(fontsize=8)

    for j in range(dim_out, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)
    path = os.path.join(fig_dir, 'output_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_output_pairwise(outputs, fig_dir, labels):
    """Pairwise scatter plots of output dimensions (only if dim_out > 1)."""
    dim_out = outputs.shape[1]
    if dim_out < 2:
        return

    fig, axes = get_2d_despined_figure(nrows=dim_out,
                                       ncols=dim_out,
                                       figsize=(3 * dim_out, 3 * dim_out),
                                       keep_ticks=True,
                                       equal_axes=False)
    axes = np.array(axes).reshape(dim_out, dim_out)
    for i in range(dim_out):
        for j in range(dim_out):
            ax = axes[i, j]
            if i == j:
                y = outputs[:, i]
                kde = gaussian_kde(y)
                grid = np.linspace(y.min(), y.max(), 300)
                ax.plot(grid, kde(grid), color='steelblue', lw=2, label='KDE')
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                ax.scatter(outputs[:, j],
                           outputs[:, i],
                           s=2,
                           alpha=0.3,
                           color='steelblue',
                           label='Samples')
            if i == dim_out - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
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
        # plot_output_pairwise(sel_outputs, fig_dir, sel_labels)
        # plot_input_output_scatter(samples, sel_outputs, fig_dir, sel_labels)
    else:
        print('  No columns selected for distribution plots')

    plot_location_heatmaps(args.results_dir, fig_dir, names, labels, outputs,
                           args.locations, args.bins, args.color)


if __name__ == '__main__':
    main()
