#!/usr/bin/env python
"""Forward Uncertainty Quantification driver script.

Draws samples from a distribution (or loads them from a file), evaluates a
forward model on each sample, and stores the outputs.

Usage:
    # Sample from a distribution defined in the config
    python forward_uq.py path/to/config.yaml
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pdmp.loader import get_config
from pdmp.forward_model import get_model
from pdmp.distributions import get_prior
from pdmp.random_field import get_field, get_jax_field
from pdmp.logger_setup import suppress_external_loggers
from pdmp.plotting_utils import get_2d_despined_figure


def plot_output_marginals(outputs, fig_dir, output_labels=None):
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

        label = output_labels[
            j] if output_labels is not None else f'Output[{j}]'
        ax.set_xlabel(label)
        ax.set_ylabel('Density' if j % n_cols == 0 else '')
        ax.legend(fontsize=8)

    for j in range(dim_out, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)
    path = os.path.join(fig_dir, 'output_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_output_pairwise(outputs, fig_dir):
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
                ax.set_xlabel(f'Output[{j}]')
            if j == 0:
                ax.set_ylabel(f'Output[{i}]')
    path = os.path.join(fig_dir, 'output_pairwise.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_input_output_scatter(samples, outputs, fig_dir):
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
                ax.set_ylabel(f'Output[{i}]')
        fig.colorbar(sc,
                     ax=axes[i, :].tolist(),
                     shrink=0.8,
                     label=f'Output[{i}]')
    path = os.path.join(fig_dir, 'input_output_scatter.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def flatten_output(out):
    """Flatten a model output to a 1-D numpy array.

    Handles both plain ndarray/scalar outputs and dict outputs (e.g. RVEModel).
    Returns (flat_array, legend) where legend is a list of column labels.
    """
    if isinstance(out, dict):
        parts, legend = [], []
        for key, val in out.items():
            arr = np.atleast_1d(np.array(val, dtype=float)).ravel()
            parts.append(arr)
            if arr.size == 1:
                legend.append(key)
            else:
                legend.extend([f'{key}[{i}]' for i in range(arr.size)])
        return np.concatenate(parts), legend
    else:
        arr = np.atleast_1d(np.array(out, dtype=float)).ravel()
        return arr, [f'out[{i}]' for i in range(arr.size)]


def main():
    parser = argparse.ArgumentParser(description='Forward UQ driver')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    config = get_config(args.config)

    # Optional field (needed by some models)
    field = None
    model_cfg = config['model']
    if isinstance(model_cfg, dict) and 'field' in model_cfg:
        field_cfg = model_cfg['field']
        field_name = field_cfg.get('name', '')
        if field_name.startswith('Jax'):
            field = get_jax_field(field_cfg)
        else:
            field = get_field(field_cfg)

    # Get samples: from file or by drawing from distribution
    uq_cfg = config.get('forward_uq', {})

    if 'samples_file' in uq_cfg:
        samples = np.loadtxt(uq_cfg['samples_file'])
        if samples.ndim == 1:
            samples = samples[:, None]
        print(
            f'Loaded {samples.shape[0]} samples from {uq_cfg["samples_file"]}')
    else:
        seed = int(uq_cfg.get('seed', 42))
        n_samples = int(uq_cfg.get('n_samples', 1000))
        rng = np.random.default_rng(seed)
        distribution = get_prior(config['distribution'], rng=rng, field=field)
        samples = distribution.get_sample(n_samples)

    model = get_model(model_cfg, field=field)
    suppress_external_loggers()

    # Evaluate model on each sample, flattening dict outputs
    legend = None
    rows = []
    for i, s in enumerate(samples):
        raw = model.eval(s)
        flat, leg = flatten_output(raw)
        if legend is None:
            legend = leg
        rows.append(flat)
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f'  Evaluated {i + 1}/{len(samples)}', flush=True)

    outputs = np.array(rows)

    # Apply model-specific LaTeX labels if available
    label_map = getattr(model, 'LATEX_LABELS', {})
    if label_map and legend is not None:
        legend = [label_map.get(l, l) for l in legend]

    # Save results
    out_dir = config.get('output', {}).get('dir', './results_forward_uq')
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir, 'samples.dat'), samples)
    np.savetxt(os.path.join(out_dir, 'outputs.dat'), outputs)

    if legend is not None:
        with open(os.path.join(out_dir, 'outputs_legend.txt'), 'w') as f:
            for i, name in enumerate(legend):
                f.write(f'{i}\t{name}\n')

    mean_out = np.mean(outputs, axis=0)
    std_out = np.std(outputs, axis=0)
    print(f'\nSaved {len(samples)} samples to {out_dir}/')
    print(f'Input dim:  {samples.shape[1]}')
    print(f'Output dim: {outputs.shape[1]}')
    for i, name in enumerate(legend
                             or [f'out[{j}]'
                                 for j in range(outputs.shape[1])]):
        print(f'  {name}: mean = {mean_out[i]:.4f}, std = {std_out[i]:.4f}')

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f'\nSaving figures to {fig_dir}/')
    plot_output_marginals(outputs, fig_dir, legend)
    plot_output_pairwise(outputs, fig_dir)
    plot_input_output_scatter(samples, outputs, fig_dir)


if __name__ == '__main__':
    main()
