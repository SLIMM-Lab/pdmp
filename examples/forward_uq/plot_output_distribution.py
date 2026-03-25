"""Analyze and plot the output distribution from a forward UQ run.

Optionally compares against the theoretical output distribution.
For a linear model y = A x + b with Gaussian input x ~ N(mu, Sigma),
the theoretical output is y ~ N(A mu + b, A Sigma A^T).

Usage:
    python examples/forward_uq/plot_output_distribution.py [results_dir] [--config config.yaml]

Default results_dir: ./results_forward_uq
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, multivariate_normal

from pdmp.loader import get_config


def load_data(results_dir):
    samples = np.loadtxt(os.path.join(results_dir, 'samples.dat'))
    outputs = np.loadtxt(os.path.join(results_dir, 'outputs.dat'))
    if samples.ndim == 1:
        samples = samples[:, None]
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    return samples, outputs


def get_theoretical(config):
    """Return (mean_out, cov_out) if model is Linear and distribution is MultivariateNormal.

    Returns None if the theoretical distribution cannot be computed.
    """
    model_cfg = config.get('model', {})
    dist_cfg = config.get('distribution', {})

    if model_cfg.get('name') != 'Linear' or dist_cfg.get(
            'name') != 'MultivariateNormal':
        return None

    A = np.array(model_cfg['A'])
    b = np.array(model_cfg['b'])
    mu = np.array(dist_cfg['mean'])
    Sigma = np.array(dist_cfg['cov'])

    mean_out = A @ mu + b
    cov_out = A @ Sigma @ A.T
    return mean_out, cov_out


def plot_output_marginals(outputs, fig_dir, theory=None):
    """Histogram + KDE for each output dimension, with optional theoretical Gaussian."""
    dim_out = outputs.shape[1]
    fig, axes = plt.subplots(1,
                             dim_out,
                             figsize=(4 * dim_out, 3.5),
                             squeeze=False)

    for j in range(dim_out):
        ax = axes[0, j]
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

        if theory is not None:
            mean_out, cov_out = theory
            mu_j = mean_out[j]
            sig_j = np.sqrt(cov_out[j, j])
            # extend grid to cover theoretical distribution too
            lo = min(grid[0], mu_j - 4 * sig_j)
            hi = max(grid[-1], mu_j + 4 * sig_j)
            theo_grid = np.linspace(lo, hi, 400)
            ax.plot(theo_grid,
                    norm.pdf(theo_grid, mu_j, sig_j),
                    color='tomato',
                    lw=2,
                    ls='-',
                    label=f'Theory N({mu_j:.3f}, {sig_j:.3f}²)')
            ax.axvline(mu_j, color='tomato', lw=1.2, ls='--')
            ax.set_xlim(lo, hi)

        ax.set_xlabel(f'Output[{j}]')
        ax.set_ylabel('Density' if j == 0 else '')
        ax.legend(fontsize=8)
        ax.set_title(f'Output[{j}]  std={y.std():.3f}' + (
            f'  (theory {np.sqrt(theory[1][j,j]):.3f})' if theory else ''))

    fig.suptitle('Output marginal distributions', y=1.02)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'output_marginals.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_output_pairwise(outputs, fig_dir, theory=None):
    """Pairwise scatter plots of output dimensions (only if dim_out > 1)."""
    dim_out = outputs.shape[1]
    if dim_out < 2:
        return

    fig, axes = plt.subplots(dim_out,
                             dim_out,
                             figsize=(3 * dim_out, 3 * dim_out))
    for i in range(dim_out):
        for j in range(dim_out):
            ax = axes[i, j]
            if i == j:
                y = outputs[:, i]
                kde = gaussian_kde(y)
                grid = np.linspace(y.min(), y.max(), 300)
                ax.plot(grid, kde(grid), color='steelblue', lw=2, label='KDE')
                if theory is not None:
                    mean_out, cov_out = theory
                    mu_j = mean_out[i]
                    sig_j = np.sqrt(cov_out[i, i])
                    lo = min(grid[0], mu_j - 4 * sig_j)
                    hi = max(grid[-1], mu_j + 4 * sig_j)
                    theo_grid = np.linspace(lo, hi, 400)
                    ax.plot(theo_grid,
                            norm.pdf(theo_grid, mu_j, sig_j),
                            color='tomato',
                            lw=2,
                            ls='--',
                            label='Theory')
                    ax.set_xlim(lo, hi)
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                ax.scatter(outputs[:, j],
                           outputs[:, i],
                           s=2,
                           alpha=0.3,
                           color='steelblue',
                           label='Samples')
                corr = np.corrcoef(outputs[:, j], outputs[:, i])[0, 1]
                ax.set_title(f'r={corr:.2f}', fontsize=9)

                if theory is not None:
                    mean_out, cov_out = theory
                    # 2D marginal contours
                    sub_mean = np.array([mean_out[j], mean_out[i]])
                    sub_cov = np.array([[cov_out[j, j], cov_out[j, i]],
                                        [cov_out[i, j], cov_out[i, i]]])
                    rv = multivariate_normal(mean=sub_mean, cov=sub_cov)
                    xlo = min(outputs[:, j].min(),
                              sub_mean[0] - 4 * np.sqrt(sub_cov[0, 0]))
                    xhi = max(outputs[:, j].max(),
                              sub_mean[0] + 4 * np.sqrt(sub_cov[0, 0]))
                    ylo = min(outputs[:, i].min(),
                              sub_mean[1] - 4 * np.sqrt(sub_cov[1, 1]))
                    yhi = max(outputs[:, i].max(),
                              sub_mean[1] + 4 * np.sqrt(sub_cov[1, 1]))
                    gx = np.linspace(xlo, xhi, 100)
                    gy = np.linspace(ylo, yhi, 100)
                    GX, GY = np.meshgrid(gx, gy)
                    Z = rv.pdf(np.stack([GX, GY], axis=-1))
                    ax.contour(GX,
                               GY,
                               Z,
                               levels=5,
                               colors='tomato',
                               linewidths=1.2)
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)

            if i == dim_out - 1:
                ax.set_xlabel(f'Output[{j}]')
            if j == 0:
                ax.set_ylabel(f'Output[{i}]')

    # legend proxy for contours
    if theory is not None:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='steelblue', lw=2, label='Samples / KDE'),
            Line2D([0], [0], color='tomato', lw=2, ls='--', label='Theory')
        ]
        fig.legend(handles=handles, loc='upper right', fontsize=9)

    fig.suptitle('Output pairwise distributions', y=1.01)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'output_pairwise.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def plot_input_output_scatter(samples, outputs, fig_dir):
    """Scatter plots: each input vs each output, coloured by output value."""
    dim_in = samples.shape[1]
    dim_out = outputs.shape[1]

    fig, axes = plt.subplots(dim_out,
                             dim_in,
                             figsize=(3.5 * dim_in, 3 * dim_out),
                             squeeze=False)

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
            corr = np.corrcoef(samples[:, j], outputs[:, i])[0, 1]
            ax.set_title(f'r={corr:.2f}', fontsize=9)
        fig.colorbar(sc,
                     ax=axes[i, :].tolist(),
                     shrink=0.8,
                     label=f'Output[{i}]')

    fig.suptitle('Input–output scatter', y=1.01)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'input_output_scatter.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved {path}')
    plt.close(fig)


def print_summary(samples, outputs, theory=None):
    print('\n--- Summary statistics ---')
    print(f'n_samples : {samples.shape[0]}')
    print(f'dim_in    : {samples.shape[1]}')
    print(f'dim_out   : {outputs.shape[1]}')
    print('\nOutputs:')
    header = f'  {"dim":>4}  {"mean":>10}  {"std":>10}'
    if theory is not None:
        header += f'  {"theo mean":>10}  {"theo std":>10}  {"mean err":>10}'
    print(header)
    for j in range(outputs.shape[1]):
        y = outputs[:, j]
        row = f'  {j:>4}  {y.mean():>10.4f}  {y.std():>10.4f}'
        if theory is not None:
            mu_j = theory[0][j]
            sig_j = np.sqrt(theory[1][j, j])
            row += f'  {mu_j:>10.4f}  {sig_j:>10.4f}  {abs(y.mean() - mu_j):>10.4f}'
        print(row)

    if outputs.shape[1] > 1:
        print('\nEmpirical output correlation matrix:')
        for row in np.corrcoef(outputs.T):
            print('  ' + '  '.join(f'{v:6.3f}' for v in row))
        if theory is not None:
            std_vec = np.sqrt(np.diag(theory[1]))
            theo_corr = theory[1] / np.outer(std_vec, std_vec)
            print('Theoretical output correlation matrix:')
            for row in theo_corr:
                print('  ' + '  '.join(f'{v:6.3f}' for v in row))


def main():
    parser = argparse.ArgumentParser(
        description='Plot forward UQ output distribution')
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='./results_forward_uq',
        help='Directory containing samples.dat and outputs.dat')
    parser.add_argument(
        '--config',
        default=None,
        help=
        'YAML config used to run forward_uq.py (enables theoretical overlay)')
    args = parser.parse_args()

    samples, outputs = load_data(args.results_dir)

    theory = None
    if args.config is not None:
        config = get_config(args.config)
        theory = get_theoretical(config)
        if theory is None:
            print(
                'Note: theoretical distribution not available for this model/distribution combination.'
            )

    fig_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print_summary(samples, outputs, theory)

    print(f'\nSaving figures to {fig_dir}/')
    plot_output_marginals(outputs, fig_dir, theory)
    plot_output_pairwise(outputs, fig_dir, theory)
    plot_input_output_scatter(samples, outputs, fig_dir)


if __name__ == '__main__':
    main()
