#!/usr/bin/env python3
"""Visualise and compare GP surrogate training strategies on a 2-D BananaDistribution.

This script trains a ``GaussianProcess`` surrogate on the banana-shaped
log-density using two strategies:

* **Laplace** – draw training points from the Laplace approximation around the
  MAP.  On a banana target this concentrates samples near the mode but misses
  the curved tails.
* **Bayesian optimisation** – start from a small Laplace batch and then
  iteratively place training points where the acquisition function
  ``weighted_variance`` is maximised, adaptively tracking the true density.

For each strategy the script produces separate PDF figures:

  ``density_true.pdf``               – true log-density contours
  ``unnorm_density_true.pdf``        – true (unnormalised) density contours
  ``density_laplace.pdf``            – Laplace-GP surrogate log-density + training pts
  ``unnorm_density_laplace.pdf``     – Laplace-GP surrogate density + training pts
  ``density_bo.pdf``                 – BO-GP surrogate log-density + training pts
  ``unnorm_density_bo.pdf``          – BO-GP surrogate density + training pts
  ``std_laplace.pdf``                – Laplace-GP posterior std (filled contours)
  ``std_bo.pdf``                     – BO-GP posterior std (filled contours)
  ``difference_laplace.pdf``         – signed error: surrogate − true
  ``difference_bo.pdf``
  ``rmse.pdf``                       – RMSE bar chart

Run
---
    python banana_bo_training.py               # default settings
    python banana_bo_training.py --n-bo 80    # more BO iterations
    python banana_bo_training.py --no-laplace  # BO only
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import gpytorch

# ── project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pdmp.distributions import BananaDistribution, TransformedDistribution, Distribution
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.surrogates import GaussianProcess

# ── global colormap (matches the rest of the project) ─────────────────────────
CMAP = sns.color_palette('rocket', as_cmap=True)

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── output directory ───────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── shared plot geometry ───────────────────────────────────────────────────────
X1_RANGE = (-6.0, 6.0)
X2_RANGE = (-7.0, 3.0)
AXES_LABEL = ('x_1', 'x_2')
FIGSIZE = (4., 4.)


# ==============================================================================
# Target definition
# ==============================================================================

def make_banana(rng: np.random.Generator) -> BananaDistribution:
    """Return a 2-D banana distribution centred near the origin."""
    mean = np.array([0.0, 4.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    return BananaDistribution(mean=mean, cov=cov, a=2.0, b=0.3, rng=rng)

def rotate_banana(banana: BananaDistribution, rot: float) -> Distribution:
    """Return a rotated version of the banana distribution."""
    # Create a rotation matrix
    theta = np.radians(rot)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # The AffineTransformation maps xi (output space) → x (banana space) via x = M @ xi.
    # For a physical CCW rotation of the banana by θ, samples satisfy xi = R @ x_banana,
    # so x_banana = R^{-1} @ xi = R.T @ xi, meaning M = R.T.
    params = {
        'transformation': 'Affine',
        'M': R.T,
        'b': np.zeros(2)
    }

    # Create the transformed distribution
    return TransformedDistribution(base_distribution=banana, params=params)


# ==============================================================================
# Evaluation grid
# ==============================================================================

def make_eval_grid(x1_range=X1_RANGE, x2_range=X2_RANGE, n: int = 60):
    """Return a 2-D meshgrid and the corresponding flat (N, 2) point array."""
    x1 = np.linspace(*x1_range, n)
    x2 = np.linspace(*x2_range, n)
    X1, X2 = np.meshgrid(x1, x2)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, pts


def eval_true(target: BananaDistribution, pts: np.ndarray) -> np.ndarray:
    return np.array([target.log_density(p) for p in pts])


def eval_surrogate(surrogate: GaussianProcess, pts: np.ndarray) -> np.ndarray:
    return np.array([surrogate.eval(p) for p in pts])


def gp_posterior_std(surrogate: GaussianProcess, pts: np.ndarray) -> np.ndarray:
    """Return the GP posterior std of the residual on the evaluation grid."""
    x_t = torch.tensor(pts, dtype=torch.float64)
    surrogate._model.eval()
    surrogate._likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(True):
        pred = surrogate._model(x_t)
    return pred.variance.sqrt().numpy()


# ==============================================================================
# GP training
# ==============================================================================

GP_KWARGS = dict(
    lbfgs_steps=20,
    n_restarts=5,
    lr=0.3,
    tolerance_grad=1e-6,
    tolerance_change=1e-9,
    print_every=5,
)


def train_laplace(target, rng, n_samples: int = 30) -> GaussianProcess:
    print(f"\n{'='*60}")
    print(f"Training GP – Laplace strategy  (n_samples={n_samples})")
    print('='*60)
    return GaussianProcess(
        target=target,
        rng=rng,
        training_strategy='laplace',
        n_samples=n_samples,
        **GP_KWARGS,
    )


def train_bo(
    target,
    rng,
    n_bo_init: int = 10,
    n_bo_iter: int = 40,
    acquisition: str = 'weighted_variance',
) -> GaussianProcess:
    print(f"\n{'='*60}")
    print(f"Training GP – BO strategy")
    print(f"  n_bo_init={n_bo_init}, n_bo_iter={n_bo_iter}, "
          f"acquisition='{acquisition}'")
    print('='*60)
    return GaussianProcess(
        target=target,
        rng=rng,
        training_strategy='bayesian_optimization',
        n_bo_init=n_bo_init,
        n_bo_iter=n_bo_iter,
        acquisition=acquisition,
        bo_bounds_scale=5.0,
        bo_retrain_interval=10,
        # bo_num_restarts=5,
        # bo_raw_samples=256,
        bo_num_restarts=20,
        bo_raw_samples=1024,
        **GP_KWARGS,
    )


# ==============================================================================
# Plotting helpers
# ==============================================================================

def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close(fig)


def _plot_limits():
    return (list(X1_RANGE), list(X2_RANGE))


# ==============================================================================
# Figure: true log-density
# ==============================================================================

def _log_levels(log_norm: np.ndarray, n: int = 20, pct: float = 10.0):
    """Return contour levels spanning from the *pct*-th percentile to 0.

    Using a lower-percentile cutoff prevents sparse contour coverage near the
    mode caused by extreme low-density regions at the grid edges.
    """
    finite = log_norm[np.isfinite(log_norm)]
    vmin = np.percentile(finite, pct)
    return np.linspace(vmin, 0.0, n)


def plot_true_density(X1, X2, true_log: np.ndarray):
    """Contour plot of the normalised true log-density."""
    true_norm = (true_log - true_log.max()).reshape(X1.shape)
    levels = _log_levels(true_norm)

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )
    ax.contour(X1, X2, true_norm, levels=levels, cmap=CMAP, zorder=2)
    ax.set_title('True log-density', fontsize=10)
    _save(fig, 'density_true.pdf')


# ==============================================================================
# Figure: surrogate density + training points
# ==============================================================================

def plot_surrogate_density(
    gp: GaussianProcess,
    X1, X2, pts,
    true_log: np.ndarray,
    label: str,
    filename: str,
):
    """Contour plot of the surrogate log-density with training points overlaid.

    The true density is shown as faint dashed grey contours for reference.
    Training points are coloured by acquisition order (cool colourmap).
    """
    surr_log = eval_surrogate(gp, pts)
    surr_norm = (surr_log - surr_log.max()).reshape(X1.shape)
    true_norm = (true_log - true_log.max()).reshape(X1.shape)
    rmse = np.sqrt(np.mean((true_log - surr_log) ** 2))

    train_x = gp._x_data.numpy()
    n_train = len(train_x)

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )

    # True density as faint dashed reference (same level cutoff for consistency)
    ax.contour(X1, X2, true_norm, levels=_log_levels(true_norm, n=15),
               colors='0.6', linewidths=0.5, alpha=0.5, linestyles='--', zorder=1)

    # Surrogate density (rocket cmap, no fill)
    ax.contour(X1, X2, surr_norm, levels=_log_levels(surr_norm), cmap=CMAP, zorder=2)

    # Training points coloured by order of acquisition (cool → warm = early → late)
    colors = plt.cm.cool(np.linspace(0.15, 0.9, n_train))
    for pt, c in zip(train_x, colors):
        ax.scatter(pt[0], pt[1], color=c, s=28, zorder=5,
                   edgecolors='white', linewidths=0.4)

    ax.set_title(f'{label}  (n={n_train}, RMSE={rmse:.2f})', fontsize=10)
    _save(fig, filename)


# ==============================================================================
# Figure: true unnormalised density
# ==============================================================================

def plot_true_unnorm_density(X1, X2, true_log: np.ndarray):
    """Contour plot of the unnormalised true density exp(log π - max log π)."""
    true_unnorm = np.exp(true_log - true_log.max()).reshape(X1.shape)

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )
    ax.contour(X1, X2, true_unnorm, levels=20, cmap=CMAP, zorder=2)
    ax.set_title('True density', fontsize=10)
    _save(fig, 'unnorm_density_true.pdf')


# ==============================================================================
# Figure: surrogate unnormalised density + training points
# ==============================================================================

def plot_surrogate_unnorm_density(
    gp: GaussianProcess,
    X1, X2, pts,
    true_log: np.ndarray,
    label: str,
    filename: str,
):
    """Contour plot of the surrogate density exp(log π̃ - max log π̃) with
    training points overlaid.  The true density is shown as faint dashed grey
    contours for reference.
    """
    surr_log = eval_surrogate(gp, pts)
    surr_unnorm = np.exp(surr_log - surr_log.max()).reshape(X1.shape)
    true_unnorm = np.exp(true_log - true_log.max()).reshape(X1.shape)

    train_x = gp._x_data.numpy()
    n_train = len(train_x)

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )

    # True density as faint dashed reference
    ax.contour(X1, X2, true_unnorm, levels=15, colors='0.6',
               linewidths=0.5, alpha=0.5, linestyles='--', zorder=1)

    # Surrogate density (rocket cmap, no fill)
    ax.contour(X1, X2, surr_unnorm, levels=20, cmap=CMAP, zorder=2)

    # Training points coloured by acquisition order
    colors = plt.cm.cool(np.linspace(0.15, 0.9, n_train))
    for pt, c in zip(train_x, colors):
        ax.scatter(pt[0], pt[1], color=c, s=28, zorder=5,
                   edgecolors='white', linewidths=0.4)

    ax.set_title(f'{label}  (n={n_train})', fontsize=10)
    _save(fig, filename)


# ==============================================================================
# Figure: GP posterior std (filled contours with rocket cmap)
# ==============================================================================

def plot_surrogate_std(
    gp: GaussianProcess,
    X1, X2, pts,
    label: str,
    filename: str,
):
    """Filled contour plot of the GP posterior standard deviation."""
    std = gp_posterior_std(gp, pts).reshape(X1.shape)
    train_x = gp._x_data.numpy()

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )

    cf = ax.contourf(X1, X2, std, levels=20, cmap=CMAP, zorder=1)
    plt.colorbar(cf, ax=ax, label='GP std (residual)')

    ax.scatter(train_x[:, 0], train_x[:, 1], c='white', s=20, zorder=5,
               edgecolors='k', linewidths=0.5,
               label=f'Training pts  (n={len(train_x)})')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.set_title(f'{label} – posterior std', fontsize=10)
    _save(fig, filename)


# ==============================================================================
# Figure: signed error (surrogate − true)
# ==============================================================================

def plot_difference(
    gp: GaussianProcess,
    X1, X2, pts,
    true_log: np.ndarray,
    label: str,
    filename: str,
):
    """Filled contour plot of the signed approximation error."""
    surr_log = eval_surrogate(gp, pts)
    diff = (surr_log - true_log).reshape(X1.shape)
    train_x = gp._x_data.numpy()

    vmax = np.percentile(np.abs(diff[np.isfinite(diff)]), 95)
    levels = np.linspace(-vmax, vmax, 21)

    fig, ax = get_2d_despined_figure(
        plot_limits=_plot_limits(),
        figsize=FIGSIZE,
        axes_label=AXES_LABEL,
        equal_axes=False,
        keep_ticks=True,
    )

    cf = ax.contourf(X1, X2, diff, levels=levels, cmap='RdBu_r', zorder=1)
    plt.colorbar(cf, ax=ax, label=r'$\log\tilde{\pi} - \log\pi$')

    ax.scatter(train_x[:, 0], train_x[:, 1], c='k', s=20, zorder=5,
               edgecolors='white', linewidths=0.4,
               label=f'Training pts  (n={len(train_x)})')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.set_title(f'{label} – signed error', fontsize=10)
    _save(fig, filename)


# ==============================================================================
# Figure: RMSE bar chart
# ==============================================================================

def plot_rmse(gp_laplace, gp_bo, true_log, pts):
    labels, rmses, counts = [], [], []
    for gp, lbl in [(gp_laplace, 'Laplace'), (gp_bo, 'BO')]:
        if gp is None:
            continue
        surr_log = eval_surrogate(gp, pts)
        labels.append(lbl)
        rmses.append(np.sqrt(np.mean((true_log - surr_log) ** 2)))
        counts.append(len(gp._x_data))

    fig, ax = plt.subplots(figsize=(4., 3.5), constrained_layout=True)
    bars = ax.bar(labels, rmses,
                  color=[CMAP(0.3), CMAP(0.7)][:len(labels)],
                  edgecolor='k', linewidth=0.8, width=0.5)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(rmses),
                f'n={n}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('RMSE (log-density)', fontsize=9)
    ax.set_title('Surrogate approximation quality', fontsize=10)
    sns.despine()
    _save(fig, 'rmse.pdf')


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--n-laplace', type=int, default=40,
                   help='Laplace training samples (default 40)')
    p.add_argument('--n-bo-init', type=int, default=10,
                   help='Initial Laplace samples before BO (default 10)')
    p.add_argument('--n-bo', type=int, default=30,
                   help='Number of BO rounds (default 30)')
    p.add_argument('--acquisition',
                   choices=['max_variance', 'weighted_variance'],
                   default='weighted_variance',
                   help='BO acquisition function (default: weighted_variance)')
    p.add_argument('--no-laplace', action='store_true',
                   help='Skip Laplace baseline')
    p.add_argument('--no-bo', action='store_true',
                   help='Skip BO training')
    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 60)
    print('Banana GP surrogate training comparison')
    print('=' * 60)
    print(f'  Target      : BananaDistribution(a=2.0, b=0.3)')
    print(f'  Laplace n   : {args.n_laplace}')
    print(f'  BO n_init   : {args.n_bo_init}')
    print(f'  BO n_iter   : {args.n_bo}')
    print(f'  Acquisition : {args.acquisition}')
    print(f'  Figures     : {FIG_DIR}')

    # ── target ────────────────────────────────────────────────────────────────
    target_orig = make_banana(np.random.default_rng(SEED))
    target = target_orig
    # target = rotate_banana(target_orig, rot=45.0)

    # ── evaluation grid ───────────────────────────────────────────────────────
    X1, X2, pts = make_eval_grid()
    print('\nEvaluating true log-density on grid ...')
    true_log = eval_true(target, pts)
    print(f'  {len(pts)} pts,  range [{true_log.min():.1f}, {true_log.max():.1f}]')

    # ── train surrogates ──────────────────────────────────────────────────────
    gp_laplace = None
    gp_bo = None

    if not args.no_laplace:
        gp_laplace = train_laplace(
            target, np.random.default_rng(SEED + 1), n_samples=args.n_laplace)

    if not args.no_bo:
        gp_bo = train_bo(
            target, np.random.default_rng(SEED + 2),
            n_bo_init=args.n_bo_init,
            n_bo_iter=args.n_bo,
            acquisition=args.acquisition,
        )

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures ...')

    plot_true_density(X1, X2, true_log)
    plot_true_unnorm_density(X1, X2, true_log)

    if gp_laplace is not None:
        plot_surrogate_density(gp_laplace, X1, X2, pts, true_log,
                               label='Laplace GP',
                               filename='density_laplace.pdf')
        plot_surrogate_unnorm_density(gp_laplace, X1, X2, pts, true_log,
                                      label='Laplace GP',
                                      filename='unnorm_density_laplace.pdf')
        plot_surrogate_std(gp_laplace, X1, X2, pts,
                           label='Laplace GP',
                           filename='std_laplace.pdf')
        plot_difference(gp_laplace, X1, X2, pts, true_log,
                        label='Laplace GP',
                        filename='difference_laplace.pdf')

    if gp_bo is not None:
        plot_surrogate_density(gp_bo, X1, X2, pts, true_log,
                               label='BO GP',
                               filename='density_bo.pdf')
        plot_surrogate_unnorm_density(gp_bo, X1, X2, pts, true_log,
                                      label='BO GP',
                                      filename='unnorm_density_bo.pdf')
        plot_surrogate_std(gp_bo, X1, X2, pts,
                           label='BO GP',
                           filename='std_bo.pdf')
        plot_difference(gp_bo, X1, X2, pts, true_log,
                        label='BO GP',
                        filename='difference_bo.pdf')

    plot_rmse(gp_laplace, gp_bo, true_log, pts)

    # ── summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    for gp, lbl in [(gp_laplace, 'Laplace'), (gp_bo, 'BO')]:
        if gp is None:
            continue
        surr_log = eval_surrogate(gp, pts)
        rmse = np.sqrt(np.mean((true_log - surr_log) ** 2))
        print(f'  {lbl:8s}  n_train={len(gp._x_data):3d}  RMSE={rmse:.4f}')

    print(f'\n✓ All figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
