#!/usr/bin/env python3
"""Analyze BPS results: log-joint in latent θ-space.

Loads the BPS trajectory, transforms samples from the affine (whitened) space
to the latent θ-space, evaluates the log-joint on a 2-D grid, and produces a
contour plot with BPS samples overlaid.
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde  # noqa: F401 – kept for future use

matplotlib.use('Agg')

from pdmp.loader import get_config, get_target
from pdmp.logger_setup import suppress_external_loggers
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.utils import sample_equidistant_along_path

# ============================================================================
# Configuration
# ============================================================================

BPS_DIR = './bps'
CACHE_DIR = './cache'
FIG_DIR = './figures'

# Grid in latent space θ
THETA1_MIN, THETA1_MAX = -2, 1  # θ₁ (→ rho via sigmoid)
THETA2_MIN, THETA2_MAX = 4, 9  # θ₂ (→ l via exp)

XI1_MIN, XI1_MAX = -5.0, 3.0
XI2_MIN, XI2_MAX = -3.0, 10.0

N_GRID = 50
N_BPS_SAMPLES = 1000  # equidistant samples from BPS trajectory

# Physical parameter bounds (derived from θ limits via sigmoid / exp)
RHO_MIN = 1.0 / (1.0 + np.exp(-THETA1_MIN))  # sigmoid(THETA1_MIN)
RHO_MAX = 1.0 / (1.0 + np.exp(-THETA1_MAX))  # sigmoid(THETA1_MAX)
L_MIN = np.exp(THETA2_MIN)
L_MAX = np.exp(THETA2_MAX)

Y_LIM = 2500

# ============================================================================
# Data loading
# ============================================================================


def load_bps_samples(bps_dir):
    """Load BPS trajectory (positions, velocities, times) and return equidistant samples."""
    pos_file = os.path.join(bps_dir, 'positions.dat')
    vel_file = os.path.join(bps_dir, 'velocities.dat')
    time_file = os.path.join(bps_dir, 'times.dat')

    if not all(os.path.exists(f) for f in [pos_file, vel_file, time_file]):
        print(f"BPS output files not found in {bps_dir}")
        return None, None, None, None

    positions = np.loadtxt(pos_file)
    velocities = np.loadtxt(vel_file)
    times = np.loadtxt(time_file)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 1)

    xi_samples = sample_equidistant_along_path(positions,
                                               velocities,
                                               times,
                                               N=N_BPS_SAMPLES)
    return positions, velocities, times, xi_samples


# ============================================================================
# Grid evaluation
# ============================================================================


def evaluate_log_joint_grid(inner_target, cache_file, force=False):
    """Evaluate inner_target.log_density and prior on a 2-D grid in latent θ-space."""
    if os.path.exists(cache_file) and not force:
        data = np.load(cache_file)
        if 'log_prior_grid' in data:
            return (data['theta1_grid'], data['theta2_grid'],
                    data['log_joint_grid'], data['log_prior_grid'])

    t1_vals = np.linspace(THETA1_MIN, THETA1_MAX, N_GRID)
    t2_vals = np.linspace(THETA2_MIN, THETA2_MAX, N_GRID)
    theta1_grid, theta2_grid = np.meshgrid(t1_vals, t2_vals)
    log_joint_grid = np.full_like(theta1_grid, -np.inf)
    log_prior_grid = np.full_like(theta1_grid, -np.inf)
    model = inner_target._likelihood._likelihood._model

    rho_l_test = np.array([0.5, 40.0])
    model.eval(rho_l_test, save_dir='data_hom'
               )  # test if model eval works at all before filling the grid

    prior = inner_target.prior
    for i in range(N_GRID):
        for j in range(N_GRID):
            theta = np.array([theta1_grid[j, i], theta2_grid[j, i]])
            try:
                log_joint_grid[j, i] = inner_target.log_density(theta)
                log_prior_grid[j, i] = prior.log_density(theta)
            except Exception:
                pass
        if (i + 1) % 5 == 0:
            print(f"  Grid progress: {i+1}/{N_GRID}")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(cache_file,
             theta1_grid=theta1_grid,
             theta2_grid=theta2_grid,
             log_joint_grid=log_joint_grid,
             log_prior_grid=log_prior_grid)
    return theta1_grid, theta2_grid, log_joint_grid, log_prior_grid


# ============================================================================
# Plotting
# ============================================================================


def plot_log_joint_latent(theta1_grid, theta2_grid, log_joint_grid,
                          log_prior_grid, bps_samples_theta):
    """Contour plot of log-joint in latent θ-space with BPS samples overlaid."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap_post = sns.color_palette('rocket', as_cmap=True)

    log_norm = log_joint_grid - np.nanmax(log_joint_grid)
    post_norm = np.exp(log_norm)

    fig, ax = get_2d_despined_figure(plot_limits=([THETA1_MIN, THETA1_MAX],
                                                  [THETA2_MIN, THETA2_MAX]),
                                     figsize=(4., 4.),
                                     axes_label=(r'\theta_1', r'\theta_2'),
                                     equal_axes=False,
                                     keep_ticks=True)

    # Prior contours
    log_prior_norm = log_prior_grid - np.nanmax(log_prior_grid)
    ax.contour(theta1_grid,
               theta2_grid,
               np.exp(log_prior_norm),
               levels=10,
               alpha=0.25,
               cmap='Blues',
               linestyles='-',
               zorder=1)

    # Posterior contours
    ax.contour(theta1_grid,
               theta2_grid,
               post_norm,
               levels=20,
               alpha=0.7,
               cmap=cmap_post,
               zorder=2)

    # BPS samples
    if bps_samples_theta is not None:
        ax.scatter(bps_samples_theta[:, 0],
                   bps_samples_theta[:, 1],
                   c='C0',
                   s=10,
                   alpha=0.6,
                   label='BPS',
                   zorder=3,
                   linewidths=0)

    ax.legend(loc='upper right', frameon=False)
    fig_path = os.path.join(FIG_DIR, 'log_joint_latent.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)


def plot_log_likelihood_latent(theta1_grid, theta2_grid, log_lik_grid):
    """Contour plot of the log-likelihood in latent θ-space."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap = sns.color_palette('mako', as_cmap=True)

    log_norm = log_lik_grid - np.nanmax(log_lik_grid)
    lik_norm = np.exp(log_norm)

    fig, ax = get_2d_despined_figure(plot_limits=([THETA1_MIN, THETA1_MAX],
                                                  [THETA2_MIN, THETA2_MAX]),
                                     figsize=(4., 4.),
                                     axes_label=(r'\theta_1', r'\theta_2'),
                                     equal_axes=False,
                                     keep_ticks=True)

    ax.contour(theta1_grid,
               theta2_grid,
               lik_norm,
               levels=20,
               alpha=0.8,
               cmap=cmap,
               zorder=1)

    fig_path = os.path.join(FIG_DIR, 'log_likelihood_latent.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)


def plot_log_likelihood_physical(theta1_grid, theta2_grid, log_lik_grid):
    """Contour plot of the log-likelihood in physical (rho, l) space.

    Transforms the theta-grid via rho = sigmoid(theta1), l = exp(theta2) —
    consistent with the Composite transformation in the config.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap = sns.color_palette('mako', as_cmap=True)

    rho_grid = 1.0 / (1.0 + np.exp(-theta1_grid))
    l_grid = np.exp(theta2_grid)

    log_norm = log_lik_grid - np.nanmax(log_lik_grid)
    lik_norm = np.exp(log_norm)

    fig, ax = get_2d_despined_figure(plot_limits=([RHO_MIN,
                                                   RHO_MAX], [L_MIN, L_MAX]),
                                     figsize=(4., 4.),
                                     axes_label=(r'\rho', r'l'),
                                     equal_axes=False,
                                     keep_ticks=True)

    ax.contour(rho_grid,
               l_grid,
               lik_norm,
               levels=20,
               alpha=0.8,
               cmap=cmap,
               zorder=1)
    # ax.set_yscale('log')
    ax.set_ylim(0, Y_LIM)

    fig_path = os.path.join(FIG_DIR, 'log_likelihood_physical.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)


def plot_log_joint_physical(theta1_grid, theta2_grid, log_joint_grid,
                            log_prior_grid, bps_samples_theta):
    """Contour plot of log-joint in physical (rho, l) space with BPS samples."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap_post = sns.color_palette('rocket', as_cmap=True)

    rho_grid = 1.0 / (1.0 + np.exp(-theta1_grid))
    l_grid = np.exp(theta2_grid)

    # Jacobian correction: log p(rho, l) = log p(theta) - log(rho) - log(1-rho) - log(l)
    log_jac = np.log(rho_grid) + np.log(1.0 - rho_grid) + np.log(l_grid)
    log_joint_physical = log_joint_grid - log_jac
    log_prior_physical = log_prior_grid - log_jac

    log_norm = log_joint_physical - np.nanmax(log_joint_physical)
    post_norm = np.exp(log_norm)

    log_prior_norm = log_prior_physical - np.nanmax(log_prior_physical)

    fig, ax = get_2d_despined_figure(plot_limits=([RHO_MIN,
                                                   RHO_MAX], [L_MIN, L_MAX]),
                                     figsize=(4., 4.),
                                     axes_label=(r'\rho', r'l'),
                                     equal_axes=False,
                                     keep_ticks=True)

    ax.contour(rho_grid,
               l_grid,
               np.exp(log_prior_norm),
               levels=10,
               alpha=0.25,
               cmap='Blues',
               linestyles='-',
               zorder=1)
    ax.contour(rho_grid,
               l_grid,
               post_norm,
               levels=20,
               alpha=0.7,
               cmap=cmap_post,
               zorder=2)

    ax.set_ylim(0, Y_LIM)

    if bps_samples_theta is not None:
        rho_samples = 1.0 / (1.0 + np.exp(-bps_samples_theta[:, 0]))
        l_samples = np.exp(bps_samples_theta[:, 1])
        ax.scatter(rho_samples,
                   l_samples,
                   c='C0',
                   s=10,
                   alpha=0.6,
                   label='BPS',
                   zorder=3,
                   linewidths=0)
        ax.legend(loc='upper right', frameon=False)

    fig_path = os.path.join(FIG_DIR, 'log_joint_physical.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)


def evaluate_log_joint_grid_affine(inner_target,
                                   transformation,
                                   cache_file,
                                   force=False):
    """Evaluate log-joint on a regular grid in affine (whitened) ξ-space."""
    if os.path.exists(cache_file) and not force:
        data = np.load(cache_file)
        if 'log_prior_grid' in data:
            return (data['xi1_grid'], data['xi2_grid'], data['log_joint_grid'],
                    data['log_prior_grid'])

    xi1_vals = np.linspace(XI1_MIN, XI1_MAX, N_GRID)
    xi2_vals = np.linspace(XI2_MIN, XI2_MAX, N_GRID)
    xi1_grid, xi2_grid = np.meshgrid(xi1_vals, xi2_vals)
    log_joint_grid = np.full_like(xi1_grid, -np.inf)
    log_prior_grid = np.full_like(xi1_grid, -np.inf)

    prior = inner_target.prior
    for i in range(N_GRID):
        for j in range(N_GRID):
            xi = np.array([xi1_grid[j, i], xi2_grid[j, i]])
            theta = transformation.transform(xi)
            try:
                log_joint_grid[j, i] = inner_target.log_density(theta)
                log_prior_grid[j, i] = prior.log_density(theta)
            except Exception:
                pass
        if (i + 1) % 5 == 0:
            print(f"  Grid progress: {i+1}/{N_GRID}")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(cache_file,
             xi1_grid=xi1_grid,
             xi2_grid=xi2_grid,
             log_joint_grid=log_joint_grid,
             log_prior_grid=log_prior_grid)
    return xi1_grid, xi2_grid, log_joint_grid, log_prior_grid


def plot_log_joint_affine(xi1_grid, xi2_grid, log_joint_grid, log_prior_grid,
                          xi_aff_samples):
    """Contour plot of log-joint in affine (whitened) ξ-space with BPS samples."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap_post = sns.color_palette('rocket', as_cmap=True)

    xi1_lim = [xi1_grid.min(), xi1_grid.max()]
    xi2_lim = [xi2_grid.min(), xi2_grid.max()]

    log_norm = log_joint_grid - np.nanmax(log_joint_grid)
    post_norm = np.exp(log_norm)
    log_prior_norm = log_prior_grid - np.nanmax(log_prior_grid)

    fig, ax = get_2d_despined_figure(plot_limits=(xi1_lim, xi2_lim),
                                     figsize=(4., 4.),
                                     axes_label=(r'\xi_1', r'\xi_2'),
                                     equal_axes=False,
                                     keep_ticks=True)

    ax.contour(xi1_grid,
               xi2_grid,
               np.exp(log_prior_norm),
               levels=10,
               alpha=0.25,
               cmap='Blues',
               linestyles='-',
               zorder=1)
    ax.contour(xi1_grid,
               xi2_grid,
               post_norm,
               levels=20,
               alpha=0.7,
               cmap=cmap_post,
               zorder=2)

    if xi_aff_samples is not None:
        ax.scatter(xi_aff_samples[:, 0],
                   xi_aff_samples[:, 1],
                   c='C0',
                   s=10,
                   alpha=0.6,
                   label='BPS',
                   zorder=3,
                   linewidths=0)
        ax.legend(loc='upper right', frameon=False)

    fig_path = os.path.join(FIG_DIR, 'log_joint_affine.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bps-dir', default='./bps')
    parser.add_argument('--force-grid', action='store_true')
    parser.add_argument('--grid-space',
                        choices=['latent', 'affine', 'both'],
                        default='latent')
    args = parser.parse_args()

    global BPS_DIR, FIG_DIR, CACHE_DIR
    BPS_DIR = args.bps_dir
    FIG_DIR = os.path.join(BPS_DIR, 'figures')
    CACHE_DIR = os.path.join(BPS_DIR, 'cache')

    rng = np.random.default_rng(42)

    # Load config — get_config converts numeric lists to numpy arrays
    config_file = os.path.join(BPS_DIR, 'config.yaml')
    config = get_config(config_file)

    # Patch observation_file to absolute path so the script works from any cwd
    obs_abs = os.path.abspath(os.path.join(BPS_DIR, 'observations.dat'))

    def _patch_obs(d):
        if isinstance(d, dict):
            if d.get('name') == 'GaussianLikelihood':
                d['observation_file'] = obs_abs
            for v in d.values():
                _patch_obs(v)

    _patch_obs(config)

    # Load inner BayesianInverse target directly — avoids triggering the Laplace
    # MAP solve that the outer TransformedDistribution would require.
    full_target = get_target(config['problem'], rng=rng)
    inner_target = get_target(config['problem']['distribution'], rng=rng)
    suppress_external_loggers()
    print(f"Inner target dim: {inner_target.dim}")

    # BPS samples live in affine (whitened) space; without the transformation
    # (which requires the MAP point) we cannot map them to θ.
    bps_theta = None
    # Uncomment once the Laplace issue is resolved and the transformation is available:
    _, _, _, xi_aff_samples = load_bps_samples(BPS_DIR)
    if xi_aff_samples is not None:
        bps_theta = np.array([
            full_target._transformation.transform(xi) for xi in xi_aff_samples
        ])

    if bps_theta is not None:
        rho_samples = 1.0 / (1.0 + np.exp(-bps_theta[:, 0]))
        l_samples = np.exp(bps_theta[:, 1])
        phys_samples = np.column_stack([rho_samples, l_samples])
        samples_path = os.path.join(BPS_DIR, 'samples.dat')
        np.savetxt(samples_path, phys_samples, header='rho l')
        print(f"Saved physical samples: {samples_path}")

    cache_latent = os.path.join(CACHE_DIR, 'log_joint_latent.npz')
    cache_affine = os.path.join(CACHE_DIR, 'log_joint_affine.npz')
    tr = full_target._transformation

    if args.grid_space in ('latent', 'both'):
        t1_grid, t2_grid, lj_grid, lp_grid = evaluate_log_joint_grid(
            inner_target, cache_latent, force=args.force_grid)
        ll_grid = lj_grid - lp_grid
        plot_log_joint_latent(t1_grid, t2_grid, lj_grid, lp_grid, bps_theta)
        plot_log_likelihood_latent(t1_grid, t2_grid, ll_grid)
        plot_log_likelihood_physical(t1_grid, t2_grid, ll_grid)
        plot_log_joint_physical(t1_grid, t2_grid, lj_grid, lp_grid, bps_theta)
        # Affine plot via cheap coordinate mapping (no extra FEM calls)
        theta_pts = np.column_stack([t1_grid.ravel(), t2_grid.ravel()])
        xi_pts = (theta_pts - tr._b) @ tr._M_inv.T
        xi1_g = xi_pts[:, 0].reshape(t1_grid.shape)
        xi2_g = xi_pts[:, 1].reshape(t1_grid.shape)
        plot_log_joint_affine(xi1_g, xi2_g, lj_grid, lp_grid, xi_aff_samples)

    if args.grid_space in ('affine', 'both'):
        xi1_g, xi2_g, lj_aff, lp_aff = evaluate_log_joint_grid_affine(
            inner_target, tr, cache_affine, force=args.force_grid)
        plot_log_joint_affine(xi1_g, xi2_g, lj_aff, lp_aff, xi_aff_samples)

    print("Done.")


if __name__ == '__main__':
    main()
