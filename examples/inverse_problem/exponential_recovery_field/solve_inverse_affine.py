#!/usr/bin/env python3
"""Inverse problem with ExponentialRecoveryField and transformed likelihood.

This script:
1. Generates synthetic observations (or loads existing ones)
2. Sets up an inverse problem with:
   - ExponentialRecoveryField with parameters [rho, l]
   - Transformed likelihood with SIGMOID for rho and EXPONENTIAL for l
3. Evaluates and plots the 2D unnormalized posterior
4. Samples using Random Walk Metropolis (RWM)
5. Samples using ZigZag Sampler (ZZS)
6. Stores all results to disk for quick re-plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import argparse
from scipy.stats import gaussian_kde
from typing import Dict, Any
import seaborn as sns

from pdmp.distributions import LOGIT, EXPONENTIAL, COMPOSITE, SIGMOID
from pdmp.loader import get_target, get_sampler, get_surrogate
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.forward_model import JaxFemModel
from pdmp.utils import central_moment_from_skeleton, sample_equidistant_along_path


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "./data"
SAMPLES_DIR = "./samples"
FIG_DIR = "./figures"
# FIG_DIR = "/home/leon/Nextcloud/Documents/documentation/itz/figures/fem_model/exp_recov"

# True parameter values for generating observations
# rho in (0, 1), l > 0 (e.g., 0.5 to 2.0)
TRUE_RHO = 0.7
TRUE_L = 1.2

# Noise level for observations
SIGMA_OBS = 0.005

# Sampling parameters (start with coarse for testing)
N_GRID_RHO = 50  # Grid points for rho
N_GRID_L = 50    # Grid points for l
N_RWM_SAMPLES = 50  # RWM samples
T_MAX_ZZS = 10.0     # ZigZag max time

# # for testing
# N_GRID_RHO = 10  # Grid points for rho
# N_GRID_L = 10    # Grid points for l
# ============================================================================
# Enable/Disable Samplers (set to False to skip sampling)
# ============================================================================
RUN_RWM = False      # Set to True to run Random Walk Metropolis
RUN_ZIGZAG = False   # Set to True to run ZigZag sampler

# File names
OBS_FILE = os.path.join(DATA_DIR, "observations.dat")
GRID_FILE = os.path.join(DATA_DIR, "posterior_grid.npz")
RWM_FILE = os.path.join(SAMPLES_DIR, "rwm_samples.npy")
ZZS_FILE = os.path.join(SAMPLES_DIR, "zzs_samples.npz")


# ============================================================================
# Configuration Dictionaries
# ============================================================================

def get_config():
    """Get problem configuration."""
    config = {
        'name': 'Transformed',
        'transformation': 'Affine',
        'M': [[0.25097, 0.59644],
       [0.59644, 2.14637]],
        'b': [1.03476, 0.73861],
        'distribution': {
            'name': 'BayesianInverse',
            'model': {
                'name': 'JaxFem',
                'd_x': 1.0,
                'd_y': 1.0,
                'd_z': 2.5,
                'h': 0.25,
                'nu': 0.3,
                'field': {
                    'name': 'JaxExponentialRecoveryField',
                    'f_infinity': 1.0,
                    'idx': 0,  # Recovery along x-direction
                    'coefficient_distribution': {
                        'name': 'MultivariateNormal',
                        'mean': [0.5, 0.9],  # [rho, l]
                        'cov': [[2., 0.0], [0.0, 2.0]]
                    }
                }
            },
            'prior': {
                'name': 'FromField'
            },
            'likelihood': {
                'name': 'TransformedLikelihood',
                'transformation': COMPOSITE,
                'transformations': [
                    {'type': SIGMOID, 'a': 0.0, 'b': 1.0},  # For rho (bounded to [0, 1])
                    EXPONENTIAL,  # For l (positive)
                ],
                'indices': [np.array([0]), np.array([1])],
                'likelihood': {
                    'name': 'GaussianLikelihood',
                    'sigma': SIGMA_OBS,
                    'observation_file': OBS_FILE
                }
            }
        }
    }
    return config


# ============================================================================
# Data Generation
# ============================================================================

def generate_observations(config: Dict[str, Any], rng: np.random.Generator, force: bool = False):
    """Generate synthetic observations from true parameters."""
    if os.path.exists(OBS_FILE) and not force:
        print(f"✓ Observations already exist: {OBS_FILE}")
        return

    print("=" * 70)
    print("Generating synthetic observations...")
    print("=" * 70)

    # Create field and model using the same approach as the loader
    from pdmp.random_field import get_jax_field
    from pdmp.forward_model import get_model

    # Get the field first
    field_cfg = config['model']['field']
    field = get_jax_field(field_cfg, rng=rng)
    print(f"  Field created: dim={field.dim}")

    # Create model WITH the field
    model = get_model(config['model'], field=field)

    print(f"  Model input dim: {model.get_dim_in()}")
    print(f"  Model output dim: {model.get_dim_out()}")

    # Verify field is attached
    if model.field is None:
        raise RuntimeError("Field was not properly attached to model!")
    print(f"  ✓ Field properly attached to model")

    # True parameters in original space [rho, l]
    theta_true_original = np.array([TRUE_RHO, TRUE_L])

    print(f"  True parameters: rho={TRUE_RHO}, l={TRUE_L}")

    # Generate observations (model works directly with field coefficients [rho, l])
    y_obs = model.eval(theta_true_original).copy()  # Make a writable copy
    # y_obs += rng.standard_normal(model.get_dim_out()) * SIGMA_OBS

    # Save observations
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savetxt(OBS_FILE, y_obs.reshape(1, -1))

    print(f"  ✓ Created observation file: {OBS_FILE}")
    print(f"  Observations shape: {y_obs.shape}")
    print(f"  Observations range: [{y_obs.min():.6f}, {y_obs.max():.6f}]")


# ============================================================================
# Posterior Evaluation
# ============================================================================

def evaluate_posterior_grid(target, rng: np.random.Generator, force: bool = False):
    """Evaluate log-posterior on a 2D grid."""
    if os.path.exists(GRID_FILE) and not force:
        print(f"✓ Loading posterior grid from: {GRID_FILE}")
        data = np.load(GRID_FILE)
        return data['xi_rho_grid'], data['xi_l_grid'], data['log_post_grid']

    print("=" * 70)
    print("Evaluating posterior on 2D grid...")
    print("=" * 70)

    # Grid in transformed space
    # rho in [0, 1] -> xi_rho in [-inf, inf], focus on [-3, 3] (covers ~0.05 to 0.95)
    # l in (0, inf) -> xi_l in [-inf, inf], focus on [-1, 1.5] (covers ~0.37 to 4.48)
    # xi_rho_min, xi_rho_max = 0.0, 1.4
    # xi_l_min, xi_l_max = -1.0, 2.5

    xi_rho_min, xi_rho_max = -3.0, 3.0
    xi_l_min, xi_l_max = -3.0, 3.0

    xi_rho_vals = np.linspace(xi_rho_min, xi_rho_max, N_GRID_RHO)
    xi_l_vals = np.linspace(xi_l_min, xi_l_max, N_GRID_L)
    xi_rho_grid, xi_l_grid = np.meshgrid(xi_rho_vals, xi_l_vals)

    # Evaluate log-posterior
    log_post_grid = np.zeros_like(xi_rho_grid)

    true_xi_rho = np.log(TRUE_RHO / (1.0 - TRUE_RHO))  # logit transform
    true_xi_l = np.log(TRUE_L)  # log transform
    true_xi = np.array([true_xi_rho, true_xi_l])

    print(f"  Grid size: {N_GRID_RHO} × {N_GRID_L} = {N_GRID_RHO * N_GRID_L} points")
    print(f"  xi_rho range: [{xi_rho_min}, {xi_rho_max}]")
    print(f"  xi_l range: [{xi_l_min}, {xi_l_max}]")

    for i in range(N_GRID_RHO):
        for j in range(N_GRID_L):
            xi = np.array([xi_rho_grid[j, i], xi_l_grid[j, i]])
            try:
                log_post_grid[j, i] = target.log_density(xi)
            except Exception as e:
                print(f"    Warning: Error at ({xi[0]:.2f}, {xi[1]:.2f}): {e}")
                log_post_grid[j, i] = -np.inf

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{N_GRID_RHO}")

    # Save grid
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(GRID_FILE, xi_rho_grid=xi_rho_grid, xi_l_grid=xi_l_grid,
             log_post_grid=log_post_grid)

    print(f"  ✓ Saved posterior grid to: {GRID_FILE}")
    print(f"  Log-posterior range: [{np.max(log_post_grid[np.isfinite(log_post_grid)]):.2f}, "
          f"{np.min(log_post_grid[np.isfinite(log_post_grid)]):.2f}]")

    return xi_rho_grid, xi_l_grid, log_post_grid


# ============================================================================
# Sampling with RWM
# ============================================================================

def run_rwm_sampling(target, rng: np.random.Generator, force: bool = False):
    """Run Random Walk Metropolis sampling."""
    if os.path.exists(RWM_FILE) and not force:
        print(f"✓ Loading RWM samples from: {RWM_FILE}")
        return np.load(RWM_FILE)

    print("=" * 70)
    print("Running Random Walk Metropolis sampling...")
    print("=" * 70)

    # Initial point: prior mean in transformed space
    # rho=0.5 -> xi_rho=0, l=1.0 -> xi_l=0
    x_0 = np.array([0.0, 0.0])

    rwm_config = {
        'name': 'RandomWalkMetropolis',
        'sigma': 0.15,  # Step size
        'x_0': x_0,
        'n_samples': N_RWM_SAMPLES
    }

    print(f"  Number of samples: {N_RWM_SAMPLES}")
    print(f"  Initial point: {x_0}")
    print(f"  Step size: {rwm_config['sigma']}")

    sampler = get_sampler(rwm_config, target=target, rng=rng)
    sampler.run()
    samples = sampler.chain

    # Save samples
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    np.save(RWM_FILE, samples)

    print(f"  ✓ Saved RWM samples to: {RWM_FILE}")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Acceptance rate: {sampler._n_accept / sampler._n_samples:.3f}")

    return samples


# ============================================================================
# Sampling with ZigZag
# ============================================================================

def run_zigzag_sampling(target, rng: np.random.Generator, force: bool = False):
    """Run ZigZag sampling."""
    if os.path.exists(ZZS_FILE) and not force:
        print(f"✓ Loading ZigZag samples from: {ZZS_FILE}")
        data = np.load(ZZS_FILE)
        return data['positions'], data['velocities'], data['times'], data['samples']

    print("=" * 70)
    print("Running ZigZag sampling...")
    print("=" * 70)

    # Initial point
    x_0 = np.array([0.0, 0.0])

    # Surrogate (Laplace approximation at mode)
    surrogate_config = {
        'name': 'Laplace',
        # 'mean': [0.0, 0.0],
        # 'cov': [[1.0, 0.0], [0.0, 1.0]]
    }

    zig_zag_config = {
        'name': 'ZigZag',
        't_max': T_MAX_ZZS,
        'dt': 0.01,
        'offset_shrinkage': 0.01,
        'x_0': x_0
    }

    print(f"  Max time: {T_MAX_ZZS}")
    print(f"  Time step: {zig_zag_config['dt']}")
    print(f"  Initial point: {x_0}")

    surrogate = get_surrogate(surrogate_config, target=target, rng=rng)
    zig_zag = get_sampler(zig_zag_config, target=target, rng=rng, surrogate=surrogate)
    zig_zag.run()

    pos = zig_zag.positions
    vel = zig_zag.velocities
    times = zig_zag.times

    # Sample equidistantly along path
    n_samples_zzs = 500
    samples_zzs = sample_equidistant_along_path(pos, vel, times, N=n_samples_zzs)

    # Save
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    np.savez(ZZS_FILE, positions=pos, velocities=vel, times=times, samples=samples_zzs)

    print(f"  ✓ Saved ZigZag samples to: {ZZS_FILE}")
    print(f"  Skeleton points: {len(times)}")
    print(f"  Equidistant samples: {n_samples_zzs}")

    return pos, vel, times, samples_zzs


# ============================================================================
# Plotting
# ============================================================================

def plot_posterior_2d(xi_rho_grid, xi_l_grid, log_post_grid,
                      rwm_samples, zzs_samples,
                      xi_rho_true, xi_l_true, target=None):
    """Plot 2D posterior with samples - creates separate PDF figures."""
    print("=" * 70)
    print("Creating 2D posterior plots...")
    print("=" * 70)

    # Convert transformed space to original space for labeling
    rho_grid = 1.0 / (1.0 + np.exp(-xi_rho_grid))
    l_grid = np.exp(xi_l_grid)

    # Normalize log-posterior for plotting
    log_post_norm = log_post_grid - np.max(log_post_grid)
    post_norm = np.exp(log_post_norm)

    # Evaluate prior on the grid if target is provided
    log_prior_grid = None
    if target is not None and hasattr(target, 'prior'):
        log_prior_grid = np.zeros_like(log_post_grid)
        for i in range(xi_rho_grid.shape[1]):
            for j in range(xi_rho_grid.shape[0]):
                xi = np.array([xi_rho_grid[j, i], xi_l_grid[j, i]])
                try:
                    log_prior_grid[j, i] = target.prior.log_density(xi)
                except:
                    log_prior_grid[j, i] = -np.inf
        # Normalize prior for plotting
        log_prior_norm = log_prior_grid - np.max(log_prior_grid)
        prior_norm = np.exp(log_prior_norm)

    # Define colormaps
    cmap_post = sns.color_palette('rocket', as_cmap=True)  # Posterior
    cmap_prior = sns.color_palette('viridis', as_cmap=True)  # Prior

    os.makedirs(FIG_DIR, exist_ok=True)

    # ========================================================================
    # Plot 1: Transformed space (xi_rho, xi_l)
    # ========================================================================

    # Determine plot limits
    xi_rho_min, xi_rho_max = xi_rho_grid.min(), xi_rho_grid.max()
    xi_l_min, xi_l_max = xi_l_grid.min(), xi_l_grid.max()

    fig1, ax1 = get_2d_despined_figure(
        plot_limits=([xi_rho_min, xi_rho_max], [xi_l_min, xi_l_max]),
        figsize=(4., 4.),
        axes_label=(r'\xi_\rho', r'\xi_l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours first (if available)
    if log_prior_grid is not None:
        ax1.contour(xi_rho_grid, xi_l_grid, prior_norm, levels=20,
                    zorder=1, alpha=0.3, cmap=cmap_prior, linestyles='dashed')

    # Plot posterior contours
    ax1.contour(xi_rho_grid, xi_l_grid, post_norm, levels=20,
                zorder=2, alpha=0.6, cmap=cmap_post)

    # Plot samples
    ax1.scatter(rwm_samples[::5, 0], rwm_samples[::5, 1], c='red', s=10, alpha=0.5,
                label='RWM', zorder=2)
    ax1.scatter(zzs_samples[::5, 0], zzs_samples[::5, 1], c='cyan', s=10, alpha=0.5,
                label='ZZS', zorder=2)

    # True value
    ax1.scatter([xi_rho_true], [xi_l_true], c='black', s=100, marker='*',
                edgecolors='white', linewidths=1.5, label='True', zorder=10)

    ax1.legend(loc='upper right', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'posterior_2d_transformed.pdf')
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Plot 2: Original space (rho, l)
    # ========================================================================

    # Apply change of variables formula
    # log p(rho, l) = log p(xi_rho, xi_l) - log|det J|
    # For logit: log|dxi/drho| = log(1/(rho(1-rho)))
    # For exponential: log|dxi/dl| = log(1/l)
    log_jacobian = np.log(rho_grid * (1.0 - rho_grid)) + np.log(l_grid)
    log_post_original = log_post_grid - log_jacobian

    # Normalize for plotting
    log_post_original_norm = log_post_original - np.nanmax(log_post_original)
    post_original_norm = np.exp(log_post_original_norm)

    # Apply change of variables to prior if available
    prior_original_norm = None
    if log_prior_grid is not None:
        log_prior_original = log_prior_grid - log_jacobian
        log_prior_original_norm = log_prior_original - np.nanmax(log_prior_original)
        prior_original_norm = np.exp(log_prior_original_norm)

    # Determine plot limits
    rho_min, rho_max = rho_grid.min(), rho_grid.max()
    l_min, l_max = l_grid.min(), l_grid.max()

    fig2, ax2 = get_2d_despined_figure(
        plot_limits=([rho_min, rho_max], [l_min, l_max]),
        figsize=(4., 4.),
        axes_label=(r'\rho', r'l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours first (if available)
    if prior_original_norm is not None:
        ax2.contour(rho_grid, l_grid, prior_original_norm, levels=20,
                    zorder=1, alpha=0.3, cmap=cmap_prior, linestyles='dashed')

    # Plot posterior contours
    ax2.contour(rho_grid, l_grid, post_original_norm, levels=20,
                zorder=2, alpha=0.6, cmap=cmap_post)

    # Convert samples to original space
    rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
    rwm_l = np.exp(rwm_samples[:, 1])
    zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
    zzs_l = np.exp(zzs_samples[:, 1])

    ax2.scatter(rwm_rho[::5], rwm_l[::5], c='red', s=10, alpha=0.5,
                label='RWM', zorder=2)
    ax2.scatter(zzs_rho[::5], zzs_l[::5], c='cyan', s=10, alpha=0.5,
                label='ZZS', zorder=2)
    ax2.scatter([TRUE_RHO], [TRUE_L], c='black', s=100, marker='*',
                edgecolors='white', linewidths=1.5, label='True', zorder=10)

    ax2.legend(loc='upper right', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'posterior_2d_original.pdf')
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    return None


def plot_marginals(rwm_samples, zzs_samples, xi_rho_true, xi_l_true):
    """Plot marginal distributions - creates separate PDF figures."""
    print("=" * 70)
    print("Creating marginal plots...")
    print("=" * 70)

    # Convert to original space
    rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
    rwm_l = np.exp(rwm_samples[:, 1])
    zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
    zzs_l = np.exp(zzs_samples[:, 1])

    os.makedirs(FIG_DIR, exist_ok=True)

    # ========================================================================
    # Marginal histogram for rho
    # ========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(4., 3.))
    ax1.hist(rwm_rho, bins=30, alpha=0.5, density=True, label='RWM', color='red')
    ax1.hist(zzs_rho, bins=30, alpha=0.5, density=True, label='ZZS', color='cyan')
    ax1.axvline(TRUE_RHO, color='black', linestyle='--', linewidth=2, label='True')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('Density')
    ax1.legend(frameon=False)
    sns.despine()

    fig_path = os.path.join(FIG_DIR, 'marginal_rho_hist.pdf')
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Marginal histogram for l
    # ========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(4., 3.))
    ax2.hist(rwm_l, bins=30, alpha=0.5, density=True, label='RWM', color='red')
    ax2.hist(zzs_l, bins=30, alpha=0.5, density=True, label='ZZS', color='cyan')
    ax2.axvline(TRUE_L, color='black', linestyle='--', linewidth=2, label='True')
    ax2.set_xlabel(r'$l$')
    ax2.set_ylabel('Density')
    ax2.legend(frameon=False)
    sns.despine()

    fig_path = os.path.join(FIG_DIR, 'marginal_l_hist.pdf')
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    # ========================================================================
    # KDE for rho
    # ========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(4., 3.))
    rho_grid = np.linspace(0, 1, 200)
    try:
        kde_rwm_rho = gaussian_kde(rwm_rho)
        ax3.plot(rho_grid, kde_rwm_rho(rho_grid), label='RWM', color='red', linewidth=2)
    except:
        pass
    try:
        kde_zzs_rho = gaussian_kde(zzs_rho)
        ax3.plot(rho_grid, kde_zzs_rho(rho_grid), label='ZZS', color='cyan', linewidth=2)
    except:
        pass
    ax3.axvline(TRUE_RHO, color='black', linestyle='--', linewidth=2, label='True')
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel('Density')
    ax3.legend(frameon=False)
    sns.despine()

    fig_path = os.path.join(FIG_DIR, 'marginal_rho_kde.pdf')
    fig3.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig3)

    # ========================================================================
    # KDE for l
    # ========================================================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(4., 3.))
    l_grid = np.linspace(0.5, 2.5, 200)
    try:
        kde_rwm_l = gaussian_kde(rwm_l)
        ax4.plot(l_grid, kde_rwm_l(l_grid), label='RWM', color='red', linewidth=2)
    except:
        pass
    try:
        kde_zzs_l = gaussian_kde(zzs_l)
        ax4.plot(l_grid, kde_zzs_l(l_grid), label='ZZS', color='cyan', linewidth=2)
    except:
        pass
    ax4.axvline(TRUE_L, color='black', linestyle='--', linewidth=2, label='True')
    ax4.set_xlabel(r'$l$')
    ax4.set_ylabel('Density')
    ax4.legend(frameon=False)
    sns.despine()

    fig_path = os.path.join(FIG_DIR, 'marginal_l_kde.pdf')
    fig4.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig4)

    return None


def plot_posterior_2d_grid_only(xi_rho_grid, xi_l_grid, log_post_grid,
                                  rwm_samples, zzs_samples,
                                  xi_rho_true, xi_l_true, target=None):
    """Plot 2D posterior grid without requiring both samplers - creates separate PDF figures."""
    print("=" * 70)
    print("Creating 2D posterior plots (grid only)...")
    print("=" * 70)


    like_trans = target._base_distribution._likelihood._transformation
    aff_trans = target._transformation

    rho_grid = np.zeros_like(xi_rho_grid)
    l_grid = np.zeros_like(xi_l_grid)
    for i in range(xi_rho_grid.shape[0]):
        for j in range(xi_rho_grid.shape[1]):
            xi_aff = np.array([xi_rho_grid[i, j], xi_l_grid[i, j]])
            xi = aff_trans.transform(xi_aff)
            x = like_trans.transform(xi)
            rho_grid[i, j] = x[0]
            l_grid[i, j] = x[1]

    xi_aff_true = aff_trans.inverse_transform(np.array([xi_rho_true, xi_l_true]))

    # # Convert transformed space to original space for labeling
    # rho_grid = 1.0 / (1.0 + np.exp(-xi_rho_grid))
    # l_grid = np.exp(xi_l_grid)

    # Normalize log-posterior for plotting
    log_post_norm = log_post_grid - np.max(log_post_grid)
    post_norm = np.exp(log_post_norm)

    # Evaluate prior on the grid if target is provided
    log_prior_grid = None
    if target is not None and hasattr(target._base_distribution, 'prior'):
        prior = target._base_distribution.prior
        log_prior_grid = np.zeros_like(log_post_grid)
        for i in range(xi_rho_grid.shape[1]):
            for j in range(xi_rho_grid.shape[0]):
                xi_aff = np.array([xi_rho_grid[j, i], xi_l_grid[j, i]])
                xi = aff_trans.transform(xi_aff)
                try:
                    log_prior_grid[j, i] = prior.log_density(xi)
                except:
                    log_prior_grid[j, i] = -np.inf
        # Normalize prior for plotting
        log_prior_norm = log_prior_grid - np.max(log_prior_grid)
        prior_norm = np.exp(log_prior_norm)

    # Define colormaps
    cmap_post = sns.color_palette('rocket', as_cmap=True)  # Posterior
    cmap_prior = sns.color_palette('viridis', as_cmap=True)  # Prior

    os.makedirs(FIG_DIR, exist_ok=True)

    # ========================================================================
    # Plot 1: Transformed space (xi_rho, xi_l)
    # ========================================================================

    # Determine plot limits
    xi_rho_min, xi_rho_max = xi_rho_grid.min(), xi_rho_grid.max()
    xi_l_min, xi_l_max = xi_l_grid.min(), xi_l_grid.max()

    fig1, ax1 = get_2d_despined_figure(
        plot_limits=([xi_rho_min, xi_rho_max], [xi_l_min, xi_l_max]),
        figsize=(4., 4.),
        axes_label=(r'\xi_\rho', r'\xi_l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours first (if available)
    if log_prior_grid is not None:
        ax1.contour(xi_rho_grid, xi_l_grid, prior_norm, levels=20,
                    zorder=1, alpha=0.3, cmap=cmap_prior, linestyles='-')

    # Plot posterior contours
    ax1.contour(xi_rho_grid, xi_l_grid, post_norm, levels=20,
                zorder=2, alpha=0.6, cmap=cmap_post)

    # Plot samples if available
    if rwm_samples is not None:
        ax1.scatter(rwm_samples[::5, 0], rwm_samples[::5, 1], c='red', s=10, alpha=0.5,
                    label='RWM', zorder=2)
    if zzs_samples is not None:
        ax1.scatter(zzs_samples[::5, 0], zzs_samples[::5, 1], c='cyan', s=10, alpha=0.5,
                    label='ZZS', zorder=2)

    # True value
    ax1.scatter(*xi_aff_true, c='black', s=100, marker='*',
                edgecolors='white', linewidths=1.5, label='True', zorder=10)

    ax1.legend(loc='upper right', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'posterior_2d_transformed.pdf')
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Plot 2: Original space (rho, l)
    # ========================================================================

    # Apply change of variables formula
    # log p(rho, l) = log p(xi_rho, xi_l) - log|det J|
    # For logit: log|dxi/drho| = log(1/(rho(1-rho)))
    # For exponential: log|dxi/dl| = log(1/l)

    log_jacobian = np.zeros_like(xi_rho_grid)
    for i in range(xi_rho_grid.shape[0]):
        for j in range(xi_rho_grid.shape[1]):
            xi_aff = np.array([xi_rho_grid[i, j], xi_l_grid[i, j]])
            xi = aff_trans.transform(xi_aff)
            try:
                log_jacobian[i, j] = like_trans.log_det_jacobian(xi) + aff_trans.log_det_jacobian(xi_aff)
            except:
                log_jacobian[i, j] = -np.inf

    # log_jacobian = np.log(rho_grid * (1.0 - rho_grid)) + np.log(l_grid)
    log_post_original = log_post_grid - log_jacobian


    # Normalize for plotting
    log_post_original_norm = log_post_original - np.nanmax(log_post_original)
    post_original_norm = np.exp(log_post_original_norm)

    # Apply change of variables to prior if available
    prior_original_norm = None
    if log_prior_grid is not None:
        log_prior_original = log_prior_grid - log_jacobian
        log_prior_original_norm = log_prior_original - np.nanmax(log_prior_original)
        prior_original_norm = np.exp(log_prior_original_norm)

    # Determine plot limits
    rho_min, rho_max = rho_grid.min(), rho_grid.max()
    l_min, l_max = l_grid.min(), l_grid.max()
    l_max = 12.5
    rho_max = 0.81

    fig2, ax2 = get_2d_despined_figure(
        plot_limits=([rho_min, rho_max], [l_min, l_max]),
        figsize=(4., 4.),
        axes_label=(r'\rho', r'l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours first (if available)
    if prior_original_norm is not None:
        ax2.contour(rho_grid, l_grid, prior_original_norm, levels=20,
                    zorder=1, alpha=0.3, cmap=cmap_prior, linestyles='-')

    # Plot posterior contours
    ax2.contour(rho_grid, l_grid, post_original_norm, levels=20,
                zorder=2, alpha=0.6, cmap=cmap_post)

    # Convert samples to original space if available
    if rwm_samples is not None:
        rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
        rwm_l = np.exp(rwm_samples[:, 1])
        ax2.scatter(rwm_rho[::5], rwm_l[::5], c='red', s=10, alpha=0.5,
                    label='RWM', zorder=2)

    if zzs_samples is not None:
        zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
        zzs_l = np.exp(zzs_samples[:, 1])
        ax2.scatter(zzs_rho[::5], zzs_l[::5], c='cyan', s=10, alpha=0.5,
                    label='ZZS', zorder=2)

    ax2.scatter([TRUE_RHO], [TRUE_L], c='black', s=100, marker='*',
                edgecolors='white', linewidths=1.5, label='True', zorder=10)

    ax2.legend(loc='upper right', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'posterior_2d_original.pdf')
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    return None


def print_summary_statistics(rwm_samples, zzs_samples):
    """Print summary statistics for the samples."""
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print(f"\nTrue values:")
    print(f"  rho = {TRUE_RHO:.3f}")
    print(f"  l   = {TRUE_L:.3f}")

    if rwm_samples is not None:
        # Convert to original space
        rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
        rwm_l = np.exp(rwm_samples[:, 1])

        print(f"\nRWM estimates (mean ± std):")
        print(f"  rho = {rwm_rho.mean():.3f} ± {rwm_rho.std():.3f}")
        print(f"  l   = {rwm_l.mean():.3f} ± {rwm_l.std():.3f}")

    if zzs_samples is not None:
        # Convert to original space
        zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
        zzs_l = np.exp(zzs_samples[:, 1])

        print(f"\nZZS estimates (mean ± std):")
        print(f"  rho = {zzs_rho.mean():.3f} ± {zzs_rho.std():.3f}")
        print(f"  l   = {zzs_l.mean():.3f} ± {zzs_l.std():.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Solve inverse problem with ExponentialRecoveryField')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation of all data')
    parser.add_argument('--force-obs', action='store_true',
                        help='Force regeneration of observations')
    parser.add_argument('--force-grid', action='store_true',
                        help='Force recomputation of posterior grid')
    parser.add_argument('--force-samples', action='store_true',
                        help='Force resampling')
    args = parser.parse_args()

    # Set up RNG
    rng = np.random.default_rng(seed=42)

    # Get configuration
    config = get_config()

    # write config to yaml file for record-keeping
    from pdmp.loader import numpy_to_yaml
    import yaml
    os.makedirs(DATA_DIR, exist_ok=True)
    config_path = os.path.join(DATA_DIR, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(numpy_to_yaml(config), f)
    print(f"✓ Configuration saved to: {config_path}")

    # Generate or load observations
    generate_observations(config, rng, force=args.force or args.force_obs)

    # Load target distribution
    print("=" * 70)
    print("Loading target distribution...")
    print("=" * 70)
    target = get_target(config, rng=rng)
    print(f"  Target dimension: {target.dim}")
    print("  ✓ Target loaded successfully")

    # Evaluate posterior grid
    xi_rho_grid, xi_l_grid, log_post_grid = evaluate_posterior_grid(
        target, rng, force=args.force or args.force_grid
    )

    # True values in transformed space
    xi_rho_true = np.log(TRUE_RHO / (1.0 - TRUE_RHO))
    xi_l_true = np.log(TRUE_L)

    # Conditionally run samplers
    rwm_samples = None
    zzs_samples = None

    if RUN_RWM:
        # Run RWM sampling
        rwm_samples = run_rwm_sampling(target, rng, force=args.force or args.force_samples)
    else:
        print("=" * 70)
        print("Skipping RWM sampling (RUN_RWM = False)")
        print("=" * 70)

    if RUN_ZIGZAG:
        # Run ZigZag sampling
        pos, vel, times, zzs_samples = run_zigzag_sampling(
            target, rng, force=args.force or args.force_samples
        )
    else:
        print("=" * 70)
        print("Skipping ZigZag sampling (RUN_ZIGZAG = False)")
        print("=" * 70)

    # Plot results (only if we have samples)
    if rwm_samples is not None and zzs_samples is not None:
        plot_posterior_2d(xi_rho_grid, xi_l_grid, log_post_grid,
                          rwm_samples, zzs_samples,
                          xi_rho_true, xi_l_true, target=target)

        plot_marginals(rwm_samples, zzs_samples, xi_rho_true, xi_l_true)

        # Print summary statistics
        print_summary_statistics(rwm_samples, zzs_samples)
    elif rwm_samples is not None:
        print("=" * 70)
        print("Only RWM samples available - creating grid plot only")
        print("=" * 70)
        plot_posterior_2d_grid_only(xi_rho_grid, xi_l_grid, log_post_grid,
                                     rwm_samples, None,
                                     xi_rho_true, xi_l_true, target=target)
    elif zzs_samples is not None:
        print("=" * 70)
        print("Only ZigZag samples available - creating grid plot only")
        print("=" * 70)
        plot_posterior_2d_grid_only(xi_rho_grid, xi_l_grid, log_post_grid,
                                     None, zzs_samples,
                                     xi_rho_true, xi_l_true, target=target)
    else:
        print("=" * 70)
        print("No samples - creating grid-only plot")
        print("=" * 70)
        plot_posterior_2d_grid_only(xi_rho_grid, xi_l_grid, log_post_grid,
                                     None, None,
                                     xi_rho_true, xi_l_true, target=target)

    print("\n" + "=" * 70)
    print("✓ All done! Figures saved to:", FIG_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

