#!/usr/bin/env python3
"""Inverse problem with ExponentialRecoveryField and affine transformation.

This script extends the basic inverse problem with an affine transformation on top
of the Bayesian inverse posterior. It works with three coordinate spaces:

1. **Parameter space (rho, l)**: The original physical parameter space
   - rho ∈ (0, 1): bounded parameter
   - l ∈ (0, ∞): positive parameter

2. **Transformed space (xi)**: After applying likelihood transformations
   - SIGMOID transformation for rho: xi_rho ∈ (-∞, ∞)
   - EXPONENTIAL transformation for l: xi_l ∈ (-∞, ∞)

3. **Affine space (xi_aff)**: After applying affine transformation M and bias b
   - xi_aff = M^(-1) @ (xi - b)
   - This is the space where sampling is performed

The script:
1. Generates synthetic observations (or loads existing ones)
2. Sets up an inverse problem with:
   - ExponentialRecoveryField with parameters [rho, l]
   - Transformed likelihood (SIGMOID for rho, EXPONENTIAL for l)
   - Affine transformation on top
3. Evaluates and plots the 2D unnormalized posterior in both spaces
4. Optionally samples using Random Walk Metropolis (RWM)
5. Optionally samples using ZigZag Sampler (ZZS)
6. Stores all results to disk for quick re-plotting

All transformations are properly accounted for using Jacobian determinants.
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

from pdmp.distributions import EXPONENTIAL, COMPOSITE, SIGMOID
from pdmp.loader import get_target, get_sampler, get_surrogate
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.utils import sample_equidistant_along_path


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

# Grid resolution for posterior evaluation
N_GRID_RHO = 50  # Grid points for rho
N_GRID_L = 50    # Grid points for l

# Sampling parameters
N_RWM_SAMPLES = 100  # Number of RWM samples
T_MAX_ZZS = 10.0    # ZigZag max time

# Uncomment for quick testing with coarser resolution
# N_GRID_RHO = 10
# N_GRID_L = 10

# ============================================================================
# Enable/Disable Samplers
# ============================================================================
RUN_RWM = True      # Set to True to run Random Walk Metropolis
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
    """Evaluate log-posterior on a 2D grid in affine space.

    Note: The grid is defined in affine space (xi_aff), which is where sampling occurs.
    The target distribution handles all transformations internally.
    """
    if os.path.exists(GRID_FILE) and not force:
        print(f"✓ Loading posterior grid from: {GRID_FILE}")
        data = np.load(GRID_FILE)
        return data['xi_rho_grid'], data['xi_l_grid'], data['log_post_grid']

    print("=" * 70)
    print("Evaluating posterior on 2D grid...")
    print("=" * 70)

    # Grid in affine space (xi_aff)
    # The affine transformation maps: xi = M @ xi_aff + b
    # where xi is in the likelihood-transformed space
    xi_rho_min, xi_rho_max = -3.0, 3.0
    xi_l_min, xi_l_max = -3.0, 3.0

    xi_rho_vals = np.linspace(xi_rho_min, xi_rho_max, N_GRID_RHO)
    xi_l_vals = np.linspace(xi_l_min, xi_l_max, N_GRID_L)
    xi_rho_grid, xi_l_grid = np.meshgrid(xi_rho_vals, xi_l_vals)

    # Evaluate log-posterior (target expects points in affine space)
    log_post_grid = np.zeros_like(xi_rho_grid)

    # True values in intermediate (likelihood-transformed) space for reference
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
    """Run Random Walk Metropolis sampling in affine space."""
    if os.path.exists(RWM_FILE) and not force:
        print(f"✓ Loading RWM samples from: {RWM_FILE}")
        return np.load(RWM_FILE)

    print("=" * 70)
    print("Running Random Walk Metropolis sampling...")
    print("=" * 70)

    # Initial point in affine space
    # Starting at origin (0, 0) which corresponds to applying inverse affine transform
    # to the prior mean in likelihood-transformed space
    x_0 = np.array([0.0, 0.0])

    rwm_config = {
        'name': 'RandomWalkMetropolis',
        'sigma': 1.0,  # Step size
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
    """Run ZigZag sampling in affine space."""
    if os.path.exists(ZZS_FILE) and not force:
        print(f"✓ Loading ZigZag samples from: {ZZS_FILE}")
        data = np.load(ZZS_FILE)
        return data['positions'], data['velocities'], data['times'], data['samples']

    print("=" * 70)
    print("Running ZigZag sampling...")
    print("=" * 70)

    # Initial point in affine space
    x_0 = np.array([0.0, 0.0])

    # Surrogate: Laplace approximation fitted at the mode
    # (parameters are automatically computed by get_surrogate)
    surrogate_config = {
        'name': 'Laplace',
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
# Coordinate Transformation Helpers
# ============================================================================

def transform_grid_to_original_space(xi_aff_rho_grid, xi_aff_l_grid, aff_trans, like_trans):
    """Transform affine space grid to original parameter space (rho, l).

    Args:
        xi_aff_rho_grid: Grid of affine-space rho values
        xi_aff_l_grid: Grid of affine-space l values
        aff_trans: Affine transformation object
        like_trans: Likelihood transformation object

    Returns:
        rho_grid, l_grid: Grids in original parameter space
    """
    rho_grid = np.zeros_like(xi_aff_rho_grid)
    l_grid = np.zeros_like(xi_aff_l_grid)

    for i in range(xi_aff_rho_grid.shape[0]):
        for j in range(xi_aff_rho_grid.shape[1]):
            xi_aff = np.array([xi_aff_rho_grid[i, j], xi_aff_l_grid[i, j]])
            xi = aff_trans.transform(xi_aff)
            x = like_trans.transform(xi)
            rho_grid[i, j] = x[0]
            l_grid[i, j] = x[1]

    return rho_grid, l_grid


def compute_log_jacobian_grid(xi_aff_rho_grid, xi_aff_l_grid, aff_trans, like_trans):
    """Compute log-Jacobian determinant for change of variables on grid.

    Args:
        xi_aff_rho_grid: Grid of affine-space rho values
        xi_aff_l_grid: Grid of affine-space l values
        aff_trans: Affine transformation object
        like_trans: Likelihood transformation object

    Returns:
        log_jacobian: Grid of log-Jacobian values
    """
    log_jacobian = np.zeros_like(xi_aff_rho_grid)

    for i in range(xi_aff_rho_grid.shape[0]):
        for j in range(xi_aff_rho_grid.shape[1]):
            xi_aff = np.array([xi_aff_rho_grid[i, j], xi_aff_l_grid[i, j]])
            xi = aff_trans.transform(xi_aff)
            try:
                log_jacobian[i, j] = (like_trans.log_det_jacobian(xi) +
                                      aff_trans.log_det_jacobian(xi_aff))
            except:
                log_jacobian[i, j] = -np.inf

    return log_jacobian


def transform_samples_to_original_space(samples, aff_trans, like_trans):
    """Transform samples from affine space to original parameter space.

    Args:
        samples: Samples in affine space (N x 2)
        aff_trans: Affine transformation object
        like_trans: Likelihood transformation object

    Returns:
        rho_samples, l_samples: Samples in original parameter space
    """
    if samples is None:
        return None, None

    rho_samples = np.zeros(len(samples))
    l_samples = np.zeros(len(samples))

    for i, xi_aff in enumerate(samples):
        xi = aff_trans.transform(xi_aff)
        x = like_trans.transform(xi)
        rho_samples[i] = x[0]
        l_samples[i] = x[1]

    return rho_samples, l_samples


# ============================================================================
# Plotting
# ============================================================================

def plot_posterior_2d(xi_rho_grid, xi_l_grid, log_post_grid,
                      rwm_samples, zzs_samples,
                      xi_rho_true, xi_l_true, target=None):
    """Plot 2D posterior with optional samples in both spaces.

    Creates separate PDF figures for transformed (affine) space and original parameter space.
    This function handles both cases: with or without samples from samplers.

    Args:
        xi_rho_grid: Grid of affine-space rho values
        xi_l_grid: Grid of affine-space l values
        log_post_grid: Log-posterior values on grid
        rwm_samples: RWM samples (can be None)
        zzs_samples: ZigZag samples (can be None)
        xi_rho_true: True value in transformed space
        xi_l_true: True value in transformed space
        target: Target distribution object (optional, used to extract transformations)
    """
    print("=" * 70)
    print("Creating 2D posterior plots...")
    print("=" * 70)

    # Extract transformations from target
    if target is not None and hasattr(target, '_transformation'):
        aff_trans = target._transformation
        like_trans = target._base_distribution._likelihood._transformation

        # Transform grid to original space
        rho_grid, l_grid = transform_grid_to_original_space(
            xi_rho_grid, xi_l_grid, aff_trans, like_trans
        )

        # Transform true values
        xi_aff_true = aff_trans.inverse_transform(np.array([xi_rho_true, xi_l_true]))
    else:
        # Fallback: assume simple transformations (backward compatibility)
        rho_grid = 1.0 / (1.0 + np.exp(-xi_rho_grid))
        l_grid = np.exp(xi_l_grid)
        xi_aff_true = np.array([xi_rho_true, xi_l_true])
        aff_trans = None
        like_trans = None

    # Normalize log-posterior for plotting
    log_post_norm = log_post_grid - np.max(log_post_grid)
    post_norm = np.exp(log_post_norm)

    # Evaluate prior on the grid if target is provided
    log_prior_grid = None
    prior_norm = None
    if target is not None:
        # Get the base distribution's prior
        if hasattr(target, '_base_distribution') and hasattr(target._base_distribution, 'prior'):
            prior = target._base_distribution.prior
            log_prior_grid = np.zeros_like(log_post_grid)

            for i in range(xi_rho_grid.shape[1]):
                for j in range(xi_rho_grid.shape[0]):
                    xi_aff = np.array([xi_rho_grid[j, i], xi_l_grid[j, i]])
                    if aff_trans is not None:
                        xi = aff_trans.transform(xi_aff)
                    else:
                        xi = xi_aff
                    try:
                        log_prior_grid[j, i] = prior.log_density(xi)
                    except:
                        log_prior_grid[j, i] = -np.inf

            # Normalize prior for plotting
            log_prior_norm = log_prior_grid - np.max(log_prior_grid)
            prior_norm = np.exp(log_prior_norm)

    # Define colormaps
    cmap_post = sns.color_palette('rocket', as_cmap=True)
    cmap_prior = sns.color_palette('viridis', as_cmap=True)

    os.makedirs(FIG_DIR, exist_ok=True)

    # ========================================================================
    # Plot 1: Transformed (affine) space
    # ========================================================================
    xi_rho_min, xi_rho_max = xi_rho_grid.min(), xi_rho_grid.max()
    xi_l_min, xi_l_max = xi_l_grid.min(), xi_l_grid.max()

    fig1, ax1 = get_2d_despined_figure(
        plot_limits=([xi_rho_min, xi_rho_max], [xi_l_min, xi_l_max]),
        figsize=(4., 4.),
        axes_label=(r'\xi_\rho', r'\xi_l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours (if available)
    if prior_norm is not None:
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
    # Plot 2: Original parameter space (rho, l)
    # ========================================================================

    # Compute log-Jacobian for change of variables
    if aff_trans is not None and like_trans is not None:
        log_jacobian = compute_log_jacobian_grid(xi_rho_grid, xi_l_grid, aff_trans, like_trans)
    else:
        # Fallback for simple transformations
        log_jacobian = np.log(rho_grid * (1.0 - rho_grid)) + np.log(l_grid)

    log_post_original = log_post_grid - log_jacobian
    log_post_original_norm = log_post_original - np.nanmax(log_post_original)
    post_original_norm = np.exp(log_post_original_norm)

    # Apply change of variables to prior if available
    prior_original_norm = None
    if log_prior_grid is not None:
        log_prior_original = log_prior_grid - log_jacobian
        log_prior_original_norm = log_prior_original - np.nanmax(log_prior_original)
        prior_original_norm = np.exp(log_prior_original_norm)

    # Determine plot limits (with optional manual override)
    rho_min, rho_max = rho_grid.min(), min(rho_grid.max(), 0.81)
    l_min, l_max = l_grid.min(), min(l_grid.max(), 12.5)

    fig2, ax2 = get_2d_despined_figure(
        plot_limits=([rho_min, rho_max], [l_min, l_max]),
        figsize=(4., 4.),
        axes_label=(r'\rho', r'l'),
        equal_axes=False,
        keep_ticks=True
    )

    # Plot prior contours (if available)
    if prior_original_norm is not None:
        ax2.contour(rho_grid, l_grid, prior_original_norm, levels=20,
                    zorder=1, alpha=0.3, cmap=cmap_prior, linestyles='-')

    # Plot posterior contours
    ax2.contour(rho_grid, l_grid, post_original_norm, levels=20,
                zorder=2, alpha=0.6, cmap=cmap_post)

    # Convert samples to original space if available
    if rwm_samples is not None:
        if aff_trans is not None and like_trans is not None:
            rwm_rho, rwm_l = transform_samples_to_original_space(rwm_samples, aff_trans, like_trans)
        else:
            rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
            rwm_l = np.exp(rwm_samples[:, 1])
        ax2.scatter(rwm_rho[::5], rwm_l[::5], c='red', s=10, alpha=0.5,
                    label='RWM', zorder=2)

    if zzs_samples is not None:
        if aff_trans is not None and like_trans is not None:
            zzs_rho, zzs_l = transform_samples_to_original_space(zzs_samples, aff_trans, like_trans)
        else:
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


def plot_marginals(rwm_samples, zzs_samples, xi_rho_true, xi_l_true, target=None):
    """Plot marginal distributions - creates separate PDF figures.

    Args:
        rwm_samples: RWM samples in affine space (can be None)
        zzs_samples: ZigZag samples in affine space (can be None)
        xi_rho_true: True value in transformed space
        xi_l_true: True value in transformed space
        target: Target distribution object (optional, used to extract transformations)
    """
    print("=" * 70)
    print("Creating marginal plots...")
    print("=" * 70)

    # Extract transformations from target
    if target is not None and hasattr(target, '_transformation'):
        aff_trans = target._transformation
        like_trans = target._base_distribution._likelihood._transformation

        # Convert samples to original space
        rwm_rho, rwm_l = transform_samples_to_original_space(rwm_samples, aff_trans, like_trans)
        zzs_rho, zzs_l = transform_samples_to_original_space(zzs_samples, aff_trans, like_trans)
    else:
        # Fallback: assume simple transformations
        if rwm_samples is not None:
            rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
            rwm_l = np.exp(rwm_samples[:, 1])
        else:
            rwm_rho, rwm_l = None, None

        if zzs_samples is not None:
            zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
            zzs_l = np.exp(zzs_samples[:, 1])
        else:
            zzs_rho, zzs_l = None, None

    os.makedirs(FIG_DIR, exist_ok=True)

    # ========================================================================
    # Marginal histogram for rho
    # ========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(4., 3.))
    if rwm_rho is not None:
        ax1.hist(rwm_rho, bins=30, alpha=0.5, density=True, label='RWM', color='red')
    if zzs_rho is not None:
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
    if rwm_l is not None:
        ax2.hist(rwm_l, bins=30, alpha=0.5, density=True, label='RWM', color='red')
    if zzs_l is not None:
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
    if rwm_rho is not None:
        try:
            kde_rwm_rho = gaussian_kde(rwm_rho)
            ax3.plot(rho_grid, kde_rwm_rho(rho_grid), label='RWM', color='red', linewidth=2)
        except:
            pass
    if zzs_rho is not None:
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
    if rwm_l is not None:
        try:
            kde_rwm_l = gaussian_kde(rwm_l)
            ax4.plot(l_grid, kde_rwm_l(l_grid), label='RWM', color='red', linewidth=2)
        except:
            pass
    if zzs_l is not None:
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



def print_summary_statistics(rwm_samples, zzs_samples, target=None):
    """Print summary statistics for the samples.

    Args:
        rwm_samples: RWM samples in affine space (can be None)
        zzs_samples: ZigZag samples in affine space (can be None)
        target: Target distribution object (optional, used to extract transformations)
    """
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print(f"\nTrue values:")
    print(f"  rho = {TRUE_RHO:.3f}")
    print(f"  l   = {TRUE_L:.3f}")

    if rwm_samples is not None:
        # Extract transformations from target
        if target is not None and hasattr(target, '_transformation'):
            aff_trans = target._transformation
            like_trans = target._base_distribution._likelihood._transformation
            rwm_rho, rwm_l = transform_samples_to_original_space(rwm_samples, aff_trans, like_trans)
        else:
            # Fallback: assume simple transformations
            rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
            rwm_l = np.exp(rwm_samples[:, 1])

        print(f"\nRWM estimates (mean ± std):")
        print(f"  rho = {rwm_rho.mean():.3f} ± {rwm_rho.std():.3f}")
        print(f"  l   = {rwm_l.mean():.3f} ± {rwm_l.std():.3f}")

    if zzs_samples is not None:
        # Extract transformations from target
        if target is not None and hasattr(target, '_transformation'):
            aff_trans = target._transformation
            like_trans = target._base_distribution._likelihood._transformation
            zzs_rho, zzs_l = transform_samples_to_original_space(zzs_samples, aff_trans, like_trans)
        else:
            # Fallback: assume simple transformations
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

    # # write config to yaml file for record-keeping
    # from pdmp.loader import numpy_to_yaml
    # import yaml
    # os.makedirs(DATA_DIR, exist_ok=True)
    # config_path = os.path.join(DATA_DIR, "config.yaml")
    # with open(config_path, 'w') as f:
    #     yaml.dump(numpy_to_yaml(config), f)
    # print(f"✓ Configuration saved to: {config_path}")

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

    # Plot results - unified plotting function handles all cases
    plot_posterior_2d(xi_rho_grid, xi_l_grid, log_post_grid,
                      rwm_samples, zzs_samples,
                      xi_rho_true, xi_l_true, target=target)

    # Plot marginals if we have any samples
    if rwm_samples is not None or zzs_samples is not None:
        plot_marginals(rwm_samples, zzs_samples, xi_rho_true, xi_l_true, target=target)

    # Print summary statistics if we have any samples
    if rwm_samples is not None or zzs_samples is not None:
        print_summary_statistics(rwm_samples, zzs_samples, target=target)

    print("\n" + "=" * 70)
    print("✓ All done! Figures saved to:", FIG_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

