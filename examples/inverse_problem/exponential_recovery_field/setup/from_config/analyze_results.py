#!/usr/bin/env python3
"""Analyze results from RWM and ZigZag samplers.

This script collects results from separate sampler runs and generates all analysis plots.
It assumes the samplers have been run using run_inference.py and their results are stored
in the rwm/ and zzs/ folders.

The script:
1. Loads configuration from the sampler folders
2. Optionally regenerates observations
3. Evaluates and caches the 2D unnormalized posterior grid
4. Loads samples from RWM and ZigZag samplers
5. Creates comprehensive plots:
   - 2D posterior in both transformed and original space
   - Marginal distributions (histograms and KDE)
   - Exponential recovery field with 95% credible intervals
6. Prints summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import os
import argparse
from scipy.stats import gaussian_kde
from typing import Dict, Any, Optional, Tuple
import seaborn as sns
import yaml

from pdmp.loader import get_target
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.utils import sample_equidistant_along_path
from pdmp.logger_setup import suppress_external_loggers
from pdmp.surrogates import GaussianProcess

# ============================================================================
# Configuration
# ============================================================================

# Directories
RWM_DIR = "./rwm"
ZZS_DIR = "./zzs"
CACHE_DIR = "./cache"
FIG_DIR = "./figures"

# Default noise level (used when regenerating observations)
SIGMA_OBS = 0.005

# Grid resolution for posterior evaluation
N_GRID_RHO = 50  # Grid points for rho
N_GRID_L = 50  # Grid points for l

N_GRID_L = 10
N_GRID_RHO = 10

# Sampling parameters for plotting
SKIP_ZZS = 1
SKIP_RWM = 2

# ============================================================================
# Plotting Options
# ============================================================================
PLOT_PRIOR = False  # Set to False to hide prior contours in plots
PLOT_ZZS_PATH = True  # Set to True to plot full ZigZag path instead of samples
PLOT_TRAINING_DATA = True  # Set to True to plot GP training data locations

# ============================================================================
# Helper Functions
# ============================================================================


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_rwm_samples(rwm_dir: str) -> Optional[np.ndarray]:
    """Load RWM samples from directory.

    Returns:
        samples: Array of shape (n_samples, dim) or None if not available
    """
    samples_file = os.path.join(rwm_dir, "samples.dat")
    if not os.path.exists(samples_file):
        print(f"  Warning: RWM samples not found at {samples_file}")
        return None

    samples = np.loadtxt(samples_file)
    print(f"  ✓ Loaded RWM samples from: {samples_file}")
    print(f"    Shape: {samples.shape}")
    return samples


def load_zzs_samples(
    zzs_dir: str,
    n_equidistant: int = 500
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray]]:
    """Load ZigZag samples from directory.

    Args:
        zzs_dir: Directory containing ZigZag results
        n_equidistant: Number of equidistant samples to extract from trajectory

    Returns:
        positions: Skeleton positions (n_bounces, dim)
        velocities: Skeleton velocities (n_bounces, dim)
        times: Skeleton times (n_bounces,)
        samples: Equidistant samples (n_equidistant, dim)
    """
    pos_file = os.path.join(zzs_dir, "positions.dat")
    vel_file = os.path.join(zzs_dir, "velocities.dat")
    times_file = os.path.join(zzs_dir, "times.dat")

    if not all(os.path.exists(f) for f in [pos_file, vel_file, times_file]):
        print(f"  Warning: ZigZag samples not found in {zzs_dir}")
        return None, None, None, None

    positions = np.loadtxt(pos_file)
    velocities = np.loadtxt(vel_file)
    times = np.loadtxt(times_file)

    # Ensure positions and velocities are 2D
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 1)

    print(f"  ✓ Loaded ZigZag trajectory from: {zzs_dir}")
    print(f"    Skeleton points: {len(times)}")

    # Sample equidistantly along path
    samples = sample_equidistant_along_path(positions,
                                            velocities,
                                            times,
                                            N=n_equidistant)
    print(f"    Equidistant samples: {len(samples)}")

    return positions, velocities, times, samples


def get_true_parameters_from_config(
        config: Dict[str, Any]) -> Tuple[float, float]:
    """Extract true parameter values from observations generation.

    This is a bit tricky since the true parameters aren't explicitly stored.
    We'll need to infer them or use defaults.
    """
    # Default values (you may want to adjust these)
    TRUE_RHO = 0.7
    TRUE_L = 1.2

    return TRUE_RHO, TRUE_L


# ============================================================================
# Data Generation
# ============================================================================


def generate_observations(config: Dict[str, Any],
                          rng: np.random.Generator,
                          obs_file: str,
                          force: bool = False) -> np.ndarray:
    """Generate synthetic observations from true parameters."""
    if os.path.exists(obs_file) and not force:
        print(f"✓ Observations already exist: {obs_file}")
        obs = np.loadtxt(obs_file)
        return obs

    print("=" * 70)
    print("Generating synthetic observations...")
    print("=" * 70)

    # Create field and model using the same approach as the loader
    from pdmp.random_field import get_jax_field
    from pdmp.forward_model import get_model

    # Get the field first
    field_cfg = config['distribution']['model']['field']
    field = get_jax_field(field_cfg, rng=rng)
    print(f"  Field created: dim={field.dim}")

    # Create model WITH the field
    model = get_model(config['distribution']['model'], field=field)

    print(f"  Model input dim: {model.get_dim_in()}")
    print(f"  Model output dim: {model.get_dim_out()}")

    # Verify field is attached
    if model.field is None:
        raise RuntimeError("Field was not properly attached to model!")
    print(f"  ✓ Field properly attached to model")

    # True parameters in original space [rho, l]
    TRUE_RHO, TRUE_L = get_true_parameters_from_config(config)
    theta_true_original = np.array([TRUE_RHO, TRUE_L])

    print(f"  True parameters: rho={TRUE_RHO}, l={TRUE_L}")

    # Generate observations (model works directly with field coefficients [rho, l])
    y_obs = model.eval(theta_true_original).copy()  # Make a writable copy

    # Save observations
    os.makedirs(os.path.dirname(obs_file), exist_ok=True)
    np.savetxt(obs_file, y_obs.reshape(1, -1))

    print(f"  ✓ Created observation file: {obs_file}")
    print(f"  Observations shape: {y_obs.shape}")
    print(f"  Observations range: [{y_obs.min():.6f}, {y_obs.max():.6f}]")

    return y_obs


# ============================================================================
# Posterior Evaluation
# ============================================================================


def evaluate_posterior_grid(target,
                            cache_file: str,
                            rng: np.random.Generator,
                            force: bool = False):
    """Evaluate log-posterior on a 2D grid in affine space.

    Note: The grid is defined in affine space (xi_aff), which is where sampling occurs.
    The target distribution handles all transformations internally.
    """
    if os.path.exists(cache_file) and not force:
        print(f"✓ Loading posterior grid from: {cache_file}")
        data = np.load(cache_file)
        return data['xi_rho_grid'], data['xi_l_grid'], data['log_post_grid']

    print("=" * 70)
    print("Evaluating posterior on 2D grid...")
    print("=" * 70)

    # Grid in affine space (xi_aff)
    xi_rho_min, xi_rho_max = -4.5, 3.0
    xi_l_min, xi_l_max = -3.5, 4.0

    xi_rho_vals = np.linspace(xi_rho_min, xi_rho_max, N_GRID_RHO)
    xi_l_vals = np.linspace(xi_l_min, xi_l_max, N_GRID_L)
    xi_rho_grid, xi_l_grid = np.meshgrid(xi_rho_vals, xi_l_vals)

    # Evaluate log-posterior (target expects points in affine space)
    log_post_grid = np.zeros_like(xi_rho_grid)

    print(
        f"  Grid size: {N_GRID_RHO} × {N_GRID_L} = {N_GRID_RHO * N_GRID_L} points"
    )
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
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(cache_file,
             xi_rho_grid=xi_rho_grid,
             xi_l_grid=xi_l_grid,
             log_post_grid=log_post_grid)

    print(f"  ✓ Saved posterior grid to: {cache_file}")
    print(
        f"  Log-posterior range: [{np.max(log_post_grid[np.isfinite(log_post_grid)]):.2f}, "
        f"{np.min(log_post_grid[np.isfinite(log_post_grid)]):.2f}]")

    return xi_rho_grid, xi_l_grid, log_post_grid


# ============================================================================
# Coordinate Transformation Helpers
# ============================================================================


def transform_grid_to_original_space(xi_aff_rho_grid, xi_aff_l_grid, aff_trans,
                                     like_trans):
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


def compute_log_jacobian_grid(xi_aff_rho_grid, xi_aff_l_grid, aff_trans,
                              like_trans):
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


def plot_posterior_2d(xi_rho_grid,
                      xi_l_grid,
                      log_post_grid,
                      rwm_samples,
                      zzs_samples,
                      xi_rho_true,
                      xi_l_true,
                      target=None,
                      zzs_path=None,
                      plot_prior=True,
                      plot_zzs_path=False,
                      include_rwm=True,
                      include_zzs=True,
                      suffix=''):
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
        zzs_path: Tuple of (positions, velocities, times) for ZigZag path (optional)
        plot_prior: Whether to plot prior contours (default: True)
        plot_zzs_path: Whether to plot full ZigZag path instead of samples (default: False)
        include_rwm: Whether to include RWM samples in the plot (default: True)
        include_zzs: Whether to include ZZS samples in the plot (default: True)
        suffix: Suffix to add to the output filename (default: '')
    """
    print("=" * 70)
    print("Creating 2D posterior plots...")
    print("=" * 70)

    gt_color = 'k'

    # Extract transformations from target
    if target is not None and hasattr(target, '_transformation'):
        aff_trans = target._transformation
        like_trans = target._base_distribution._likelihood._transformation

        # Transform grid to original space
        rho_grid, l_grid = transform_grid_to_original_space(
            xi_rho_grid, xi_l_grid, aff_trans, like_trans)

        # Transform true values
        xi_aff_true = aff_trans.inverse_transform(
            np.array([xi_rho_true, xi_l_true]))
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
        if hasattr(target, '_base_distribution') and hasattr(
                target._base_distribution, 'prior'):
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

    fig1, ax1 = get_2d_despined_figure(plot_limits=([xi_rho_min, xi_rho_max],
                                                    [xi_l_min, xi_l_max]),
                                       figsize=(4., 4.),
                                       axes_label=(r'\xi_\rho', r'\xi_l'),
                                       equal_axes=False,
                                       keep_ticks=True)

    # Plot prior contours (if available and enabled)
    if prior_norm is not None and plot_prior:
        ax1.contour(xi_rho_grid,
                    xi_l_grid,
                    prior_norm,
                    levels=20,
                    zorder=1,
                    alpha=0.3,
                    cmap=cmap_prior,
                    linestyles='-')

    # Plot posterior contours
    ax1.contour(xi_rho_grid,
                xi_l_grid,
                post_norm,
                levels=20,
                zorder=2,
                alpha=0.6,
                cmap=cmap_post)

    # Plot samples if available and requested
    if rwm_samples is not None and include_rwm:
        ax1.plot(rwm_samples[::SKIP_RWM, 0],
                 rwm_samples[::SKIP_RWM, 1],
                 'o',
                 c='C3',
                 ms=3,
                 alpha=0.5,
                 label='RWM',
                 zorder=2,
                 mec='none')

    # Plot ZigZag: either full path or samples
    if include_zzs:
        if plot_zzs_path and zzs_path is not None:
            # Plot the full piecewise linear path
            positions, velocities, times = zzs_path
            ax1.plot(positions[:, 0],
                     positions[:, 1],
                     c='C0',
                     linewidth=1.5,
                     alpha=0.7,
                     zorder=2)
        elif zzs_samples is not None:
            # Plot equidistant samples
            ax1.scatter(zzs_samples[::5, 0],
                        zzs_samples[::1, 1],
                        c='cyan',
                        s=10,
                        alpha=0.5,
                        label='ZZS',
                        zorder=2)

    # True value
    ax1.scatter(*xi_aff_true,
                c=gt_color,
                s=120,
                marker='*',
                edgecolors='none',
                linewidths=1.5,
                label='True',
                zorder=10)

    ax1.legend(loc='upper right', frameon=False)

    # Save
    filename = 'posterior_2d_transformed' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Plot 2: Original parameter space (rho, l)
    # ========================================================================

    # Compute log-Jacobian for change of variables
    if aff_trans is not None and like_trans is not None:
        log_jacobian = compute_log_jacobian_grid(xi_rho_grid, xi_l_grid,
                                                 aff_trans, like_trans)
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
        log_prior_original_norm = log_prior_original - np.nanmax(
            log_prior_original)
        prior_original_norm = np.exp(log_prior_original_norm)

    # Determine plot limits
    rho_min, rho_max = 0.0, 1.0  # Force limits for rho since it's bounded
    l_min, l_max = 0.0, 15.  # Force limits for l based on true value and prior

    fig2, ax2 = get_2d_despined_figure(plot_limits=([rho_min,
                                                     rho_max], [l_min, l_max]),
                                       figsize=(4., 4.),
                                       axes_label=(r'\rho', r'l'),
                                       equal_axes=False,
                                       keep_ticks=True)

    # Plot prior contours (if available and enabled)
    if prior_original_norm is not None and plot_prior:
        ax2.contour(rho_grid,
                    l_grid,
                    prior_original_norm,
                    levels=20,
                    zorder=1,
                    alpha=0.3,
                    cmap=cmap_prior,
                    linestyles='-')

    # Plot posterior contours
    ax2.contour(rho_grid,
                l_grid,
                post_original_norm,
                levels=20,
                zorder=2,
                alpha=0.6,
                cmap=cmap_post)

    # Convert samples to original space if available and requested
    if rwm_samples is not None and include_rwm:
        if aff_trans is not None and like_trans is not None:
            rwm_rho, rwm_l = transform_samples_to_original_space(
                rwm_samples, aff_trans, like_trans)
        else:
            rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
            rwm_l = np.exp(rwm_samples[:, 1])
        ax2.plot(rwm_rho[::SKIP_RWM],
                 rwm_l[::SKIP_RWM],
                 'o',
                 c='C3',
                 ms=3,
                 alpha=0.5,
                 label='RWM',
                 zorder=2,
                 mec='none')

    if zzs_samples is not None and include_zzs:
        # Plot equidistant samples
        if aff_trans is not None and like_trans is not None:
            zzs_rho, zzs_l = transform_samples_to_original_space(
                zzs_samples, aff_trans, like_trans)
        else:
            zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
            zzs_l = np.exp(zzs_samples[:, 1])
        ax2.plot(zzs_rho[::SKIP_ZZS],
                 zzs_l[::SKIP_ZZS],
                 'o',
                 c='C0',
                 ms=3,
                 alpha=0.5,
                 label='ZZS',
                 zorder=2,
                 mec='none')

    # True value
    TRUE_RHO, TRUE_L = 0.7, 1.2  # Default values
    ax2.scatter([TRUE_RHO], [TRUE_L],
                c=gt_color,
                s=120,
                marker='*',
                edgecolors='none',
                linewidths=1.5,
                label='True',
                zorder=10)

    ax2.legend(loc='upper right', frameon=False)

    # Save
    filename = 'posterior_2d_original' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    return None


def plot_marginals(rwm_samples,
                   zzs_samples,
                   xi_rho_true,
                   xi_l_true,
                   target=None,
                   include_rwm=True,
                   include_zzs=True,
                   suffix=''):
    """Plot marginal distributions - creates separate PDF figures.

    Args:
        rwm_samples: RWM samples in affine space (can be None)
        zzs_samples: ZigZag samples in affine space (can be None)
        xi_rho_true: True value in transformed space
        xi_l_true: True value in transformed space
        target: Target distribution object (optional, used to extract transformations)
        include_rwm: Whether to include RWM samples in the plot (default: True)
        include_zzs: Whether to include ZZS samples in the plot (default: True)
        suffix: Suffix to add to the output filename (default: '')
    """
    print("=" * 70)
    print("Creating marginal plots...")
    print("=" * 70)

    # Extract transformations from target
    if target is not None and hasattr(target, '_transformation'):
        aff_trans = target._transformation
        like_trans = target._base_distribution._likelihood._transformation

        # Convert samples to original space
        rwm_rho, rwm_l = transform_samples_to_original_space(
            rwm_samples, aff_trans, like_trans)
        zzs_rho, zzs_l = transform_samples_to_original_space(
            zzs_samples, aff_trans, like_trans)
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

    TRUE_RHO, TRUE_L = 0.7, 1.2  # Default values

    # ========================================================================
    # Marginal histogram for rho
    # ========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(4., 3.))
    if rwm_rho is not None and include_rwm:
        ax1.hist(rwm_rho,
                 bins=30,
                 alpha=0.5,
                 density=True,
                 label='RWM',
                 color='red')
    if zzs_rho is not None and include_zzs:
        ax1.hist(zzs_rho,
                 bins=30,
                 alpha=0.5,
                 density=True,
                 label='ZZS',
                 color='cyan')
    ax1.axvline(TRUE_RHO,
                color='black',
                linestyle='--',
                linewidth=2,
                label='True')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('Density')
    ax1.legend(frameon=False)
    sns.despine()

    filename = 'marginal_rho_hist' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Marginal histogram for l
    # ========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(4., 3.))
    if rwm_l is not None and include_rwm:
        ax2.hist(rwm_l,
                 bins=30,
                 alpha=0.5,
                 density=True,
                 label='RWM',
                 color='red')
    if zzs_l is not None and include_zzs:
        ax2.hist(zzs_l,
                 bins=30,
                 alpha=0.5,
                 density=True,
                 label='ZZS',
                 color='cyan')
    ax2.axvline(TRUE_L,
                color='black',
                linestyle='--',
                linewidth=2,
                label='True')
    ax2.set_xlabel(r'$l$')
    ax2.set_ylabel('Density')
    ax2.legend(frameon=False)
    sns.despine()

    filename = 'marginal_l_hist' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    # ========================================================================
    # KDE for rho
    # ========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(4., 3.))
    rho_grid = np.linspace(0, 1, 200)
    if rwm_rho is not None and include_rwm:
        try:
            kde_rwm_rho = gaussian_kde(rwm_rho)
            ax3.plot(rho_grid,
                     kde_rwm_rho(rho_grid),
                     label='RWM',
                     color='red',
                     linewidth=2)
        except:
            pass
    if zzs_rho is not None and include_zzs:
        try:
            kde_zzs_rho = gaussian_kde(zzs_rho)
            ax3.plot(rho_grid,
                     kde_zzs_rho(rho_grid),
                     label='ZZS',
                     color='cyan',
                     linewidth=2)
        except:
            pass
    ax3.axvline(TRUE_RHO,
                color='black',
                linestyle='--',
                linewidth=2,
                label='True')
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel('Density')
    ax3.legend(frameon=False)
    sns.despine()

    filename = 'marginal_rho_kde' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig3.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig3)

    # ========================================================================
    # KDE for l
    # ========================================================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(4., 3.))
    l_grid = np.linspace(0.5, 2.5, 200)
    if rwm_l is not None and include_rwm:
        try:
            kde_rwm_l = gaussian_kde(rwm_l)
            ax4.plot(l_grid,
                     kde_rwm_l(l_grid),
                     label='RWM',
                     color='red',
                     linewidth=2)
        except:
            pass
    if zzs_l is not None and include_zzs:
        try:
            kde_zzs_l = gaussian_kde(zzs_l)
            ax4.plot(l_grid,
                     kde_zzs_l(l_grid),
                     label='ZZS',
                     color='cyan',
                     linewidth=2)
        except:
            pass
    ax4.axvline(TRUE_L,
                color='black',
                linestyle='--',
                linewidth=2,
                label='True')
    ax4.set_xlabel(r'$l$')
    ax4.set_ylabel('Density')
    ax4.legend(frameon=False)
    sns.despine()

    filename = 'marginal_l_kde' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig4.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig4)


def plot_exponential_field(rwm_samples,
                           zzs_samples,
                           target=None,
                           config=None,
                           include_rwm=True,
                           include_zzs=True,
                           suffix=''):
    """Plot exponential recovery field with 95% credible intervals.

    The field is F(x) = F_inf * (1 - (1 - rho) * exp(-x/l))

    Args:
        rwm_samples: RWM samples in affine space (can be None)
        zzs_samples: ZigZag samples in affine space (can be None)
        target: Target distribution object (optional, used to extract transformations)
        config: Configuration dictionary containing field parameters
        include_rwm: Whether to include RWM samples in the plot (default: True)
        include_zzs: Whether to include ZZS samples in the plot (default: True)
        suffix: Suffix to add to the output filename (default: '')
    """
    print("=" * 70)
    print("Creating exponential field plots with 95% credible intervals...")
    print("=" * 70)

    # Get field parameters from config
    if config is None:
        print("  Warning: No config provided, using defaults")
        f_infinity = 1.0
        d_x = 1.0
    else:
        model_config = config.get('distribution', {}).get('model', {})
        field_config = model_config.get('field', {})
        f_infinity = field_config.get('f_infinity', 1.0)
        d_x = model_config.get('d_z', 1.0)

    print(f"  F_infinity = {f_infinity}")
    print(f"  x range: [0, {d_x}]")

    # Create x grid
    x_vals = np.linspace(0, d_x, 200)

    # Function to evaluate field
    def eval_field(rho, l, x):
        """Evaluate F(x) = F_inf * (1 - (1 - rho) * exp(-x/l))"""
        return f_infinity * (1.0 - (1.0 - rho) * np.exp(-x / l))

    # Extract transformations from target
    if target is not None and hasattr(target, '_transformation'):
        aff_trans = target._transformation
        like_trans = target._base_distribution._likelihood._transformation
    else:
        aff_trans = None
        like_trans = None

    os.makedirs(FIG_DIR, exist_ok=True)

    # Initialize variables
    rwm_median = rwm_lower = rwm_upper = None
    zzs_median = zzs_lower = zzs_upper = None
    prior_median = prior_lower = prior_upper = None

    # Generate prior samples and compute credible interval
    if target is not None and hasattr(target, '_base_distribution'):
        base_dist = target._base_distribution
        if hasattr(base_dist, 'prior'):
            print("  Generating prior samples...")
            n_prior_samples = 5000

            # Sample from prior in xi space (transformed space)
            prior = base_dist.prior
            prior_samples_xi = np.array(
                [prior.get_sample() for _ in range(n_prior_samples)])

            # Transform to (rho, l) space using likelihood transformation
            if like_trans is not None:
                prior_rho = np.zeros(n_prior_samples)
                prior_l = np.zeros(n_prior_samples)

                for i in range(n_prior_samples):
                    xi = prior_samples_xi[i]
                    x = like_trans.transform(xi)
                    prior_rho[i] = x[0]
                    prior_l[i] = x[1]

                # Evaluate field for all prior samples
                prior_fields = np.zeros((n_prior_samples, len(x_vals)))
                for i in range(n_prior_samples):
                    prior_fields[i, :] = eval_field(prior_rho[i], prior_l[i],
                                                    x_vals)

                # Compute percentiles
                prior_median = np.percentile(prior_fields, 50, axis=0)
                prior_lower = np.percentile(prior_fields, 2.5, axis=0)
                prior_upper = np.percentile(prior_fields, 97.5, axis=0)

                print(f"  Prior: {n_prior_samples} samples processed")
            else:
                print(
                    "  Warning: No likelihood transformation available, skipping prior"
                )

    # Process RWM samples
    if rwm_samples is not None:
        if aff_trans is not None and like_trans is not None:
            rwm_rho, rwm_l = transform_samples_to_original_space(
                rwm_samples, aff_trans, like_trans)
        else:
            rwm_rho = 1.0 / (1.0 + np.exp(-rwm_samples[:, 0]))
            rwm_l = np.exp(rwm_samples[:, 1])

        # Evaluate field for all samples
        rwm_fields = np.zeros((len(rwm_rho), len(x_vals)))
        for i in range(len(rwm_rho)):
            rwm_fields[i, :] = eval_field(rwm_rho[i], rwm_l[i], x_vals)

        # Compute percentiles
        rwm_median = np.percentile(rwm_fields, 50, axis=0)
        rwm_lower = np.percentile(rwm_fields, 2.5, axis=0)
        rwm_upper = np.percentile(rwm_fields, 97.5, axis=0)

        print(f"  RWM: {len(rwm_rho)} samples processed")

    # Process ZZS samples
    if zzs_samples is not None:
        if aff_trans is not None and like_trans is not None:
            zzs_rho, zzs_l = transform_samples_to_original_space(
                zzs_samples, aff_trans, like_trans)
        else:
            zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
            zzs_l = np.exp(zzs_samples[:, 1])

        # Evaluate field for all samples
        zzs_fields = np.zeros((len(zzs_rho), len(x_vals)))
        for i in range(len(zzs_rho)):
            zzs_fields[i, :] = eval_field(zzs_rho[i], zzs_l[i], x_vals)

        # Compute percentiles
        zzs_median = np.percentile(zzs_fields, 50, axis=0)
        zzs_lower = np.percentile(zzs_fields, 2.5, axis=0)
        zzs_upper = np.percentile(zzs_fields, 97.5, axis=0)

        print(f"  ZZS: {len(zzs_rho)} samples processed")

    # True field
    TRUE_RHO, TRUE_L = 0.7, 1.2  # Default values
    true_field = eval_field(TRUE_RHO, TRUE_L, x_vals)

    # Create plot
    fig, ax = get_2d_despined_figure(figsize=(5., 3.),
                                     equal_axes=False,
                                     keep_ticks=True)

    if prior_median is not None:
        ax.plot(x_vals,
                prior_median,
                'C0',
                linewidth=1.5,
                linestyle='-',
                label='Prior (median)',
                zorder=3,
                alpha=1)
        ax.fill_between(x_vals,
                        prior_lower,
                        prior_upper,
                        color='C0',
                        alpha=0.15,
                        label='Prior (95% CI)',
                        zorder=0)

    # Plot RWM
    if rwm_samples is not None and rwm_median is not None and include_rwm:
        ax.plot(x_vals,
                rwm_median,
                '-',
                color='C3',
                linewidth=2,
                label='RWM (median)',
                zorder=3)
        ax.fill_between(x_vals,
                        rwm_lower,
                        rwm_upper,
                        color='C3',
                        alpha=0.2,
                        label='RWM (95% CI)',
                        zorder=2)

    # Plot ZZS
    if zzs_samples is not None and zzs_median is not None and include_zzs:
        ax.plot(x_vals,
                zzs_median,
                '-',
                c='C1',
                linewidth=2,
                label='ZZS (median)',
                zorder=3)
        ax.fill_between(x_vals,
                        zzs_lower,
                        zzs_upper,
                        color='C1',
                        alpha=0.2,
                        label='ZZS (95% CI)',
                        zorder=2)

    # Plot true field
    ax.plot(x_vals,
            true_field,
            'k--',
            linewidth=2,
            label='True field',
            zorder=4)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$F(x)$')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    sns.despine()

    # Save
    filename = 'exponential_field' + suffix + '.pdf'
    fig_path = os.path.join(FIG_DIR, filename)
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig)


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

    TRUE_RHO, TRUE_L = 0.7, 1.2  # Default values
    print(f"\nTrue values:")
    print(f"  rho = {TRUE_RHO:.3f}")
    print(f"  l   = {TRUE_L:.3f}")

    if rwm_samples is not None:
        # Extract transformations from target
        if target is not None and hasattr(target, '_transformation'):
            aff_trans = target._transformation
            like_trans = target._base_distribution._likelihood._transformation
            rwm_rho, rwm_l = transform_samples_to_original_space(
                rwm_samples, aff_trans, like_trans)
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
            zzs_rho, zzs_l = transform_samples_to_original_space(
                zzs_samples, aff_trans, like_trans)
        else:
            # Fallback: assume simple transformations
            zzs_rho = 1.0 / (1.0 + np.exp(-zzs_samples[:, 0]))
            zzs_l = np.exp(zzs_samples[:, 1])

        print(f"\nZZS estimates (mean ± std):")
        print(f"  rho = {zzs_rho.mean():.3f} ± {zzs_rho.std():.3f}")
        print(f"  l   = {zzs_l.mean():.3f} ± {zzs_l.std():.3f}")


def load_zzs_surrogate(zzs_dir: str, target,
                       rng: np.random.Generator) -> Optional[GaussianProcess]:
    """Load ZigZag Gaussian Process surrogate from directory.

    Args:
        zzs_dir: Directory containing ZigZag results
        target: Target distribution (needed to initialize surrogate)
        rng: Random number generator

    Returns:
        GaussianProcess surrogate model or None if not available
    """
    model_dir = os.path.join(zzs_dir, "model_params")

    if not os.path.exists(model_dir):
        print(f"  Warning: Surrogate model directory not found at {model_dir}")
        return None

    model_file = os.path.join(model_dir, "model_params.th")
    if not os.path.exists(model_file):
        print(f"  Warning: Surrogate model file not found at {model_file}")
        return None

    print(f"  Loading ZigZag surrogate from: {model_dir}")

    # Create GaussianProcess surrogate (without training)
    surrogate = GaussianProcess(target=target,
                                rng=rng,
                                train_on_init=False,
                                mean=np.zeros(2),
                                cov=np.eye(2))

    # Load the trained model
    surrogate.load_model(model_dir)

    print(
        f"  ✓ Loaded surrogate with {len(surrogate._x_data)} training points")

    return surrogate


def plot_surrogate_vs_posterior(xi_rho_grid,
                                xi_l_grid,
                                log_post_grid,
                                surrogate,
                                xi_rho_true,
                                xi_l_true,
                                target,
                                plot_training_data=True):
    """Plot surrogate vs true posterior on 2D grid.

    Args:
        xi_rho_grid: Grid of affine-space rho values
        xi_l_grid: Grid of affine-space l values
        log_post_grid: Log-posterior values on grid
        surrogate: GaussianProcess surrogate model
        xi_rho_true: True value in transformed space
        xi_l_true: True value in transformed space
        plot_training_data: Whether to plot training data locations (default: True)
    """
    print("=" * 70)
    print("Creating surrogate vs posterior plots...")
    print("=" * 70)

    print("xi rho true, xi l true:", xi_rho_true, xi_l_true)

    aff_trans = target._transformation
    like_trans = target._base_distribution._likelihood._transformation

    # Transform true values
    xi_aff_true = aff_trans.inverse_transform(
        np.array([xi_rho_true, xi_l_true]))

    # Evaluate surrogate on the grid
    log_surrogate_grid = np.zeros_like(log_post_grid)

    for i in range(xi_rho_grid.shape[1]):
        for j in range(xi_rho_grid.shape[0]):
            xi = np.array([xi_rho_grid[j, i], xi_l_grid[j, i]])
            try:
                # Surrogate.eval returns log-density
                log_surrogate_grid[j, i] = surrogate.eval(xi)
            except Exception as e:
                print(
                    f"    Warning: Error evaluating surrogate at ({xi[0]:.2f}, {xi[1]:.2f}): {e}"
                )
                log_surrogate_grid[j, i] = -np.inf

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{xi_rho_grid.shape[1]}")

    # Normalize both for plotting
    log_post_norm = log_post_grid - np.max(log_post_grid)
    post_norm = np.exp(log_post_norm)

    log_surrogate_norm = log_surrogate_grid - np.max(log_surrogate_grid)
    surrogate_norm = np.exp(log_surrogate_norm)

    # Define colormaps
    cmap_post = sns.color_palette('rocket', as_cmap=True)
    cmap_surr = sns.color_palette('viridis', as_cmap=True)

    os.makedirs(FIG_DIR, exist_ok=True)

    # Extract training data from surrogate
    X_train = None
    if plot_training_data and hasattr(surrogate, '_x_data'):
        X_train = surrogate._x_data.numpy()
        print(f"  Training data: {len(X_train)} points")

    # ========================================================================
    # Plot 1: Surrogate vs Posterior overlay
    # ========================================================================
    xi_rho_min, xi_rho_max = xi_rho_grid.min(), xi_rho_grid.max()
    xi_l_min, xi_l_max = xi_l_grid.min(), xi_l_grid.max()

    fig1, ax1 = get_2d_despined_figure(plot_limits=([xi_rho_min, xi_rho_max],
                                                    [xi_l_min, xi_l_max]),
                                       figsize=(4., 4.),
                                       axes_label=(r'\xi_\rho', r'\xi_l'),
                                       equal_axes=False,
                                       keep_ticks=True)

    # Plot surrogate contours (background)
    ax1.contour(xi_rho_grid,
                xi_l_grid,
                surrogate_norm,
                levels=20,
                zorder=1,
                alpha=0.4,
                cmap=cmap_surr,
                linestyles='--',
                linewidths=1.5)

    # Plot posterior contours (foreground)
    ax1.contour(xi_rho_grid,
                xi_l_grid,
                post_norm,
                levels=20,
                zorder=2,
                alpha=0.6,
                cmap=cmap_post)

    # Plot training data locations
    if X_train is not None:
        ax1.scatter(X_train[:, 0],
                    X_train[:, 1],
                    c='green',
                    s=15,
                    alpha=0.6,
                    marker='x',
                    linewidths=1.5,
                    label='Training data',
                    zorder=5)

    # True value
    ax1.scatter(*xi_aff_true,
                c='k',
                s=120,
                marker='*',
                edgecolors='none',
                linewidths=1.5,
                label='True',
                zorder=10)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='C3', linewidth=2, label='Posterior'),
        Line2D([0], [0],
               color='C0',
               linewidth=2,
               linestyle='--',
               label='Surrogate'),
    ]
    if X_train is not None:
        legend_elements.append(
            Line2D([0], [0],
                   marker='x',
                   color='w',
                   markeredgecolor='green',
                   markersize=8,
                   label=f'Training ({len(X_train)})',
                   linestyle='None',
                   markeredgewidth=1.5))
    legend_elements.append(
        Line2D([0], [0],
               marker='*',
               color='w',
               markerfacecolor='k',
               markersize=10,
               label='True',
               linestyle='None'))
    ax1.legend(handles=legend_elements, loc='lower left', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'surrogate_vs_posterior.pdf')
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig1)

    # ========================================================================
    # Plot 2: Difference plot (Posterior - Surrogate)
    # ========================================================================
    difference = log_post_grid - log_surrogate_grid

    fig2, ax2 = get_2d_despined_figure(plot_limits=([xi_rho_min, xi_rho_max],
                                                    [xi_l_min, xi_l_max]),
                                       figsize=(5., 4.),
                                       axes_label=(r'\xi_\rho', r'\xi_l'),
                                       equal_axes=False,
                                       keep_ticks=True)

    # Plot difference as contourf
    vmax = np.percentile(np.abs(difference[np.isfinite(difference)]), 95)
    levels = np.linspace(-vmax, vmax, 21)

    contourf = ax2.contourf(xi_rho_grid,
                            xi_l_grid,
                            difference,
                            levels=levels,
                            cmap='RdBu_r',
                            zorder=1)

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax2)
    cbar.set_label(r'$\log \pi - \log \tilde{\pi}$')

    # Plot training data locations
    if X_train is not None:
        ax2.scatter(X_train[:, 0],
                    X_train[:, 1],
                    c='green',
                    s=15,
                    alpha=0.7,
                    marker='x',
                    linewidths=1.5,
                    edgecolors='white',
                    zorder=5)

    # True value
    ax2.scatter(*xi_aff_true,
                c='k',
                s=120,
                marker='*',
                edgecolors='white',
                linewidths=1.5,
                label='True',
                zorder=10)

    ax2.legend(loc='upper right', frameon=False)

    # Save
    fig_path = os.path.join(FIG_DIR, 'surrogate_difference.pdf')
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig2)

    # ========================================================================
    # Plot 3: Side-by-side comparison
    # ========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8., 3.5))
    # fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8., 3.5))

    # Posterior
    ax3a.contour(xi_rho_grid,
                 xi_l_grid,
                 post_norm,
                 levels=20,
                 cmap=cmap_post,
                 alpha=0.6)
    if X_train is not None:
        ax3a.scatter(X_train[:, 0],
                     X_train[:, 1],
                     c='green',
                     s=10,
                     alpha=0.6,
                     marker='x',
                     linewidths=1.0,
                     zorder=5)
    ax3a.scatter(*xi_aff_true,
                 c='k',
                 s=100,
                 marker='*',
                 edgecolors='white',
                 linewidths=1.5,
                 zorder=10)
    ax3a.set_xlabel(r'$\xi_\rho$')
    ax3a.set_ylabel(r'$\xi_l$')
    ax3a.set_title('Posterior')
    sns.despine(ax=ax3a)

    # Surrogate
    ax3b.contour(xi_rho_grid,
                 xi_l_grid,
                 surrogate_norm,
                 levels=20,
                 cmap=cmap_surr,
                 alpha=0.6)
    if X_train is not None:
        ax3b.scatter(X_train[:, 0],
                     X_train[:, 1],
                     c='green',
                     s=10,
                     alpha=0.6,
                     marker='x',
                     linewidths=1.0,
                     zorder=5)
    ax3b.scatter(*xi_aff_true,
                 c='k',
                 s=100,
                 marker='*',
                 edgecolors='white',
                 linewidths=1.5,
                 zorder=10)
    ax3b.set_xlabel(r'$\xi_\rho$')
    ax3b.set_ylabel(r'$\xi_l$')
    ax3b.set_title('Surrogate')
    sns.despine(ax=ax3b)

    # Save
    fig_path = os.path.join(FIG_DIR, 'surrogate_comparison.pdf')
    fig3.savefig(fig_path, bbox_inches='tight')
    print(f"  ✓ Saved figure: {fig_path}")
    plt.close(fig3)


# ============================================================================
# Main
# ============================================================================


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze results from RWM and ZigZag samplers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - use cached data
  python analyze_results.py
  
  # Regenerate observations
  python analyze_results.py --force-obs
  
  # Recompute posterior grid
  python analyze_results.py --force-grid
  
  # Force recomputation of everything
  python analyze_results.py --force
        """)
    parser.add_argument('--force',
                        action='store_true',
                        help='Force recomputation of all cached data')
    parser.add_argument('--force-obs',
                        action='store_true',
                        help='Force regeneration of observations')
    parser.add_argument('--force-grid',
                        action='store_true',
                        help='Force recomputation of posterior grid')
    parser.add_argument(
        '--rwm-dir',
        type=str,
        default='./rwm',
        help='Directory containing RWM results (default: ./rwm)')
    parser.add_argument(
        '--zzs-dir',
        type=str,
        default='./zzs',
        help='Directory containing ZigZag results (default: ./zzs)')
    parser.add_argument('--cache-dir',
                        type=str,
                        default='./cache',
                        help='Directory for cached data (default: ./cache)')
    parser.add_argument(
        '--fig-dir',
        type=str,
        default='./figures',
        help='Directory for output figures (default: ./figures)')
    args = parser.parse_args()

    # Update global directories
    global RWM_DIR, ZZS_DIR, CACHE_DIR, FIG_DIR
    RWM_DIR = args.rwm_dir
    ZZS_DIR = args.zzs_dir
    CACHE_DIR = args.cache_dir
    FIG_DIR = args.fig_dir

    # Set up RNG
    rng = np.random.default_rng(seed=42)

    print("=" * 70)
    print("ANALYZING SAMPLER RESULTS")
    print("=" * 70)
    print(f"RWM directory: {RWM_DIR}")
    print(f"ZigZag directory: {ZZS_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Figure directory: {FIG_DIR}")
    print("=" * 70)

    # Load configuration from one of the sampler directories
    # (both should have the same config)
    config_file = os.path.join(RWM_DIR, "config.yaml")
    if not os.path.exists(config_file):
        config_file = os.path.join(ZZS_DIR, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"No config.yaml found in {RWM_DIR} or {ZZS_DIR}. "
            "Please run the samplers first using run_inference.py")

    print(f"Loading configuration from: {config_file}")
    config = load_yaml_config(config_file)
    config = config['problem']

    # Update observation file path in config to use shared location
    obs_file = os.path.join(CACHE_DIR, "observations.dat")

    # Generate or load observations
    generate_observations(config,
                          rng,
                          obs_file,
                          force=args.force or args.force_obs)

    # Update config to point to the observation file
    if 'distribution' in config and 'likelihood' in config['distribution']:
        config['distribution']['likelihood']['likelihood'][
            'observation_file'] = obs_file

    # Load target distribution
    print("=" * 70)
    print("Loading target distribution...")
    print("=" * 70)
    target = get_target(config, rng=rng)

    # Suppress verbose output from external libraries after target is created
    suppress_external_loggers()

    print(f"  Target dimension: {target.dim}")
    print("  ✓ Target loaded successfully")

    # Evaluate posterior grid
    cache_file = os.path.join(CACHE_DIR, "posterior_grid.npz")
    xi_rho_grid, xi_l_grid, log_post_grid = evaluate_posterior_grid(
        target, cache_file, rng, force=args.force or args.force_grid)

    # True values in transformed space
    TRUE_RHO, TRUE_L = get_true_parameters_from_config(config)
    xi_rho_true = np.log(TRUE_RHO / (1.0 - TRUE_RHO))
    xi_l_true = np.log(TRUE_L)

    # Load samples from sampler directories
    print("=" * 70)
    print("Loading sampler results...")
    print("=" * 70)

    rwm_samples = load_rwm_samples(RWM_DIR)
    zzs_pos, zzs_vel, zzs_times, zzs_samples = load_zzs_samples(ZZS_DIR)

    # Package ZigZag path data
    zzs_path = None
    if zzs_pos is not None and zzs_vel is not None and zzs_times is not None:
        zzs_path = (zzs_pos, zzs_vel, zzs_times)

    # Check if we have any samples
    if rwm_samples is None and zzs_samples is None:
        print("\n" + "!" * 70)
        print("WARNING: No samples found in either RWM or ZigZag directories!")
        print("Please run the samplers first using run_inference.py")
        print("!" * 70)
        return

    # Plot results - Combined (both samplers)
    print("\n" + "=" * 70)
    print("CREATING COMBINED PLOTS (RWM + ZZS)")
    print("=" * 70)
    plot_posterior_2d(xi_rho_grid,
                      xi_l_grid,
                      log_post_grid,
                      rwm_samples,
                      zzs_samples,
                      xi_rho_true,
                      xi_l_true,
                      target=target,
                      zzs_path=zzs_path,
                      plot_prior=PLOT_PRIOR,
                      plot_zzs_path=PLOT_ZZS_PATH,
                      include_rwm=True,
                      include_zzs=True,
                      suffix='')

    # Plot marginals if we have any samples
    if rwm_samples is not None or zzs_samples is not None:
        plot_marginals(rwm_samples,
                       zzs_samples,
                       xi_rho_true,
                       xi_l_true,
                       target=target,
                       include_rwm=True,
                       include_zzs=True,
                       suffix='')

    # Plot exponential field with credible intervals if we have any samples
    if rwm_samples is not None or zzs_samples is not None:
        plot_exponential_field(rwm_samples,
                               zzs_samples,
                               target=target,
                               config=config,
                               include_rwm=True,
                               include_zzs=True,
                               suffix='')

    # Plot results - ZZS only
    if zzs_samples is not None:
        print("\n" + "=" * 70)
        print("CREATING ZZS-ONLY PLOTS")
        print("=" * 70)
        plot_posterior_2d(xi_rho_grid,
                          xi_l_grid,
                          log_post_grid,
                          rwm_samples,
                          zzs_samples,
                          xi_rho_true,
                          xi_l_true,
                          target=target,
                          zzs_path=zzs_path,
                          plot_prior=PLOT_PRIOR,
                          plot_zzs_path=PLOT_ZZS_PATH,
                          include_rwm=False,
                          include_zzs=True,
                          suffix='_zzs')

        plot_marginals(rwm_samples,
                       zzs_samples,
                       xi_rho_true,
                       xi_l_true,
                       target=target,
                       include_rwm=False,
                       include_zzs=True,
                       suffix='_zzs')

        plot_exponential_field(rwm_samples,
                               zzs_samples,
                               target=target,
                               config=config,
                               include_rwm=False,
                               include_zzs=True,
                               suffix='_zzs')

    # Plot results - RWM only
    if rwm_samples is not None:
        print("\n" + "=" * 70)
        print("CREATING RWM-ONLY PLOTS")
        print("=" * 70)
        plot_posterior_2d(xi_rho_grid,
                          xi_l_grid,
                          log_post_grid,
                          rwm_samples,
                          zzs_samples,
                          xi_rho_true,
                          xi_l_true,
                          target=target,
                          zzs_path=zzs_path,
                          plot_prior=PLOT_PRIOR,
                          plot_zzs_path=PLOT_ZZS_PATH,
                          include_rwm=True,
                          include_zzs=False,
                          suffix='_rwm')

        plot_marginals(rwm_samples,
                       zzs_samples,
                       xi_rho_true,
                       xi_l_true,
                       target=target,
                       include_rwm=True,
                       include_zzs=False,
                       suffix='_rwm')

        plot_exponential_field(rwm_samples,
                               zzs_samples,
                               target=target,
                               config=config,
                               include_rwm=True,
                               include_zzs=False,
                               suffix='_rwm')

    # Print summary statistics if we have any samples
    if rwm_samples is not None or zzs_samples is not None:
        print_summary_statistics(rwm_samples, zzs_samples, target=target)

    # Load and plot ZigZag surrogate vs posterior
    print("=" * 70)
    print("Loading ZigZag surrogate model...")
    print("=" * 70)
    zzs_surrogate = load_zzs_surrogate(ZZS_DIR, target, rng)

    if zzs_surrogate is not None:
        plot_surrogate_vs_posterior(xi_rho_grid,
                                    xi_l_grid,
                                    log_post_grid,
                                    zzs_surrogate,
                                    xi_rho_true,
                                    xi_l_true,
                                    target,
                                    plot_training_data=PLOT_TRAINING_DATA)
    else:
        print("  Skipping surrogate plots (no surrogate model available)")

    print("\n" + "=" * 70)
    print("✓ All done! Figures saved to:", FIG_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
