# Analysis of RWM and ZigZag Sampler Results

This directory contains scripts for running samplers and analyzing their results.

## Directory Structure

```
from_config/
├── rwm/                    # RWM sampler results
│   ├── config.yaml         # Configuration used
│   ├── samples.dat         # Sampled chain
│   ├── accepted.dat        # Accepted proposals
│   └── ...
├── zzs/                    # ZigZag sampler results
│   ├── config.yaml         # Configuration used
│   ├── positions.dat       # Trajectory positions
│   ├── velocities.dat      # Trajectory velocities
│   ├── times.dat           # Event times
│   └── ...
├── cache/                  # Cached data (created by analyze_results.py)
│   ├── observations.dat    # Synthetic observations
│   └── posterior_grid.npz  # Cached posterior grid evaluation
├── figures/                # Generated plots (created by analyze_results.py)
│   ├── posterior_2d_*.pdf  # 2D posterior plots (combined, ZZS-only, RWM-only)
│   ├── marginal_*.pdf      # Marginal plots (combined, ZZS-only, RWM-only)
│   ├── exponential_field*.pdf  # Field plots (combined, ZZS-only, RWM-only)
│   └── surrogate_*.pdf     # Surrogate comparison plots
├── analyze_results.py      # Main analysis script
└── README.md               # This file
```

## Workflow

### Step 1: Run the Samplers

Run each sampler separately using the `run_inference.py` script from the repository root:

```bash
# Run RWM sampler
cd rwm/
python ../../../run_inference.py config.yaml

# Run ZigZag sampler
cd ../zzs/
python ../../../run_inference.py config.yaml
```

Each sampler will create its output files in its respective directory.

### Step 2: Analyze Results

Once both samplers have been run, use the analysis script to collect results and generate plots:

```bash
# Basic usage - use all cached data
python analyze_results.py

# Force regeneration of observations
python analyze_results.py --force-obs

# Force recomputation of posterior grid
python analyze_results.py --force-grid

# Force recomputation of everything
python analyze_results.py --force
```

### Command-Line Options

The `analyze_results.py` script supports the following options:

- `--force`: Force recomputation of all cached data
- `--force-obs`: Force regeneration of observations only
- `--force-grid`: Force recomputation of posterior grid only
- `--rwm-dir DIR`: Specify RWM results directory (default: `./rwm`)
- `--zzs-dir DIR`: Specify ZigZag results directory (default: `./zzs`)
- `--cache-dir DIR`: Specify cache directory (default: `./cache`)
- `--fig-dir DIR`: Specify output figures directory (default: `./figures`)

## What the Analysis Script Does

The `analyze_results.py` script:

1. **Loads configuration** from the sampler directories
2. **Generates/loads observations** (optionally regenerates with `--force-obs`)
3. **Evaluates posterior grid** on a 2D grid in the transformed space
   - Result is cached in `cache/posterior_grid.npz`
   - Recompute with `--force-grid`
4. **Loads samples** from both RWM and ZigZag samplers
5. **Loads ZigZag surrogate** (Gaussian Process) if available
6. **Creates comprehensive plots** in three sets:
   - **Combined plots**: Both RWM and ZZS samples overlaid
   - **ZZS-only plots**: Only ZigZag samples (if available)
   - **RWM-only plots**: Only RWM samples (if available)
   
   Each set includes:
   - 2D posterior in transformed (affine) space
   - 2D posterior in original (rho, l) parameter space
   - Marginal distributions (histograms)
   - Marginal distributions (KDE)
   - Exponential recovery field with 95% credible intervals
   
   Additionally:
   - Surrogate vs posterior comparison (if ZigZag surrogate available)
7. **Prints summary statistics** for both samplers

## Generated Plots

All plots are saved as PDF files in the `figures/` directory. The script generates three sets of plots:

### Combined Plots (Both Samplers)
- `posterior_2d_transformed.pdf`: Posterior in transformed space with both RWM and ZZS samples
- `posterior_2d_original.pdf`: Posterior in original parameter space with both samplers
- `marginal_rho_hist.pdf`: Marginal histogram for ρ parameter (both samplers)
- `marginal_rho_kde.pdf`: Marginal KDE for ρ parameter (both samplers)
- `marginal_l_hist.pdf`: Marginal histogram for l parameter (both samplers)
- `marginal_l_kde.pdf`: Marginal KDE for l parameter (both samplers)
- `exponential_field.pdf`: Exponential recovery field with credible intervals (both samplers)

### ZigZag-Only Plots
- `posterior_2d_transformed_zzs.pdf`: Posterior with only ZZS samples
- `posterior_2d_original_zzs.pdf`: Posterior in original space with only ZZS samples
- `marginal_rho_hist_zzs.pdf`: Marginal histogram for ρ (ZZS only)
- `marginal_rho_kde_zzs.pdf`: Marginal KDE for ρ (ZZS only)
- `marginal_l_hist_zzs.pdf`: Marginal histogram for l (ZZS only)
- `marginal_l_kde_zzs.pdf`: Marginal KDE for l (ZZS only)
- `exponential_field_zzs.pdf`: Exponential recovery field (ZZS only)

### RWM-Only Plots
- `posterior_2d_transformed_rwm.pdf`: Posterior with only RWM samples
- `posterior_2d_original_rwm.pdf`: Posterior in original space with only RWM samples
- `marginal_rho_hist_rwm.pdf`: Marginal histogram for ρ (RWM only)
- `marginal_rho_kde_rwm.pdf`: Marginal KDE for ρ (RWM only)
- `marginal_l_hist_rwm.pdf`: Marginal histogram for l (RWM only)
- `marginal_l_kde_rwm.pdf`: Marginal KDE for l (RWM only)
- `exponential_field_rwm.pdf`: Exponential recovery field (RWM only)

### Surrogate Plots (ZigZag GP model)
- `surrogate_vs_posterior.pdf`: Overlay of ZigZag surrogate and true posterior with training data
- `surrogate_difference.pdf`: Difference between posterior and surrogate
- `surrogate_comparison.pdf`: Side-by-side comparison of posterior and surrogate

**Note**: The ZZS-only and RWM-only plots are only generated if the respective sampler results are available.

## Caching

The script caches computationally expensive operations:

1. **Observations** (`cache/observations.dat`):
   - Generated once using the true parameters
   - Regenerate with `--force-obs`

2. **Posterior grid** (`cache/posterior_grid.npz`):
   - Evaluates log-posterior on a 50×50 grid
   - Recompute with `--force-grid`

Note: The script does **not** cache sampler results. These are always loaded fresh from the `rwm/` and `zzs/` directories.

## Customization

You can customize the plotting behavior by editing the configuration at the top of `analyze_results.py`:

```python
# Plotting Options
PLOT_PRIOR = False           # Show/hide prior contours
PLOT_ZZS_PATH = True         # Plot full ZigZag path vs. equidistant samples
PLOT_TRAINING_DATA = True    # Show GP training data locations (green crosses)

# Grid resolution
N_GRID_RHO = 50
N_GRID_L = 50

# Sample thinning for plots
SKIP_RWM = 2
SKIP_ZZS = 1
```

## Dependencies

The script requires the same dependencies as the main `pdmp` package:
- numpy
- matplotlib
- scipy
- seaborn
- pyyaml
- pdmp (the package itself)

