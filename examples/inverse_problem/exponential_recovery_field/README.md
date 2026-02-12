# Inverse Problem with Exponential Recovery Field

This example demonstrates Bayesian inverse problem with:
- **ExponentialRecoveryField** with parameters [ρ, l]
- **Transformed likelihood** using LOGIT for ρ (bounded to [0,1]) and EXPONENTIAL for l (positive)
- **2D posterior visualization**
- **Sampling** with Random Walk Metropolis (RWM) and ZigZag Sampler (ZZS)

## Field Model

The exponential recovery field is:
```
F(x) = F_∞ × (1 - (1 - ρ) × exp(-x/l))
```

where:
- ρ ∈ (0, 1): recovery ratio
- l > 0: length scale
- F_∞: asymptotic field value (fixed)

## Transformations

To ensure parameters stay in their valid domains during sampling:
- **ρ**: LOGIT transformation (maps unbounded ξ_ρ ∈ ℝ to ρ ∈ [0, 1])
- **l**: EXPONENTIAL transformation (maps unbounded ξ_l ∈ ℝ to l ∈ (0, ∞))

## Usage

### First run (generates all data):
```bash
python solve_inverse.py
```

### Subsequent runs (uses cached data):
```bash
python solve_inverse.py
```

### Force recomputation:
```bash
# Recompute everything
python solve_inverse.py --force

# Recompute only specific parts
python solve_inverse.py --force-obs      # Regenerate observations
python solve_inverse.py --force-grid     # Recompute posterior grid
python solve_inverse.py --force-samples  # Resample
```

## Output Structure

```
exponential_recovery_field/
├── solve_inverse.py          # Main script
├── README.md                 # This file
├── data/                     # Cached computation results
│   ├── observations.dat      # Synthetic observations
│   └── posterior_grid.npz    # 2D posterior evaluation
├── samples/                  # Sampling results
│   ├── rwm_samples.npy       # Random Walk Metropolis samples
│   └── zzs_samples.npz       # ZigZag samples (path + equidistant)
└── figures/                  # Output plots
    ├── posterior_2d.png      # 2D posterior with samples
    └── marginals.png         # Marginal distributions
```

## Configuration

Edit the script to change:
- **True parameters**: `TRUE_RHO`, `TRUE_L`
- **Noise level**: `SIGMA_OBS`
- **Grid resolution**: `N_GRID_RHO`, `N_GRID_L`
- **Sample counts**: `N_RWM_SAMPLES`, `T_MAX_ZZS`

## Key Features

1. **Efficient caching**: All expensive computations are cached to disk
2. **Quick re-plotting**: Re-run without recomputation to adjust plots
3. **Transformed sampling**: Samples in unbounded space, automatically satisfies constraints
4. **2D visualization**: Both transformed and original parameter spaces
5. **Marginal distributions**: Compare RWM and ZZS posteriors

## Example Output

The script produces:
1. **posterior_2d.png**: 2D contour plots showing:
   - Posterior in transformed space (ξ_ρ, ξ_l)
   - Posterior in original space (ρ, l)
   - RWM and ZZS samples overlaid
   - True parameter values marked

2. **marginals.png**: Marginal distributions for both parameters:
   - Histograms from RWM and ZZS
   - Kernel density estimates (KDE)
   - True values marked

## Notes

- Start with coarse resolution (15×15 grid, 500 samples) for testing
- Increase resolution once everything runs correctly
- The model evaluation is fast enough for dense grids (~225 evaluations in seconds)
- ZigZag requires a surrogate (Laplace approximation used here)

