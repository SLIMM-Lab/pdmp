# Understanding the Three Coordinate Spaces in solve_inverse_affine.py

This document explains the three coordinate spaces used in the affine transformation setup and how transformations between them work.

## The Three Coordinate Spaces

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COORDINATE SPACE DIAGRAM                        │
└─────────────────────────────────────────────────────────────────────┘

   Parameter Space          Transformed Space         Affine Space
   ───────────────          ─────────────────         ────────────
        (x)           ←→           (ξ)          ←→      (ξ_aff)
                         likelihood                   affine
                         transform                    transform

   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
   │ rho ∈ (0, 1) │       │ ξ_rho ∈ ℝ    │       │ ξ_aff_rho ∈ ℝ│
   │ l   ∈ (0, ∞) │  ←→   │ ξ_l   ∈ ℝ    │  ←→   │ ξ_aff_l   ∈ ℝ│
   └──────────────┘       └──────────────┘       └──────────────┘
    Physical params        Unconstrained          Sampling space
```

## 1. Parameter Space (x): Physical Parameters

**Variables:** `rho`, `l`

**Domain:**
- `rho ∈ (0, 1)`: Bounded parameter (recovery ratio)
- `l ∈ (0, ∞)`: Positive parameter (length scale)

**Purpose:** 
- The original physical parameter space
- Where forward model evaluations occur
- Where observations are generated

**Example:**
```python
TRUE_RHO = 0.7  # In parameter space
TRUE_L = 1.2    # In parameter space
```

## 2. Transformed Space (ξ): Likelihood-Transformed Space

**Variables:** `xi_rho`, `xi_l`

**Domain:**
- `xi_rho ∈ ℝ`: Unbounded (after SIGMOID transform)
- `xi_l ∈ ℝ`: Unbounded (after EXPONENTIAL transform)

**Transformations:**
- **Forward (x → ξ):**
  - `xi_rho = logit(rho) = log(rho / (1 - rho))`
  - `xi_l = log(l)`

- **Inverse (ξ → x):**
  - `rho = sigmoid(xi_rho) = 1 / (1 + exp(-xi_rho))`
  - `l = exp(xi_l)`

**Purpose:**
- Makes constrained parameters unconstrained
- Prior and likelihood are defined here
- Intermediate space in the transformation chain

**Jacobian:**
```python
# For density transformation
log_det_J = log(rho * (1 - rho)) + log(l)
```

**Example:**
```python
# True value in transformed space
xi_rho_true = np.log(TRUE_RHO / (1.0 - TRUE_RHO))  # ≈ 0.847
xi_l_true = np.log(TRUE_L)                          # ≈ 0.182
```

## 3. Affine Space (ξ_aff): Sampling Space

**Variables:** `xi_aff_rho`, `xi_aff_l`

**Domain:**
- `xi_aff_rho ∈ ℝ`: Unbounded
- `xi_aff_l ∈ ℝ`: Unbounded

**Transformation:**
- **Forward (ξ → ξ_aff):**
  ```python
  ξ_aff = M^(-1) @ (ξ - b)
  ```
  where:
  - `M` is a 2×2 matrix (affine scaling/rotation)
  - `b` is a 2D bias vector

- **Inverse (ξ_aff → ξ):**
  ```python
  ξ = M @ ξ_aff + b
  ```

**Purpose:**
- The space where MCMC sampling occurs
- Allows exploration of transformed posterior geometry
- Can improve sampling efficiency

**Jacobian:**
```python
log_det_J = log(det(M))  # Constant for affine transform
```

**Example Configuration:**
```python
config = {
    'transformation': 'Affine',
    'M': [[0.25097, 0.59644],
          [0.59644, 2.14637]],
    'b': [1.03476, 0.73861],
}
```

## Transformation Pipeline

### Forward Direction (Parameter → Affine)

```python
# Starting point: parameter space
rho, l = 0.7, 1.2

# Step 1: Apply likelihood transformations
xi_rho = np.log(rho / (1 - rho))  # logit
xi_l = np.log(l)                   # log
xi = np.array([xi_rho, xi_l])

# Step 2: Apply affine transformation
xi_aff = np.linalg.solve(M, xi - b)
```

### Reverse Direction (Affine → Parameter)

```python
# Starting point: affine space (e.g., MCMC sample)
xi_aff = np.array([0.5, -0.3])

# Step 1: Undo affine transformation
xi = M @ xi_aff + b

# Step 2: Undo likelihood transformations
rho = 1.0 / (1.0 + np.exp(-xi[0]))  # sigmoid
l = np.exp(xi[1])                    # exp
```

## Density Transformations

When we transform a probability density, we need to account for the Jacobian:

```python
# Posterior in affine space
log p(ξ_aff | y) = log p(ξ | y) + log |det J_affine|

# Posterior in parameter space
log p(x | y) = log p(ξ | y) + log |det J_likelihood|

# Combined transformation
log p(x | y) = log p(ξ_aff | y) - log |det J_affine| + log |det J_likelihood|
```

This is handled automatically by the helper functions:
```python
log_jacobian = compute_log_jacobian_grid(xi_aff_grid, aff_trans, like_trans)
log_post_original = log_post_affine - log_jacobian
```

## Code Implementation

### Helper Functions

The three coordinate spaces are connected by helper functions:

```python
# Grid transformation (affine → parameter)
rho_grid, l_grid = transform_grid_to_original_space(
    xi_aff_rho_grid, xi_aff_l_grid, aff_trans, like_trans
)

# Jacobian computation
log_jacobian = compute_log_jacobian_grid(
    xi_aff_rho_grid, xi_aff_l_grid, aff_trans, like_trans
)

# Sample transformation (affine → parameter)
rho_samples, l_samples = transform_samples_to_original_space(
    affine_samples, aff_trans, like_trans
)
```

### Extracting Transformations from Target

```python
# Get transformation objects from target distribution
aff_trans = target._transformation
like_trans = target._base_distribution._likelihood._transformation

# Use them for transformations
xi = aff_trans.transform(xi_aff)        # affine → transformed
x = like_trans.transform(xi)             # transformed → parameter
```

## Workflow Summary

1. **Setup:** Define problem in parameter space (physical parameters)
2. **Observation:** Generate observations using forward model in parameter space
3. **Prior/Likelihood:** Define distributions in transformed space (ξ)
4. **Affine Transform:** Apply affine transformation to reach sampling space (ξ_aff)
5. **Sampling:** Run MCMC samplers in affine space
6. **Visualization:** 
   - Plot posterior in affine space (where sampling occurs)
   - Transform and plot posterior in parameter space (for interpretation)
7. **Analysis:** Compute statistics in parameter space

## Key Takeaways

✅ **Sampling happens in affine space (ξ_aff)**
✅ **Prior and likelihood are defined in transformed space (ξ)**
✅ **Interpretation and physics happen in parameter space (x)**
✅ **All transformations are properly accounted for via Jacobians**
✅ **Helper functions handle transformations consistently**

## References

See the following functions in `solve_inverse_affine.py`:
- `transform_grid_to_original_space()`: Grid transformation
- `compute_log_jacobian_grid()`: Jacobian computation
- `transform_samples_to_original_space()`: Sample transformation
- `plot_posterior_2d()`: Visualization in both spaces

