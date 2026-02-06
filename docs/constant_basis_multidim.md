# Multi-Dimensional ConstantBasis Implementation

## Overview

The `ConstantBasis` class has been extended to support **arbitrary spatial dimensions** (1D, 2D, 3D, and beyond). This enables the creation of spatially uniform but uncertain random fields in multi-dimensional domains, which is essential for FEM simulations and other applications.

## Key Features

### 1. **Multi-Dimensional Support**
- **1D**: Interval [a, b]
- **2D**: Rectangle [a₁, b₁] × [a₂, b₂]
- **3D**: Box [a₁, b₁] × [a₂, b₂] × [a₃, b₃]
- **ND**: Arbitrary dimensions

### 2. **Backward Compatibility**
- Original 1D tuple interface `(a, b)` still works
- All existing code continues to function without changes

### 3. **Proper Volume-Based Normalization**
- 1D: ||φ|| = √(length)
- 2D: ||φ|| = √(area)
- 3D: ||φ|| = √(volume)
- ND: ||φ|| = √(hypervolume)

### 4. **JAX Gradient Support**
- Full autodiff support in all dimensions
- Efficient gradient computation w.r.t. coefficients
- No performance penalty for multi-dimensional cases

## Usage

### 1D Constant Field (Backward Compatible)

```python
from pdmp.project_field import ConstantBasis
from pdmp.random_field import JaxRandomFieldBase
from pdmp.distributions import MultivariateNormal

# Original tuple interface still works
basis = ConstantBasis((0.0, 1.0))

mean = np.array([2.0])
cov = np.array([[0.5**2]])
dist = MultivariateNormal(mean, cov)

field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

# Evaluate at 1D points
x = jnp.array([0.0, 0.5, 1.0])
coeffs = jnp.array([3.0])
values = field.evaluate(coeffs, x)  # [3.0, 3.0, 3.0]
```

### 2D Constant Field

```python
# 2D rectangular domain [0, 10] × [0, 5]
domain = [[0.0, 10.0], [0.0, 5.0]]
basis = ConstantBasis(domain)

print(f"Spatial dimension: {basis.spatial_dim}")  # 2
print(f"Area: {np.prod(basis.domain[:, 1] - basis.domain[:, 0])}")  # 50.0
print(f"Norm: {basis.get_norms()[0, 0]}")  # sqrt(50) ≈ 7.071

# Create field with uncertain material property
mean = np.array([200.0])  # Young's modulus in GPa
cov = np.array([[15.0**2]])
dist = MultivariateNormal(mean, cov)
field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

# Evaluate at 2D points
x = jnp.array([
    [0.0, 0.0],
    [5.0, 2.5],
    [10.0, 5.0]
])
coeffs = jnp.array([210.0])
values = field.evaluate(coeffs, x)  # [210.0, 210.0, 210.0]
```

### 3D Constant Field

```python
# 3D cubic domain [0, 1]³
domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
basis = ConstantBasis(domain)

print(f"Spatial dimension: {basis.spatial_dim}")  # 3
print(f"Volume: {np.prod(basis.domain[:, 1] - basis.domain[:, 0])}")  # 1.0

# Uncertain permeability field
mean = np.array([1e-5])  # m²
cov = np.array([[2e-6**2]])
dist = MultivariateNormal(mean, cov)
field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

# Evaluate at 3D points
x = jnp.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0]
])
coeffs = jnp.array([1.2e-5])
values = field.evaluate(coeffs, x)  # All 1.2e-5
```

## Integration with FEM

### Realistic 3D FEM Scenario

```python
# 3D structural component: 100mm × 50mm × 20mm
domain = [[0.0, 0.1], [0.0, 0.05], [0.0, 0.02]]
basis = ConstantBasis(domain)

# Uncertain material density (kg/m³)
mean = np.array([7850.0])  # Steel
cov = np.array([[50.0**2]])
dist = MultivariateNormal(mean, cov)
field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

# Simulate FEM mesh with quadrature points
n_cells = 100
n_quads_per_cell = 8  # 8 Gauss points per hex element
quad_points = generate_quadrature_points(n_cells, n_quads_per_cell, domain)

# Sample material property
sampled_density = field.coefficient_distribution.get_sample()

# Evaluate at all quadrature points (shape: n_cells * n_quads_per_cell, 3)
density_field = field.evaluate(jnp.array(sampled_density), quad_points)

# All values are constant (spatially uniform)
assert jnp.allclose(density_field, sampled_density[0])
```

## Gradient Computation

### 2D Gradient Example

```python
domain = [[0.0, 1.0], [0.0, 1.0]]
basis = ConstantBasis(domain)
field = JaxRandomFieldBase(basis=basis, coefficient_dist=dist)

def loss_fn(coeffs):
    # Evaluate at a 3×3 grid
    x = jnp.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])
    values = field.evaluate(coeffs, x)
    return jnp.sum(values**2)

coeffs = jnp.array([2.0])
grad = jax.grad(loss_fn)(coeffs)  # Works correctly!
```

## Implementation Details

### Domain Specification

The domain is specified as an array of shape `(spatial_dim, 2)`:

```python
# 1D: Can use tuple (backward compatible)
domain_1d = (0.0, 1.0)  # or [[0.0, 1.0]]

# 2D: Array with 2 rows
domain_2d = [[0.0, 1.0], [0.0, 2.0]]

# 3D: Array with 3 rows
domain_3d = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

# Each row: [lower_bound, upper_bound] for that dimension
```

### Basis Evaluation

The basis function returns ones for all input points:

```python
# 1D input: shape (n_points,) or (n_points, 1)
x_1d = np.array([0.0, 0.5, 1.0])
phi = basis(x_1d)  # shape (3, 1) if i=None, (3,) if i=0

# Multi-D input: shape (n_points, spatial_dim)
x_2d = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
phi = basis(x_2d)  # shape (3, 1) if i=None, (3,) if i=0
```

### Properties

```python
basis = ConstantBasis([[0.0, 2.0], [0.0, 3.0]])

# Access properties
basis.spatial_dim     # 2
basis.domain          # [[0.0, 2.0], [0.0, 3.0]]
basis.get_n()         # 1 (single basis function)
basis.get_norms()     # [[sqrt(6)]] (sqrt of area)
```

## Changes to Existing Code

### Updated Files

1. **`pdmp/project_field.py`**: 
   - Extended `ConstantBasis` to handle multi-dimensional domains
   - Added `spatial_dim` and `domain` properties
   - Maintained backward compatibility with tuple interface

2. **`pdmp/random_field.py`**:
   - Updated `JaxRandomFieldBase.evaluate()` to preserve multi-dimensional structure
   - Updated `JaxGaussianRandomField.evaluate()` similarly
   - Removed `.ravel()` that was flattening multi-dimensional inputs

### No Breaking Changes

- All existing 1D code continues to work
- Tuple interface `(a, b)` still supported
- API remains the same
- Tests confirm backward compatibility

## Testing

### Test Coverage

- **Unit tests**: 31 tests covering 1D, 2D, 3D cases
  - `tests/test_constant_basis.py`: 13 tests (1D, backward compatibility)
  - `tests/test_constant_basis_multidim.py`: 18 tests (multi-dimensional)
  
- **Examples**:
  - `examples/constant_basis_example.py`: 1D examples
  - `examples/constant_basis_multidim_example.py`: 2D/3D examples

All tests pass successfully (31/31).

### Running Tests

```bash
# Run all ConstantBasis tests
pytest tests/test_constant_basis*.py -v

# Run multi-dimensional tests only
pytest tests/test_constant_basis_multidim.py -v

# Run examples
python examples/constant_basis_multidim_example.py
```

## Use Cases

### 1. **Uncertain Material Properties in FEM**
- Young's modulus (2D/3D)
- Poisson's ratio (2D/3D)
- Density (2D/3D)
- Thermal conductivity (2D/3D)

### 2. **Porous Media Flow**
- Spatially uniform but uncertain permeability (2D/3D)
- Porosity fields (2D/3D)

### 3. **Thermal Analysis**
- Uniform but uncertain thermal diffusivity (2D/3D)
- Heat capacity (2D/3D)

### 4. **Parameter Inference**
- Bayesian inference of global material parameters
- MCMC sampling with gradient-based methods
- Uncertainty quantification

## Advantages Over PiecewiseConstantBasis

| Feature | ConstantBasis | PiecewiseConstantBasis |
|---------|---------------|------------------------|
| Number of coefficients | 1 | n (multiple) |
| Spatial variation | None | Piecewise constant |
| Use case | Global uncertainty | Spatially varying |
| Computational cost | Minimal | Higher |
| Gradient complexity | Simple | Complex |
| Multi-dimensional | ✓ Yes | ✗ 1D only |

## Future Extensions

The multi-dimensional framework is easily extensible:

1. **PiecewiseConstantBasis2D/3D**: Extend piecewise constant basis to 2D/3D
2. **Other basis types**: Polynomial, Fourier, etc. in multiple dimensions
3. **Anisotropic domains**: Different resolutions in different dimensions
4. **Non-rectangular domains**: Cylindrical, spherical coordinates

## Summary

The multi-dimensional `ConstantBasis` implementation provides:

✅ **Full multi-dimensional support** (1D, 2D, 3D, ND)  
✅ **Backward compatibility** with existing 1D code  
✅ **Proper volume-based normalization**  
✅ **JAX gradient support** in all dimensions  
✅ **FEM integration** for realistic simulations  
✅ **Comprehensive testing** (31 tests, all passing)  
✅ **Extensive documentation** and examples  

This makes `ConstantBasis` a powerful tool for uncertainty quantification in multi-dimensional FEM simulations!
