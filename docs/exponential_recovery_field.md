# JaxExponentialRecoveryField Documentation

## Overview

The `JaxExponentialRecoveryField` implements a spatially-varying field with exponential recovery behavior along one spatial dimension. The field is constant along other dimensions.

## Mathematical Model

The field follows the equation:

```
F(x) = F_∞ · (1 - (1 - ρ) · exp(-x/l))
```

Where:
- `F_∞` (F_infinity): Known far-field value (fixed parameter)
- `ρ` (rho): Ratio of initial to far-field value, F(0)/F_∞ (inferred parameter)
- `l`: Length scale controlling recovery rate (inferred parameter)
- `x`: Spatial coordinate (first dimension)

## Boundary Conditions

The field satisfies:
- **At x = 0**: `F(0) = F_∞ · ρ`
- **As x → ∞**: `F(∞) → F_∞`

## Parameters to Infer

The field has 2 parameters that can be inferred:
1. **ρ (rho)**: Recovery ratio, typically in range [0, 1]
   - ρ = 0: Field starts at 0
   - ρ = 1: Field is constant at F_∞ everywhere
   - 0 < ρ < 1: Field recovers from F_∞·ρ to F_∞

2. **l (lengthscale)**: Recovery length scale, l > 0
   - Larger l: Slower recovery
   - Smaller l: Faster recovery
   - At x = l: Field has recovered to ~63% of (F_∞ - F_0)

## Usage

### Basic Setup

```python
from pdmp.random_field import get_jax_field

# Configuration
config = {
    'name': 'JaxExponentialRecoveryField',
    'f_infinity': 100.0,  # Known far-field value
    'coefficient_distribution': {
        'name': 'MultivariateNormal',
        'mean': [0.5, 2.0],       # Prior mean for [rho, l]
        'cov': [[0.01, 0.0],      # Prior covariance
                [0.0, 0.5]]
    }
}

# Create field
field = get_jax_field(config)
```

### Evaluation

```python
import jax.numpy as jnp

# Define coefficients [rho, l]
coeffs = jnp.array([0.3, 1.5])

# Evaluate at spatial locations
x = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
F_values = field.evaluate(coeffs, x)

# For multi-dimensional coordinates (uses first dimension)
x_3d = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0]
])
F_values_3d = field.evaluate(coeffs, x_3d)  # Only x[:, 0] matters
```

### Gradient Computation

The field is fully differentiable with respect to both parameters:

```python
import jax

# Define objective function
def objective(coeffs):
    x = jnp.array([0.0, 1.0, 2.0])
    return jnp.sum(field.evaluate(coeffs, x))

# Compute gradients
grad_fn = jax.grad(objective)
grads = grad_fn(coeffs)  # [dF/d(rho), dF/d(l)]
```

### Use with JaxFemModel

```python
from pdmp.forward_model import JaxFemModel

# Create FEM model with exponential recovery field for Young's modulus
model = JaxFemModel(
    d_x=1.0, d_y=1.0, d_z=2.5,
    h=0.5,
    n_params=2,  # rho and l
    d_obs=1,
    field=field
)

# Evaluate forward model
observations = model(coeffs)

# Compute gradients w.r.t. field parameters
grad_fn = jax.grad(lambda c: jnp.sum(model(c)))
grads = grad_fn(coeffs)
```

## Physical Interpretation

This field type is useful for modeling:

1. **Material property recovery**: Young's modulus recovering from a damaged region
2. **Healing processes**: Properties recovering after treatment
3. **Boundary layer effects**: Property transition from boundary to bulk
4. **Manufacturing defects**: Property variation due to processing

## Special Cases

### Case 1: No Initial Damage (ρ = 1)
```python
coeffs = jnp.array([1.0, 1.0])  # rho=1, any l
# Result: F(x) = F_∞ everywhere (constant field)
```

### Case 2: Maximum Initial Damage (ρ = 0)
```python
coeffs = jnp.array([0.0, 1.0])  # rho=0, l=1
# Result: F(0) = 0, F(∞) = F_∞ (full recovery)
```

### Case 3: Fast Recovery (small l)
```python
coeffs = jnp.array([0.5, 0.1])  # rho=0.5, l=0.1
# Result: Rapid recovery over short distance
```

### Case 4: Slow Recovery (large l)
```python
coeffs = jnp.array([0.5, 10.0])  # rho=0.5, l=10.0
# Result: Gradual recovery over long distance
```

## Verification

All functionality is tested in `tests/test_exponential_recovery_field.py`:
- Boundary conditions: F(0) = F_∞·ρ, F(∞) = F_∞
- Profile shape matches analytical formula
- Multi-dimensional coordinate handling
- Gradient correctness (automatic differentiation)
- Special cases (ρ=0, ρ=1)

Run tests with:
```bash
pytest tests/test_exponential_recovery_field.py -v
```

## Comparison with Other Fields

| Field Type | Parameters | Spatial Structure | Use Case |
|------------|------------|-------------------|----------|
| `JaxConstantField` | 1 | Uniform | Simple material property |
| `JaxGaussianRandomField` | N | Gaussian process | Complex spatial correlation |
| `JaxRandomFieldBase` | N | Basis expansion | Flexible distributions |
| `JaxExponentialRecoveryField` | 2 (ρ, l) | Exponential decay | Recovery/boundary layers |

## Tips for Inference

1. **Prior selection for ρ**:
   - Use narrow prior around expected recovery ratio
   - Constrain to [0, 1] using appropriate distribution
   - Consider Beta or truncated Normal distribution

2. **Prior selection for l**:
   - Scale should match domain size
   - Use log-normal or Gamma to ensure l > 0
   - For domain [0, L], typical range: l ∈ [0.1L, L]

3. **Identifiability**:
   - Need observations at multiple x locations
   - Observations near x=0 constrain ρ
   - Observations at intermediate x constrain l
   - Need at least 3-4 observation points for robust inference

