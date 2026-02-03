# JAX-Compatible Random Fields for FEM Models

This document describes the JAX-compatible random field implementation for use with the `JaxFemModel`.

## Overview

The `JaxRandomField` interface allows you to define spatially-varying material properties that can be used with JAX-based automatic differentiation. This is particularly useful for Bayesian inverse problems where you want to infer spatially-varying parameters.

## Quick Start

### 1. Using a Constant Field

The simplest random field is `JaxConstantField`, which represents a single parameter that is constant throughout the entire domain:

```python
from pdmp.random_field import JaxConstantField
from pdmp.forward_model import JaxFemModel

# Define a constant field for Young's modulus
field = JaxConstantField(mean=1e6, std=1e5)

# Create a FEM model with the field
model = JaxFemModel(
    d_x=1.0, d_y=1.0, d_z=2.5,
    h=0.5,
    field=field
)

# The model automatically infers n_params from the field
print(f"Number of parameters: {model.get_dim_in()}")  # Output: 1

# Evaluate the model
import numpy as np
params = np.array([1.2e6])
y = model.eval(params)
```

### 2. Using Configuration Files

You can also create models from configuration dictionaries:

```python
from pdmp.random_field import get_jax_field
from pdmp.forward_model import JaxFemModel

# Field configuration
field_config = {
    'name': 'JaxConstantField',
    'mean': 1e6,
    'std': 2e5
}
field = get_jax_field(field_config)

# Model configuration
model_config = {
    'name': 'JaxFem',
    'd_x': 1.0,
    'd_y': 1.0,
    'd_z': 2.5,
    'h': 0.5,
    'nu': 0.3
}
model = JaxFemModel.from_dict(model_config, field=field)
```

### 3. Integration with Bayesian Inference

The field integrates seamlessly with the existing `get_target` and `get_prior` infrastructure:

**Configuration file (YAML):**
```yaml
name: BayesianInverse

model:
  name: JaxFem
  d_x: 1.0
  d_y: 1.0
  d_z: 2.5
  h: 0.5
  nu: 0.3
  field:
    name: JaxConstantField
    mean: 1.0e6
    std: 2.0e5

prior:
  name: FromField  # Extract prior from field

likelihood:
  name: Gaussian
  sigma: 0.01
```

**Python code:**
```python
from pdmp.loader import get_target
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

rng = np.random.default_rng(0)
posterior = get_target(config, rng)

# Sample from prior
theta_0 = posterior.get_prior_sample()

# Evaluate posterior
log_prob = posterior.log_pdf(theta_0)
```

## How It Works

### Parameter Mapping

The random field maps coefficient vectors to spatially-varying material properties:

1. **Coefficients** → **Field Values** → **Material Properties**

For `JaxConstantField`:
- Input: Single coefficient (scalar or array with 1 element)
- Output: Constant value throughout the entire domain

For future extensions (e.g., piecewise constant, Gaussian process):
- Input: Multiple coefficients representing basis weights
- Output: Spatially-varying values evaluated at cell centers or quadrature points

### Gradient Computation

The field evaluation is fully compatible with JAX's automatic differentiation:

```python
import jax

# The gradient flows through the field evaluation
def loss_fn(params):
    y = model.eval(params)
    return jnp.sum(y**2)

grad_fn = jax.grad(loss_fn)
gradient = grad_fn(params)
```

## Available Fields

### JaxConstantField

**Description:** Single parameter constant throughout the domain

**Configuration:**
```python
{
    'name': 'JaxConstantField',
    'mean': 1.0e6,    # Prior mean
    'std': 2.0e5      # Prior standard deviation
}
```

**Properties:**
- `dim`: Always 1
- Prior distribution: Normal(mean, std²)

### Future Extensions

Additional field types can be added by implementing the `JaxRandomField` protocol:

```python
from pdmp.random_field import JaxRandomField

class MyCustomField:
    @property
    def dim(self) -> int:
        """Number of parameters/coefficients."""
        ...
    
    def evaluate(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate field at spatial locations."""
        ...
    
    def coefficient_distribution(self, rng=None):
        """Return prior distribution over coefficients."""
        ...
```

## Comparison with GaussianRandomField

| Feature | GaussianRandomField | JaxRandomField |
|---------|-------------------|----------------|
| Autodiff | NumPy-based | JAX-based ✓ |
| Compatible with | PiecewiseConstantModel | JaxFemModel |
| Spatial variation | Piecewise constant basis | Flexible (depends on implementation) |
| Prior | MultivariateNormal with kernel | Depends on field type |

## Best Practices

1. **Start simple:** Use `JaxConstantField` first, then extend to more complex fields
2. **Check dimensions:** Always verify `model.get_dim_in()` matches your expectations
3. **Use FromField prior:** Let the field define the prior distribution via `prior: {name: FromField}`
4. **Test gradients:** Verify that `eval_grad` and `eval_vjp` work correctly
5. **Memory efficiency:** Use VJP (`linearize`, `eval_vjp`) instead of full Jacobian when possible

## Troubleshooting

### Issue: "JaxConstantField expects 1 coefficient, got N"

**Solution:** Make sure you're passing a single parameter value when using `JaxConstantField`.

### Issue: Shape mismatch in field evaluation

**Solution:** Check that the spatial coordinates `x` have the expected shape `(n_points, spatial_dim)`.

### Issue: Prior dimension mismatch

**Solution:** Ensure the prior configuration matches the field dimension:
```python
field = get_jax_field({'name': 'JaxConstantField', 'mean': 1e6, 'std': 1e5})
# field.dim == 1

# Prior should extract from field:
prior_config = {'name': 'FromField'}
prior = get_prior(prior_config, field=field)
# prior.dim == 1
```

## Examples

See `test_jax_field.py` for complete working examples.
