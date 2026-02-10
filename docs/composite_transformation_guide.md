# CompositeTransformation User Guide

## Overview

The `CompositeTransformation` class allows you to apply different transformations to different subsets of variables. This is particularly useful when working with joint distributions where each component may require a different transformation (e.g., exponential for positive parameters, sigmoid for bounded parameters, etc.).

## Basic Usage

### Creating a Composite Transformation

```python
from pdmp.distributions import (
    CompositeTransformation,
    SigmoidTransformation,
    ExponentialTransformation,
)
import numpy as np

# Apply sigmoid to dimensions [0, 1] and exponential to dimensions [2, 3, 4]
trans = CompositeTransformation(
    transformations=[
        SigmoidTransformation(a=0, b=1),  # Maps to [0, 1]
        ExponentialTransformation()        # Maps to (0, ∞)
    ],
    indices=[
        np.array([0, 1]),      # Sigmoid applies to these dims
        np.array([2, 3, 4])    # Exponential applies to these dims
    ]
)

# Transform from unbounded to bounded/positive space
xi = np.array([0.0, 1.0, 0.0, 0.5, -0.5])
x = trans.transform(xi)
print(x)  # x[0:2] in [0,1], x[2:5] > 0
```

### Using with Slices

You can use slice notation for contiguous index ranges:

```python
trans = CompositeTransformation(
    transformations=[SigmoidTransformation(0, 1), ExponentialTransformation()],
    indices=[slice(0, 2), slice(2, 5)]  # Equivalent to the above
)
```

## Integration with Distributions

### With Generic Distribution

```python
from pdmp.distributions import Gaussian, TransformedDistribution, COMPOSITE

# Create a 5D Gaussian base distribution
base_dist = Gaussian(mean=np.zeros(5), cov=np.eye(5))

# Apply composite transformation
params = {
    'transformation': COMPOSITE,
    'transformations': ['Exponential', 'Sigmoid'],
    'indices': [np.array([0, 1, 2]), np.array([3, 4])]
}

trans_dist = TransformedDistribution(base_dist, params)

# Sample from the transformed distribution
xi_sample = trans_dist.get_sample()
log_p = trans_dist.log_density(xi_sample)
grad = trans_dist.grad_log_density(xi_sample)
```

### Automatic Indices with JointDistribution

When using a `JointDistribution`, indices are automatically created to match the child distributions:

```python
from pdmp.distributions import JointDistribution

# Create a joint distribution
dist1 = Gaussian(np.zeros(2), np.eye(2))  # 2D
dist2 = Gaussian(np.zeros(3), np.eye(3))  # 3D
dist3 = Gaussian(np.zeros(1), np.eye(1))  # 1D
joint = JointDistribution([dist1, dist2, dist3])

# Indices will automatically be: [0,1], [2,3,4], [5]
params = {
    'transformation': COMPOSITE,
    'transformations': ['Exponential', 'Sigmoid', 'Exponential']
    # No 'indices' needed - automatically created!
}

trans_dist = TransformedDistribution(joint, params)
```

## Advanced Usage

### Multiple Transformation Specifications

You can specify transformations in three ways:

1. **String names** (for simple transformations):
   ```python
   'transformations': ['Exponential', 'Sigmoid']
   ```

2. **Dictionary specifications** (for transformations with parameters):
   ```python
   'transformations': [
       {'type': 'Sigmoid', 'a': -1, 'b': 2},
       {'type': 'Exponential'}
   ]
   ```

3. **Transformation objects** (for full control):
   ```python
   'transformations': [
       SigmoidTransformation(a=-1, b=2),
       ExponentialTransformation(),
       AffineTransformation(M=some_matrix, b=some_vector)
   ]
   ```

### Three-Way Composite Example

```python
# Different transformation for three subsets
M = np.array([[2.0, 0.5], [0.5, 1.0]])
b = np.array([1.0, -1.0])

trans = CompositeTransformation(
    transformations=[
        SigmoidTransformation(0, 10),         # Bounds to [0, 10]
        ExponentialTransformation(),          # Positive reals
        AffineTransformation(M, b)            # Affine map
    ],
    indices=[
        np.array([0, 1]),     # Sigmoid on first 2 dims
        np.array([2]),        # Exponential on dim 2
        np.array([3, 4])      # Affine on last 2 dims
    ]
)
```

## Key Features

### Block-Diagonal Structure

The Jacobian of a composite transformation is block-diagonal, which makes computations efficient:

```python
xi = np.array([...])
J = trans.jacobian(xi)  # Block-diagonal matrix
log_det = trans.log_det_jacobian(xi)  # Efficient computation
```

### Gradient Support

All gradient computations respect the block structure:

```python
grad = trans.grad_log_det_jacobian(xi)
H = trans.hessian_log_det_jacobian(xi)  # Block-diagonal Hessian
```

### Invertibility

The composite transformation is invertible (assuming all component transformations are):

```python
x = trans.transform(xi)
xi_recovered = trans.inverse_transform(x)
assert np.allclose(xi, xi_recovered)
```

## Common Use Cases

### 1. Mixed Parameter Constraints

Different parameters often have different constraints:

```python
# Variance parameters: must be positive (exponential)
# Probability parameters: must be in [0,1] (sigmoid)
# Location parameters: unconstrained (identity or affine)

trans = CompositeTransformation(
    [ExponentialTransformation(), SigmoidTransformation(0, 1)],
    [np.array([0, 1]), np.array([2, 3])]  # variances, then probabilities
)
```

### 2. Physics-Based Constraints

Physical parameters often have domain restrictions:

```python
# Temperature: positive (exponential)
# Concentration: bounded [0, 1] (sigmoid)
# Spatial coordinates: unbounded (identity)
```

### 3. Hierarchical Models

When sampling hierarchical models with joint distributions:

```python
# Different transformation per level of hierarchy
hyperparams = Gaussian(...)  # Level 1: may need exponential
params = Gaussian(...)        # Level 2: may need sigmoid
joint = JointDistribution([hyperparams, params])

params = {
    'transformation': COMPOSITE,
    'transformations': ['Exponential', 'Sigmoid']
}
```

## Validation

The class performs several validations:

- **Disjoint indices**: Ensures no dimension appears in multiple transformations
- **Complete partition**: Ensures all dimensions are covered
- **Dimension consistency**: Checks that indices match the actual dimension of inputs

```python
# This will raise ValueError (overlapping indices):
CompositeTransformation(
    [trans1, trans2],
    [np.array([0, 1, 2]), np.array([2, 3])]  # 2 appears twice!
)

# This will raise ValueError on first transform call (incomplete):
trans = CompositeTransformation(
    [trans1, trans2],
    [np.array([0, 1]), np.array([3, 4])]  # Missing dimension 2!
)
trans.transform(np.array([...]))  # ValueError here
```

## Performance Considerations

- Block-diagonal structure means O(n) complexity for many operations instead of O(n²) or O(n³)
- Each transformation is independent, allowing for potential parallelization
- Log-determinant computation is a sum of block log-determinants

## Comparison to Chaining

**CompositeTransformation** (this feature):
- Applies **different** transformations to **different subsets**
- Block-diagonal Jacobian structure
- Use case: `x[0:2] = sigmoid(ξ[0:2]), x[2:5] = exp(ξ[2:5])`

**Chained Transformations** (alternative approach):
- Applies transformations **sequentially** to **all variables**
- Full Jacobian (generally not diagonal)
- Use case: `x = affine(sigmoid(ξ))`

Both can be combined if needed by nesting CompositeTransformations or by chaining a composite transformation with other transformations.

