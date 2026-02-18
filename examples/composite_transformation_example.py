"""Example demonstrating CompositeTransformation usage.

This example shows how to use CompositeTransformation to apply different
transformations to different subsets of variables.
"""
import numpy as np
from pdmp.distributions import (
    CompositeTransformation,
    SigmoidTransformation,
    ExponentialTransformation,
    AffineTransformation,
    MultivariateNormal,
    JointDistribution,
    TransformedDistribution,
    COMPOSITE,
    LOG,
)


def example_1_basic_composite():
    """Example 1: Basic composite transformation."""
    print("=" * 70)
    print("Example 1: Basic Composite Transformation")
    print("=" * 70)

    # Create composite: sigmoid on [0,1], exponential on [2,3,4]
    trans = CompositeTransformation(
        transformations=[
            SigmoidTransformation(a=0, b=1),
            ExponentialTransformation()
        ],
        indices=[
            np.array([0, 1]),
            np.array([2, 3, 4])
        ]
    )

    # Transform
    xi = np.array([0.0, 1.0, 0.0, 0.5, -0.5])
    x = trans.transform(xi)

    print(f"Input (unbounded):     ξ = {xi}")
    print(f"Output (transformed):  x = {x}")
    print(f"  - x[0:2] ∈ [0,1]:    {x[0:2]}")
    print(f"  - x[2:5] > 0:        {x[2:5]}")

    # Test invertibility
    xi_recovered = trans.inverse_transform(x)
    print(f"Inverse correct: {np.allclose(xi, xi_recovered)}")

    # Log-determinant
    log_det = trans.log_det_jacobian(xi)
    print(f"Log |det(J)|: {log_det:.4f}")
    print()


def example_2_with_distribution():
    """Example 2: Composite transformation with a distribution.

    Important: In TransformedDistribution:
    - Base distribution lives in x space (constrained/original)
    - Transformation maps ξ → x (transform method)
    - For exponential: x = exp(ξ), so x > 0 and ξ is unbounded
    - For sigmoid: x = a + (b-a)*sigmoid(ξ), so x ∈ [a,b] and ξ is unbounded

    Therefore, the base distribution must have support matching the constrained x space.
    """
    print("=" * 70)
    print("Example 2: Composite Transformation with Distribution")
    print("=" * 70)

    # Create joint distribution with appropriate support in x space:
    # - Components [0,1,2]: positive (for exponential transform x = exp(ξ))
    # - Components [3,4]: in [0,1] (for sigmoid transform x = sigmoid(ξ))

    # Use log-normal-like distributions (positive support) for first 3 components
    # and beta-like distributions (support on [0,1]) for last 2 components
    from pdmp.distributions import JointDistribution

    # For simplicity, we'll use MultivariateNormal with positive means
    # In practice, you'd use distributions with proper support
    dist1 = MultivariateNormal(mean=np.array([2.0, 3.0, 1.5]),
                               cov=np.diag([0.5, 0.5, 0.5]))
    dist2 = MultivariateNormal(mean=np.array([0.5, 0.5]),
                               cov=np.diag([0.1, 0.1]))

    base_dist = JointDistribution([dist1, dist2])

    # Apply composite transformation
    # Indices will be automatically inferred: [0,1,2] and [3,4]
    params = {
        'transformation': COMPOSITE,
        'transformations': [
            {'type': 'Exponential'},  # x[0:3] = exp(ξ[0:3])
            {'type': 'Sigmoid', 'a': 0, 'b': 1}  # x[3:5] = sigmoid(ξ[3:5])
        ]
    }

    trans_dist = TransformedDistribution(base_dist, params)

    # Sample and evaluate
    xi = trans_dist.get_sample()
    log_p = trans_dist.log_density(xi)
    grad = trans_dist.grad_log_density(xi)

    # Transform to x space to verify constraints
    x = trans_dist._transformation.transform(xi)

    print(f"Sample in unconstrained space ξ: {xi}")
    print(f"Transformed to constrained space x: {x}")
    print(f"  - x[0:3] > 0 (exponential): {x[0:3]}")
    print(f"  - x[3:5] ∈ [0,1] (sigmoid): {x[3:5]}")
    print(f"\nLog density: {log_p:.4f}")
    print(f"Gradient: {grad}")
    print(f"All gradients finite: {np.all(np.isfinite(grad))}")

    # Verify constraints are satisfied
    print(f"\nConstraints satisfied:")
    print(f"  All x[0:3] > 0: {np.all(x[0:3] > 0)}")
    print(f"  All x[3:5] ∈ [0,1]: {np.all((x[3:5] >= 0) & (x[3:5] <= 1))}")
    print()


def example_3_joint_distribution():
    """Example 3: LogTransformation with Gaussian base (automatic indices).

    This demonstrates using LogTransformation with a Gaussian base distribution,
    which solves the problem of infinite support.

    Convention:
    - Base distribution is defined in x space (Gaussian, unbounded)
    - Transformation defines the mapping ξ → x (via transform method)
    - LogTransformation: transform(ξ) = log(ξ), so ξ must be positive

    Why LogTransformation works with Gaussian base:
    - Base samples x from Gaussian (can be negative)
    - get_sample() returns inverse_transform(x) = exp(x), which gives ξ > 0
    - log_density(ξ) computes x = transform(ξ) = log(ξ), then evaluates p(x)
    - For MCMC: we work in ξ space (positive), gradients tell us how p(ξ) changes

    Similarly, LogitTransformation maps Gaussian x to bounded ξ ∈ [a,b].
    """
    print("=" * 70)
    print("Example 3: JointDistribution with Automatic Indices (Log/Logit)")
    print("=" * 70)

    # Create joint of three distributions (base distributions in x space)
    # These are Gaussians with infinite support
    dist1 = MultivariateNormal(mean=np.zeros(2), cov=np.eye(2))  # 2D
    dist2 = MultivariateNormal(mean=np.zeros(3), cov=np.eye(3))  # 3D
    dist3 = MultivariateNormal(mean=np.zeros(1), cov=np.eye(1))  # 1D

    joint = JointDistribution([dist1, dist2, dist3])
    print(f"Joint distribution dimension: {joint.dim}")
    print(f"Child dimensions: {[d.dim for d in joint.distributions]}")

    # Use Log and Logit transformations
    # Log: inverse_transform(x) = exp(x), so Gaussian x → positive ξ
    # Logit: inverse_transform(x) = sigmoid(x), so Gaussian x → [a,b] ξ
    params = {
        'transformation': COMPOSITE,
        'transformations': [
            {'type': 'Log'},                      # x (Gaussian) → ξ = exp(x) > 0
            {'type': 'Logit', 'a': 0, 'b': 10},  # x (Gaussian) → ξ = sigmoid(x) ∈ [0,10]
            {'type': 'Log'}                       # x (Gaussian) → ξ = exp(x) > 0
        ]
    }

    trans_dist = TransformedDistribution(joint, params)

    # Sample - this returns ξ values (transformed from Gaussian x via inverse_transform)
    xi = trans_dist.get_sample()
    print(f"Sample ξ (transformed space):")
    print(f"  - ξ[0:2] from Log (should be > 0): {xi[0:2]}")
    print(f"  - ξ[2:5] from Logit (should be ∈ [0,10]): {xi[2:5]}")
    print(f"  - ξ[5] from Log (should be > 0): {xi[5]}")

    # Verify constraints on ξ
    print(f"\nConstraints on ξ:")
    print(f"  All ξ[0:2] > 0: {np.all(xi[0:2] > 0)}")
    print(f"  All ξ[2:5] ∈ [0,10]: {np.all((xi[2:5] >= 0) & (xi[2:5] <= 10))}")
    print(f"  ξ[5] > 0: {xi[5] > 0}")

    # We can also transform back to x space (Gaussian) if needed
    x = trans_dist._transformation.transform(xi)
    print(f"\nOriginal x space (Gaussian, unbounded): {x}")
    print()


def example_4_three_way():
    """Example 4: Three different transformations."""
    print("=" * 70)
    print("Example 4: Three-Way Composite")
    print("=" * 70)

    # Affine transformation matrix
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, -1.0])

    trans = CompositeTransformation(
        transformations=[
            SigmoidTransformation(0, 10),
            ExponentialTransformation(),
            AffineTransformation(M, b)
        ],
        indices=[
            np.array([0, 1]),    # Sigmoid
            np.array([2]),       # Exponential
            np.array([3, 4])     # Affine
        ]
    )

    xi = np.array([0.5, -0.5, 1.0, 0.0, 0.0])
    x = trans.transform(xi)

    print(f"Input: {xi}")
    print(f"Output: {x}")
    print(f"  - Sigmoid [0,1] → [0,10]: {xi[0:2]} → {x[0:2]}")
    print(f"  - Exponential [2]: {xi[2]} → {x[2]}")
    print(f"  - Affine [3,4]: {xi[3:5]} → {x[3:5]}")
    print()


def example_5_physics_model():
    """Example 5: Physics-based model with mixed constraints.

    Important: Base distribution must have support in the constrained x space.
    - Temperature, diffusion, reaction rate need x > 0
    - Concentration needs x ∈ [0,1]
    """
    print("=" * 70)
    print("Example 5: Physics Model with Mixed Constraints")
    print("=" * 70)

    # Model parameters with physically meaningful priors in x space:
    # - Temperature (positive, around 300K): centered at 300
    # - Concentration (in [0,1]): centered at 0.5
    # - Diffusion coefficient (positive, small): centered at 0.01
    # - Reaction rate (positive, small): centered at 0.1

    # Create base distributions with appropriate support
    temp_dist = MultivariateNormal(mean=np.array([300.0]), cov=np.array([[100.0]]))
    conc_dist = MultivariateNormal(mean=np.array([0.5]), cov=np.array([[0.05]]))
    diff_dist = MultivariateNormal(mean=np.array([0.01]), cov=np.array([[0.001]]))
    rate_dist = MultivariateNormal(mean=np.array([0.1]), cov=np.array([[0.01]]))

    base = JointDistribution([temp_dist, conc_dist, diff_dist, rate_dist])

    params = {
        'transformation': COMPOSITE,
        'transformations': [
            {'type': 'Exponential'},              # Temperature: x = exp(ξ)
            {'type': 'Sigmoid', 'a': 0, 'b': 1},  # Concentration: x = sigmoid(ξ)
            {'type': 'Exponential'},              # Diffusion: x = exp(ξ)
            {'type': 'Exponential'}               # Reaction rate: x = exp(ξ)
        ]
        # Indices automatically inferred from JointDistribution: [0], [1], [2], [3]
    }

    trans_dist = TransformedDistribution(base, params)

    # Sample parameters in unconstrained space ξ
    xi = trans_dist.get_sample()
    # Transform to constrained space x
    x = trans_dist._transformation.transform(xi)

    print("Sampled parameters:")
    print(f"  ξ (unconstrained): {xi}")
    print(f"  x (constrained):   {x}")
    print(f"\nPhysical parameters (x space):")
    print(f"  Temperature:       {x[0]:.4f} K (must be > 0)")
    print(f"  Concentration:     {x[1]:.4f} (must be in [0,1])")
    print(f"  Diffusion coeff:   {x[2]:.6f} (must be > 0)")
    print(f"  Reaction rate:     {x[3]:.4f} (must be > 0)")

    # Verify constraints
    print(f"\nConstraints satisfied:")
    print(f"  Temperature > 0:        {x[0] > 0}")
    print(f"  0 ≤ Concentration ≤ 1:  {0 <= x[1] <= 1}")
    print(f"  Diffusion > 0:          {x[2] > 0}")
    print(f"  Reaction rate > 0:      {x[3] > 0}")
    print()


def example_6_log_transformation_demo():
    """Example 6: Demonstrate LogTransformation with Gaussian base.

    This shows how LogTransformation solves the problem of using Gaussian
    base distributions (infinite support) with transformations that require
    positive values.
    """
    print("=" * 70)
    print("Example 6: LogTransformation with Gaussian Base")
    print("=" * 70)

    # Problem: Gaussian can produce negative values
    # If we use ExponentialTransformation with Gaussian base:
    # - Base samples x can be negative
    # - inverse_transform(x) = log(x) fails for negative x!

    # Solution: Use LogTransformation instead
    # - Base samples x can be any value (Gaussian)
    # - inverse_transform(x) = exp(x) always works!
    # - Resulting ξ values are always positive

    # Create Gaussian in x space (unbounded)
    base = MultivariateNormal(mean=np.zeros(3), cov=np.eye(3))

    # Apply Log transformation
    params = {'transformation': LOG}
    trans_dist = TransformedDistribution(base, params)

    # Sample multiple times to show it always works
    print("Sampling from Log-transformed Gaussian:")
    for i in range(5):
        xi = trans_dist.get_sample()
        print(f"  Sample {i+1} (ξ, always positive): {xi}")
        # Transform back to get original Gaussian sample
        x = trans_dist._transformation.transform(xi)
        print(f"  Original (x, can be negative): {x}")
        # Verify round-trip
        xi_recovered = trans_dist._transformation.inverse_transform(x)
        print(f"  Round-trip matches: {np.allclose(xi, xi_recovered)}")
        print()

    print("Key insight:")
    print("  - LogTransformation: transform(x) = log(x), inverse_transform(ξ) = exp(ξ)")
    print("  - With Gaussian base in x: samples x, returns ξ = exp(x) > 0")
    print("  - This avoids taking log of negative values!")
    print()


if __name__ == "__main__":
    example_1_basic_composite()
    example_2_with_distribution()
    example_3_joint_distribution()
    example_4_three_way()
    example_5_physics_model()
    example_6_log_transformation_demo()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

