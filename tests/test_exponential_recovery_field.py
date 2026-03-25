"""Tests for JaxExponentialRecoveryField."""
import numpy as np
import jax.numpy as jnp
import pytest

from pdmp.random_field import JaxExponentialRecoveryField, get_jax_field


def test_exponential_recovery_field_boundary_conditions():
    """Test that F(0) = F_inf * rho and F(∞) ≈ F_inf."""
    print("\n" + "=" * 70)
    print("Testing JaxExponentialRecoveryField Boundary Conditions")
    print("=" * 70)

    # Configuration
    F_inf = 100.0
    rho = 0.3
    l = 2.0

    config = {
        'name': 'JaxExponentialRecoveryField',
        'f_infinity': F_inf,
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': [rho, l],
            'cov': [[0.01, 0.0], [0.0, 0.1]]
        }
    }

    field = get_jax_field(config)

    # Test with specific coefficients
    coeffs = jnp.array([rho, l])

    # Test at x = 0
    x_zero = jnp.array([0.0])
    F_zero = field.evaluate(coeffs, x_zero)
    F_zero_expected = F_inf * rho

    print(f"\nTest at x = 0:")
    print(f"  F_inf = {F_inf}")
    print(f"  rho = {rho}")
    print(f"  F(0) = {float(F_zero[0]):.6f}")
    print(f"  Expected F(0) = F_inf * rho = {F_zero_expected:.6f}")
    print(f"  Difference: {abs(float(F_zero[0]) - F_zero_expected):.2e}")

    assert np.allclose(F_zero[0], F_zero_expected, rtol=1e-5, atol=1e-8), \
        f"F(0) should equal F_inf * rho, got {F_zero[0]} vs {F_zero_expected}"

    # Test at x → ∞ (use large x)
    x_large = jnp.array([100.0])  # 100.0 >> l = 2.0, so exp(-100/2) ≈ 0
    F_inf_approx = field.evaluate(coeffs, x_large)

    print(f"\nTest at x → ∞ (x = {float(x_large[0])}):")
    print(f"  F(∞) ≈ {float(F_inf_approx[0]):.6f}")
    print(f"  Expected F(∞) = F_inf = {F_inf:.6f}")
    print(f"  Difference: {abs(float(F_inf_approx[0]) - F_inf):.2e}")
    print(
        f"  exp(-x/l) = exp(-{float(x_large[0])}/{l}) = {np.exp(-float(x_large[0])/l):.2e}"
    )

    assert np.allclose(F_inf_approx[0], F_inf, rtol=1e-6), \
        f"F(∞) should approach F_inf, got {F_inf_approx[0]} vs {F_inf}"

    print("\n✓ Boundary conditions test passed!")


def test_exponential_recovery_field_profile():
    """Test the full exponential recovery profile."""
    print("\n" + "=" * 70)
    print("Testing JaxExponentialRecoveryField Profile")
    print("=" * 70)

    F_inf = 100.0
    rho = 0.5
    l = 1.0

    config = {
        'name': 'JaxExponentialRecoveryField',
        'f_infinity': F_inf,
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': [rho, l],
            'cov': [[0.01, 0.0], [0.0, 0.1]]
        }
    }

    field = get_jax_field(config)
    coeffs = jnp.array([rho, l])

    # Test at multiple points
    x_values = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    F_values = field.evaluate(coeffs, x_values)

    print(f"\nField parameters:")
    print(f"  F_inf = {F_inf}")
    print(f"  rho = {rho}")
    print(f"  l = {l}")
    print(f"  F(0) = F_inf * rho = {F_inf * rho}")
    print(f"  F(∞) = F_inf = {F_inf}")

    print(f"\nProfile values:")
    print(
        f"  {'x':>8s} | {'F(x)':>10s} | {'(F-F0)/(Finf-F0)':>20s} | {'1-exp(-x/l)':>15s}"
    )
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*20}-+-{'-'*15}")

    F_0 = F_inf * rho
    for x, F in zip(x_values, F_values):
        normalized = (F - F_0) / (F_inf - F_0)
        expected_norm = 1.0 - np.exp(-x / l)
        print(
            f"  {float(x):8.2f} | {float(F):10.4f} | {float(normalized):20.6f} | {float(expected_norm):15.6f}"
        )

    # Verify that the field is monotonically increasing (for rho < 1)
    for i in range(len(F_values) - 1):
        assert F_values[i] < F_values[i+1], \
            f"Field should be monotonically increasing for rho < 1"

    # Verify analytical formula at a specific point
    x_test = 2.0
    F_analytical = F_inf * (1.0 - (1.0 - rho) * np.exp(-x_test / l))
    F_computed = field.evaluate(coeffs, jnp.array([x_test]))[0]

    print(f"\nAnalytical verification at x = {x_test}:")
    print(f"  Analytical: F = {F_analytical:.6f}")
    print(f"  Computed:   F = {float(F_computed):.6f}")
    print(f"  Difference: {abs(float(F_computed) - F_analytical):.2e}")

    assert np.allclose(F_computed, F_analytical, rtol=1e-4, atol=1e-6), \
        f"Field evaluation doesn't match analytical formula"

    print("\n✓ Profile test passed!")


def test_exponential_recovery_field_multidim():
    """Test that the field works with multi-dimensional coordinates."""
    print("\n" + "=" * 70)
    print("Testing JaxExponentialRecoveryField with Multi-dimensional Input")
    print("=" * 70)

    F_inf = 50.0
    rho = 0.2
    l = 1.5

    config = {
        'name': 'JaxExponentialRecoveryField',
        'f_infinity': F_inf,
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': [rho, l],
            'cov': [[0.01, 0.0], [0.0, 0.1]]
        }
    }

    field = get_jax_field(config)
    coeffs = jnp.array([rho, l])

    # Test with 2D coordinates (should use only first dimension)
    x_2d = jnp.array([[0.0, 5.0], [1.0, 10.0], [2.0, 15.0], [3.0, 20.0]])

    F_2d = field.evaluate(coeffs, x_2d)

    # Test with 1D coordinates
    x_1d = x_2d[:, 0]
    F_1d = field.evaluate(coeffs, x_1d)

    print(f"\n2D input (using first column only):")
    print(f"  x_2d[:, 0] = {x_2d[:, 0]}")
    print(f"  F(x_2d) = {F_2d}")

    print(f"\n1D input:")
    print(f"  x_1d = {x_1d}")
    print(f"  F(x_1d) = {F_1d}")

    print(f"\nComparison:")
    for i in range(len(F_1d)):
        print(
            f"  x={float(x_1d[i]):.1f}: F_2d={float(F_2d[i]):.6f}, F_1d={float(F_1d[i]):.6f}, diff={abs(float(F_2d[i] - F_1d[i])):.2e}"
        )

    assert np.allclose(F_2d, F_1d, rtol=1e-10), \
        "2D and 1D inputs should give same results (using first dimension)"

    print("\n✓ Multi-dimensional input test passed!")


def test_exponential_recovery_field_gradient():
    """Test that gradients can be computed through the field."""
    print("\n" + "=" * 70)
    print("Testing JaxExponentialRecoveryField Gradient Computation")
    print("=" * 70)

    import jax

    F_inf = 100.0
    rho = 0.4
    l = 2.0

    config = {
        'name': 'JaxExponentialRecoveryField',
        'f_infinity': F_inf,
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': [rho, l],
            'cov': [[0.01, 0.0], [0.0, 0.1]]
        }
    }

    field = get_jax_field(config)

    # Define a function that computes sum of field values
    def field_sum(coeffs):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        return jnp.sum(field.evaluate(coeffs, x))

    coeffs = jnp.array([rho, l])

    # Compute gradient
    grad_fn = jax.grad(field_sum)
    grads = grad_fn(coeffs)

    print(f"\nField parameters:")
    print(f"  coeffs = [rho={rho}, l={l}]")

    print(f"\nGradient of sum(F(x)) w.r.t. coefficients:")
    print(f"  dF/d(rho) = {float(grads[0]):.6f}")
    print(f"  dF/d(l) = {float(grads[1]):.6f}")

    # Verify gradients using finite differences
    h = 1e-4

    # Gradient w.r.t. rho
    coeffs_rho_plus = jnp.array([rho + h, l])
    coeffs_rho_minus = jnp.array([rho - h, l])
    grad_rho_fd = (field_sum(coeffs_rho_plus) -
                   field_sum(coeffs_rho_minus)) / (2 * h)

    # Gradient w.r.t. l
    coeffs_l_plus = jnp.array([rho, l + h])
    coeffs_l_minus = jnp.array([rho, l - h])
    grad_l_fd = (field_sum(coeffs_l_plus) - field_sum(coeffs_l_minus)) / (2 *
                                                                          h)

    print(f"\nFinite difference verification:")
    print(
        f"  dF/d(rho) FD = {float(grad_rho_fd):.6f}, diff = {abs(float(grads[0]) - float(grad_rho_fd)):.2e}"
    )
    print(
        f"  dF/d(l) FD = {float(grad_l_fd):.6f}, diff = {abs(float(grads[1]) - float(grad_l_fd)):.2e}"
    )

    # Note: Finite differences have their own numerical errors, so we use looser tolerances
    assert np.allclose(grads[0], grad_rho_fd, rtol=2e-3, atol=1e-1), \
        "Gradient w.r.t. rho doesn't match finite difference"
    assert np.allclose(grads[1], grad_l_fd, rtol=2e-2, atol=1e-1), \
        "Gradient w.r.t. l doesn't match finite difference"

    print("\n✓ Gradient test passed!")


def test_exponential_recovery_field_special_cases():
    """Test special cases: rho=0, rho=1."""
    print("\n" + "=" * 70)
    print("Testing JaxExponentialRecoveryField Special Cases")
    print("=" * 70)

    F_inf = 100.0
    l = 1.0

    # Case 1: rho = 0 (starts at 0, recovers to F_inf)
    print("\nCase 1: rho = 0")
    rho = 0.0
    config = {
        'name': 'JaxExponentialRecoveryField',
        'f_infinity': F_inf,
        'coefficient_distribution': {
            'name': 'MultivariateNormal',
            'mean': [rho, l],
            'cov': [[0.01, 0.0], [0.0, 0.1]]
        }
    }
    field = get_jax_field(config)
    coeffs = jnp.array([rho, l])

    F_0 = field.evaluate(coeffs, jnp.array([0.0]))[0]
    F_inf_approx = field.evaluate(coeffs, jnp.array([100.0]))[0]

    print(f"  F(0) = {float(F_0):.6f} (expected 0.0)")
    print(f"  F(∞) ≈ {float(F_inf_approx):.6f} (expected {F_inf:.6f})")

    assert np.allclose(F_0, 0.0, atol=1e-10), "F(0) should be 0 when rho=0"
    assert np.allclose(F_inf_approx, F_inf,
                       rtol=1e-6), "F(∞) should be F_inf when rho=0"

    # Case 2: rho = 1 (constant field at F_inf)
    print("\nCase 2: rho = 1")
    rho = 1.0
    config['coefficient_distribution']['mean'] = [rho, l]
    field = get_jax_field(config)
    coeffs = jnp.array([rho, l])

    x_values = jnp.array([0.0, 1.0, 2.0, 5.0, 100.0])
    F_values = field.evaluate(coeffs, x_values)

    print(f"  x values: {x_values}")
    print(f"  F values: {F_values}")
    print(f"  Expected: all equal to {F_inf:.6f}")

    assert np.allclose(F_values, F_inf, rtol=1e-5, atol=1e-8), \
        "When rho=1, field should be constant at F_inf everywhere"

    print("\n✓ Special cases test passed!")


if __name__ == "__main__":
    test_exponential_recovery_field_boundary_conditions()
    test_exponential_recovery_field_profile()
    test_exponential_recovery_field_multidim()
    test_exponential_recovery_field_gradient()
    test_exponential_recovery_field_special_cases()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
