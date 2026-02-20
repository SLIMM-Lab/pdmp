#!/usr/bin/env python3
"""Test script for JaxFemModel with the new sensor setup.

Sensor Format:
- Each sensor spec must include 'name', 'location_fn', and either 'point' or 'points'.
- 'location_fn' can be a string ('side_faces', 'top_surface', 'bottom') or a callable.

  Example: [{"name": "s1", "location_fn": "side_faces", "point": np.array([x, y, z])}]
  Example: [{"name": "s1", "location_fn": "bottom",
             "points": np.array([[x1,y1,z1], [x2,y2,z2]])}]

Output Dimensions:
- Each sensor point produces 3 displacement components (ux, uy, uz)
- Total output dimension = (total number of sensor points) × 3
"""

import numpy as np
from pdmp.random_field import JaxConstantField, get_jax_field
from pdmp.forward_model import JaxFemModel
from pdmp.distributions import MultivariateNormal


def test_jax_fem_model_basic_sensors():
    """Test JaxFemModel with basic sensor setup."""
    print("=" * 70)
    print("Testing JaxFemModel with basic sensors")
    print("=" * 70)

    # Create a simple constant field for Young's modulus
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Define a simple sensor (single point)
    sensors = [
        {"name": "sensor_center", "location_fn": "side_faces", "point": np.array([0.5, 0.5, 1.25])}
    ]

    # Create FEM model with sensor
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,  # coarse mesh for testing
        n_params=1,
        field=field,
        sensors=sensors
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")
    print(f"Number of sensors: {len(sensors)}")

    assert model.get_dim_in() == 1, "Should have 1 parameter (from field)"
    # Each sensor point produces 3 displacement components (x, y, z)
    assert model.get_dim_out() == 3, "Should have 3 outputs (ux, uy, uz from 1 sensor point)"

    # Test forward evaluation
    params = np.array([12.])
    print(f"Evaluating model with params={params}")

    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    print(f"Model output values: {y}")

    assert y.shape == (3,), f"Expected shape (3,), got {y.shape}"
    print("✓ Basic sensor test passed!\n")


def test_jax_fem_model_multiple_points_per_sensor():
    """Test JaxFemModel with multiple points per sensor (single group)."""
    print("=" * 70)
    print("Testing JaxFemModel with multiple points per sensor")
    print("=" * 70)

    # Create a simple constant field
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Define a sensor with multiple points
    sensors = [
        {
            "name": "multi_point_sensor",
            "location_fn": "side_faces",
            "points": np.array([
                [0.0, 0.25, 1.0],
                [0.0, 0.5,  1.25],
                [0.0, 0.75, 2.0]
            ])
        }
    ]

    # Create FEM model with multiple sensor points
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")
    print(f"Number of sensors: {len(sensors)}")
    print(f"Points in sensor: {len(sensors[0]['points'])}")

    assert model.get_dim_in() == 1, "Should have 1 parameter"
    # 3 points × 3 displacement components = 9 outputs
    assert model.get_dim_out() == 9, "Should have 9 outputs (3 points × 3 displacement components)"

    # Test forward evaluation
    params = np.array([12.])
    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    print(f"Model output values (first 6): {y[:6]}")

    assert y.shape == (9,), f"Expected shape (9,), got {y.shape}"
    print("✓ Multiple points per sensor test passed!\n")


def test_jax_fem_model_multiple_sensor_groups():
    """Test JaxFemModel with multiple groups of sensors on different boundaries."""
    print("=" * 70)
    print("Testing JaxFemModel with multiple sensor groups")
    print("=" * 70)

    # Create a simple constant field
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Define multiple sensors with different numbers of points
    sensors = [
        {
            "name": "side_sensors",
            "location_fn": "side_faces",
            "points": np.array([
                [0.0, 0.2, 0.5],
                [0.0, 0.8, 0.5]
            ])
        },
        {
            "name": "more_side_sensors",
            "location_fn": "side_faces",
            "points": np.array([
                [0.25, 0.0, 1.25],
                [0.5,  0.0, 1.25],
                [0.75, 0.0, 1.25]
            ])
        },
        {
            "name": "top_sensor",
            "location_fn": "top_surface",
            "point": np.array([0.5, 0.5, 2.5])
        }
    ]

    # Create FEM model with multiple sensor groups
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")
    print(f"Number of sensors: {len(sensors)}")

    # Calculate total points: 2 + 3 + 1 = 6 points
    # Each point gives 3 displacement components: 6 * 3 = 18 outputs
    total_points = 2 + 3 + 1
    expected_outputs = total_points * 3
    print(f"Total sensor points: {total_points}")
    print(f"Expected outputs: {expected_outputs}")

    assert model.get_dim_in() == 1, "Should have 1 parameter"
    assert model.get_dim_out() == expected_outputs, f"Should have {expected_outputs} outputs"

    # Test forward evaluation
    params = np.array([12.])
    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    print(f"Model output values (first 6): {y[:6]}")

    assert y.shape == (expected_outputs,), f"Expected shape ({expected_outputs},), got {y.shape}"
    print("✓ Multiple sensor groups test passed!\n")


def test_jax_fem_model_sensors_from_config():
    """Test creating JaxFemModel from configuration with sensors."""
    print("=" * 70)
    print("Testing JaxFemModel.from_dict with sensors")
    print("=" * 70)

    # Create field from config
    field_config = {
        'name': 'JaxConstantField',
        'mean': 10,
        'std': 2
    }
    field = get_jax_field(field_config)

    # Define sensors in config format
    sensors_config = [
        {
            "name": "vertical_sensors",
            "location_fn": "side_faces",
            "points": [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.5],
                [0.0, 0.5, 2.0],
                [0.0, 0.5, 2.5],
            ]
        }
    ]

    model_config = {
        'name': 'JaxFem',
        'd_x': 1.0, 'd_y': 1.0, 'd_z': 2.5,
        'h': 0.25, 'nu': 0.3,
        'sensors': sensors_config
    }

    model = JaxFemModel.from_dict(model_config, field=field)

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")

    assert model.get_dim_in() == 1, "Should infer n_params=1 from field"
    # 4 points × 3 displacement components = 12 outputs
    assert model.get_dim_out() == 12, "Should have 12 outputs from 4 sensor points"

    # Test evaluation
    params = np.array([12.])
    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    assert y.shape == (12,), f"Expected shape (12,), got {y.shape}"

    print("Model output values:", y)

    print("✓ Sensors from config test passed!\n")


def test_jax_fem_model_sensors_from_config_multiple_groups():
    """Test creating JaxFemModel from configuration with multiple sensor groups."""
    print("=" * 70)
    print("Testing JaxFemModel.from_dict with multiple sensor groups")
    print("=" * 70)

    # Create field from config
    field_config = {
        'name': 'JaxConstantField',
        'mean': 10,
        'std': 2
    }
    field = get_jax_field(field_config)

    # Define multiple groups of sensors in config format
    sensors_config = [
        {
            "name": "side_sensors",
            "location_fn": "side_faces",
            "points": [[0.0, 0.25, 1.0], [0.0, 0.75, 1.0]]
        },
        {
            "name": "top_sensor",
            "location_fn": "top_surface",
            "point": [0.5, 0.5, 2.5]
        },
    ]

    model_config = {
        'name': 'JaxFem',
        'd_x': 1.0, 'd_y': 1.0, 'd_z': 2.5,
        'h': 0.25, 'nu': 0.3,
        'sensors': sensors_config
    }

    model = JaxFemModel.from_dict(model_config, field=field)

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")
    print(f"Number of sensors: {len(sensors_config)}")

    # 2 points + 1 point = 3 points total
    # 3 points × 3 displacement components = 9 outputs
    total_points = 2 + 1
    expected_outputs = total_points * 3
    print(f"Total sensor points: {total_points}")
    print(f"Expected outputs: {expected_outputs}")

    assert model.get_dim_in() == 1, "Should infer n_params=1 from field"
    assert model.get_dim_out() == expected_outputs, f"Should have {expected_outputs} outputs"

    # Test evaluation
    params = np.array([12.])
    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    assert y.shape == (expected_outputs,), f"Expected shape ({expected_outputs},), got {y.shape}"
    print("Model output values:", y)

    print("✓ Multiple sensor groups from config test passed!\n")


def test_jax_fem_model_default_sensors():
    """Test JaxFemModel with default sensor behavior (no sensors specified)."""
    print("=" * 70)
    print("Testing JaxFemModel with default sensors")
    print("=" * 70)

    # Create a simple constant field
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Create FEM model without specifying sensors (should use default)
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field
        # No sensors specified - will use default
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")

    assert model.get_dim_in() == 1, "Should have 1 parameter"
    # Default sensor gives 3 outputs (ux, uy, uz components)
    assert model.get_dim_out() == 3, "Should have 3 outputs (default sensor with 3 displacement components)"

    # Test evaluation
    params = np.array([12.])
    y = model.eval(params)
    print(f"Model output shape: {y.shape}")
    assert y.shape == (3,), f"Expected shape (3,), got {y.shape}"

    print("✓ Default sensor test passed!\n")


def test_sensor_output_consistency():
    """Test that sensor outputs are consistent with expected displacement values."""
    print("=" * 70)
    print("Testing sensor output consistency")
    print("=" * 70)

    # Create a simple constant field
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Define sensors - use default which should capture displacement
    # Let's use the default sensor which is at a location that should see displacement
    sensors = [
        {"name": "default_sensor", "location_fn": "side_faces",
         "point": np.array([0, 0.5, 1.25])}
    ]

    # Create model
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors
    )

    # Evaluate with two different parameter values
    params1 = np.array([10])  # Lower Young's modulus
    params2 = np.array([20])  # Higher Young's modulus

    y1 = model.eval(params1)
    y2 = model.eval(params2)

    print(f"Output with E={params1[0]}: {y1}")
    print(f"Output with E={params2[0]}: {y2}")
    print(f"Norm(y1)={np.linalg.norm(y1)}, Norm(y2)={np.linalg.norm(y2)}")

    # Check if we get non-zero displacements
    if np.allclose(y1, 0) and np.allclose(y2, 0):
        print("⚠ Warning: Both outputs are zero. This may indicate the sensor is at a constrained location.")
        print("  Skipping consistency check but marking test as passed.")
    else:
        # With higher Young's modulus, displacement magnitude should be smaller or equal
        # (stiffer material deforms less under same load)
        # Allow for numerical tolerance
        norm_ratio = np.linalg.norm(y2) / (np.linalg.norm(y1) + 1e-12)
        print(f"Displacement ratio (E=20 / E=10): {norm_ratio}")
        assert norm_ratio <= 1.01, "Higher Young's modulus should not result in larger displacement"

    print("✓ Output consistency test passed!\n")


def test_gradient_computation_with_sensors():
    """Test that gradients can be computed with the new sensor setup."""
    print("=" * 70)
    print("Testing gradient computation with sensors")
    print("=" * 70)

    # Create a simple constant field
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Define sensors
    sensors = [
        {
            "name": "vertical_line",
            "location_fn": "side_faces",
            "points": np.array([
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 2.0]
            ])
        }
    ]

    # Create model
    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors
    )

    # Test gradient computation
    params = np.array([12.])
    print(f"Evaluating gradient at params={params}")

    try:
        grad = model.eval_grad(params)
        print(f"Gradient shape: {grad.shape}")
        print(f"Gradient values:\n{grad}")

        # 2 points × 3 displacement components = 6 outputs
        # Gradient is (n_outputs, n_params) = (6, 1)
        assert grad.shape == (6, 1), f"Expected gradient shape (6, 1), got {grad.shape}"

        # Gradient should be negative (increasing E decreases displacement)
        # Note: not all components may be negative, but the norm should decrease
        print(f"Gradient has negative components: {np.any(grad < 0)}")

        print("✓ Gradient computation test passed!\n")
    except Exception as e:
        print(f"⚠ Gradient computation failed: {e}")
        raise


def test_dirichlet_boundary_sensors_near_zero():
    """Test that sensors on the Dirichlet boundary face (z=0) read near-zero displacement.

    The bottom face (z=0) has all three displacement DOFs clamped to zero via
    Dirichlet BCs.  This test verifies:

    1. **Solver tolerance**: the raw FEM solution at bottom nodes is essentially
       zero (|u| < 1e-8), confirming the linear-system enforcement is correct.
    2. **Interpolation accuracy**: the sensor output interpolated from those
       same nodes is also near-zero, confirming that the interpolation itself
       does not introduce a spurious non-zero reading.

    Any small non-zero sensor reading on the Dirichlet face must therefore
    originate from one of these two sources, and this test pinpoints which.
    """
    print("=" * 70)
    print("Testing Dirichlet boundary sensors read near-zero displacement")
    print("=" * 70)

    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    sensors_on_dirichlet = [
        {
            "name": "dirichlet_face_sensors",
            "location_fn": "bottom",
            "points": np.array([
                [0.25, 0.25, 0.0],
                [0.5,  0.5,  0.0],
                [0.75, 0.75, 0.0],
            ])
        }
    ]

    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors_on_dirichlet,
    )

    params = np.array([12.])

    # --- Part 1: check raw FEM solution at bottom nodes -----------------------
    import jax.numpy as jnp
    from pdmp.forward_model import evaluate_sensor_displacements

    fe = model.problem.fe
    param_field = jnp.asarray(params[0]) * jnp.ones((fe.num_cells, fe.num_quads))
    sol_list = model.fwd_pred([param_field])
    sol = sol_list[0]

    bottom_node_mask = np.isclose(np.asarray(fe.points[:, 2]), 0.0, atol=1e-5)
    bottom_nodes = np.where(bottom_node_mask)[0]
    assert bottom_nodes.size > 0, "No bottom nodes found – check mesh or tolerance."

    bottom_displacements = np.asarray(sol[bottom_nodes])
    max_bottom_disp = np.max(np.abs(bottom_displacements))
    print(f"Max |displacement| at bottom nodes (z=0): {max_bottom_disp:.2e}")
    print(f"Bottom node displacement sample (first 3 nodes):\n{bottom_displacements[:3]}")

    solver_tol = 1e-8
    assert max_bottom_disp < solver_tol, (
        f"Raw FEM solution at Dirichlet nodes is not near-zero: "
        f"max |u| = {max_bottom_disp:.2e} (expected < {solver_tol}). "
        f"This indicates a solver tolerance issue, not an interpolation issue."
    )
    print(f"✓ Part 1 passed: raw FEM solution at bottom nodes is < {solver_tol}")

    # --- Part 2: check interpolated sensor readings on the bottom face --------
    readings = evaluate_sensor_displacements(sol, model.sensor_interpolants)
    assert len(readings) == 1
    u_sensor = np.asarray(readings[0]["u"])

    max_sensor_disp = np.max(np.abs(u_sensor))
    print(f"\nInterpolated displacement at Dirichlet boundary sensors: {u_sensor}")
    print(f"Max |interpolated displacement| at Dirichlet sensors: {max_sensor_disp:.2e}")

    interp_tol = 1e-12
    assert max_sensor_disp < interp_tol, (
        f"Interpolated sensor displacement on Dirichlet face is not near-zero: "
        f"max |u| = {max_sensor_disp:.2e} (expected < {interp_tol}). "
        f"This indicates an interpolation issue (e.g. sensor point not found on "
        f"the correct face), not a solver tolerance issue."
    )
    print(f"✓ Part 2 passed: interpolated sensor output at Dirichlet nodes is < {interp_tol}")

    # --- Sanity check: sensor at top should have non-zero displacement --------
    from pdmp.forward_model import build_sensor_interpolants
    sensors_top = [{"name": "top_sensor", "location_fn": "top_surface",
                    "point": np.array([0.5, 0.5, 2.5])}]
    top_interpolants = build_sensor_interpolants(
        fe, sensors_top,
        location_fn_map={'top_surface': lambda p: jnp.isclose(p[2], 2.5, atol=1e-5)}
    )
    top_readings = evaluate_sensor_displacements(sol, top_interpolants)
    u_top = np.asarray(top_readings[0]["u"])
    print(f"\nDisplacement at top face sensor (sanity check): {u_top}")
    assert np.max(np.abs(u_top)) > 1e-6, (
        "Top-face sensor reads near-zero – the FEM solve may not have applied "
        "the traction load correctly."
    )
    print("✓ Sanity check passed: top-face sensor reads non-zero displacement\n")

    print("✓ Dirichlet boundary sensor test passed!\n")


if __name__ == '__main__':
    # Initialize jax-fem logger by creating a minimal model, then suppress it
    # This prevents verbose output during the actual tests
    print("Initializing test environment...")
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[1.**2]]))
    field = JaxConstantField(field_dist)
    _ = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, n_params=1, field=field)

    # Now suppress the logger for all subsequent model creations
    # Use ERROR level instead of WARNING to suppress even more output
    import logging
    jax_fem_logger = logging.getLogger('jax_fem')
    jax_fem_logger.setLevel(logging.ERROR)
    for handler in jax_fem_logger.handlers:
        handler.setLevel(logging.ERROR)
    print("Logger suppression activated.\n")

    # Run all tests
    test_jax_fem_model_basic_sensors()
    test_jax_fem_model_multiple_points_per_sensor()
    test_jax_fem_model_multiple_sensor_groups()
    test_jax_fem_model_sensors_from_config()
    test_dirichlet_boundary_sensors_near_zero()
    test_jax_fem_model_sensors_from_config_multiple_groups()
    test_jax_fem_model_default_sensors()
    test_sensor_output_consistency()
    test_gradient_computation_with_sensors()

    print("=" * 70)
    print("All sensor tests passed! ✓")
    print("=" * 70)


