#!/usr/bin/env python3
"""Test script for JaxFemModel with the new sensor setup.

Sensor Format:
- Each sensor spec must include 'name', 'location_fn', and either 'point' or 'points'.
- 'location_fn' can be a string ('side_faces', 'top_face', 'bottom_face') or a callable.

  Example: [{"name": "s1", "location_fn": "side_faces", "point": np.array([x, y, z])}]
  Example: [{"name": "s1", "location_fn": "bottom_face",
             "points": np.array([[x1,y1,z1], [x2,y2,z2]])}]

Output Dimensions:
- Each sensor point produces 3 displacement components (ux, uy, uz)
- Total output dimension = (total number of sensor points) × 3
"""

import numpy as np
from pdmp.random_field import JaxConstantField, get_jax_field
from pdmp.forward_model import (
    JaxFemModel,
    quad_bilinear_weights,
    point_in_triangle,
    build_sensor_interpolants,
    evaluate_sensor_displacements,
)
from pdmp.distributions import MultivariateNormal


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the low-level interpolation primitives
# ─────────────────────────────────────────────────────────────────────────────

def test_quad_weights_at_nodes():
    """At each corner node the corresponding weight must be 1 and all others 0."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights at corner nodes")
    print("=" * 70)

    # Axis-aligned unit square in the z=0 plane, CCW ordering.
    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    for i, node in enumerate(quad):
        w = quad_bilinear_weights(node, quad, tol)
        assert w is not None, f"Node {i}: weights returned None"
        assert np.allclose(w[i], 1.0, atol=1e-10), \
            f"Node {i}: expected w[{i}]=1, got {w}"
        other = [j for j in range(4) if j != i]
        assert np.allclose(w[other], 0.0, atol=1e-10), \
            f"Node {i}: expected other weights=0, got {w}"
        print(f"  node {i}: weights = {w}  ✓")

    print("✓ quad_weights_at_nodes passed!\n")


def test_quad_weights_at_centre():
    """At the face centre all four weights should equal 0.25."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights at face centre")
    print("=" * 70)

    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    centre = np.array([0.5, 0.5, 0.0])
    tol = 1e-8

    w = quad_bilinear_weights(centre, quad, tol)
    assert w is not None, "Centre: weights returned None"
    assert np.allclose(w, 0.25, atol=1e-10), \
        f"Centre: expected all weights=0.25, got {w}"
    print(f"  centre weights = {w}  ✓")
    print("✓ quad_weights_at_centre passed!\n")


def test_quad_weights_on_edge():
    """On an edge only the two nodes spanning that edge should be non-zero."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights on edge")
    print("=" * 70)

    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    # Mid-point of edge 0-1 (y=0 bottom edge): nodes 0 and 1 share weight 0.5
    p_edge01 = np.array([0.5, 0.0, 0.0])
    w = quad_bilinear_weights(p_edge01, quad, tol)
    assert w is not None
    assert np.allclose(w[0], 0.5, atol=1e-10) and np.allclose(w[1], 0.5, atol=1e-10), \
        f"Edge 0-1 midpoint: expected [0.5, 0.5, 0, 0], got {w}"
    assert np.allclose(w[2:], 0.0, atol=1e-10), \
        f"Edge 0-1 midpoint: expected w[2]=w[3]=0, got {w}"
    print(f"  edge 0-1 midpoint weights = {w}  ✓")

    # Point at y=0.49 on left edge (x=0): nodes 0 and 3, weights (0.51, 0.49) approx
    p_left = np.array([0.0, 0.49, 0.0])
    w2 = quad_bilinear_weights(p_left, quad, tol)
    assert w2 is not None
    assert np.allclose(w2[1], 0.0, atol=1e-10) and np.allclose(w2[2], 0.0, atol=1e-10), \
        f"Left edge y=0.49: expected w[1]=w[2]=0, got {w2}"
    assert np.isclose(w2[0] + w2[3], 1.0, atol=1e-10), \
        f"Left edge y=0.49: weights should sum to 1, got {w2}"
    assert np.isclose(w2[3], 0.49, atol=1e-10) and np.isclose(w2[0], 0.51, atol=1e-10), \
        f"Left edge y=0.49: expected w[0]=0.51, w[3]=0.49, got {w2}"
    print(f"  left edge (x=0, y=0.49) weights = {w2}  ✓")

    print("✓ quad_weights_on_edge passed!\n")


def test_quad_weights_interior_all_nonzero():
    """For a generic interior point all four bilinear weights must be non-zero.

    This is the key test that was broken by the old triangle-splitting approach,
    which would assign zero weight to one node even for interior points.
    """
    print("=" * 70)
    print("Unit test: quad_bilinear_weights — all 4 weights non-zero at interior")
    print("=" * 70)

    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    interior_points = [
        np.array([0.3, 0.4, 0.0]),
        np.array([0.7, 0.2, 0.0]),
        np.array([0.1, 0.9, 0.0]),
        np.array([0.6, 0.6, 0.0]),
    ]

    for p in interior_points:
        w = quad_bilinear_weights(p, quad, tol)
        assert w is not None, f"Interior point {p}: weights returned None"
        assert np.all(w > tol), \
            f"Interior point {p}: expected all 4 weights > 0, got {w}"
        assert np.isclose(np.sum(w), 1.0, atol=1e-10), \
            f"Interior point {p}: weights don't sum to 1, got sum={np.sum(w)}"
        # Verify reconstruction: sum_i w_i * node_i == point (x, y components)
        reconstructed = w @ quad[:, :2]
        assert np.allclose(reconstructed, p[:2], atol=1e-10), \
            f"Interior point {p}: reconstruction failed: got {reconstructed}"
        print(f"  p={p[:2]}  weights={np.round(w,4)}  reconstructed={np.round(reconstructed,4)}  ✓")

    print("✓ quad_weights_interior_all_nonzero passed!\n")


def test_quad_weights_off_plane_rejected():
    """A point that is not coplanar with the face must return None."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights rejects off-plane points")
    print("=" * 70)

    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    off_plane_points = [
        np.array([0.5, 0.5, 0.1]),    # above face, inside projected boundary
        np.array([0.3, 0.3, -0.05]),  # below face
        np.array([0.0, 0.49, 2.5]),   # the original bug case: on x=0 side face, z=2.5
    ]

    for p in off_plane_points:
        w = quad_bilinear_weights(p, quad, tol)
        assert w is None, \
            f"Off-plane point {p}: expected None, got {w}"
        print(f"  p={p}  correctly rejected (returned None)  ✓")

    print("✓ quad_weights_off_plane_rejected passed!\n")


def test_quad_weights_outside_face_rejected():
    """A coplanar point that lies outside the face boundary must return None."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights rejects out-of-face coplanar points")
    print("=" * 70)

    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    outside_points = [
        np.array([1.5, 0.5, 0.0]),   # outside in x
        np.array([0.5, -0.1, 0.0]),  # outside in y
        np.array([-0.1, 0.5, 0.0]),  # outside in x (negative)
    ]

    for p in outside_points:
        w = quad_bilinear_weights(p, quad, tol)
        assert w is None, \
            f"Out-of-face point {p}: expected None, got {w}"
        print(f"  p={p}  correctly rejected (returned None)  ✓")

    print("✓ quad_weights_outside_face_rejected passed!\n")


def test_quad_weights_non_square_face():
    """Bilinear interpolation should work on a rectangular (non-square) face."""
    print("=" * 70)
    print("Unit test: quad_bilinear_weights on a rectangular face")
    print("=" * 70)

    # 2 x 0.5 rectangle in z=1 plane
    quad = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    ])
    tol = 1e-8

    # Centre should give all weights = 0.25
    centre = np.array([1.0, 0.25, 1.0])
    w = quad_bilinear_weights(centre, quad, tol)
    assert w is not None
    assert np.allclose(w, 0.25, atol=1e-10), f"Rectangle centre: expected all 0.25, got {w}"
    print(f"  rectangle centre weights = {w}  ✓")

    # Interior point (1.2, 0.1): all 4 weights non-zero
    p = np.array([1.2, 0.1, 1.0])
    w2 = quad_bilinear_weights(p, quad, tol)
    assert w2 is not None
    assert np.all(w2 > tol), f"Rectangle interior: expected all weights > 0, got {w2}"
    reconstructed = w2 @ quad[:, :2]
    assert np.allclose(reconstructed, p[:2], atol=1e-10), \
        f"Rectangle interior: reconstruction failed: {reconstructed} != {p[:2]}"
    print(f"  rectangle interior p={p[:2]}  weights={np.round(w2,4)}  ✓")

    print("✓ quad_weights_non_square_face passed!\n")


def test_point_in_triangle_coplanarity():
    """point_in_triangle must reject points that are not on the triangle's plane."""
    print("=" * 70)
    print("Unit test: point_in_triangle coplanarity check")
    print("=" * 70)

    tri = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tol = 1e-8

    # Off-plane: z != 0 but (x,y) projects inside the triangle
    inside, w = point_in_triangle(np.array([0.2, 0.2, 0.1]), tri, tol)
    assert not inside, "Off-plane point should be rejected"
    print("  off-plane point correctly rejected  ✓")

    # On-plane, inside
    inside, w = point_in_triangle(np.array([0.2, 0.2, 0.0]), tri, tol)
    assert inside and w is not None
    assert np.isclose(sum(w), 1.0, atol=1e-10)
    print(f"  on-plane interior point: weights={np.round(w,4)}  ✓")

    # On-plane, outside
    inside, _ = point_in_triangle(np.array([0.8, 0.8, 0.0]), tri, tol)
    assert not inside, "Out-of-triangle point should be rejected"
    print("  on-plane exterior point correctly rejected  ✓")

    print("✓ point_in_triangle_coplanarity passed!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: verify all-4-node contribution through a full FEM solve
# ─────────────────────────────────────────────────────────────────────────────

def test_interior_face_sensor_uses_all_four_nodes():
    """An interior face sensor must interpolate from all 4 nodes of its quad face.

    With the old triangle-splitting approach one of the four nodes received zero
    weight even at interior points.  This test verifies that the bilinear
    implementation gives non-zero weight to all four nodes for a point that is
    strictly inside the face (not on any edge or at any corner).
    """
    print("=" * 70)
    print("Integration test: interior face sensor uses all 4 nodes")
    print("=" * 70)

    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # Place sensor strictly inside a top-face quad: x and y are not on any mesh
    # node (h=0.25 → nodes at 0, 0.25, 0.5, …).  x=0.1, y=0.1 falls in the
    # quad spanned by (0,0), (0.25,0), (0.25,0.25), (0,0.25).
    sensors = [
        {"name": "interior_top",
         "location_fn": "top_face",
         "point": np.array([0.1, 0.1, 2.5])}
    ]

    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors,
    )

    interp = model.sensor_interpolants[0]
    weights = interp["weights"][0]
    nodes   = interp["nodes"][0]
    coords  = model.problem.fe.points[nodes]

    print(f"  Sensor node IDs  : {nodes}")
    print(f"  Node coordinates :")
    for nid, c, w in zip(nodes, coords, weights):
        print(f"    node {nid:4d}  {c}  weight={w:.6f}")

    tol = 1e-8
    assert np.all(weights > tol), (
        f"Expected all 4 weights > 0 for interior point, got {weights}. "
        "This indicates the old triangle-splitting bug is still present."
    )
    assert np.isclose(np.sum(weights), 1.0, atol=1e-10), \
        f"Weights do not sum to 1: {weights}"

    # Also verify the weighted sum of node coordinates recovers the sensor point
    reconstructed = weights @ coords
    expected = np.array([0.1, 0.1, 2.5])
    assert np.allclose(reconstructed, expected, atol=1e-10), \
        f"Weight reconstruction failed: {reconstructed} != {expected}"
    print(f"  Reconstructed point: {reconstructed}  ✓")

    print("✓ interior_face_sensor_uses_all_four_nodes passed!\n")


def test_shared_edge_sensors_agree():
    """Two sensors at the same point on the edge shared by two boundary faces
    must produce identical interpolation weights (on the two shared edge nodes)
    and identical displacement readings after a FEM solve.
    """
    print("=" * 70)
    print("Integration test: shared-edge sensors agree")
    print("=" * 70)

    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # y=0.49 is strictly between mesh nodes 0.25 and 0.5 so the point lies on
    # an interior edge of each boundary face (not at a corner node).
    shared_pt = np.array([0.0, 0.49, 2.5])

    sensors = [
        {"name": "side_edge",  "location_fn": "side_faces", "point": shared_pt},
        {"name": "top_edge",   "location_fn": "top_face",   "point": shared_pt},
    ]

    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,
        n_params=1,
        field=field,
        sensors=sensors,
    )

    s0, s1 = model.sensor_interpolants

    # Non-zero weights must sit on exactly the two nodes of the shared edge
    nz0 = set(s0["nodes"][0][s0["weights"][0] > 1e-10].tolist())
    nz1 = set(s1["nodes"][0][s1["weights"][0] > 1e-10].tolist())

    print(f"  side_edge  non-zero-weight nodes: {sorted(nz0)}")
    print(f"  top_edge   non-zero-weight nodes: {sorted(nz1)}")

    assert nz0 == nz1, (
        f"Both sensors should activate the same two edge nodes, "
        f"but side={sorted(nz0)}, top={sorted(nz1)}"
    )
    assert len(nz0) == 2, f"Expected exactly 2 non-zero-weight nodes, got {len(nz0)}"

    # The weights on those two shared nodes should be identical
    def shared_weights(interp):
        nz_mask = interp["weights"][0] > 1e-10
        return dict(zip(interp["nodes"][0][nz_mask].tolist(),
                        interp["weights"][0][nz_mask].tolist()))

    w0 = shared_weights(s0)
    w1 = shared_weights(s1)
    for node_id in nz0:
        assert np.isclose(w0[node_id], w1[node_id], atol=1e-10), (
            f"Node {node_id}: side weight={w0[node_id]}, top weight={w1[node_id]}"
        )
    print(f"  Shared edge weights: {w0}  ✓")

    # Full FEM solve: displacement readings must be identical
    import jax.numpy as jnp
    fe = model.problem.fe
    param_field = 10. * jnp.ones((fe.num_cells, fe.num_quads))
    sol = model.fwd_pred([param_field])[0]
    readings = evaluate_sensor_displacements(sol, model.sensor_interpolants)
    u0 = np.asarray(readings[0]["u"])
    u1 = np.asarray(readings[1]["u"])
    diff = np.max(np.abs(u0 - u1))
    print(f"  side displacement : {u0}")
    print(f"  top  displacement : {u1}")
    print(f"  max |difference|  : {diff:.2e}")
    assert diff == 0.0, f"Expected exactly zero difference, got {diff:.2e}"

    print("✓ shared_edge_sensors_agree passed!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Existing integration tests (unchanged)
# ─────────────────────────────────────────────────────────────────────────────



def test_jax_fem_model_basic_sensors():
    """Test JaxFemModel with basic sensor setup.

    Also verifies that specifying a sensor point that does not lie on the
    declared face raises a ``ValueError`` during model construction.
    """
    print("=" * 70)
    print("Testing JaxFemModel with basic sensors")
    print("=" * 70)

    # Create a simple constant field for Young's modulus
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[2.**2]]))
    field = JaxConstantField(field_dist)

    # --- Part 1: point NOT on side_faces should raise ValueError --------------
    # [0.5, 0.5, 1.25] is an interior point; none of its x/y coordinates equal
    # 0 or d_x/d_y, so build_sensor_interpolants must raise a ValueError.
    invalid_sensors = [
        {"name": "sensor_interior", "location_fn": "side_faces",
         "point": np.array([0.5, 0.5, 1.25])}
    ]

    print("Part 1: Verifying that an off-face sensor point raises ValueError...")
    raised = False
    try:
        _ = JaxFemModel(
            d_x=1.0, d_y=1.0, d_z=2.5,
            h=0.25,
            n_params=1,
            field=field,
            sensors=invalid_sensors
        )
    except ValueError as e:
        raised = True
        print(f"  \u2713 ValueError raised as expected: {e}")
        assert "sensor_interior" in str(e) or "not located" in str(e), (
            f"Unexpected error message: {e}"
        )

    assert raised, (
        "Expected a ValueError when the sensor point is not on the declared face, "
        "but no exception was raised."
    )

    # --- Part 2: valid sensor point on side_faces creates model correctly -----
    # [0.0, 0.5, 1.25] lies on the x=0 side face.
    print("Part 2: Creating model with a valid side-face sensor point...")
    valid_sensors = [
        {"name": "sensor_side", "location_fn": "side_faces",
         "point": np.array([0.0, 0.5, 1.25])}
    ]

    model = JaxFemModel(
        d_x=1.0, d_y=1.0, d_z=2.5,
        h=0.25,  # coarse mesh for testing
        n_params=1,
        field=field,
        sensors=valid_sensors
    )

    print(f"Model input dimension: {model.get_dim_in()}")
    print(f"Model output dimension: {model.get_dim_out()}")
    print(f"Number of sensors: {len(valid_sensors)}")

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
    print("\u2713 Basic sensor test passed!\n")


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
            "location_fn": "top_face",
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
            "location_fn": "top_face",
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
            "location_fn": "bottom_face",
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
    sensors_top = [{"name": "top_sensor", "location_fn": "top_face",
                    "point": np.array([0.5, 0.5, 2.5])}]
    top_interpolants = build_sensor_interpolants(
        fe, sensors_top,
        location_fn_map={'top_face': lambda p: jnp.isclose(p[2], 2.5, atol=1e-5)}
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
    # ── Unit tests (no FEM model needed) ──────────────────────────────────────
    test_quad_weights_at_nodes()
    test_quad_weights_at_centre()
    test_quad_weights_on_edge()
    test_quad_weights_interior_all_nonzero()
    test_quad_weights_off_plane_rejected()
    test_quad_weights_outside_face_rejected()
    test_quad_weights_non_square_face()
    test_point_in_triangle_coplanarity()

    # ── Integration tests (need a FEM solve) ──────────────────────────────────
    # Initialize jax-fem logger by creating a minimal model, then suppress it
    print("Initializing test environment...")
    field_dist = MultivariateNormal(mean=np.array([10.]), cov=np.array([[1.**2]]))
    field = JaxConstantField(field_dist)
    _ = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, n_params=1, field=field,
                    sensors=[{"name": "warmup", "location_fn": "side_faces",
                               "point": np.array([0.0, 0.5, 1.0])}])

    import logging
    jax_fem_logger = logging.getLogger('jax_fem')
    jax_fem_logger.setLevel(logging.ERROR)
    for handler in jax_fem_logger.handlers:
        handler.setLevel(logging.ERROR)
    print("Logger suppression activated.\n")

    test_interior_face_sensor_uses_all_four_nodes()
    test_shared_edge_sensors_agree()
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


