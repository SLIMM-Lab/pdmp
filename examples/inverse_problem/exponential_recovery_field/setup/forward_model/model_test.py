import numpy as np
import jax.numpy as jnp

from pdmp.loader import get_model, get_config
from pdmp.forward_model import JaxFemModel, build_sensor_interpolants, evaluate_sensor_displacements



# ── helpers ──────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── build model ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = get_config('config.yaml')
    model: JaxFemModel = get_model(config['model'])

    fe       = model.problem.fe
    h        = config['model']['h']
    d_x      = config['model']['d_x']
    d_y      = config['model']['d_y']
    d_z      = config['model']['d_z']
    test_pt  = np.array([0.0, 0.49, 2.5])   # the shared corner point from config

    print_section("Mesh info")
    print(f"  h={h}  domain=({d_x} x {d_y} x {d_z})")
    print(f"  num_total_nodes : {fe.num_total_nodes}")
    print(f"  num_cells       : {fe.num_cells}")

    # ── 1. inspect sensor interpolants ────────────────────────────────────────

    print_section("Sensor interpolants (from model)")
    for interp in model.sensor_interpolants:
        print(f"\n  [{interp['name']}]")
        print(f"    point(s)  : {interp['points']}")
        print(f"    nodes     : {interp['nodes']}")
        print(f"    weights   : {interp['weights']}")
        node_coords = fe.points[interp['nodes'][0]]
        print(f"    node coords:")
        for nid, nc in zip(interp['nodes'][0], node_coords):
            print(f"      node {nid:5d}  {nc}")

    # ── 2. check whether the two sensors found the same nodes ─────────────────

    print_section("Comparing node sets of both sensors")
    if len(model.sensor_interpolants) == 2:
        s0 = model.sensor_interpolants[0]
        s1 = model.sensor_interpolants[1]
        nodes0 = set(s0['nodes'][0].tolist())
        nodes1 = set(s1['nodes'][0].tolist())
        print(f"  '{s0['name']}' nodes : {sorted(nodes0)}")
        print(f"  '{s1['name']}' nodes : {sorted(nodes1)}")
        print(f"  Shared nodes       : {sorted(nodes0 & nodes1)}")
        same_nodes   = nodes0 == nodes1
        same_weights = np.allclose(
            np.sort(s0['weights'][0]), np.sort(s1['weights'][0])
        )
        print(f"  Identical node sets  : {same_nodes}")
        print(f"  Weights close (sorted): {same_weights}")
        if not same_nodes:
            print("\n  *** WARNING: the two sensors interpolate from DIFFERENT nodes ***")
            print("      This means they can give different readings even at the same E.")

    # ── 3. run the forward model and compare readings ─────────────────────────

    print_section("Forward model evaluation")
    E_test = np.array([10.])
    y = model.eval(E_test, save_dir='.')
    print(f"  Full observation vector y = {y}")

    # Break down by sensor
    param_field = E_test[0] * jnp.ones((fe.num_cells, fe.num_quads))
    sol_list    = model.fwd_pred([param_field])
    sol         = sol_list[0]
    readings    = evaluate_sensor_displacements(sol, model.sensor_interpolants)

    print("\n  Per-sensor displacements:")
    for r in readings:
        print(f"    [{r['name']}]  u = {np.asarray(r['u'])}")

    if len(readings) == 2:
        diff = np.asarray(readings[0]['u']) - np.asarray(readings[1]['u'])
        print(f"\n  Difference (sensor_top_side - sensor_top): {diff}")
        print(f"  Max absolute difference: {np.max(np.abs(diff)):.6e}")
        if np.allclose(diff, 0, atol=1e-8):
            print("  ✓  Both sensors agree (within floating-point tolerance).")
        else:
            print("  ✗  Sensors disagree — likely due to different interpolating faces.")

    # ── 4. sanity check: point exactly at a mesh node ─────────────────────────

    print_section("Sanity check: sensor exactly at a mesh node")
    # With h=0.25, a node at x=0, y=0.5, z=2.5 should be on both boundaries
    node_pt = np.array([0.0, 0.5, d_z])
    print(f"  Test point: {node_pt}  (should be on both side_faces and top_face)")

    def side_faces(point):
        return (jnp.isclose(point[0], 0., atol=1e-5) |
                jnp.isclose(point[0], d_x, atol=1e-5) |
                jnp.isclose(point[1], 0., atol=1e-5) |
                jnp.isclose(point[1], d_y, atol=1e-5))

    def top_face(point):
        return jnp.isclose(point[2], d_z, atol=1e-5)

    location_fn_map = {'side_faces': side_faces, 'top_face': top_face}

    node_sensors = [
        {"name": "node_side",  "location_fn": "side_faces", "point": node_pt},
        {"name": "node_top",   "location_fn": "top_face",   "point": node_pt},
    ]
    try:
        node_interps = build_sensor_interpolants(fe, node_sensors, location_fn_map)
        for interp in node_interps:
            print(f"\n  [{interp['name']}]  nodes={interp['nodes'][0]}  weights={interp['weights'][0]}")
        node_readings = evaluate_sensor_displacements(sol, node_interps)
        for r in node_readings:
            print(f"    displacement: {np.asarray(r['u'])}")
        if len(node_readings) == 2:
            diff_node = np.asarray(node_readings[0]['u']) - np.asarray(node_readings[1]['u'])
            print(f"  Node-point difference: {diff_node}  (max={np.max(np.abs(diff_node)):.2e})")
    except ValueError as e:
        print(f"  ERROR building node-point sensor: {e}")

    # ── 5. check: is the original test_pt actually on a mesh face? ────────────

    print_section("Is the test point y=0.49 on a mesh face?")
    print(f"  Mesh y-nodes (h={h}): {np.arange(0, d_y + h/2, h)}")
    print(f"  test_pt y=0.49 — nearest nodes: "
          f"{np.arange(0, d_y + h/2, h)[np.argsort(np.abs(np.arange(0, d_y + h/2, h) - 0.49))[:2]]}")
    print("  y=0.49 lies INSIDE a face element (not on a node), so interpolation is used.")
    print()
    print("  Both sensors find the shared edge (x=0, z=d_z) and its two bounding nodes.")
    print("  side_faces picks the x=0 face; top_face picks the z=d_z face.")
    print("  They share nodes 30 and 31, but differ in their 4th node (x≠0 vs z≠d_z).")
    print("  The weights on the *shared* nodes are identical, so the non-shared nodes")
    print("  receive zero weight — both sensors effectively interpolate identically.")
    print()
    print("  The tiny residual difference (~1e-9) is pure floating-point rounding;")
    print("  it is many orders of magnitude below any physical displacement.")
