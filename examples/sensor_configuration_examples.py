#!/usr/bin/env python3
"""
Example: Using JaxFemModel with custom sensor configurations

This example demonstrates:
1. Single-point sensors on the same boundary
2. Sensors on different boundaries (mixed location_fn)
3. Loading sensor configuration from a dict (YAML-compatible)
4. Custom callable sensor location function

Each sensor spec requires a 'location_fn' key identifying which mesh boundary
to search. Built-in string options: 'side_faces', 'top_surface', 'bottom'.
"""

import numpy as np
from pdmp.forward_model import JaxFemModel

print("=" * 70)
print("EXAMPLE 1: Sensors on the same boundary")
print("=" * 70)

sensors_same = [
    {"name": "sensor_left",  "location_fn": "side_faces",
     "point": np.array([0.0, 0.5, 1.25])},
    {"name": "sensor_right", "location_fn": "side_faces",
     "point": np.array([1.0, 0.5, 1.25])},
    {
        "name": "sensor_array",
        "location_fn": "side_faces",
        "points": np.array([
            [0.5, 0.0, 0.5],
            [0.5, 0.0, 1.25],
            [0.5, 0.0, 2.0],
        ])
    },
]

# model_1 = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, sensors=sensors_same)

print("✓ Same-boundary sensor configuration created")
for s in sensors_same:
    n = 1 if "point" in s else len(s["points"])
    print(f"  - {s['name']} ({s['location_fn']}): {n} point(s)")
total_obs = (1 + 1 + 3) * 3
print(f"  Total observation dimensions: {total_obs}")


print("\n" + "=" * 70)
print("EXAMPLE 2: Sensors on different boundaries")
print("=" * 70)

sensors_mixed = [
    {
        "name": "side_array",
        "location_fn": "side_faces",
        "points": np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 1.5]])
    },
    {
        "name": "top_point",
        "location_fn": "top_surface",
        "point": np.array([0.5, 0.5, 2.5])
    },
    {
        "name": "bottom_check",
        "location_fn": "bottom",
        "point": np.array([0.5, 0.5, 0.0])
    },
]

# model_2 = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, sensors=sensors_mixed)

print("✓ Mixed-boundary sensor configuration created")
for s in sensors_mixed:
    n = 1 if "point" in s else len(s["points"])
    print(f"  - {s['name']} ({s['location_fn']}): {n} point(s)")
total_obs = (2 + 1 + 1) * 3
print(f"  Total observation dimensions: {total_obs}")


print("\n" + "=" * 70)
print("EXAMPLE 3: Loading from configuration dict (YAML-compatible)")
print("=" * 70)

# Equivalent YAML:
#
# sensors:
#   - name: 'sensor_left'
#     location_fn: 'side_faces'
#     point: [0.0, 0.5, 1.25]
#   - name: 'sensor_right'
#     location_fn: 'side_faces'
#     point: [1.0, 0.5, 1.25]
#   - name: 'sensor_array'
#     location_fn: 'side_faces'
#     points:
#       - [0.5, 0.0, 0.5]
#       - [0.5, 0.0, 1.25]
#       - [0.5, 0.0, 2.0]
#   - name: 'top_sensor'
#     location_fn: 'top_surface'
#     point: [0.5, 0.5, 2.5]

config = {
    'd_x': 1.0, 'd_y': 1.0, 'd_z': 2.5,
    'h': 0.5, 'ele_type': 'HEX8', 'nu': 0.3,
    'n_params': 2,
    'sensors': [
        {'name': 'sensor_left',  'location_fn': 'side_faces',
         'point': [0.0, 0.5, 1.25]},
        {'name': 'sensor_right', 'location_fn': 'side_faces',
         'point': [1.0, 0.5, 1.25]},
        {
            'name': 'sensor_array',
            'location_fn': 'side_faces',
            'points': [
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 1.25],
                [0.5, 0.0, 2.0],
            ]
        },
        {'name': 'top_sensor', 'location_fn': 'top_surface',
         'point': [0.5, 0.5, 2.5]},
    ],
}

# model_3 = JaxFemModel.from_dict(config)

print("✓ Configuration dict created")
print(f"  Sensors defined in config: {len(config['sensors'])}")
for s in config['sensors']:
    n = 1 if 'point' in s else len(s['points'])
    print(f"  - {s['name']} ({s['location_fn']}): {n} point(s)")


print("\n" + "=" * 70)
print("EXAMPLE 4: Custom callable sensor location function")
print("=" * 70)

import jax.numpy as jnp

def left_face_only(point):
    """Only match the x=0 face."""
    return jnp.isclose(point[0], 0.0, atol=1e-5)

sensors_custom = [
    {"name": "x0_sensor_1", "location_fn": left_face_only,
     "point": np.array([0.0, 0.25, 1.0])},
    {"name": "x0_sensor_2", "location_fn": left_face_only,
     "point": np.array([0.0, 0.75, 1.5])},
]

# model_4 = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, sensors=sensors_custom)

print("✓ Custom callable location function defined (x=0 face only)")
for s in sensors_custom:
    print(f"  - {s['name']}: {s['point']}")


print("\n" + "=" * 70)
print("Summary of Sensor Configuration Options")
print("=" * 70)
print("""
Required keys per sensor spec:
  name        : identifier string
  location_fn : 'side_faces' | 'top_surface' | 'bottom' | callable
  point       : [x, y, z]          (single point)
  points      : [[x1,y1,z1], ...]  (multiple points; use instead of 'point')

Built-in location_fn strings:
  'side_faces'   – x=0, x=d_x, y=0, y=d_y faces
  'top_surface'  – z=d_z face
  'bottom'       – z=0 face (Dirichlet-clamped, reads near-zero displacement)

Observation dimensions:
  d_obs = (total sensor points across all groups) × 3
""")

print("✓ All examples completed successfully!")
