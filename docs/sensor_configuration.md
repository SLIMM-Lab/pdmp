# Sensor Configuration Guide for JaxFemModel

## Overview

The `JaxFemModel` supports flexible sensor configurations, allowing you to:
- Define multiple sensors with single or multiple observation points
- Place each sensor group on a different mesh boundary via a per-sensor `location_fn`
- Configure sensors via YAML configuration files

## Sensor Structure

Each sensor is defined as a dictionary with the following required and optional keys:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | ✓ | Identifier string for the sensor group |
| `location_fn` | ✓ | Boundary face to search for sensor points (see below) |
| `point` | ✓ (or `points`) | Single observation point `[x, y, z]` |
| `points` | ✓ (or `point`) | Multiple observation points `[[x1,y1,z1], ...]` |

### Single-Point Sensor
```python
{
    "name": "sensor_name",
    "location_fn": "side_faces",
    "point": [x, y, z]
}
```

### Multi-Point Sensor (Sensor Group)
```python
{
    "name": "sensor_group_name",
    "location_fn": "bottom",
    "points": [
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3],
    ]
}
```

## Sensor Location Functions

The `location_fn` key on each sensor spec defines which mesh boundary that sensor group
is placed on. Different sensor groups can use different boundaries.

### Built-in String Identifiers

| String | Boundary |
|--------|----------|
| `'side_faces'` | x=0, x=d_x, y=0, y=d_y faces |
| `'top_surface'` | z=d_z face |
| `'bottom'` | z=0 face (Dirichlet-clamped) |

### Callable Function

You can also pass a callable directly as `location_fn`:

```python
import jax.numpy as jnp

def left_face_only(point):
    return jnp.isclose(point[0], 0.0, atol=1e-5)

sensors = [
    {"name": "left_sensor", "location_fn": left_face_only,
     "point": np.array([0.0, 0.5, 1.25])},
]
```

## Usage Examples

### Example 1: Sensors on the Same Boundary

```python
from pdmp.forward_model import JaxFemModel
import numpy as np

sensors = [
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

model = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, sensors=sensors)
```

### Example 2: Sensors on Different Boundaries

Each sensor group independently declares its boundary:

```python
sensors = [
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

model = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, h=0.5, sensors=sensors)
```

### Example 3: YAML Configuration

```yaml
sensors:
  - name: 'sensor_left'
    location_fn: 'side_faces'
    point: [0.0, 0.5, 1.25]

  - name: 'sensor_right'
    location_fn: 'side_faces'
    point: [1.0, 0.5, 1.25]

  - name: 'sensor_array'
    location_fn: 'side_faces'
    points:
      - [0.5, 0.0, 0.5]
      - [0.5, 0.0, 1.25]
      - [0.5, 0.0, 2.0]

  - name: 'top_sensor'
    location_fn: 'top_surface'
    point: [0.5, 0.5, 2.5]
```

Load it with:

```python
import yaml
from pdmp.forward_model import JaxFemModel

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = JaxFemModel.from_dict(config['model'], field=field)
```

## Observation Dimensions

The total number of observations (`d_obs`) is automatically computed from the sensors:

```
d_obs = (total number of sensor points across all groups) × 3
```

Where each sensor point contributes 3 degrees of freedom (displacement in x, y, z).

### Example:
```python
sensors = [
    {"name": "s1", "location_fn": "side_faces", "point": [0, 0.5, 1]},       # 1 point
    {"name": "s2", "location_fn": "side_faces", "point": [1, 0.5, 1]},       # 1 point
    {"name": "group", "location_fn": "top_surface",
     "points": [[...], [...], [...]]},                                         # 3 points
]
# Total: 5 points × 3 DOF = 15 observations
```

## Default Behavior

If no sensors are specified, the model uses a single default sensor on the side faces:

```python
sensors = [
    {"name": "sensor_left_center", "location_fn": "side_faces",
     "point": np.array([0, 0.5*d_y, 0.5*d_z])},
]
```

## Best Practices

1. **Boundary matching**: Make sure each sensor point actually lies on the face declared
   by its `location_fn`. A `ValueError` is raised at construction time if no matching
   face is found, so mistakes are caught early.
2. **Naming**: Use descriptive names for sensors to aid debugging and visualisation.
3. **Groups**: Use sensor groups for spatially-related observation points that share a
   boundary.
4. **Dirichlet face**: Sensors on the clamped bottom face (`location_fn: 'bottom'`)
   will read near-zero displacement by construction. This is sometimes useful as a
   consistency check.

## See Also

- `tests/test_jax_fem_sensors.py` — comprehensive test suite including a
  Dirichlet-boundary sensor check
- `pdmp/forward_model.py` — implementation details
