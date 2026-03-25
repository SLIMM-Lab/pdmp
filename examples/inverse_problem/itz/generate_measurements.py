#!/usr/bin/env python3
"""Generate synthetic displacement observations from the coarsened ITZ FEM model.

Distributes N_SENSORS evenly across all non-Dirichlet boundary faces (4 side
faces + top face), placing sensors only at non-void coarse element face centres.
Saves the sensor configuration to a reusable YAML file and the flattened
displacement observations to a .npy file.

Usage:
    python generate_measurements.py [N_SENSORS] [POOL] [--plot] [--noise-std=S] [--seed=N]

    N_SENSORS       total number of sensors (default 20)
    POOL            coarsening factor per axis (default 2, must divide 50,50,110)
    --plot          save sensor position figures to data/sensor_positions_3d.png
                    and data/sensor_positions_faces.png
    --noise-std=S   standard deviation of Gaussian noise added to observations,
                    in microns (default 0.1)
    --seed=N        integer RNG seed for reproducibility (default: random)
"""
import sys
import numpy as onp
import jax
import jax.numpy as np
import os
import yaml
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh

from pdmp.forward_model import build_sensor_interpolants, evaluate_sensor_displacements
from pdmp.loader import numpy_to_yaml

# ── Phase definitions ────────────────────────────────────────────────────────
PORE = 1
OUTER_CSH = 2
INNER_CSH = 3
ANHYDROUS = 4
AGGREGATE = 7

E_MAP = {OUTER_CSH: 25.0, INNER_CSH: 31.0, ANHYDROUS: 130.0, AGGREGATE: 70.0}
NU = 0.18

VOXEL_SIZE = 2.0  # µm
Z_THRES = 110.0  # µm — load applied above this z-coordinate
TOTAL_FORCE = 40.0  # mN

# ── CLI arguments ────────────────────────────────────────────────────────────
PLOT = '--plot' in sys.argv
_flags = {
    a.lstrip('-').split('=')[0]: (a.split('=')[1] if '=' in a else None)
    for a in sys.argv[1:] if a.startswith('--')
}
SEED = int(_flags['seed']) if 'seed' in _flags else None
NOISE_STD = float(_flags['noise-std']) if 'noise-std' in _flags else 0.05
_pos = [a for a in sys.argv[1:] if not a.startswith('--')]
N_SENSORS = int(_pos[0]) if len(_pos) > 0 else 20
POOL = int(_pos[1]) if len(_pos) > 1 else 2

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ITZ_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
#                        os.pardir, 'jax-fem', 'itz')
ITZ_DIR = SCRIPT_DIR
GEOM_FILE = os.path.join(ITZ_DIR, 'itz_geom.npy')
OUT_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load geometry ────────────────────────────────────────────────────────
geom = onp.load(GEOM_FILE)
Nx, Ny, Nz = geom.shape
domain_x = Nx * VOXEL_SIZE
domain_y = Ny * VOXEL_SIZE
domain_z = Nz * VOXEL_SIZE

assert Nx % POOL == 0 and Ny % POOL == 0 and Nz % POOL == 0, \
    f"POOL={POOL} must divide all geometry dimensions ({Nx}, {Ny}, {Nz})"

Cx, Cy, Cz = Nx // POOL, Ny // POOL, Nz // POOL
elem_size = POOL * VOXEL_SIZE
print(f"Geometry: {geom.shape}, coarse grid: ({Cx},{Cy},{Cz}), POOL={POOL}")

# ── 2. Coarsen ──────────────────────────────────────────────────────────────
E_voxel = onp.zeros(geom.shape, dtype=onp.float64)
for phase, E_val in E_MAP.items():
    E_voxel[geom == phase] = E_val

E_coarse = E_voxel.reshape(Cx, POOL, Cy, POOL, Cz, POOL).mean(axis=(1, 3, 5))

# ── 3. Remove pores + connected component filter ───────────────────────────
solid_mask_3d = E_coarse > 0.0

labels_3d, n_components = label(solid_mask_3d)
if n_components > 1:
    component_sizes = onp.bincount(labels_3d.ravel())[1:]
    largest = onp.argmax(component_sizes) + 1
    solid_mask_3d = labels_3d == largest
    print(f"Connected components: {n_components}, kept largest")

print(f"Solid elements: {int(solid_mask_3d.sum())}")

# ── Visualization ───────────────────────────────────────────────────────────
# Per-face display settings: (mask_slice, horiz_coord_idx, vert_coord_idx,
#                              horiz_extent, vert_extent, xlabel, ylabel)
_FACE_DISPLAY = {
    'face_x_min':
    (lambda sm, cx, cy, cz: sm[0, :, :], 1, 2, 'y (µm)', 'z (µm)'),
    'face_x_max':
    (lambda sm, cx, cy, cz: sm[cx - 1, :, :], 1, 2, 'y (µm)', 'z (µm)'),
    'face_y_min':
    (lambda sm, cx, cy, cz: sm[:, 0, :], 0, 2, 'x (µm)', 'z (µm)'),
    'face_y_max':
    (lambda sm, cx, cy, cz: sm[:, cy - 1, :], 0, 2, 'x (µm)', 'z (µm)'),
    'face_z_max': (lambda sm, cx, cy, cz: sm[:, :, cz - 1], 0, 1, 'x (µm)',
                   'y (µm)'),
}
_FACE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def _plot_sensors(sensor_specs, solid_mask_3d, cx, cy, cz, elem_size, domain_x,
                  domain_y, domain_z, out_dir):
    """Save two figures: 3D scatter overview and per-face 2D panels."""
    face_names = list(_FACE_DISPLAY.keys())
    color_map = dict(zip(face_names, _FACE_COLORS))
    sensor_pts = {s['name']: onp.array(s['points']) for s in sensor_specs}

    # ── Figure 1: 3D scatter ──────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Domain bounding box wireframe
    corners = onp.array([
        [0, 0, 0],
        [domain_x, 0, 0],
        [domain_x, domain_y, 0],
        [0, domain_y, 0],
        [0, 0, domain_z],
        [domain_x, 0, domain_z],
        [domain_x, domain_y, domain_z],
        [0, domain_y, domain_z],
    ])
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(corners[i], corners[j]), color='gray', lw=0.6, alpha=0.4)

    # Selected sensor positions
    for name, pts in sensor_pts.items():
        c = color_map.get(name, 'C5')
        ax.scatter(pts[:, 0],
                   pts[:, 1],
                   pts[:, 2],
                   s=40,
                   color=c,
                   label=name,
                   zorder=5,
                   edgecolors='k',
                   linewidths=0.4)

    ax.set_xlabel('x (µm)', labelpad=4)
    ax.set_ylabel('y (µm)', labelpad=4)
    ax.set_zlabel('z (µm)', labelpad=4)
    ax.set_title(
        f'{sum(len(p) for p in sensor_pts.values())} sensors, POOL={POOL}')
    ax.legend(loc='upper left', fontsize=7, markerscale=1.2)
    fig.tight_layout()

    path_3d = os.path.join(out_dir, 'sensor_positions_3d.png')
    fig.savefig(path_3d, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path_3d}")

    # ── Figure 2: per-face 2D panels ─────────────────────────────────────
    present = [n for n in face_names if n in sensor_pts]
    n_panels = len(present)
    fig2, axes = plt.subplots(1,
                              n_panels,
                              figsize=(3.5 * n_panels, 4),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    extents = {
        'face_x_min': [0, domain_y, 0, domain_z],
        'face_x_max': [0, domain_y, 0, domain_z],
        'face_y_min': [0, domain_x, 0, domain_z],
        'face_y_max': [0, domain_x, 0, domain_z],
        'face_z_max': [0, domain_x, 0, domain_y],
    }

    for ax2, fname in zip(axes, present):
        mask_fn, hcoord, vcoord, xlabel, ylabel = _FACE_DISPLAY[fname]
        mask = mask_fn(solid_mask_3d, cx, cy, cz).astype(float)
        # transpose so horizontal axis = hcoord, vertical = vcoord
        ax2.imshow(mask.T,
                   origin='lower',
                   extent=extents[fname],
                   cmap='Greys',
                   vmin=0,
                   vmax=1,
                   aspect='auto',
                   alpha=0.5)

        # Selected sensors
        if fname in sensor_pts:
            spts = sensor_pts[fname]
            ax2.scatter(spts[:, hcoord],
                        spts[:, vcoord],
                        s=50,
                        color=color_map.get(fname, 'C5'),
                        edgecolors='k',
                        linewidths=0.5,
                        zorder=5,
                        label=f'n={len(spts)}')
            ax2.legend(fontsize=8, loc='upper right')

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(fname.replace('_', ' '))
        ax2.set_xlim(extents[fname][:2])
        ax2.set_ylim(extents[fname][2:])

    fig2.suptitle(
        'Sensor placement on boundary faces (dark = solid, light = void)',
        fontsize=9)
    path_faces = os.path.join(out_dir, 'sensor_positions_faces.png')
    fig2.savefig(path_faces, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {path_faces}")


# ── 4. Generate sensor positions on non-Dirichlet, non-void faces ───────────
# Each face is described by: the solid boundary layer (2D bool array),
# the two free-axis extents, and a function that assembles the 3D point.
def _build_face_table(solid_mask_3d, cx, cy, cz, es, domain_x, domain_y,
                      domain_z):
    """Return ordered dict {name: (loc_fn, layer2d, dim0, dim1, make_pt)}."""
    return {
        'face_x_min': ('side_faces', solid_mask_3d[0, :, :], domain_y,
                       domain_z, cy, cz, lambda u, v: onp.array([0.0, u, v])),
        'face_x_max':
        ('side_faces', solid_mask_3d[cx - 1, :, :], domain_y, domain_z, cy, cz,
         lambda u, v: onp.array([domain_x, u, v])),
        'face_y_min': ('side_faces', solid_mask_3d[:, 0, :], domain_x,
                       domain_z, cx, cz, lambda u, v: onp.array([u, 0.0, v])),
        'face_y_max':
        ('side_faces', solid_mask_3d[:, cy - 1, :], domain_x, domain_z, cx, cz,
         lambda u, v: onp.array([u, domain_y, v])),
        'face_z_max': ('top_face', solid_mask_3d[:, :,
                                                 cz - 1], domain_x, domain_y,
                       cx, cy, lambda u, v: onp.array([u, v, domain_z])),
    }


def _random_sensors(solid_mask_3d, cx, cy, cz, es, domain_x, domain_y,
                    domain_z, n_sensors, rng):
    """Place n_sensors randomly on non-void boundary faces via rejection sampling.

    Sensors are distributed proportionally to the solid area of each face.
    Coordinates are continuous (not snapped to element centres).
    """
    faces = _build_face_table(solid_mask_3d, cx, cy, cz, es, domain_x,
                              domain_y, domain_z)

    # Count solid elements per face for proportional allocation
    solid_counts = {
        name: int(info[1].sum())
        for name, info in faces.items() if info[1].any()
    }
    face_names = list(solid_counts.keys())
    total = sum(solid_counts.values())

    # Proportional allocation with largest-remainder rounding
    raw = {name: n_sensors * solid_counts[name] / total for name in face_names}
    allocated = {name: int(r) for name, r in raw.items()}
    shortfall = n_sensors - sum(allocated.values())
    for name in sorted(face_names,
                       key=lambda n: raw[n] - allocated[n],
                       reverse=True)[:shortfall]:
        allocated[name] += 1

    sensor_specs = []
    for name in face_names:
        n = allocated[name]
        if n == 0:
            continue
        loc_fn, layer, dim0, dim1, n0, n1, make_pt = faces[name]

        # Rejection sampling: draw random (u, v) in [0, dim0] x [0, dim1],
        # accept if the coarse element at that position is solid.
        accepted = []
        while len(accepted) < n:
            batch = max(n * 10, 200)
            u = rng.uniform(0.0, dim0, batch)
            v = rng.uniform(0.0, dim1, batch)
            i0 = onp.minimum((u / es).astype(int), n0 - 1)
            i1 = onp.minimum((v / es).astype(int), n1 - 1)
            mask = layer[i0, i1]
            for uu, vv in zip(u[mask], v[mask]):
                accepted.append(make_pt(uu, vv))
                if len(accepted) == n:
                    break

        accepted = onp.array(accepted.copy())
        sensor_specs.append({
            'name': name,
            'location_fn': loc_fn,
            'points': accepted,
        })

    return sensor_specs


rng = onp.random.default_rng(SEED)
sensor_specs = _random_sensors(solid_mask_3d, Cx, Cy, Cz, elem_size, domain_x,
                               domain_y, domain_z, N_SENSORS, rng)
total_placed = sum(len(s['points']) for s in sensor_specs)
print(f"Placed {total_placed} sensors (seed={SEED}):")
for s in sensor_specs:
    print(f"  {s['name']} ({s['location_fn']}): {len(s['points'])} sensors")

# ── 5. Save sensor configuration to YAML ────────────────────────────────────
sensors_path = os.path.join(OUT_DIR, 'sensors.yml')
with open(sensors_path, 'w') as f:
    yaml.dump(numpy_to_yaml({'sensors': sensor_specs}),
              f,
              default_flow_style=None,
              sort_keys=False)
print(f"\nSensor config saved to {sensors_path}")

# ── 6. Build coarse HEX8 mesh ──────────────────────────────────────────────
x = onp.linspace(0, domain_x, Cx + 1)
y = onp.linspace(0, domain_y, Cy + 1)
z = onp.linspace(0, domain_z, Cz + 1)
xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
all_points = onp.stack((xv, yv, zv), axis=3).reshape(-1, 3)

points_inds = onp.arange(len(all_points)).reshape(Cx + 1, Cy + 1, Cz + 1)
inds1 = points_inds[:-1, :-1, :-1]
inds2 = points_inds[1:, :-1, :-1]
inds3 = points_inds[1:, 1:, :-1]
inds4 = points_inds[:-1, 1:, :-1]
inds5 = points_inds[:-1, :-1, 1:]
inds6 = points_inds[1:, :-1, 1:]
inds7 = points_inds[1:, 1:, 1:]
inds8 = points_inds[:-1, 1:, 1:]
all_cells = onp.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8),
                      axis=3).reshape(-1, 8)

solid_mask = solid_mask_3d.ravel()
cells = all_cells[solid_mask]
E_per_element = E_coarse.ravel()[solid_mask]

used_node_ids = onp.unique(cells.ravel())
old_to_new = onp.full(len(all_points), -1, dtype=onp.int64)
old_to_new[used_node_ids] = onp.arange(len(used_node_ids))
cells = old_to_new[cells]
points = all_points[used_node_ids]

print(f"\nMesh: {len(points)} nodes, {len(cells)} elements")

# ── 7. Problem class and BCs ───────────────────────────────────────────────
_traction_y = [0.0]


class LinearElasticity(Problem):

    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):

        def stress(u_grad, E):
            nu = NU
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(3) + 2. * mu * epsilon
            return sigma

        return stress

    def get_surface_maps(self):

        def surface_map(u, x):
            return np.array([0., _traction_y[0], 0.])

        return [surface_map]

    def set_params(self, params):
        self.internal_vars = [params]


def bottom(point):
    return np.isclose(point[2], 0., atol=1e-3)


def zero_dirichlet_val(point):
    return 0.


def load_face(point):
    return np.isclose(point[1], domain_y, atol=1e-3) * (point[2] > Z_THRES)


def side_faces(point):
    return (np.isclose(point[0], 0., atol=1e-5) +
            np.isclose(point[0], domain_x, atol=1e-5) +
            np.isclose(point[1], 0., atol=1e-5) +
            np.isclose(point[1], domain_y, atol=1e-5))


def top_face(point):
    return np.isclose(point[2], domain_z, atol=1e-5)


dirichlet_bc_info = [
    [bottom, bottom, bottom],
    [0, 1, 2],
    [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val],
]
location_fns = [load_face]

location_fn_map = {
    'side_faces': side_faces,
    'top_face': top_face,
}

# ── 8. Assemble and solve ──────────────────────────────────────────────────
mesh = Mesh(points, cells, ele_type='HEX8')
problem = LinearElasticity(
    mesh,
    vec=3,
    dim=3,
    ele_type='HEX8',
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)

sensor_interpolants = build_sensor_interpolants(problem.fe, sensor_specs,
                                                location_fn_map)

A_loaded = float(onp.sum(problem.nanson_scale[0][:, 0, :]))
_traction_y[0] = TOTAL_FORCE / A_loaded
print(
    f"Load surface area: {A_loaded:.2f} µm², traction: {_traction_y[0]:.6e} GPa"
)

num_quads = problem.fe.num_quads
E_arr = np.repeat(E_per_element[:, None], num_quads, axis=1)

print("Solving FEM...")
fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(E_arr)

# ── 9. Extract sensor displacements ────────────────────────────────────────
sensor_readings = evaluate_sensor_displacements(sol_list[0],
                                                sensor_interpolants)

all_displacements = []
print("\nSensor displacements:")
for reading in sensor_readings:
    u = onp.array(reading['u'])
    all_displacements.append(u)
    print(f"  {reading['name']}: {u.shape[0]} points, "
          f"|u| range [{onp.linalg.norm(u, axis=1).min():.6e}, "
          f"{onp.linalg.norm(u, axis=1).max():.6e}]")

observations = onp.concatenate([u.ravel() for u in all_displacements])
noise = onp.random.default_rng(SEED).normal(0.0,
                                            NOISE_STD,
                                            size=observations.shape)
observations = observations + noise
print(
    f"\nObservation vector: shape {observations.shape}, noise std={NOISE_STD} µm"
)

# ── 10. Save observations ──────────────────────────────────────────────────
obs_path = os.path.join(OUT_DIR, 'observations.dat')
onp.savetxt(obs_path, observations.reshape(1, -1), encoding='utf-8')
print(f"Observations saved to {obs_path}")

print(f"\nDone. {N_SENSORS} sensors, {len(observations)} DOFs "
      f"({N_SENSORS}×3), POOL={POOL}, noise std={NOISE_STD} µm")

if PLOT:
    _plot_sensors(sensor_specs, solid_mask_3d, Cx, Cy, Cz, elem_size, domain_x,
                  domain_y, domain_z, OUT_DIR)
