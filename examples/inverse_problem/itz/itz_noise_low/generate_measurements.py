#!/usr/bin/env python3
"""Generate synthetic observations from all ITZ geometry realizations.

Produces two consistent datasets from a single loop over the geometry files:

  joint/   — all geometries stacked into one observation vector and one
              config.yaml, ready for joint KO inference.
  separate/ — one sub-folder per geometry, each with its own observations
              and config.yaml, ready for independent KO inference runs.
              Comparing the mixture of per-geometry posteriors with the joint
              posterior is the primary scientific goal.

Usage:
    python generate_measurements.py [N_SENSORS_PER_GEOM] [POOL] [--plot]
                                    [--noise-std=S] [--seed=N]

    N_SENSORS_PER_GEOM  sensors per geometry realization (default 5)
    POOL                coarsening factor per axis (default 2, must divide 50,50,110)
    --plot              save sensor position figures for each geometry
    --noise-std=S       Gaussian noise std in microns (default 0.05)
    --seed=N            base RNG seed; geometry i uses seed+i (default 42)
    --recompute         ignore cached FEM solutions and re-run all solves
"""
import copy
import sys
import os
from glob import glob

import numpy as onp
import jax
import jax.numpy as np
import yaml
from scipy.ndimage import label

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh

from pdmp.forward_model import (build_sensor_interpolants,
                                evaluate_sensor_displacements)
from pdmp.loader import numpy_to_yaml, dump_yaml_custom_format
from pdmp.logger_setup import suppress_external_loggers

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
TOTAL_FORCE = 60.0  # mN

# ── CLI arguments ────────────────────────────────────────────────────────────
# Supports both --flag=value and --flag value forms.
PLOT = '--plot' in sys.argv
RECOMPUTE = '--recompute' in sys.argv
_argv = sys.argv[1:]
_flags = {}
_skip_idx = set()
for _i, _a in enumerate(_argv):
    if _a.startswith('--') and _a not in ('--plot', '--recompute'):
        if '=' in _a:
            _k, _v = _a.lstrip('-').split('=', 1)
            _flags[_k] = _v
        elif _i + 1 < len(_argv) and not _argv[_i + 1].startswith('--'):
            _flags[_a.lstrip('-')] = _argv[_i + 1]
            _skip_idx.add(_i + 1)
        else:
            _flags[_a.lstrip('-')] = None

SEED = int(_flags['seed']) if 'seed' in _flags else 42
NOISE_STD = float(_flags['noise-std']) if 'noise-std' in _flags else 0.01
_pos = [
    _a for _i, _a in enumerate(_argv)
    if not _a.startswith('--') and _i not in _skip_idx
]
N_SENSORS_PER_GEOM = int(_pos[0]) if len(_pos) > 0 else 5
POOL = int(_pos[1]) if len(_pos) > 1 else 2

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOM_DIR = os.path.join(SCRIPT_DIR, 'geometries')
JOINT_DIR = os.path.join(SCRIPT_DIR, 'joint')
SEPARATE_DIR = os.path.join(SCRIPT_DIR, 'separate')

CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')

geom_files = sorted(glob(os.path.join(GEOM_DIR, '*.npy')))
if not geom_files:
    raise FileNotFoundError(f"No .npy files found in {GEOM_DIR}")
print(f"Found {len(geom_files)} geometry files")


# ── FEM solution cache ───────────────────────────────────────────────────────
def _cache_path(geom_name):
    return os.path.join(CACHE_DIR, f'{geom_name}_pool{POOL}.npz')


def _load_cached_u(geom_name):
    """Return cached displacement field or None if missing / --recompute set."""
    if RECOMPUTE:
        return None
    path = _cache_path(geom_name)
    if os.path.exists(path):
        return onp.load(path)['u']
    return None


def _save_cached_u(geom_name, u):
    os.makedirs(CACHE_DIR, exist_ok=True)
    onp.savez(_cache_path(geom_name), u=u)


# ── Inference config template (shared by joint and separate) ─────────────────
# Hyperparameter values are calibrated from prior_exploration.py.
_CONFIG_TEMPLATE = {
    'problem': {
        'name': 'Transformed',
        'transformation': 'Affine',
        'distribution': {
            'name': 'BayesianInverse',
            'prior': {
                'name': 'FromField'
            },
            'likelihood': {
                'name':
                'TransformedLikelihood',
                'transformation':
                'Composite',
                'transformations': [
                    {
                        'a': 0.0,
                        'b': 1.0,
                        'type': 'Sigmoid'
                    },
                    'Exponential',
                    {
                        'a': 0.0,
                        'b': 130,
                        'type': 'Sigmoid'
                    },
                    'Identity',
                ],
                'indices': [[0], [1], [2], [3, 4, 5]],
                'likelihood': {
                    'name': 'KOGaussianLikelihood',
                    'observation_file': 'observations.dat',
                    'kernel': 'isotropic',
                    'psi_prior': {
                        'name':
                        'MultivariateNormal',
                        'mean': [-2.64916, -3.06994, 4.66406],
                        'cov': [
                            [0.693147, 0.0, 0.0],
                            [0.0, 0.14842, 0.0],
                            [0.0, 0.0, 0.693147],
                        ],
                    },
                },
            },
            'model': {
                'd_x': 100.0,
                'd_y': 100.0,
                'd_z': 220.0,
                'field': {
                    'coefficient_distribution': {
                        'mean': [0.448315, 4.89906, 0.559249],
                        'cov': [
                            [0.470359, 0.0, 0.0],
                            [0.0, 0.223144, 0.0],
                            [0.0, 0.0, 0.904241],
                        ],
                        'name':
                        'MultivariateNormal',
                    },
                    'infer_f_infinity': True,
                    'idx': 2,
                    'name': 'JaxExponentialRecoveryField',
                },
                'h': 15.0,
                'name': 'JaxFem',
                'nu': 0.18,
                'indenter_loc': 110.0,
                'total_force': 60.0,
                'sensors': [],  # populated per geometry or for joint
            },
        },
    },
    'sampler': {
        'name': 'RandomWalkMetropolis',
        'n_samples': 10000,
        'sigma': 0.94,
        'x_0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    'output': {
        'dir': '.',
        'logging': {
            'level': 'INFO',
            'log_file': 'inference.log',
        },
    },
    'seed': 0,
}


def _build_config(sensor_specs):
    """Return a deep-copied config dict with the given sensor specs."""
    cfg = copy.deepcopy(_CONFIG_TEMPLATE)
    cfg['problem']['distribution']['model']['sensors'] = numpy_to_yaml(
        sensor_specs)
    return cfg


def _strip_geom_prefix(sensor_specs):
    """Strip the 'geomNN_' prefix from sensor names (e.g. 'geom01_face_x_min' → 'face_x_min')."""
    result = []
    for s in sensor_specs:
        s2 = dict(s)
        s2['name'] = s2['name'].split('_', 1)[1]
        result.append(s2)
    return result


# ── Sensor placement helpers ─────────────────────────────────────────────────
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


def _build_face_table(solid_mask_3d, cx, cy, cz, es, domain_x, domain_y,
                      domain_z):
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
    """Place n_sensors randomly on non-void boundary faces (rejection sampling)."""
    faces = _build_face_table(solid_mask_3d, cx, cy, cz, es, domain_x,
                              domain_y, domain_z)

    solid_counts = {
        name: int(info[1].sum())
        for name, info in faces.items() if info[1].any()
    }
    face_names = list(solid_counts.keys())
    total = sum(solid_counts.values())

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

        sensor_specs.append({
            'name': name,
            'location_fn': loc_fn,
            'points': onp.array(accepted),
        })

    return sensor_specs


# ── BCs and location functions ───────────────────────────────────────────────
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


def _make_bc_fns(domain_x, domain_y, domain_z, z_thres):

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-3)

    def zero_dirichlet_val(point):
        return 0.

    def load_face(point):
        return np.isclose(point[1], domain_y, atol=1e-3) * (point[2] > z_thres)

    def side_faces(point):
        return (np.isclose(point[0], 0., atol=1e-5) +
                np.isclose(point[0], domain_x, atol=1e-5) +
                np.isclose(point[1], 0., atol=1e-5) +
                np.isclose(point[1], domain_y, atol=1e-5))

    def top_face(point):
        return np.isclose(point[2], domain_z, atol=1e-5)

    return bottom, zero_dirichlet_val, load_face, side_faces, top_face


# ── Main loop over geometries ────────────────────────────────────────────────
all_sensor_specs = []  # all geoms, for joint config
all_observations = []
suppress_external_loggers()

for geom_idx, geom_path in enumerate(geom_files):
    geom_name = os.path.splitext(os.path.basename(geom_path))[0]  # e.g. '01'
    print(f"\n{'='*60}")
    print(f"Geometry {geom_name} ({geom_idx + 1}/{len(geom_files)})")
    print(f"{'='*60}")

    # ── Load and coarsen geometry ────────────────────────────────────────────
    geom = onp.load(geom_path)
    Nx, Ny, Nz = geom.shape
    domain_x = Nx * VOXEL_SIZE
    domain_y = Ny * VOXEL_SIZE
    domain_z = Nz * VOXEL_SIZE

    assert Nx % POOL == 0 and Ny % POOL == 0 and Nz % POOL == 0, \
        f"POOL={POOL} must divide all geometry dimensions ({Nx}, {Ny}, {Nz})"

    Cx, Cy, Cz = Nx // POOL, Ny // POOL, Nz // POOL
    elem_size = POOL * VOXEL_SIZE
    print(f"Domain: {domain_x:.0f}×{domain_y:.0f}×{domain_z:.0f} µm, "
          f"coarse grid: ({Cx},{Cy},{Cz}), elem_size={elem_size:.1f} µm")

    E_voxel = onp.zeros(geom.shape, dtype=onp.float64)
    for phase, E_val in E_MAP.items():
        E_voxel[geom == phase] = E_val

    E_coarse = E_voxel.reshape(Cx, POOL, Cy, POOL, Cz,
                               POOL).mean(axis=(1, 3, 5))

    solid_mask_3d = E_coarse > 0.0
    labels_3d, n_components = label(solid_mask_3d)
    if n_components > 1:
        component_sizes = onp.bincount(labels_3d.ravel())[1:]
        largest = onp.argmax(component_sizes) + 1
        solid_mask_3d = labels_3d == largest
        print(f"Connected components: {n_components}, kept largest")
    print(f"Solid elements: {int(solid_mask_3d.sum())}")

    # ── Place sensors (fresh random locations for this geometry) ─────────────
    geom_rng = onp.random.default_rng(SEED + geom_idx)
    sensor_specs = _random_sensors(solid_mask_3d, Cx, Cy, Cz, elem_size,
                                   domain_x, domain_y, domain_z,
                                   N_SENSORS_PER_GEOM, geom_rng)

    # Prefix sensor names with geometry index so all specs can coexist in the
    # joint config (the separate configs use clean names without the prefix).
    for s in sensor_specs:
        s['name'] = f'geom{geom_name}_{s["name"]}'

    total_placed = sum(len(s['points']) for s in sensor_specs)
    print(f"Placed {total_placed} sensors (seed={SEED + geom_idx}):")
    for s in sensor_specs:
        print(
            f"  {s['name']} ({s['location_fn']}): {len(s['points'])} sensors")

    # ── Build coarse HEX8 mesh ───────────────────────────────────────────────
    x = onp.linspace(0, domain_x, Cx + 1)
    y = onp.linspace(0, domain_y, Cy + 1)
    z = onp.linspace(0, domain_z, Cz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
    all_points_mesh = onp.stack((xv, yv, zv), axis=3).reshape(-1, 3)

    points_inds = onp.arange(len(all_points_mesh)).reshape(
        Cx + 1, Cy + 1, Cz + 1)
    inds1 = points_inds[:-1, :-1, :-1]
    inds2 = points_inds[1:, :-1, :-1]
    inds3 = points_inds[1:, 1:, :-1]
    inds4 = points_inds[:-1, 1:, :-1]
    inds5 = points_inds[:-1, :-1, 1:]
    inds6 = points_inds[1:, :-1, 1:]
    inds7 = points_inds[1:, 1:, 1:]
    inds8 = points_inds[:-1, 1:, 1:]
    all_cells = onp.stack(
        (inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8),
        axis=3).reshape(-1, 8)

    solid_mask_flat = solid_mask_3d.ravel()
    cells = all_cells[solid_mask_flat]
    E_per_element = E_coarse.ravel()[solid_mask_flat]

    used_node_ids = onp.unique(cells.ravel())
    old_to_new = onp.full(len(all_points_mesh), -1, dtype=onp.int64)
    old_to_new[used_node_ids] = onp.arange(len(used_node_ids))
    cells = old_to_new[cells]
    points = all_points_mesh[used_node_ids]
    print(f"Mesh: {len(points)} nodes, {len(cells)} elements")

    # ── BCs for this domain ──────────────────────────────────────────────────
    bottom_fn, zero_fn, load_face_fn, side_faces_fn, top_face_fn = _make_bc_fns(
        domain_x, domain_y, domain_z, Z_THRES)

    dirichlet_bc_info = [
        [bottom_fn, bottom_fn, bottom_fn],
        [0, 1, 2],
        [zero_fn, zero_fn, zero_fn],
    ]
    location_fns_fem = [load_face_fn]
    location_fn_map_geom = {
        'side_faces': side_faces_fn,
        'top_face': top_face_fn,
    }

    # ── Assemble and solve ───────────────────────────────────────────────────
    mesh = Mesh(points, cells, ele_type='HEX8')
    problem = LinearElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns_fem,
    )

    sensor_interpolants = build_sensor_interpolants(problem.fe, sensor_specs,
                                                    location_fn_map_geom)

    A_loaded = float(onp.sum(problem.nanson_scale[0][:, 0, :]))
    _traction_y[0] = TOTAL_FORCE / A_loaded
    print(
        f"Traction: {_traction_y[0]:.6e} GPa  (loaded area {A_loaded:.2f} µm²)"
    )

    # ── Solve FEM (or load cached displacement field) ────────────────────────
    u_sol = _load_cached_u(geom_name)
    if u_sol is not None:
        print("FEM: loaded from cache.")
    else:
        num_quads = problem.fe.num_quads
        E_arr = np.repeat(E_per_element[:, None], num_quads, axis=1)
        print("Solving FEM...")
        fwd_pred = ad_wrapper(problem)
        sol_list = fwd_pred(E_arr)
        u_sol = onp.array(sol_list[0])
        _save_cached_u(geom_name, u_sol)
        print(f"FEM: solution cached to {_cache_path(geom_name)}")

    # ── Extract sensor displacements and add noise ───────────────────────────
    sensor_readings = evaluate_sensor_displacements(u_sol, sensor_interpolants)

    obs_parts = []
    for reading in sensor_readings:
        u = onp.array(reading['u'])
        obs_parts.append(u.ravel())
        print(f"  {reading['name']}: {u.shape[0]} pts, "
              f"|u| ∈ [{onp.linalg.norm(u, axis=1).min():.4e}, "
              f"{onp.linalg.norm(u, axis=1).max():.4e}] µm")

    obs_i = onp.concatenate(obs_parts)
    noise_rng = onp.random.default_rng(SEED + geom_idx + 1000)
    obs_i = obs_i + noise_rng.normal(0.0, NOISE_STD, obs_i.shape)

    # ── Write separate/NN/rwm/ output ────────────────────────────────────────
    sep_rwm_dir = os.path.join(SEPARATE_DIR, geom_name, 'rwm')
    os.makedirs(sep_rwm_dir, exist_ok=True)

    onp.savetxt(os.path.join(sep_rwm_dir, 'observations.dat'),
                obs_i.reshape(1, -1),
                encoding='utf-8')

    sep_sensor_specs = _strip_geom_prefix(sensor_specs)
    sep_cfg = _build_config(sep_sensor_specs)
    dump_yaml_custom_format(sep_cfg, os.path.join(sep_rwm_dir, 'config.yaml'))
    print(f"  → separate/{geom_name}/rwm/ written "
          f"({obs_i.shape[0]} DOFs, {len(sep_sensor_specs)} sensor groups)")

    all_sensor_specs.extend(sensor_specs)
    all_observations.append(obs_i)

# ── Write joint output ───────────────────────────────────────────────────────
observations = onp.concatenate(all_observations)

joint_obs_path = os.path.join(JOINT_DIR, 'observations.dat')
onp.savetxt(joint_obs_path, observations.reshape(1, -1), encoding='utf-8')
print(f"\nJoint observation vector: shape {observations.shape}  "
      f"({len(geom_files)} geoms × {N_SENSORS_PER_GEOM} sensors × 3 DOF), "
      f"noise std={NOISE_STD} µm")
print(f"Saved {joint_obs_path}")

joint_rwm_dir = os.path.join(JOINT_DIR, 'rwm')
os.makedirs(joint_rwm_dir, exist_ok=True)
joint_rwm_obs_path = os.path.join(joint_rwm_dir, 'observations.dat')
onp.savetxt(joint_rwm_obs_path, observations.reshape(1, -1), encoding='utf-8')

joint_cfg = _build_config(all_sensor_specs)
joint_cfg_path = os.path.join(joint_rwm_dir, 'config.yaml')
dump_yaml_custom_format(joint_cfg, joint_cfg_path)
print(f"Joint config written: {joint_cfg_path}")

print(f"\nDone.")
print(
    f"  joint:    {len(geom_files)} geometries, {len(observations)} total DOFs"
)
print(f"  separate: {len(geom_files)} folders under separate/")
