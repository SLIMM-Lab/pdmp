"""
ITZ voxel-to-FEM with configurable coarsening: pools POOL³ voxels into one
HEX8 element using Voigt-averaged stiffness (pore voxels contribute E=0).

Usage:
    python itz_fem_coarse.py [POOL]

    POOL  coarsening factor per axis (default 2).
          Must divide all geometry dimensions.
          Valid for (50,50,110): 1, 2, 5, 10.
"""
import sys
import numpy as onp
import jax
import jax.numpy as np
import os
from scipy.ndimage import label
from scipy.stats import mode

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh

from pdmp.forward_model import build_sensor_interpolants, evaluate_sensor_displacements

# ── Phase definitions ────────────────────────────────────────────────────────
PORE = 1
OUTER_CSH = 2
INNER_CSH = 3
ANHYDROUS = 4
AGGREGATE = 7

# Young's modulus per phase (GPa)
E_MAP = {OUTER_CSH: 25.0, INNER_CSH: 31.0, ANHYDROUS: 99.0, AGGREGATE: 70.0}
NU = 0.18

# Voxel geometry
VOXEL_SIZE = 2.0  # µm
Z_THRES = 110.0  # µm — load applied above this z-coordinate
TOTAL_FORCE = 40.0  # mN (GPa·µm² consistent units) — total y-force on load face

# Coarsening factor
POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 2

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOM_FILE = os.path.join(SCRIPT_DIR, 'itz_geom.npy')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# ── 1. Load geometry ────────────────────────────────────────────────────────
geom = onp.load(GEOM_FILE)  # (Nx, Ny, Nz), dtype uint8
Nx, Ny, Nz = geom.shape
domain_x = Nx * VOXEL_SIZE
domain_y = Ny * VOXEL_SIZE
domain_z = Nz * VOXEL_SIZE

assert Nx % POOL == 0 and Ny % POOL == 0 and Nz % POOL == 0, \
    f"POOL={POOL} must divide all geometry dimensions ({Nx}, {Ny}, {Nz})"

print(f"Geometry shape: {geom.shape}")
print(f"Domain size: {domain_x} x {domain_y} x {domain_z} µm")

# ── 2. Coarsen: pool POOL³ voxels → 1 element ───────────────────────────────
Cx, Cy, Cz = Nx // POOL, Ny // POOL, Nz // POOL
print(f"Coarsening: POOL={POOL}, coarse grid: ({Cx}, {Cy}, {Cz}), "
      f"{Nx*Ny*Nz} → {Cx*Cy*Cz} elements ({POOL**3}x reduction)")

# Map each voxel to its E value (pore → 0)
E_voxel = onp.zeros(geom.shape, dtype=onp.float64)
for phase, E_val in E_MAP.items():
    E_voxel[geom == phase] = E_val

# Voigt average over each POOL³ block
E_coarse = E_voxel.reshape(Cx, POOL, Cy, POOL, Cz, POOL).mean(axis=(1, 3, 5))

# Dominant phase per block (mode of non-pore voxels; fallback to PORE if all pore)
geom_blocks = geom.reshape(Cx, POOL, Cy, POOL, Cz, POOL)
geom_blocks_flat = geom_blocks.transpose(0, 2, 4, 1, 3,
                                         5).reshape(Cx, Cy, Cz, POOL**3)
# Mask out pore voxels for mode computation
masked = onp.where(geom_blocks_flat == PORE, -1, geom_blocks_flat)
dominant_phase = onp.zeros((Cx, Cy, Cz), dtype=onp.int32)
for ix in range(Cx):
    for iy in range(Cy):
        for iz in range(Cz):
            solid = masked[ix, iy, iz]
            solid = solid[solid >= 0]
            if len(solid) > 0:
                dominant_phase[ix, iy,
                               iz] = int(mode(solid, keepdims=False).mode)
            else:
                dominant_phase[ix, iy, iz] = PORE

# Pore fraction per block
pore_count = (geom_blocks_flat == PORE).sum(axis=-1)
pore_fraction = pore_count / POOL**3

# ── 3. Remove all-pore elements ─────────────────────────────────────────────
solid_mask_3d = E_coarse > 0.0

# ── 4. Connected component filter ───────────────────────────────────────────
labels_3d, n_components = label(solid_mask_3d)
if n_components > 1:
    component_sizes = onp.bincount(labels_3d.ravel())[1:]
    largest = onp.argmax(component_sizes) + 1
    solid_mask_3d = labels_3d == largest
    removed = int((labels_3d > 0).sum() - solid_mask_3d.sum())
    print(
        f"Connected components: {n_components}, kept largest (label {largest}, "
        f"{int(solid_mask_3d.sum())} elements, removed {removed})")
else:
    print(
        f"Single connected component, {int(solid_mask_3d.sum())} solid elements"
    )

# ── 5. Build structured HEX8 mesh on coarse grid ────────────────────────────
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
cell_phases = dominant_phase.ravel()[solid_mask]
cell_pore_frac = pore_fraction.ravel()[solid_mask]

# ── 6. Compact node numbering ───────────────────────────────────────────────
used_node_ids = onp.unique(cells.ravel())
old_to_new = onp.full(len(all_points), -1, dtype=onp.int64)
old_to_new[used_node_ids] = onp.arange(len(used_node_ids))
cells = old_to_new[cells]
points = all_points[used_node_ids]

print(f"Mesh: {len(points)} nodes, {len(cells)} elements")
print(f"E range: [{E_per_element.min():.2f}, {E_per_element.max():.2f}] GPa")

# ── 7. Problem class ────────────────────────────────────────────────────────
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


# ── 8. Boundary conditions ──────────────────────────────────────────────────
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

sensor_specs = [
    {
        "name":
        "sensor_left_center",
        "location_fn":
        "side_faces",
        "points":
        onp.array([[0, 0.5 * domain_y, 0.5 * domain_z],
                   [0, 0.2 * domain_y, 0.8 * domain_z]])
    },
    {
        "name": "sensor_top",
        "location_fn": "top_face",
        "point": onp.array([0.4 * domain_x, 0.3 * domain_y, domain_z])
    },
]

# ── 9. Create mesh, solve ───────────────────────────────────────────────────
mesh = Mesh(points, cells, ele_type='HEX8')
problem = LinearElasticity(
    mesh,
    vec=3,
    dim=3,
    ele_type='HEX8',
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)

sensor_interpolants = build_sensor_interpolants(
    problem.fe, sensor_specs, location_fn_map) if sensor_specs else []

A_loaded = float(onp.sum(problem.nanson_scale[0][:, 0, :]))
_traction_y[0] = TOTAL_FORCE / A_loaded
print(
    f"Load surface area: {A_loaded:.2f} µm², traction: {_traction_y[0]:.6e} GPa"
)

num_quads = problem.fe.num_quads
E_arr = np.repeat(E_per_element[:, None], num_quads, axis=1)

fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(E_arr)

# ── 10. Post-process: strains, stresses, von Mises ─────────────────────────
u_grads = problem.fe.sol_to_grad(sol_list[0])
epsilon = 0.5 * (u_grads + np.swapaxes(u_grads, -1, -2))

lmbda = E_arr * NU / ((1. + NU) * (1. - 2. * NU))
mu = E_arr / (2. * (1. + NU))

tr_eps = epsilon[:, :, 0, 0] + epsilon[:, :, 1, 1] + epsilon[:, :, 2, 2]
sigma = (lmbda[:, :, None, None] * tr_eps[:, :, None, None] * np.eye(3) +
         2. * mu[:, :, None, None] * epsilon)

JxW = problem.fe.JxW
w = JxW / JxW.sum(axis=1, keepdims=True)

eps_cell = onp.array(np.sum(epsilon * w[:, :, None, None], axis=1))
sigma_cell = onp.array(np.sum(sigma * w[:, :, None, None], axis=1))

s11, s22, s33 = sigma_cell[:, 0, 0], sigma_cell[:, 1, 1], sigma_cell[:, 2, 2]
s12, s13, s23 = sigma_cell[:, 0, 1], sigma_cell[:, 0, 2], sigma_cell[:, 1, 2]
von_mises = onp.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 +
                            (s33 - s11)**2 + 6. * (s12**2 + s13**2 + s23**2)))

vtk_path = os.path.join(DATA_DIR, 'vtk', f'itz_coarse_{POOL}.vtu')
save_sol(problem.fe,
         sol_list[0],
         vtk_path,
         cell_infos=[
             ('phase', cell_phases.astype(onp.float32)),
             ('E', E_per_element.astype(onp.float32)),
             ('pore_frac', cell_pore_frac.astype(onp.float32)),
             ('stress_xx', s11.astype(onp.float32)),
             ('stress_yy', s22.astype(onp.float32)),
             ('stress_zz', s33.astype(onp.float32)),
             ('stress_xy', s12.astype(onp.float32)),
             ('stress_xz', s13.astype(onp.float32)),
             ('stress_yz', s23.astype(onp.float32)),
             ('von_mises', von_mises.astype(onp.float32)),
             ('strain_xx', eps_cell[:, 0, 0].astype(onp.float32)),
             ('strain_yy', eps_cell[:, 1, 1].astype(onp.float32)),
             ('strain_zz', eps_cell[:, 2, 2].astype(onp.float32)),
             ('strain_xy', (2. * eps_cell[:, 0, 1]).astype(onp.float32)),
             ('strain_xz', (2. * eps_cell[:, 0, 2]).astype(onp.float32)),
             ('strain_yz', (2. * eps_cell[:, 1, 2]).astype(onp.float32)),
         ])

if sensor_interpolants:
    sensor_readings = evaluate_sensor_displacements(sol_list[0],
                                                    sensor_interpolants)
    print("Sensor displacements (ux, uy, uz):")
    for reading in sensor_readings:
        print(f"  {reading['name']} @ {reading['points']} -> {reading['u']}")

print(f"Solution saved to {vtk_path}")
