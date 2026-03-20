"""
ITZ voxel-to-FEM: converts a 3D phase array (CT scan) into a structured HEX8
FEM mesh, removes pore elements, assigns phase-dependent material properties,
and solves a linear elasticity problem with cantilever-like BCs.
"""
import numpy as onp
import jax
import jax.numpy as np
import os
from scipy.ndimage import label

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh

# ── Phase definitions ────────────────────────────────────────────────────────
PORE = 1
OUTER_CSH = 2
INNER_CSH = 3
ANHYDROUS = 4
AGGREGATE = 7

# Young's modulus per phase (GPa)
E_MAP = {OUTER_CSH: 22.0, INNER_CSH: 30.0, ANHYDROUS: 130.0, AGGREGATE: 70.0}
NU = 0.3

# Voxel geometry
VOXEL_SIZE = 2.0   # µm
Z_THRES = 110.0    # µm — load applied above this z-coordinate
TRACTION = 0.01    # GPa traction on y_max face above Z_THRES

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOM_FILE = os.path.join(SCRIPT_DIR, 'itz_geom.npy')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# ── 1. Load geometry ─────────────────────────────────────────────────────────
geom = onp.load(GEOM_FILE)  # (Nx, Ny, Nz), dtype uint8
Nx, Ny, Nz = geom.shape
domain_x = Nx * VOXEL_SIZE
domain_y = Ny * VOXEL_SIZE
domain_z = Nz * VOXEL_SIZE

print(f"Geometry shape: {geom.shape}")
print(f"Domain size: {domain_x} x {domain_y} x {domain_z} µm")
print(f"Phase counts: "
      f"{ {p: int((geom == p).sum()) for p in [PORE, OUTER_CSH, INNER_CSH, ANHYDROUS, AGGREGATE]} }")

# ── 2. Build structured HEX8 mesh (jax_fem box_mesh convention) ──────────────
x = onp.linspace(0, domain_x, Nx + 1)
y = onp.linspace(0, domain_y, Ny + 1)
z = onp.linspace(0, domain_z, Nz + 1)
xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
all_points = onp.stack((xv, yv, zv), axis=3).reshape(-1, 3)

points_inds = onp.arange(len(all_points)).reshape(Nx + 1, Ny + 1, Nz + 1)
inds1 = points_inds[:-1, :-1, :-1]
inds2 = points_inds[1:,  :-1, :-1]
inds3 = points_inds[1:,  1:,  :-1]
inds4 = points_inds[:-1, 1:,  :-1]
inds5 = points_inds[:-1, :-1, 1:]
inds6 = points_inds[1:,  :-1, 1:]
inds7 = points_inds[1:,  1:,  1:]
inds8 = points_inds[:-1, 1:,  1:]
all_cells = onp.stack((inds1, inds2, inds3, inds4,
                        inds5, inds6, inds7, inds8), axis=3).reshape(-1, 8)

# Phase per element — C-order flattening matches meshgrid indexing='ij'
phases = geom.ravel()

# ── 3. Remove pore elements ──────────────────────────────────────────────────
solid_mask_3d = geom != PORE

# ── 4. Connected component filter (prevent singular stiffness) ───────────────
labels_3d, n_components = label(solid_mask_3d)
if n_components > 1:
    component_sizes = onp.bincount(labels_3d.ravel())[1:]  # skip background=0
    largest = onp.argmax(component_sizes) + 1
    solid_mask_3d = labels_3d == largest
    removed = int((labels_3d > 0).sum() - solid_mask_3d.sum())
    print(f"Connected components: {n_components}, kept largest (label {largest}, "
          f"{int(solid_mask_3d.sum())} voxels, removed {removed})")
else:
    print(f"Single connected component, {int(solid_mask_3d.sum())} solid voxels")

solid_mask = solid_mask_3d.ravel()
cells = all_cells[solid_mask]
cell_phases = phases[solid_mask]

# ── 5. Compact node numbering ────────────────────────────────────────────────
used_node_ids = onp.unique(cells.ravel())
old_to_new = onp.full(len(all_points), -1, dtype=onp.int64)
old_to_new[used_node_ids] = onp.arange(len(used_node_ids))
cells = old_to_new[cells]
points = all_points[used_node_ids]

print(f"Mesh: {len(points)} nodes, {len(cells)} elements")
print(f"Phase distribution: "
      f"{ {int(p): int((cell_phases == p).sum()) for p in onp.unique(cell_phases)} }")

# ── 6. Material properties ───────────────────────────────────────────────────
E_per_element = onp.zeros(len(cell_phases), dtype=onp.float64)
for phase, E_val in E_MAP.items():
    E_per_element[cell_phases == phase] = E_val


# ── 7. Problem class ─────────────────────────────────────────────────────────
class LinearElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(3) + 2. * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., TRACTION, 0.])
        return [surface_map]

    def set_params(self, params):
        self.internal_vars = [params]


# ── 8. Boundary conditions ───────────────────────────────────────────────────
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-3)


def zero_dirichlet_val(point):
    return 0.


def load_face(point):
    return np.isclose(point[1], domain_y, atol=1e-3) * (point[2] > Z_THRES)


dirichlet_bc_info = [
    [bottom, bottom, bottom],
    [0, 1, 2],
    [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val],
]
location_fns = [load_face]

# ── 9. Create mesh, solve, save VTK ─────────────────────────────────────────
mesh = Mesh(points, cells, ele_type='HEX8')
problem = LinearElasticity(
    mesh, vec=3, dim=3, ele_type='HEX8',
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
)

num_quads = problem.fe.num_quads
E_arr = np.repeat(E_per_element[:, None], num_quads, axis=1)

fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(E_arr)

vtk_path = os.path.join(DATA_DIR, 'vtk', 'itz.vtu')
save_sol(problem.fe, sol_list[0], vtk_path,
         cell_infos=[('phase', cell_phases.astype(onp.float32)),
                     ('E', E_per_element.astype(onp.float32))])
print(f"Solution saved to {vtk_path}")
