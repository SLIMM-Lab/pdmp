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
E_MAP = {OUTER_CSH: 25.0, INNER_CSH: 31.0, ANHYDROUS: 130.0, AGGREGATE: 70.0}
NU = 0.18
ST_MAP = {OUTER_CSH: 58., INNER_CSH: 92, ANHYDROUS: 683., AGGREGATE: 70.0}
NU_MAP = {}

# Voxel geometry
VOXEL_SIZE = 2.0   # µm
Z_THRES = 110.0    # µm — load applied above this z-coordinate
TOTAL_FORCE = 40.0  # mN (GPa·µm² consistent units) — total y-force on load face

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
_traction_y = [0.0]  # computed after problem construction

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

A_loaded = float(onp.sum(problem.nanson_scale[0][:, 0, :]))
_traction_y[0] = TOTAL_FORCE / A_loaded
print(f"Load surface area: {A_loaded:.2f} µm², traction: {_traction_y[0]:.6e} GPa")

num_quads = problem.fe.num_quads
E_arr = np.repeat(E_per_element[:, None], num_quads, axis=1)

fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(E_arr)

# ── 10. Post-process: strains, stresses, von Mises ──────────────────────────
# u_grads: (num_cells, num_quads, 3, 3)
u_grads = problem.fe.sol_to_grad(sol_list[0])

# Strain (symmetric part of grad u): (num_cells, num_quads, 3, 3)
epsilon = 0.5 * (u_grads + np.swapaxes(u_grads, -1, -2))

# Per-quad Lamé parameters from element E and constant NU
lmbda = E_arr * NU / ((1. + NU) * (1. - 2. * NU))   # (num_cells, num_quads)
mu    = E_arr / (2. * (1. + NU))                      # (num_cells, num_quads)

# Stress: (num_cells, num_quads, 3, 3)
tr_eps = epsilon[:, :, 0, 0] + epsilon[:, :, 1, 1] + epsilon[:, :, 2, 2]
sigma = (lmbda[:, :, None, None] * tr_eps[:, :, None, None] * np.eye(3)
         + 2. * mu[:, :, None, None] * epsilon)

# JxW-weighted cell averages
JxW = problem.fe.JxW                                  # (num_cells, num_quads)
w   = JxW / JxW.sum(axis=1, keepdims=True)            # normalised weights

eps_cell   = onp.array(np.sum(epsilon * w[:, :, None, None], axis=1))  # (nc, 3, 3)
sigma_cell = onp.array(np.sum(sigma   * w[:, :, None, None], axis=1))  # (nc, 3, 3)

s11, s22, s33 = sigma_cell[:, 0, 0], sigma_cell[:, 1, 1], sigma_cell[:, 2, 2]
s12, s13, s23 = sigma_cell[:, 0, 1], sigma_cell[:, 0, 2], sigma_cell[:, 1, 2]
von_mises = onp.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                             + 6. * (s12**2 + s13**2 + s23**2)))

vtk_path = os.path.join(DATA_DIR, 'vtk', 'itz.vtu')
save_sol(problem.fe, sol_list[0], vtk_path,
         cell_infos=[
             ('phase',      cell_phases.astype(onp.float32)),
             ('E',          E_per_element.astype(onp.float32)),
             ('stress_xx',  s11.astype(onp.float32)),
             ('stress_yy',  s22.astype(onp.float32)),
             ('stress_zz',  s33.astype(onp.float32)),
             ('stress_xy',  s12.astype(onp.float32)),
             ('stress_xz',  s13.astype(onp.float32)),
             ('stress_yz',  s23.astype(onp.float32)),
             ('von_mises',  von_mises.astype(onp.float32)),
             ('strain_xx',  eps_cell[:, 0, 0].astype(onp.float32)),
             ('strain_yy',  eps_cell[:, 1, 1].astype(onp.float32)),
             ('strain_zz',  eps_cell[:, 2, 2].astype(onp.float32)),
             ('strain_xy',  (2. * eps_cell[:, 0, 1]).astype(onp.float32)),
             ('strain_xz',  (2. * eps_cell[:, 0, 2]).astype(onp.float32)),
             ('strain_yz',  (2. * eps_cell[:, 1, 2]).astype(onp.float32)),
         ])
print(f"Solution saved to {vtk_path}")
