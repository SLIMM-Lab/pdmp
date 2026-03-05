"""2D RVE with a circular aggregate — periodic boundary conditions.

Solves 2D plane-strain linear elasticity on a square representative volume
element (RVE) containing a single circular aggregate.  A macroscopic strain
is prescribed and the periodic fluctuation displacement field is computed.

The problem is formulated as:

    u(x) = eps_macro . x  +  u_tilde(x)

where eps_macro is the prescribed macroscopic strain tensor and u_tilde is
the periodic fluctuation field (unknown).  The weak form becomes

    integral  C : (eps_macro + sym(grad u_tilde)) : sym(grad v)  dOmega = 0

for all periodic test functions v, which is solved directly by jax-fem with
the macroscopic strain baked into the constitutive law.

Usage
-----
    python solve.py
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import gmsh
import meshio
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.basis import get_elements
from jax_fem.utils import save_sol

# ── Parameters ───────────────────────────────────────────────────────────

L = 1.0                   # RVE side length
R = 0.2                   # aggregate radius
cx, cy = L / 2, L / 2     # aggregate centre

E_matrix = 30e3            # Young's modulus, matrix  [MPa]
nu_matrix = 0.2            # Poisson ratio, matrix
E_aggregate = 60e3         # Young's modulus, aggregate [MPa]
nu_aggregate = 0.2         # Poisson ratio, aggregate

mesh_size = 0.04           # characteristic element length

# Prescribed macroscopic strain  (Voigt: [eps_xx, eps_yy, gamma_xy])
eps_macro_voigt = np.array([1e-3, 0.0, 0.0])   # uniaxial tension in x


# ── 1. Mesh generation ──────────────────────────────────────────────────

def generate_rve_mesh(L, R, cx, cy, mesh_size, data_dir, ele_type="TRI3"):
    """Square RVE [0,L]^2 with a circular inclusion — periodic-ready mesh.

    Uses gmsh's ``setPeriodic`` to guarantee matching node pairs on
    opposite edges, which is required for the periodic P_mat.

    Returns
    -------
    mesh : jax_fem.generate_mesh.Mesh
    phys_tags : ndarray of int, shape (num_cells,)
        1 = matrix, 2 = aggregate.
    """
    _, _, _, _, degree, _ = get_elements(ele_type)
    cell_type = get_meshio_cell_type(ele_type)

    msh_dir = os.path.join(data_dir, "msh")
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, "rve.msh")

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("rve")

    # Square boundary
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(L, L, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, L, 0, mesh_size)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right  = gmsh.model.geo.addLine(p2, p3)
    l_top    = gmsh.model.geo.addLine(p3, p4)
    l_left   = gmsh.model.geo.addLine(p4, p1)

    # Circular aggregate (four arcs)
    pc  = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size)
    pa1 = gmsh.model.geo.addPoint(cx + R, cy, 0, mesh_size)
    pa2 = gmsh.model.geo.addPoint(cx, cy + R, 0, mesh_size)
    pa3 = gmsh.model.geo.addPoint(cx - R, cy, 0, mesh_size)
    pa4 = gmsh.model.geo.addPoint(cx, cy - R, 0, mesh_size)

    c1 = gmsh.model.geo.addCircleArc(pa1, pc, pa2)
    c2 = gmsh.model.geo.addCircleArc(pa2, pc, pa3)
    c3 = gmsh.model.geo.addCircleArc(pa3, pc, pa4)
    c4 = gmsh.model.geo.addCircleArc(pa4, pc, pa1)

    circle_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    square_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])

    # Matrix = square minus circle;  aggregate = circle interior
    s_matrix    = gmsh.model.geo.addPlaneSurface([square_loop, circle_loop])
    s_aggregate = gmsh.model.geo.addPlaneSurface([circle_loop])

    gmsh.model.geo.synchronize()

    # Physical groups (material tags)
    gmsh.model.addPhysicalGroup(2, [s_matrix],    tag=1, name="matrix")
    gmsh.model.addPhysicalGroup(2, [s_aggregate], tag=2, name="aggregate")

    # Periodic meshing  (slave = master + translation)
    # Right edge is the image of Left edge under (x, y) -> (x + L, y)
    gmsh.model.mesh.setPeriodic(
        1, [l_right], [l_left],
        [1, 0, 0, L,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1],
    )
    # Top edge is the image of Bottom edge under (x, y) -> (x, y + L)
    gmsh.model.mesh.setPeriodic(
        1, [l_top], [l_bottom],
        [1, 0, 0, 0,  0, 1, 0, L,  0, 0, 1, 0,  0, 0, 0, 1],
    )

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    # Read back with meshio
    meshio_mesh = meshio.read(msh_file)
    points = meshio_mesh.points[:, :2]               # drop z coordinate
    cells  = meshio_mesh.cells_dict[cell_type]
    phys_tags = meshio_mesh.cell_data_dict["gmsh:physical"][cell_type]

    return Mesh(points, cells, ele_type=ele_type), phys_tags


# ── 2. Periodic constraint matrix ───────────────────────────────────────

def build_periodic_pmat(mesh, L, vec):
    """Sparse P_mat that enforces full periodicity on [0, L]^2.

    Nodes on the right / top edges are slaved to their left / bottom
    counterparts.  All four corners are *pinned* to zero displacement
    directly inside P_mat (their rows are zero), which simultaneously
    satisfies the corner periodicity constraints and removes the
    rigid-body translation mode without needing a separate Dirichlet BC.

    Parameters
    ----------
    mesh : Mesh
    L : float
    vec : int   (DOFs per node, 2 for 2-D elasticity)

    Returns
    -------
    P_mat : scipy.sparse.csr_array, shape (N, M)
        N = total DOFs, M = independent (reduced) DOFs.
    """
    points = mesh.points
    num_nodes = len(points)
    num_dofs  = num_nodes * vec
    EPS = 1e-5

    # Identify boundary node sets
    left   = np.where(np.abs(points[:, 0])     < EPS)[0]
    right  = np.where(np.abs(points[:, 0] - L) < EPS)[0]
    bottom = np.where(np.abs(points[:, 1])     < EPS)[0]
    top    = np.where(np.abs(points[:, 1] - L) < EPS)[0]

    # Corners
    bl = np.intersect1d(left,  bottom)   # (0, 0)
    br = np.intersect1d(right, bottom)   # (L, 0)
    tl = np.intersect1d(left,  top)      # (0, L)
    tr = np.intersect1d(right, top)      # (L, L)
    corners = np.concatenate([bl, br, tl, tr])

    # Interior edge nodes (excluding corners)
    left_int   = np.setdiff1d(left,   corners)
    right_int  = np.setdiff1d(right,  corners)
    bottom_int = np.setdiff1d(bottom, corners)
    top_int    = np.setdiff1d(top,    corners)

    slave_nodes  = []
    master_nodes = []

    # Right interior → matching Left interior  (same y)
    for ri in right_int:
        dists = np.abs(points[left_int, 1] - points[ri, 1])
        best  = np.argmin(dists)
        assert dists[best] < EPS, f"No left match for right node {ri}"
        slave_nodes.append(ri)
        master_nodes.append(left_int[best])

    # Top interior → matching Bottom interior  (same x)
    for ti in top_int:
        dists = np.abs(points[bottom_int, 0] - points[ti, 0])
        best  = np.argmin(dists)
        assert dists[best] < EPS, f"No bottom match for top node {ti}"
        slave_nodes.append(ti)
        master_nodes.append(bottom_int[best])

    slave_nodes  = np.array(slave_nodes)
    master_nodes = np.array(master_nodes)

    # Expand to DOFs
    slave_dofs  = np.concatenate([slave_nodes  * vec + v for v in range(vec)])
    master_dofs = np.concatenate([master_nodes * vec + v for v in range(vec)])

    # Corner DOFs: pinned to zero (rows of P will be zero)
    pinned_dofs = np.concatenate([corners * vec + v for v in range(vec)])

    is_slave  = np.zeros(num_dofs, dtype=bool)
    is_pinned = np.zeros(num_dofs, dtype=bool)
    master_of = np.zeros(num_dofs, dtype=int)

    is_slave[slave_dofs]   = True
    is_pinned[pinned_dofs] = True
    for s, m in zip(slave_dofs, master_dofs):
        master_of[s] = m

    is_free = ~(is_slave | is_pinned)
    M = int(is_free.sum())

    reduced_idx = np.full(num_dofs, -1, dtype=int)
    reduced_idx[is_free] = np.arange(M)

    # Assemble sparse P  (full_dof = P @ reduced_dof)
    # Pinned rows are absent → zero rows in CSR.
    I_list, J_list, V_list = [], [], []
    for i in range(num_dofs):
        if is_pinned[i]:
            continue                         # zero row → u = 0
        elif is_slave[i]:
            I_list.append(i)
            J_list.append(reduced_idx[master_of[i]])
            V_list.append(1.0)
        else:
            I_list.append(i)
            J_list.append(reduced_idx[i])
            V_list.append(1.0)

    P_mat = scipy.sparse.csr_array(
        (np.array(V_list), (np.array(I_list), np.array(J_list))),
        shape=(num_dofs, M),
    )
    return P_mat


# ── 3. Problem definition ───────────────────────────────────────────────

def make_rve_problem(mesh, phys_tags, eps_macro_voigt,
                     E_mat, nu_mat, E_agg, nu_agg,
                     ele_type="TRI3"):
    """Create the jax-fem Problem for the periodic RVE.

    The macroscopic strain is captured in the tensor_map closure so the
    solver finds the periodic fluctuation u_tilde directly.
    """

    eps_macro = jnp.array([
        [eps_macro_voigt[0],       0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])

    # Pre-compute material fields  (num_cells, num_quads)
    # (evaluated once; not a function of solver parameters)
    E_cell  = np.where(phys_tags == 2, E_agg, E_mat).astype(np.float64)
    nu_cell = np.where(phys_tags == 2, nu_agg, nu_mat).astype(np.float64)

    class PlaneStrainRVE(Problem):

        def custom_init(self):
            self.fe = self.fes[0]
            nq = self.fe.num_quads
            E_q  = jnp.tile(E_cell[:, None],  (1, nq))
            nu_q = jnp.tile(nu_cell[:, None], (1, nq))
            self.internal_vars = [E_q, nu_q]

        def get_tensor_map(self):
            def stress(u_grad, E, nu):
                mu    = E / (2.0 * (1.0 + nu))
                lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                eps_fluct = 0.5 * (u_grad + u_grad.T)
                eps_total = eps_fluct + eps_macro
                sigma = (lmbda * jnp.trace(eps_total) * jnp.eye(2)
                         + 2.0 * mu * eps_total)
                return sigma
            return stress

        def set_params(self, params):
            # Material map is baked in via custom_init — nothing to update.
            pass

    # Rigid-body translation is removed by pinning all four corners
    # to zero inside P_mat (see build_periodic_pmat), so no Dirichlet
    # BCs are needed here.
    problem = PlaneStrainRVE(
        mesh, vec=2, dim=2, ele_type=ele_type,
    )
    return problem


# ── 4. Post-processing ──────────────────────────────────────────────────

def compute_cell_stress_strain(mesh, sol, phys_tags, eps_macro_voigt,
                               E_mat, nu_mat, E_agg, nu_agg):
    """Per-element stress and strain for TRI3 (constant-strain triangle).

    Returns
    -------
    centroids : (num_cells, 2)
    eps_cells : (num_cells, 2, 2)   total strain tensor
    sigma_cells : (num_cells, 2, 2) Cauchy stress tensor (in-plane)
    """
    points = mesh.points
    cells  = mesh.cells
    num_cells = len(cells)

    eps_macro = np.array([
        [eps_macro_voigt[0],       0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])

    cell_coords = points[cells]                       # (nc, 3, 2)
    cell_disp   = sol[cells]                           # (nc, 3, 2)
    centroids   = cell_coords.mean(axis=1)             # (nc, 2)

    # Shape-function gradients for TRI3 (constant per element)
    x1, y1 = cell_coords[:, 0, 0], cell_coords[:, 0, 1]
    x2, y2 = cell_coords[:, 1, 0], cell_coords[:, 1, 1]
    x3, y3 = cell_coords[:, 2, 0], cell_coords[:, 2, 1]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)   # 2 * area

    dNdx = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det[:, None]
    dNdy = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det[:, None]

    # Fluctuation displacement gradient
    du_dx = np.einsum("cn,cn->c", cell_disp[:, :, 0], dNdx)
    du_dy = np.einsum("cn,cn->c", cell_disp[:, :, 0], dNdy)
    dv_dx = np.einsum("cn,cn->c", cell_disp[:, :, 1], dNdx)
    dv_dy = np.einsum("cn,cn->c", cell_disp[:, :, 1], dNdy)

    # Total strain = fluctuation + macroscopic
    eps_xx = du_dx + eps_macro[0, 0]
    eps_yy = dv_dy + eps_macro[1, 1]
    eps_xy = 0.5 * (du_dy + dv_dx) + eps_macro[0, 1]

    eps_cells = np.zeros((num_cells, 2, 2))
    eps_cells[:, 0, 0] = eps_xx
    eps_cells[:, 1, 1] = eps_yy
    eps_cells[:, 0, 1] = eps_xy
    eps_cells[:, 1, 0] = eps_xy

    # Stress (plane strain: sigma_zz = lmbda * tr(eps), but not stored here)
    E_c  = np.where(phys_tags == 2, E_agg, E_mat)
    nu_c = np.where(phys_tags == 2, nu_agg, nu_mat)
    mu_c    = E_c / (2.0 * (1.0 + nu_c))
    lmbda_c = E_c * nu_c / ((1.0 + nu_c) * (1.0 - 2.0 * nu_c))

    tr_eps = eps_xx + eps_yy
    sigma_cells = np.zeros((num_cells, 2, 2))
    sigma_cells[:, 0, 0] = lmbda_c * tr_eps + 2.0 * mu_c * eps_xx
    sigma_cells[:, 1, 1] = lmbda_c * tr_eps + 2.0 * mu_c * eps_yy
    sigma_cells[:, 0, 1] = 2.0 * mu_c * eps_xy
    sigma_cells[:, 1, 0] = 2.0 * mu_c * eps_xy

    return centroids, eps_cells, sigma_cells


def compute_von_mises(sigma_cells, phys_tags, nu_mat, nu_agg):
    """Von Mises stress accounting for plane-strain sigma_zz."""
    s11 = sigma_cells[:, 0, 0]
    s22 = sigma_cells[:, 1, 1]
    s12 = sigma_cells[:, 0, 1]
    nu_c = np.where(phys_tags == 2, nu_agg, nu_mat)
    # plane strain: sigma_zz = nu * (sigma_xx + sigma_yy)
    s33 = nu_c * (s11 + s22)
    vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2
                         + (s33 - s11)**2 + 6.0 * s12**2))
    return vm


# ── 5. Visualisation ────────────────────────────────────────────────────

def plot_results(mesh, sol, phys_tags, eps_cells, sigma_cells, vm,
                 L, R, cx, cy, fig_dir):
    """Save overview plots to *fig_dir*."""
    os.makedirs(fig_dir, exist_ok=True)
    points = mesh.points
    cells  = mesh.cells

    tri = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    def _add_circle(ax):
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0,
                            ls="--"))

    # ── Mesh + phases ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(phys_tags == 2, 1.0, 0.0)
    ax.tripcolor(tri, facecolors=colors, cmap="coolwarm", alpha=0.5)
    ax.triplot(tri, "k-", lw=0.2)
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title("Mesh and material phases")
    fig.savefig(os.path.join(fig_dir, "mesh.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Fluctuation displacement magnitude ───────────────────────────
    u_mag = np.sqrt(sol[:, 0]**2 + sol[:, 1]**2)
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, u_mag, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax, label=r"$|\tilde{u}|$")
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title("Fluctuation displacement")
    fig.savefig(os.path.join(fig_dir, "displacement.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Stress sigma_xx ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=sigma_cells[:, 0, 0], cmap="RdBu_r")
    fig.colorbar(tc, ax=ax, label=r"$\sigma_{xx}$  [MPa]")
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title(r"Stress $\sigma_{xx}$")
    fig.savefig(os.path.join(fig_dir, "stress_xx.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Stress sigma_yy ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=sigma_cells[:, 1, 1], cmap="RdBu_r")
    fig.colorbar(tc, ax=ax, label=r"$\sigma_{yy}$  [MPa]")
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title(r"Stress $\sigma_{yy}$")
    fig.savefig(os.path.join(fig_dir, "stress_yy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Von Mises stress ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=vm, cmap="hot")
    fig.colorbar(tc, ax=ax, label=r"$\sigma_\mathrm{vM}$  [MPa]")
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title("von Mises stress")
    fig.savefig(os.path.join(fig_dir, "stress_vonmises.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Strain eps_xx ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=eps_cells[:, 0, 0], cmap="RdBu_r")
    fig.colorbar(tc, ax=ax, label=r"$\varepsilon_{xx}$")
    _add_circle(ax)
    ax.set_aspect("equal"); ax.set_title(r"Total strain $\varepsilon_{xx}$")
    fig.savefig(os.path.join(fig_dir, "strain_xx.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


# ── 6. Main ─────────────────────────────────────────────────────────────

def main():
    ele_type = "TRI3"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    fig_dir  = os.path.join(data_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Mesh
    print("Generating mesh ...")
    mesh, phys_tags = generate_rve_mesh(
        L, R, cx, cy, mesh_size, data_dir, ele_type,
    )
    n_matrix = np.sum(phys_tags == 1)
    n_agg    = np.sum(phys_tags == 2)
    print(f"  {len(mesh.points)} nodes, {len(mesh.cells)} elements "
          f"({n_matrix} matrix, {n_agg} aggregate)")

    # 2. Periodic constraints
    print("Building periodic constraint matrix ...")
    P_mat = build_periodic_pmat(mesh, L, vec=2)
    print(f"  full DOFs: {P_mat.shape[0]},  reduced DOFs: {P_mat.shape[1]}")

    # 3. Assemble problem
    print("Setting up problem ...")
    problem = make_rve_problem(
        mesh, phys_tags, eps_macro_voigt,
        E_matrix, nu_matrix, E_aggregate, nu_aggregate,
        ele_type=ele_type,
    )
    problem.P_mat = P_mat

    # 4. Solve
    print("Solving ...")
    sol_list = solver(problem)
    sol = np.array(sol_list[0])          # (num_nodes, 2)
    u_max = np.max(np.linalg.norm(sol, axis=1))
    print(f"  max |u_tilde| = {u_max:.6e}")

    # 5. Post-process
    print("Post-processing ...")
    centroids, eps_cells, sigma_cells = compute_cell_stress_strain(
        mesh, sol, phys_tags, eps_macro_voigt,
        E_matrix, nu_matrix, E_aggregate, nu_aggregate,
    )
    vm = compute_von_mises(sigma_cells, phys_tags, nu_matrix, nu_aggregate)

    # Volume-averaged quantities (cell area as weight)
    coords = mesh.points[mesh.cells]
    areas = 0.5 * np.abs(
        (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1])
      - (coords[:, 2, 0] - coords[:, 0, 0]) * (coords[:, 1, 1] - coords[:, 0, 1])
    )
    total_area = areas.sum()
    sig_avg = np.einsum("c,cij->ij", areas, sigma_cells) / total_area
    eps_avg = np.einsum("c,cij->ij", areas, eps_cells)   / total_area

    print(f"\n  Volume-averaged stress [MPa]:")
    print(f"    sigma_xx = {sig_avg[0,0]:.4f}   sigma_yy = {sig_avg[1,1]:.4f}"
          f"   sigma_xy = {sig_avg[0,1]:.4f}")
    print(f"  Volume-averaged strain:")
    print(f"    eps_xx   = {eps_avg[0,0]:.6f}   eps_yy   = {eps_avg[1,1]:.6f}"
          f"   eps_xy   = {eps_avg[0,1]:.6f}")

    C_eff_xxxx = sig_avg[0, 0] / eps_avg[0, 0] if eps_avg[0, 0] != 0 else 0
    print(f"\n  Effective stiffness  C_xxxx = <sigma_xx> / <eps_xx>"
          f" = {C_eff_xxxx:.1f} MPa")
    print(f"  (Matrix stiffness = {E_matrix:.0f},  "
          f"aggregate stiffness = {E_aggregate:.0f})")

    # 6. Plots
    plot_results(mesh, sol, phys_tags, eps_cells, sigma_cells, vm,
                 L, R, cx, cy, fig_dir)

    # 7. VTK output
    vtk_dir = os.path.join(data_dir, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_path = os.path.join(vtk_dir, "rve.vtu")
    save_sol(problem.fes[0], sol_list[0], vtk_path)
    print(f"  VTK saved to {vtk_path}")


if __name__ == "__main__":
    main()
