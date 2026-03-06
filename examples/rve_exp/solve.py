"""2D RVE with exponential recovery E-field — multiple fibers, periodic BCs.

The matrix Young's modulus varies spatially based on distance from the
nearest fiber surface, following an exponential recovery profile:

    E(d) = E_inf * (1 - (1-rho) * exp(-d/l))

where d is the distance to the nearest fiber surface, E_inf is the
far-field matrix modulus, rho is the recovery ratio (E at fiber surface
= E_inf * rho), and l is the recovery length scale.

Usage
-----
    python solve.py
    python solve.py --rho 0.5 --l-scale 0.05
    python solve.py --eps-xx 5e-3 --mesh-size 0.02
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import gmsh
import meshio
import scipy.sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.basis import get_elements
from jax_fem.utils import save_sol

from rve_model import LinearElasticRVE

# ── Default parameters ────────────────────────────────────────────────────

L = 1.0           # RVE side length
mesh_size = 0.03  # Characteristic element length

# Default fiber layout: (cx, cy, R)
DEFAULT_FIBERS = [
    (0.25, 0.25, 0.10),
    (0.70, 0.30, 0.08),
    (0.30, 0.75, 0.07),
    (0.75, 0.70, 0.12),
]

E_inf = 30e3       # Far-field matrix E [MPa]
E_fiber = 200e3    # Fiber E [MPa]
nu_matrix = 0.35   # Matrix Poisson ratio
nu_fiber = 0.2     # Fiber Poisson ratio


# ── 1. Validation ─────────────────────────────────────────────────────────

def validate_fiber_placement(fibers, L, mesh_size):
    """Check that fibers don't overlap each other or intersect boundaries."""
    for i, (cx, cy, R) in enumerate(fibers):
        if cx - R < mesh_size or cx + R > L - mesh_size:
            raise ValueError(
                f"Fiber {i} (cx={cx}, R={R}) too close to x-boundary")
        if cy - R < mesh_size or cy + R > L - mesh_size:
            raise ValueError(
                f"Fiber {i} (cy={cy}, R={R}) too close to y-boundary")

    for i in range(len(fibers)):
        for j in range(i + 1, len(fibers)):
            cx_i, cy_i, R_i = fibers[i]
            cx_j, cy_j, R_j = fibers[j]
            dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            min_dist = R_i + R_j + mesh_size
            if dist < min_dist:
                raise ValueError(
                    f"Fibers {i} and {j} overlap or too close: "
                    f"dist={dist:.4f} < {min_dist:.4f}")


# ── 2. Mesh generation ───────────────────────────────────────────────────

def generate_multi_fiber_rve_mesh(L, fibers, mesh_size, data_dir,
                                  l_scale=0.1, ele_type="TRI3"):
    """Square RVE [0,L]^2 with multiple circular fibers — periodic-ready mesh.

    Returns
    -------
    mesh : jax_fem.generate_mesh.Mesh
    phys_tags : ndarray of int, shape (num_cells,)
        1 = matrix, 2 = fiber.
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

    square_loop = gmsh.model.geo.addCurveLoop(
        [l_bottom, l_right, l_top, l_left])

    # Create all fiber circles
    circle_loops = []
    fiber_surfaces = []
    all_fiber_curves = []

    for cx, cy, R in fibers:
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
        circle_loops.append(circle_loop)
        all_fiber_curves.extend([c1, c2, c3, c4])

    # Matrix = square minus all circles
    s_matrix = gmsh.model.geo.addPlaneSurface(
        [square_loop] + circle_loops)

    # Each fiber interior
    for cl in circle_loops:
        s_fiber = gmsh.model.geo.addPlaneSurface([cl])
        fiber_surfaces.append(s_fiber)

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [s_matrix], tag=1)
    gmsh.model.setPhysicalName(2, 1, "matrix")
    gmsh.model.addPhysicalGroup(2, fiber_surfaces, tag=2)
    gmsh.model.setPhysicalName(2, 2, "fiber")

    # Mesh refinement near fiber boundaries
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", all_fiber_curves)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", mesh_size / 2)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", l_scale)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Periodic meshing
    gmsh.model.mesh.setPeriodic(
        1, [l_right], [l_left],
        [1, 0, 0, L,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1],
    )
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
    points = meshio_mesh.points[:, :2]
    cells  = meshio_mesh.cells_dict[cell_type]
    phys_tags = meshio_mesh.cell_data_dict["gmsh:physical"][cell_type]

    return Mesh(points, cells, ele_type=ele_type), phys_tags


# ── 3. Periodic constraint matrix ────────────────────────────────────────

def build_periodic_pmat(mesh, L, vec):
    """Sparse P_mat that enforces full periodicity on [0, L]^2.

    Nodes on the right / top edges are slaved to their left / bottom
    counterparts.  All four corners are pinned to zero displacement.
    """
    points = mesh.points
    num_nodes = len(points)
    num_dofs  = num_nodes * vec
    EPS = 1e-5

    left   = np.where(np.abs(points[:, 0])     < EPS)[0]
    right  = np.where(np.abs(points[:, 0] - L) < EPS)[0]
    bottom = np.where(np.abs(points[:, 1])     < EPS)[0]
    top    = np.where(np.abs(points[:, 1] - L) < EPS)[0]

    bl = np.intersect1d(left,  bottom)
    br = np.intersect1d(right, bottom)
    tl = np.intersect1d(left,  top)
    tr = np.intersect1d(right, top)
    corners = np.concatenate([bl, br, tl, tr])

    left_int   = np.setdiff1d(left,   corners)
    right_int  = np.setdiff1d(right,  corners)
    bottom_int = np.setdiff1d(bottom, corners)
    top_int    = np.setdiff1d(top,    corners)

    slave_nodes  = []
    master_nodes = []

    for ri in right_int:
        dists = np.abs(points[left_int, 1] - points[ri, 1])
        best  = np.argmin(dists)
        assert dists[best] < EPS, f"No left match for right node {ri}"
        slave_nodes.append(ri)
        master_nodes.append(left_int[best])

    for ti in top_int:
        dists = np.abs(points[bottom_int, 0] - points[ti, 0])
        best  = np.argmin(dists)
        assert dists[best] < EPS, f"No bottom match for top node {ti}"
        slave_nodes.append(ti)
        master_nodes.append(bottom_int[best])

    slave_nodes  = np.array(slave_nodes)
    master_nodes = np.array(master_nodes)

    slave_dofs  = np.concatenate([slave_nodes  * vec + v for v in range(vec)])
    master_dofs = np.concatenate([master_nodes * vec + v for v in range(vec)])

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

    I_list, J_list, V_list = [], [], []
    for i in range(num_dofs):
        if is_pinned[i]:
            continue
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


# ── 4. Helper: broadcast eps_macro to quad-point shape ───────────────────

def make_eps_macro_q(eps_macro_voigt, nc, nq):
    """Convert Voigt strain vector to (num_cells, num_quads, 2, 2)."""
    eps_macro = jnp.array([
        [eps_macro_voigt[0],       0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])
    return jnp.broadcast_to(eps_macro[None, None, :, :], (nc, nq, 2, 2))


# ── 5. Distance and E-field computation ──────────────────────────────────

def compute_distance_to_nearest_fiber(quad_points, fibers):
    """Compute minimum distance from each quad point to nearest fiber surface.

    Parameters
    ----------
    quad_points : ndarray (nc, nq, 2)
    fibers : list of (cx, cy, R)

    Returns
    -------
    distances : ndarray (nc, nq)
        Distance to nearest fiber surface, clamped at 0.
    """
    nc, nq, _ = quad_points.shape
    distances = np.full((nc, nq), np.inf)

    for cx, cy, R in fibers:
        d = np.sqrt((quad_points[:, :, 0] - cx)**2
                    + (quad_points[:, :, 1] - cy)**2) - R
        distances = np.minimum(distances, d)

    return np.maximum(distances, 0.0)


def compute_E_field(distances, E_inf, rho, l_scale):
    """Exponential recovery E-field: E(d) = E_inf * (1 - (1-rho) * exp(-d/l))."""
    return E_inf * (1.0 - (1.0 - rho) * np.exp(-distances / l_scale))


# ── 6. Post-processing ──────────────────────────────────────────────────

def compute_von_mises_from_cell(sigma_cells, nu_cell):
    """Von Mises stress accounting for plane-strain sigma_zz."""
    s11 = sigma_cells[:, 0, 0]
    s22 = sigma_cells[:, 1, 1]
    s12 = sigma_cells[:, 0, 1]
    s33 = nu_cell * (s11 + s22)
    vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2
                         + (s33 - s11)**2 + 6.0 * s12**2))
    return vm


# ── 7. Visualisation ────────────────────────────────────────────────────

def plot_field(mesh, facecolors, fibers, L,
               title, label, cmap, fig_path):
    """Plot a per-cell scalar field on the RVE mesh with multiple fiber circles."""
    tri = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=facecolors, cmap=cmap)
    fig.colorbar(tc, ax=ax, label=label)
    for cx, cy, R in fibers:
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0, ls="--"))
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_displacement(mesh, sol, fibers, L, fig_dir, prefix=""):
    """Plot fluctuation displacement magnitude."""
    sol_np = np.array(sol)
    u_mag = np.sqrt(sol_np[:, 0]**2 + sol_np[:, 1]**2)
    tri = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, u_mag, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax, label=r"$|\tilde{u}|$")
    for cx, cy, R in fibers:
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0, ls="--"))
    ax.set_aspect("equal")
    ax.set_title("Fluctuation displacement")
    fig.savefig(os.path.join(fig_dir, f"{prefix}displacement.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 8. Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2D RVE with exponential recovery E-field")
    parser.add_argument("--rho", type=float, default=0.3,
                        help="Recovery ratio (default: 0.3)")
    parser.add_argument("--l-scale", type=float, default=0.1,
                        help="Recovery length scale (default: 0.1)")
    parser.add_argument("--eps-xx", type=float, default=1e-3,
                        help="Macroscopic eps_xx (default: 1e-3)")
    parser.add_argument("--eps-yy", type=float, default=0.0,
                        help="Macroscopic eps_yy (default: 0)")
    parser.add_argument("--gamma-xy", type=float, default=0.0,
                        help="Macroscopic gamma_xy (default: 0)")
    parser.add_argument("--mesh-size", type=float, default=mesh_size,
                        help=f"Mesh size (default: {mesh_size})")
    args = parser.parse_args()

    rho = args.rho
    l_scale = args.l_scale
    ms = args.mesh_size
    fibers = DEFAULT_FIBERS

    ele_type = "TRI3"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    fig_dir  = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Validate fiber placement ──────────────────────────────────
    validate_fiber_placement(fibers, L, ms)
    print(f"Fiber layout: {len(fibers)} fibers")
    for i, (cx, cy, R) in enumerate(fibers):
        print(f"  Fiber {i}: center=({cx:.2f}, {cy:.2f}), R={R:.3f}")

    # ── 2. Mesh ──────────────────────────────────────────────────────
    print("\nGenerating mesh ...")
    mesh, phys_tags = generate_multi_fiber_rve_mesh(
        L, fibers, ms, data_dir, l_scale=l_scale, ele_type=ele_type)
    n_matrix = np.sum(phys_tags == 1)
    n_fiber  = np.sum(phys_tags == 2)
    print(f"  {len(mesh.points)} nodes, {len(mesh.cells)} elements "
          f"({n_matrix} matrix, {n_fiber} fiber)")

    # ── 3. Periodic constraints ──────────────────────────────────────
    print("Building periodic constraint matrix ...")
    P_mat = build_periodic_pmat(mesh, L, vec=2)
    print(f"  full DOFs: {P_mat.shape[0]},  reduced DOFs: {P_mat.shape[1]}")

    # ── 4. Create problem (initial uniform E) ────────────────────────
    mat_props = dict(
        E_matrix=E_inf,
        nu_matrix=nu_matrix,
        E_aggregate=E_fiber,
        nu_aggregate=nu_fiber,
    )
    problem = LinearElasticRVE(
        mesh, vec=2, dim=2, ele_type=ele_type,
        additional_info=(phys_tags, mat_props),
    )
    problem.P_mat = P_mat

    nc = len(problem.fe.cells)
    nq = problem.fe.num_quads

    # ── 5. Compute distance field ────────────────────────────────────
    quad_points = np.array(problem.physical_quad_points)  # (nc, nq, 2)
    distances = compute_distance_to_nearest_fiber(quad_points, fibers)

    print(f"\nDistance field stats:")
    print(f"  min={distances.min():.4f}, max={distances.max():.4f}, "
          f"mean={distances.mean():.4f}")

    # ── 6. Compute E field ───────────────────────────────────────────
    E_field_values = compute_E_field(distances, E_inf, rho, l_scale)

    # Fiber cells keep constant E_fiber, matrix cells get the field E
    is_fiber = (phys_tags == 2)
    E_q_np = np.tile(E_field_values, 1)  # copy
    E_q_np[is_fiber, :] = E_fiber

    # nu is constant per phase
    nu_q_np = np.where(
        is_fiber[:, None],
        nu_fiber,
        nu_matrix,
    ) * np.ones((nc, nq))

    print(f"\nE field stats (matrix only):")
    matrix_E = E_q_np[~is_fiber]
    print(f"  min={matrix_E.min():.1f}, max={matrix_E.max():.1f}, "
          f"mean={matrix_E.mean():.1f} MPa")
    print(f"  E at fiber surface = E_inf * rho = {E_inf * rho:.1f} MPa")
    print(f"  E far field = E_inf = {E_inf:.1f} MPa")

    # ── 7. Set params and solve ──────────────────────────────────────
    eps_macro_voigt = np.array([args.eps_xx, args.eps_yy, args.gamma_xy])
    eps_macro_q = make_eps_macro_q(eps_macro_voigt, nc, nq)

    E_q = jnp.array(E_q_np)
    nu_q = jnp.array(nu_q_np)
    problem.set_params([E_q, nu_q, eps_macro_q])

    print(f"\nSolving for eps = [{eps_macro_voigt[0]:.2e}, "
          f"{eps_macro_voigt[1]:.2e}, {eps_macro_voigt[2]:.2e}] ...")
    sol_list = solver(problem)

    # ── 8. Post-processing ───────────────────────────────────────────
    sigma_avg, sigma_cell = problem.compute_avg_stress(
        sol_list[0], problem.internal_vars)

    print(f"\nVolume-averaged stress:")
    print(f"  <sigma_xx> = {float(sigma_avg[0,0]):.4f} MPa")
    print(f"  <sigma_yy> = {float(sigma_avg[1,1]):.4f} MPa")
    print(f"  <sigma_xy> = {float(sigma_avg[0,1]):.4f} MPa")

    # ── 9. Plots ─────────────────────────────────────────────────────
    # Cell-averaged distance and E for plotting
    JxW = np.array(problem.fe.JxW)
    w = JxW / JxW.sum(axis=1, keepdims=True)
    dist_cell = np.sum(distances * w, axis=1)
    E_cell = np.sum(E_q_np * w, axis=1)

    plot_field(mesh, dist_cell, fibers, L,
               "Distance to nearest fiber", "distance", "viridis",
               os.path.join(fig_dir, "distance_field.png"))

    plot_field(mesh, E_cell, fibers, L,
               "Young's modulus E", "E [MPa]", "viridis",
               os.path.join(fig_dir, "E_field.png"))

    sigma_cell_np = np.array(sigma_cell)
    nu_cell_arr = np.sum(nu_q_np * w, axis=1)

    plot_field(mesh, sigma_cell_np[:, 0, 0], fibers, L,
               r"Stress $\sigma_{xx}$", r"$\sigma_{xx}$ [MPa]", "RdBu_r",
               os.path.join(fig_dir, "stress_xx.png"))

    plot_field(mesh, sigma_cell_np[:, 1, 1], fibers, L,
               r"Stress $\sigma_{yy}$", r"$\sigma_{yy}$ [MPa]", "RdBu_r",
               os.path.join(fig_dir, "stress_yy.png"))

    vm = compute_von_mises_from_cell(sigma_cell_np, nu_cell_arr)
    plot_field(mesh, vm, fibers, L,
               "von Mises stress", r"$\sigma_\mathrm{vM}$ [MPa]", "hot",
               os.path.join(fig_dir, "stress_vonmises.png"))

    plot_displacement(mesh, sol_list[0], fibers, L, fig_dir)

    print(f"  Figures saved to {fig_dir}/")

    # ── 10. VTK output ───────────────────────────────────────────────
    eps_cell = np.array(problem.compute_avg_strain(sol_list[0], eps_macro_q))
    s11 = sigma_cell_np[:, 0, 0]
    s22 = sigma_cell_np[:, 1, 1]
    s12 = sigma_cell_np[:, 0, 1]
    s33 = nu_cell_arr * (s11 + s22)
    von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2
                                + (s33 - s11)**2 + 6.0 * s12**2))

    vtk_dir = os.path.join(data_dir, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_path = os.path.join(vtk_dir, "rve_exp.vtu")
    save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[
        ('E_field',    E_cell),
        ('distance',   dist_cell),
        ('stress_xx',  s11),
        ('stress_yy',  s22),
        ('stress_xy',  s12),
        ('von_mises',  von_mises),
        ('strain_xx',  eps_cell[:, 0, 0]),
        ('strain_yy',  eps_cell[:, 1, 1]),
        ('strain_xy',  2.0 * eps_cell[:, 0, 1]),
        ('phys_tag',   phys_tags.astype(np.float64)),
    ])
    print(f"  VTK saved to {vtk_path}")


if __name__ == "__main__":
    main()
