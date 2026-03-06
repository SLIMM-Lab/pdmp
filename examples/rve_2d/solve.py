"""2D RVE with a circular aggregate — periodic boundary conditions.

Solves 2D plane-strain elastoplasticity (J2 with isotropic hardening) on a
square representative volume element (RVE) containing a single circular
aggregate.  A macroscopic strain is prescribed in incremental load steps
and the periodic fluctuation displacement field is computed at each step.

The problem is formulated as:

    u(x) = eps_macro . x  +  u_tilde(x)

where eps_macro is the prescribed macroscopic strain tensor and u_tilde is
the periodic fluctuation field (unknown).  The weak form becomes

    integral  sigma(eps_macro + sym(grad u_tilde), state) : sym(grad v)  dOmega = 0

for all periodic test functions v.

Usage
-----
    python solve.py                          # default 20 load steps
    python solve.py --n-steps 50             # 50 load steps
    python solve.py --eps-xx 5e-3 --n-steps 40
"""

import argparse
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

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.basis import get_elements
from jax_fem.utils import save_sol

from rve_model import PlaneStrainRVE

# ── Parameters ───────────────────────────────────────────────────────────

L = 1.0                   # RVE side length
R = 0.2                   # aggregate radius
cx, cy = L / 2, L / 2     # aggregate centre

mesh_size = 0.04           # characteristic element length


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


# ── 3. Helper: broadcast eps_macro to quad-point shape ───────────────────

def make_eps_macro_q(eps_macro_voigt, nc, nq):
    """Convert Voigt strain vector to (num_cells, num_quads, 2, 2)."""
    eps_macro = jnp.array([
        [eps_macro_voigt[0],       0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])
    return jnp.broadcast_to(eps_macro[None, None, :, :], (nc, nq, 2, 2))


# ── 4. Post-processing ──────────────────────────────────────────────────

def compute_von_mises_from_cell(sigma_cells, phys_tags, nu_mat, nu_agg):
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

def plot_field(mesh, facecolors, phys_tags, L, R, cx, cy,
               title, label, cmap, fig_path):
    """Plot a per-cell scalar field on the RVE mesh."""
    tri = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, facecolors=facecolors, cmap=cmap)
    fig.colorbar(tc, ax=ax, label=label)
    ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0, ls="--"))
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stress_strain_curve(eps_history, sigma_history, fig_path):
    """Plot volume-averaged stress-strain curve."""
    eps_xx = [e[0, 0] for e in eps_history]
    sig_xx = [s[0, 0] for s in sigma_history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_xx, sig_xx, "o-", lw=1.5, markersize=3)
    ax.set_xlabel(r"$\langle\varepsilon_{xx}\rangle$")
    ax.set_ylabel(r"$\langle\sigma_{xx}\rangle$ [MPa]")
    ax.set_title("Effective stress-strain curve")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(mesh, sol, phys_tags, sigma_cell, L, R, cx, cy,
                 nu_mat, nu_agg, fig_dir, step_label=""):
    """Save overview plots to *fig_dir*."""
    os.makedirs(fig_dir, exist_ok=True)
    sigma_cells_np = np.array(sigma_cell)

    prefix = f"step{step_label}_" if step_label else ""

    # Stress sigma_xx
    plot_field(mesh, sigma_cells_np[:, 0, 0], phys_tags, L, R, cx, cy,
               r"Stress $\sigma_{xx}$", r"$\sigma_{xx}$  [MPa]", "RdBu_r",
               os.path.join(fig_dir, f"{prefix}stress_xx.png"))

    # Stress sigma_yy
    plot_field(mesh, sigma_cells_np[:, 1, 1], phys_tags, L, R, cx, cy,
               r"Stress $\sigma_{yy}$", r"$\sigma_{yy}$  [MPa]", "RdBu_r",
               os.path.join(fig_dir, f"{prefix}stress_yy.png"))

    # Von Mises
    vm = compute_von_mises_from_cell(sigma_cells_np, phys_tags, nu_mat, nu_agg)
    plot_field(mesh, vm, phys_tags, L, R, cx, cy,
               "von Mises stress", r"$\sigma_\mathrm{vM}$  [MPa]", "hot",
               os.path.join(fig_dir, f"{prefix}stress_vonmises.png"))

    # Displacement magnitude
    sol_np = np.array(sol)
    u_mag = np.sqrt(sol_np[:, 0]**2 + sol_np[:, 1]**2)
    tri = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = ax.tripcolor(tri, u_mag, shading="gouraud", cmap="viridis")
    fig.colorbar(tc, ax=ax, label=r"$|\tilde{u}|$")
    ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0, ls="--"))
    ax.set_aspect("equal")
    ax.set_title("Fluctuation displacement")
    fig.savefig(os.path.join(fig_dir, f"{prefix}displacement.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── 6. Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2D RVE with J2 plasticity")
    parser.add_argument("--n-steps", type=int, default=20,
                        help="Number of load steps (default: 20)")
    parser.add_argument("--eps-xx", type=float, default=1e-3,
                        help="Final macroscopic eps_xx (default: 5e-3)")
    parser.add_argument("--eps-yy", type=float, default=0.0,
                        help="Final macroscopic eps_yy (default: 0)")
    parser.add_argument("--gamma-xy", type=float, default=0.0,
                        help="Final macroscopic gamma_xy (default: 0)")
    parser.add_argument("--plot-every", type=int, default=0,
                        help="Plot fields every N steps (0 = final only)")
    args = parser.parse_args()

    ele_type = "TRI3"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    fig_dir  = os.path.join(data_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)

    # ── Material properties ──────────────────────────────────────────
    mat_props = dict(
        E_matrix=30e3,           # Young's modulus, matrix  [MPa]
        nu_matrix=0.2,           # Poisson ratio, matrix
        E_aggregate=60e3,        # Young's modulus, aggregate [MPa]
        nu_aggregate=0.2,        # Poisson ratio, aggregate
        sigma_y_matrix=20.0,     # yield stress, matrix [MPa]
        H_matrix=1e3,            # hardening modulus, matrix [MPa]
        sigma_y_aggregate=1e10,  # effectively elastic
        H_aggregate=0.0,         # irrelevant (never yields)
    )

    eps_macro_final = np.array([args.eps_xx, args.eps_yy, args.gamma_xy])
    n_steps = args.n_steps

    # ── 1. Mesh ──────────────────────────────────────────────────────
    print("Generating mesh ...")
    mesh, phys_tags = generate_rve_mesh(
        L, R, cx, cy, mesh_size, data_dir, ele_type,
    )
    n_matrix = np.sum(phys_tags == 1)
    n_agg    = np.sum(phys_tags == 2)
    print(f"  {len(mesh.points)} nodes, {len(mesh.cells)} elements "
          f"({n_matrix} matrix, {n_agg} aggregate)")

    # ── 2. Periodic constraints ──────────────────────────────────────
    print("Building periodic constraint matrix ...")
    P_mat = build_periodic_pmat(mesh, L, vec=2)
    print(f"  full DOFs: {P_mat.shape[0]},  reduced DOFs: {P_mat.shape[1]}")

    # ── 3. Create problem ────────────────────────────────────────────
    print("Setting up problem ...")
    problem = PlaneStrainRVE(
        mesh, vec=2, dim=2, ele_type=ele_type,
        additional_info=(phys_tags, mat_props),
    )
    problem.P_mat = P_mat

    nc = len(problem.fe.cells)
    nq = problem.fe.num_quads

    # ── 4. Load stepping ─────────────────────────────────────────────
    print(f"\nLoad stepping: {n_steps} steps, "
          f"eps_final = [{eps_macro_final[0]:.2e}, {eps_macro_final[1]:.2e}, "
          f"{eps_macro_final[2]:.2e}]")
    print(f"  sigma_y (matrix) = {mat_props['sigma_y_matrix']:.1f} MPa, "
          f"H (matrix) = {mat_props['H_matrix']:.1f} MPa\n")

    # Extract material constants from internal_vars (set by custom_init)
    E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, _ = problem.internal_vars

    # Initialise
    sol_list = None  # no initial guess for first step
    eps_history = []
    sigma_history = []

    for k in range(1, n_steps + 1):
        # Incremental macroscopic strain
        frac = k / n_steps
        eps_mac_k = frac * eps_macro_final
        eps_macro_q = make_eps_macro_q(eps_mac_k, nc, nq)

        # Update parameters
        params = [E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q]
        problem.set_params(params)

        # Solve (use previous solution as initial guess in reduced space)
        # The solver with P_mat expects dofs in reduced space (M,).
        # sol_list is in full space (N,) — project back via P_mat.T.
        solver_opts = {}
        if sol_list is not None:
            full_dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            reduced_dofs = P_mat.T @ np.array(full_dofs)
            solver_opts['initial_guess'] = [jnp.array(reduced_dofs)]
        sol_list = solver(problem, solver_options=solver_opts)

        # Record stress *before* updating state
        sigma_avg, sigma_cell = problem.compute_avg_stress(
            sol_list[0], problem.internal_vars)

        # Update plastic state
        eps_p_q, alpha_q = problem.update_int_vars_gp(
            sol_list[0], problem.internal_vars)

        # Store history
        eps_macro_tensor = jnp.array([
            [eps_mac_k[0],       0.5 * eps_mac_k[2]],
            [0.5 * eps_mac_k[2], eps_mac_k[1]],
        ])

        eps_history.append(eps_macro_tensor)
        sigma_history.append(sigma_avg)

        # Print progress
        alpha_max = float(jnp.max(alpha_q))
        n_yielded = int(jnp.sum(alpha_q > 0))
        total_qp = nc * nq
        print(f"  Step {k:3d}/{n_steps}  |  "
              f"<sig_xx>={float(sigma_avg[0,0]):8.2f}  "
              f"<sig_yy>={float(sigma_avg[1,1]):8.2f}  |  "
              f"alpha_max={alpha_max:.4e}  "
              f"yielded={n_yielded}/{total_qp} qp")

        # Intermediate plots
        if args.plot_every > 0 and k % args.plot_every == 0:
            plot_results(mesh, sol_list[0], phys_tags, sigma_cell,
                         L, R, cx, cy,
                         mat_props['nu_matrix'], mat_props['nu_aggregate'],
                         fig_dir, step_label=f"{k:03d}")

    print(f"\n Eps hist: {[float(e[0,0]) for e in eps_history]}")
    print(f" Sigma hist: {[float(s[0,0]) for s in sigma_history]}")

    # ── 5. Final post-processing ─────────────────────────────────────
    print("\nFinal results:")
    print(f"  <sigma_xx> = {float(sigma_history[-1][0,0]):.4f} MPa")
    print(f"  <sigma_yy> = {float(sigma_history[-1][1,1]):.4f} MPa")
    print(f"  <sigma_xy> = {float(sigma_history[-1][0,1]):.4f} MPa")

    # Plot final fields
    plot_results(mesh, sol_list[0], phys_tags, sigma_cell,
                 L, R, cx, cy,
                 mat_props['nu_matrix'], mat_props['nu_aggregate'],
                 fig_dir, step_label="final")

    # Stress-strain curve
    plot_stress_strain_curve(
        eps_history, sigma_history,
        os.path.join(fig_dir, "stress_strain_curve.png"))
    print(f"  Figures saved to {fig_dir}/")

    # VTK output (final step)
    vtk_dir = os.path.join(data_dir, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_path = os.path.join(vtk_dir, "rve_final.vtu")
    save_sol(problem.fes[0], sol_list[0], vtk_path)
    print(f"  VTK saved to {vtk_path}")


if __name__ == "__main__":
    main()
