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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

from jax_fem.solver import solver
from jax_fem.utils import save_sol

from pdmp.rve_utils import (
    validate_fiber_placement,
    generate_multi_fiber_rve_mesh,
    build_periodic_pmat,
    compute_distance_to_nearest_fiber,
    make_eps_macro_q,
    compute_von_mises_from_cell,
    LinearElasticRVE,
)

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


# ── E-field computation ──────────────────────────────────────────────────

def compute_E_field(distances, E_inf, rho, l_scale):
    """Exponential recovery E-field: E(d) = E_inf * (1 - (1-rho) * exp(-d/l))."""
    return E_inf * (1.0 - (1.0 - rho) * np.exp(-distances / l_scale))


# ── Visualisation ────────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────

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
