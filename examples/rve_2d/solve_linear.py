"""2D RVE with a circular aggregate — linear elastic, periodic BCs.

Solves 2D plane-strain linear elasticity on a square RVE containing a
single circular aggregate under a prescribed macroscopic strain.

Usage
-----
    python solve_linear.py
    python solve_linear.py --eps-xx 5e-3
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from rve_model import LinearElasticRVE

# Re-use mesh/BC helpers from solve.py
from solve_j2 import (
    generate_rve_mesh,
    build_periodic_pmat,
    make_eps_macro_q,
    plot_results,
    L, R, cx, cy, mesh_size,
)


def main():
    parser = argparse.ArgumentParser(description="2D RVE — linear elasticity")
    parser.add_argument("--eps-xx",   type=float, default=1e-3)
    parser.add_argument("--eps-yy",   type=float, default=0.0)
    parser.add_argument("--gamma-xy", type=float, default=0.0)
    args = parser.parse_args()

    ele_type = "TRI3"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data_linear")
    fig_dir  = os.path.join(data_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)

    mat_props = dict(
        E_matrix=30e3,       # [MPa]
        nu_matrix=0.2,
        E_aggregate=60e3,    # [MPa]
        nu_aggregate=0.2,
    )

    eps_macro_voigt = np.array([args.eps_xx, args.eps_yy, args.gamma_xy])

    # ── 1. Mesh ──────────────────────────────────────────────────────
    print("Generating mesh ...")
    mesh, phys_tags = generate_rve_mesh(
        L, R, cx, cy, mesh_size, data_dir, ele_type,
    )
    print(f"  {len(mesh.points)} nodes, {len(mesh.cells)} elements")

    # ── 2. Periodic constraints ──────────────────────────────────────
    print("Building periodic constraint matrix ...")
    P_mat = build_periodic_pmat(mesh, L, vec=2)
    print(f"  full DOFs: {P_mat.shape[0]},  reduced DOFs: {P_mat.shape[1]}")

    # ── 3. Create problem and solve ──────────────────────────────────
    print("Setting up problem ...")
    problem = LinearElasticRVE(
        mesh, vec=2, dim=2, ele_type=ele_type,
        additional_info=(phys_tags, mat_props),
    )
    problem.P_mat = P_mat

    nc = len(problem.fe.cells)
    nq = problem.fe.num_quads
    E_q, nu_q, _ = problem.internal_vars

    eps_macro_q = make_eps_macro_q(eps_macro_voigt, nc, nq)
    problem.set_params([E_q, nu_q, eps_macro_q])

    print(f"Solving for eps = [{eps_macro_voigt[0]:.2e}, "
          f"{eps_macro_voigt[1]:.2e}, {eps_macro_voigt[2]:.2e}] ...")
    sol_list = solver(problem)

    sigma_avg, sigma_cell = problem.compute_avg_stress(
        sol_list[0], problem.internal_vars)

    # ── 4. Results ───────────────────────────────────────────────────
    print(f"  <sigma_xx> = {float(sigma_avg[0,0]):.4f} MPa")
    print(f"  <sigma_yy> = {float(sigma_avg[1,1]):.4f} MPa")
    print(f"  <sigma_xy> = {float(sigma_avg[0,1]):.4f} MPa")

    plot_results(mesh, sol_list[0], phys_tags, sigma_cell,
                 L, R, cx, cy,
                 mat_props['nu_matrix'], mat_props['nu_aggregate'],
                 fig_dir)

    # Total displacement: u_total = u_tilde + eps_macro . x
    eps_macro_tensor = np.array([
        [eps_macro_voigt[0],       0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])
    u_fluct = np.array(sol_list[0])
    u_total = u_fluct + mesh.points @ eps_macro_tensor.T

    tri = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    for component, label, fname in [
        (u_total[:, 0], r"$u_x^{\mathrm{total}}$", "displacement_total_x.png"),
        (u_total[:, 1], r"$u_y^{\mathrm{total}}$", "displacement_total_y.png"),
        (np.sqrt(u_total[:, 0]**2 + u_total[:, 1]**2),
         r"$|u^{\mathrm{total}}|$", "displacement_total_mag.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        tc = ax.tripcolor(tri, component, shading="gouraud", cmap="viridis")
        fig.colorbar(tc, ax=ax, label=label)
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=1.0, ls="--"))
        ax.set_aspect("equal")
        ax.set_title(f"Total displacement {label}")
        fig.savefig(os.path.join(fig_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    vtk_dir = os.path.join(data_dir, "vtk")
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_path = os.path.join(vtk_dir, "rve_linear.vtu")
    save_sol(problem.fes[0], sol_list[0], vtk_path)
    print(f"  Figures saved to {fig_dir}/")
    print(f"  VTK saved to {vtk_path}")


if __name__ == "__main__":
    main()
