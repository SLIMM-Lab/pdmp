#!/usr/bin/env python3
"""Gradient test: AD vs finite differences for the RVE exponential field model.

Verifies that JAX automatic differentiation (via ad_wrapper) correctly
computes gradients of a stress-based cost function w.r.t. the exponential
recovery field parameters rho and l_scale.

Usage
-----
    python test_gradient.py
    python test_gradient.py --h 1e-5
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp

from jax_fem.solver import ad_wrapper

from pdmp.rve_utils import (
    generate_multi_fiber_rve_mesh,
    build_periodic_pmat,
    compute_distance_to_nearest_fiber,
    make_eps_macro_q,
    validate_fiber_placement,
    LinearElasticRVE,
)
from solve import DEFAULT_FIBERS, L, E_inf, E_fiber, nu_matrix, nu_fiber, mesh_size

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="AD vs FD gradient test")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--l-scale", type=float, default=0.1)
    parser.add_argument("--h", type=float, default=1e-5,
                        help="FD step size (default: 1e-5)")
    parser.add_argument("--mesh-size", type=float, default=mesh_size)
    args = parser.parse_args()

    fibers = DEFAULT_FIBERS
    ele_type = "TRI3"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── 1. Setup mesh and problem ─────────────────────────────────────
    validate_fiber_placement(fibers, L, args.mesh_size)
    print("Generating mesh ...")
    mesh, phys_tags = generate_multi_fiber_rve_mesh(
        L, fibers, args.mesh_size, data_dir, l_scale=args.l_scale,
        ele_type=ele_type)
    print(f"  {len(mesh.points)} nodes, {len(mesh.cells)} elements")

    print("Building periodic constraint matrix ...")
    P_mat = build_periodic_pmat(mesh, L, vec=2)

    mat_props = dict(
        E_matrix=E_inf, nu_matrix=nu_matrix,
        E_aggregate=E_fiber, nu_aggregate=nu_fiber,
    )
    problem = LinearElasticRVE(
        mesh, vec=2, dim=2, ele_type=ele_type,
        additional_info=(phys_tags, mat_props),
    )
    problem.P_mat = P_mat

    nc = len(problem.fe.cells)
    nq = problem.fe.num_quads

    # ── 2. Precompute constants ───────────────────────────────────────
    quad_points = np.array(problem.physical_quad_points)
    distances = compute_distance_to_nearest_fiber(quad_points, fibers)
    distances_jnp = jnp.array(distances)

    is_fiber = (phys_tags == 2)
    is_fiber_q = jnp.array(np.broadcast_to(is_fiber[:, None], (nc, nq)))

    nu_q = jnp.array(np.where(
        is_fiber[:, None], nu_fiber, nu_matrix,
    ) * np.ones((nc, nq)))

    eps_macro_voigt = np.array([1e-3, 0.0, 0.0])
    eps_macro_q = make_eps_macro_q(eps_macro_voigt, nc, nq)

    # ── 3. AD wrapper ─────────────────────────────────────────────────
    print("Creating ad_wrapper ...")
    fwd_pred = ad_wrapper(problem, adjoint_solver_options={"umfpack_solver": {}})

    # ── 4. Differentiable cost function ───────────────────────────────
    def cost_fn(rho, l_scale):
        # E field from parameters (JAX ops — traced by autodiff)
        E_matrix_q = E_inf * (1.0 - (1.0 - rho) * jnp.exp(-distances_jnp / l_scale))
        E_q = jnp.where(is_fiber_q, E_fiber, E_matrix_q)

        params = [E_q, nu_q, eps_macro_q]
        problem.set_params(params)
        sol = fwd_pred(params)[0]

        _, sigma_cell = problem.compute_avg_stress(sol, params)
        return jnp.sum(sigma_cell**2)

    # ── 5. Evaluate AD gradient ───────────────────────────────────────
    rho_val = jnp.float64(args.rho)
    l_val = jnp.float64(args.l_scale)

    print(f"\nComputing AD gradient at rho={args.rho}, l_scale={args.l_scale} ...")
    cost, (grad_rho_ad, grad_l_ad) = jax.value_and_grad(cost_fn, argnums=(0, 1))(rho_val, l_val)
    print(f"  cost = {float(cost):.6e}")
    print(f"  dJ/d(rho)     [AD] = {float(grad_rho_ad):.6e}")
    print(f"  dJ/d(l_scale) [AD] = {float(grad_l_ad):.6e}")

    # ── 6. Finite differences ─────────────────────────────────────────
    h = args.h
    print(f"\nComputing FD gradient (central, h={h}) ...")

    cost_rho_p = cost_fn(rho_val + h, l_val)
    cost_rho_m = cost_fn(rho_val - h, l_val)
    grad_rho_fd = (float(cost_rho_p) - float(cost_rho_m)) / (2 * h)

    cost_l_p = cost_fn(rho_val, l_val + h)
    cost_l_m = cost_fn(rho_val, l_val - h)
    grad_l_fd = (float(cost_l_p) - float(cost_l_m)) / (2 * h)

    print(f"  dJ/d(rho)     [FD] = {grad_rho_fd:.6e}")
    print(f"  dJ/d(l_scale) [FD] = {grad_l_fd:.6e}")

    # ── 7. Comparison ─────────────────────────────────────────────────
    def rel_err(ad, fd):
        denom = max(abs(ad), abs(fd), 1e-30)
        return abs(ad - fd) / denom

    err_rho = rel_err(float(grad_rho_ad), grad_rho_fd)
    err_l = rel_err(float(grad_l_ad), grad_l_fd)

    print(f"\n{'Parameter':<12} {'AD':>14} {'FD':>14} {'rel. error':>12}")
    print("-" * 54)
    print(f"{'rho':<12} {float(grad_rho_ad):>14.6e} {grad_rho_fd:>14.6e} {err_rho:>12.2e}")
    print(f"{'l_scale':<12} {float(grad_l_ad):>14.6e} {grad_l_fd:>14.6e} {err_l:>12.2e}")

    tol = 1e-3
    passed = err_rho < tol and err_l < tol
    print(f"\nTolerance: {tol:.0e}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
