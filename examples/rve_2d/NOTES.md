# 2D RVE Example — Implementation Notes

## Overview

`solve.py` solves 2D plane-strain linear elasticity on a square RVE `[0, L]^2` containing a single circular aggregate, using periodic boundary conditions and a prescribed macroscopic strain.  The purpose is to demonstrate the effect of a spatially heterogeneous material (two phases: matrix + aggregate) on local stress/strain fields.  This is a stepping stone toward propagating a calibrated `ExponentialRecoveryField` (from the existing `JaxFemModel` inverse problem) through a 2D geometry for probabilistic estimates of quantities like maximum strain or stress.

## Mathematical formulation

The total displacement is decomposed as

```
u(x) = eps_macro . x  +  u_tilde(x)
```

where `eps_macro` is the prescribed (known) macroscopic strain tensor and `u_tilde` is the periodic fluctuation (unknown).  Substituting into the equilibrium weak form gives

```
integral  C : (eps_macro + sym(grad u_tilde)) : sym(grad v)  dOmega  =  0
    for all periodic v
```

In practice the macroscopic strain is baked into the constitutive law:

```python
def stress(u_grad, E, nu):
    eps_fluct = 0.5 * (u_grad + u_grad.T)
    eps_total = eps_fluct + eps_macro        # <-- closure captures eps_macro
    sigma = lmbda * tr(eps_total) * I + 2*mu * eps_total
    return sigma
```

jax-fem then solves for `u_tilde` using its Newton solver.  Because the problem is linear, convergence is achieved in a single Newton step.

## Key implementation details

### Mesh generation (`generate_rve_mesh`)

- Uses the **gmsh Python API** to create an unstructured TRI3 mesh with the circular inclusion as a conforming internal boundary (two physical surfaces: matrix=1, aggregate=2).
- `gmsh.model.mesh.setPeriodic(1, slave_curves, master_curves, affine_transform)` forces gmsh to produce **matching node pairs on opposite edges**, which is essential for the periodic P_mat.  The affine transforms are simple translations `(x,y) -> (x+L,y)` and `(x,y) -> (x,y+L)`.
- After meshing, `meshio` reads the `.msh` file.  Physical-group tags per cell are extracted from `cell_data_dict["gmsh:physical"]` and used for material assignment.

### Periodic constraint matrix (`build_periodic_pmat`)

This is a custom implementation (not using jax-fem's built-in `periodic_boundary_conditions` helper) because of a corner-handling subtlety.

The P_mat is a sparse `(N, M)` matrix mapping reduced (independent) DOFs to full DOFs: `u_full = P_mat @ u_reduced`.

**Three categories of DOFs:**

| Category | Treatment in P_mat | Count |
|---|---|---|
| **Interior** (not on any boundary) | Identity row: own column in reduced system | bulk of DOFs |
| **Edge slaves** (right interior -> left interior, top interior -> bottom interior) | Row copies the master's column | ~48 node pairs |
| **Corner pinned** (all 4 corners) | Zero row (DOF forced to 0) | 4 nodes x 2 = 8 DOFs |

**Why corners are pinned via P_mat, not via Dirichlet BCs:**

The initial implementation used a Dirichlet BC at the bottom-left corner (`u = 0`) with the three other corners slaved to it in P_mat.  This produced an incorrect solution (large rigid-body offset ~0.7 instead of ~7e-5 fluctuation).  The root cause: jax-fem applies Dirichlet BCs in the *full* DOF space **before** the P_mat reduction (`res = P^T @ apply_bc(res_full)`).  When a Dirichlet node is also a P_mat master with slaves pointing to it, the slave contributions to the reduced residual contaminate the Dirichlet constraint.

The fix: **remove all four corner DOFs from the reduced system entirely** by giving them zero rows in P_mat.  This simultaneously satisfies corner periodicity (all corners share the same value = 0) and removes the rigid-body translation mode, with no Dirichlet BC needed.

### Problem class (`PlaneStrainRVE`)

- Subclasses `jax_fem.problem.Problem` (`vec=2, dim=2`).
- Material properties (E, nu per phase) are set in `custom_init` via `self.internal_vars = [E_q, nu_q]` — both `(num_cells, num_quads)` arrays.  The `tensor_map` receives `(E, nu)` as scalars at each quadrature point (jax-fem vmaps over cells and quads).
- `set_params(self, params)` is a no-op: the material map is baked in from `phys_tags`.  If you later want to make material properties differentiable (e.g., for inverse problems or uncertainty propagation), modify `set_params` to actually use `params`.
- No `get_surface_maps` or `location_fns`: there are no Neumann (traction) BCs.  All loading comes from `eps_macro` in the `tensor_map`.
- No `dirichlet_bc_info`: the rigid-body mode is handled by P_mat corner pinning (see above).

### Solver

```python
sol_list = solver(problem)
```

Uses jax-fem's default JAX-based Newton solver.  The P_mat is detected automatically via `hasattr(problem, 'P_mat')` and the solver operates in the reduced space:

```
dofs_reduced = zeros(M)
loop:
    dofs_full = P_mat @ dofs_reduced
    res_full = compute_residual(dofs_full)
    res_reduced = P_mat^T @ res_full
    A_reduced = P_mat^T @ A @ P_mat
    delta = solve(A_reduced, -res_reduced)
    dofs_reduced += delta
```

For `ad_wrapper` (JAX-differentiable forward model, needed for future inverse problems), replace:
```python
from jax_fem.solver import ad_wrapper
fwd_pred = ad_wrapper(problem, adjoint_solver_options={"umfpack_solver": {}})
sol_list = fwd_pred(params)   # params passed to set_params
```

### Post-processing

- **Strain**: For TRI3 elements, strain is constant per element (linear shape functions).  `compute_cell_stress_strain` computes the B-matrix analytically from the triangle vertex coordinates and adds `eps_macro` to get the total strain.
- **Stress**: Plane-strain constitutive law with per-element (E, nu) from the physical tags.
- **Von Mises**: Includes `sigma_zz = nu * (sigma_xx + sigma_yy)` from the plane-strain constraint.
- **Effective properties**: Volume-averaged stress and strain using triangle areas as weights.  The reported `C_xxxx = <sigma_xx> / <eps_xx>` is one component of the effective stiffness tensor (not the effective Young's modulus, since `eps_yy` is constrained to zero, not free).

## Current parameters and results

| Parameter | Value |
|---|---|
| RVE side length L | 1.0 |
| Aggregate radius R | 0.2 (volume fraction ~12.6%) |
| E_matrix / nu_matrix | 30 000 MPa / 0.2 |
| E_aggregate / nu_aggregate | 60 000 MPa / 0.2 |
| Mesh size | 0.04 (~829 nodes, ~1556 TRI3 elements) |
| Macroscopic strain | eps_xx = 1e-3 (uniaxial tension in x) |

| Result | Value |
|---|---|
| max \|u_tilde\| | 7.03e-5 |
| <sigma_xx> | 36.03 MPa |
| <sigma_yy> | 9.01 MPa |
| <eps_xx> | 1.000e-3 (recovers prescribed macro strain) |
| C_xxxx effective | 36 031 MPa (between matrix ~33 333 and aggregate ~66 667 plane-strain stiffness) |

## Output files

```
examples/rve_2d/
├── solve.py
├── NOTES.md
└── data/
    ├── msh/rve.msh              # gmsh mesh
    ├── vtk/rve.vtu              # VTK displacement field (for ParaView)
    └── figures/
        ├── mesh.png
        ├── displacement.png
        ├── stress_xx.png
        ├── stress_yy.png
        ├── stress_vonmises.png
        └── strain_xx.png
```

## Next steps toward the full workflow

1. **Integrate with `forward_model.py`**: Wrap the RVE solver as a new `Model` subclass (e.g., `RVEModel`) with `@register_model('RVE2D')`.  The `set_params` method would accept random-field coefficients and evaluate the field at quadrature points (same pattern as `JaxFemModel._eval_obs`).

2. **Connect the ExponentialRecoveryField**: Instead of constant E per phase, use `JaxExponentialRecoveryField` to make the matrix stiffness vary spatially (e.g., recovery from damage with distance from the aggregate interface).  This requires modifying the material assignment logic to evaluate the field at quadrature-point physical coordinates.

3. **Enable AD**: Switch from `solver(problem)` to `ad_wrapper(problem)` so that gradients of observations w.r.t. field parameters are available for PDMP/MCMC samplers.

4. **Multiple aggregates**: Extend `generate_rve_mesh` to place multiple circles (or arbitrary shapes).  gmsh handles boolean operations natively.

5. **Richer element types**: Switch from TRI3 to TRI6 for smoother stress fields (change `ele_type` and gmsh will produce quadratic elements automatically).

6. **Full effective stiffness tensor**: Run three load cases (eps_xx, eps_yy, gamma_xy) and assemble the 3x3 plane-strain stiffness matrix C_eff.

7. **Probabilistic forward propagation**: Sample from the posterior of the calibrated field, run the RVE forward model for each sample, and collect distributions of max stress, effective stiffness, etc.
