"""PlaneStrainRVE — periodic 2-D RVE with J2 plasticity.

Problem subclass for jax-fem that implements:
- 2D plane-strain linear elasticity with J2 isotropic hardening
- Macroscopic strain loading via ``set_params``
- ``get_maps()`` dual-return: tensor_map + update_int_vars_map
- ``update_int_vars_gp()`` for per-quad-point plastic state update
- ``compute_avg_stress()`` for volume-averaged stress recording

The J2 return mapping is closed-form (radial return) and fully
differentiable via standard JAX autodiff (no Newton solver or
``custom_jvp`` needed).

Usage
-----
    from rve_model import PlaneStrainRVE

    problem = PlaneStrainRVE(mesh, vec=2, dim=2, ele_type="TRI3",
                             additional_info=(phys_tags, mat_props))
    problem.set_params([E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q])
    sol_list = solver(problem)
    sigma_avg = problem.compute_avg_stress(sol_list[0], problem.internal_vars)
    eps_p_new, alpha_new = problem.update_int_vars_gp(sol_list[0], problem.internal_vars)
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.problem import Problem


class PlaneStrainRVE(Problem):
    """2-D plane-strain RVE with J2 plasticity and periodic BCs.

    Parameters
    ----------
    mesh, vec, dim, ele_type :
        Standard jax-fem Problem arguments.
    additional_info : tuple (phys_tags, mat_props)
        phys_tags : ndarray (num_cells,)
            Physical group tag per cell (1 = matrix, 2 = aggregate).
        mat_props : dict with keys
            ``E_matrix``, ``nu_matrix``, ``E_aggregate``, ``nu_aggregate``,
            ``sigma_y_matrix``, ``H_matrix``,
            ``sigma_y_aggregate``, ``H_aggregate``.
    """

    def custom_init(self, phys_tags, mat_props):
        self.fe = self.fes[0]
        nq = self.fe.num_quads
        nc = len(self.fe.cells)

        # --- Material constants (num_cells, num_quads) ---
        is_agg = (phys_tags == 2)
        E_cell = np.where(is_agg, mat_props['E_aggregate'],
                          mat_props['E_matrix']).astype(np.float64)
        nu_cell = np.where(is_agg, mat_props['nu_aggregate'],
                           mat_props['nu_matrix']).astype(np.float64)
        sigma_y_cell = np.where(is_agg, mat_props['sigma_y_aggregate'],
                                mat_props['sigma_y_matrix']).astype(np.float64)
        H_cell = np.where(is_agg, mat_props['H_aggregate'],
                          mat_props['H_matrix']).astype(np.float64)

        E_q = jnp.tile(E_cell[:, None], (1, nq))
        nu_q = jnp.tile(nu_cell[:, None], (1, nq))
        sigma_y_q = jnp.tile(sigma_y_cell[:, None], (1, nq))
        H_q = jnp.tile(H_cell[:, None], (1, nq))

        # --- Evolving state (num_cells, num_quads, ...) ---
        # eps_p: plastic strain tensor (2x2, symmetric)
        eps_p_q = jnp.zeros((nc, nq, 2, 2))
        # alpha: accumulated plastic strain (scalar)
        alpha_q = jnp.zeros((nc, nq))

        # --- Load parameter: macroscopic strain (num_cells, num_quads, 2, 2) ---
        eps_macro_q = jnp.zeros((nc, nq, 2, 2))

        # internal_vars layout:
        #   [0] E_q           (nc, nq)
        #   [1] nu_q          (nc, nq)
        #   [2] sigma_y_q     (nc, nq)
        #   [3] H_q           (nc, nq)
        #   [4] eps_p_q       (nc, nq, 2, 2)
        #   [5] alpha_q       (nc, nq)
        #   [6] eps_macro_q   (nc, nq, 2, 2)
        self.internal_vars = [E_q, nu_q, sigma_y_q, H_q,
                              eps_p_q, alpha_q, eps_macro_q]

    # ------------------------------------------------------------------
    # Constitutive law
    # ------------------------------------------------------------------

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        """Return (tensor_map, update_int_vars_map).

        Both functions have the signature::

            f(u_grad, E, nu, sigma_y, H, eps_p_old, alpha_old, eps_macro)

        where every argument is a *single quad-point* value (scalars for
        material props, (2,2) tensors for eps_p and eps_macro, etc.).

        tensor_map returns sigma (2,2).
        update_int_vars_map returns (eps_p_new, alpha_new).
        """

        def _return_mapping(u_grad, E, nu, sigma_y, H,
                            eps_p_old, alpha_old, eps_macro):
            """J2 radial return for 2-D plane strain.

            Works in full 3-D stress space (sigma_zz != 0 for plane
            strain) but stores only the 2-D in-plane plastic strain.

            Plane-strain constraint: total eps_zz = 0.
            Plastic incompressibility: eps_p_zz = -(eps_p_xx + eps_p_yy).
            Therefore elastic eps_zz_e = -eps_p_zz = eps_p_xx + eps_p_yy.
            """
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            # Total strain (2-D in-plane)
            eps_fluct = 0.5 * (u_grad + u_grad.T)
            eps_total_2d = eps_fluct + eps_macro  # (2, 2)

            # Elastic trial strain
            eps_e_trial_2d = eps_total_2d - eps_p_old  # (2, 2)

            # Out-of-plane elastic strain from plastic incompressibility
            # eps_p_zz = -(eps_p_xx + eps_p_yy), eps_zz_total = 0
            # => eps_zz_e = 0 - eps_p_zz = eps_p_xx + eps_p_yy
            eps_zz_e_trial = eps_p_old[0, 0] + eps_p_old[1, 1]

            # Full 3-D trace of elastic trial strain
            tr_eps_e = (eps_e_trial_2d[0, 0] + eps_e_trial_2d[1, 1]
                        + eps_zz_e_trial)

            # Trial stress (using full 3-D trace for lambda term)
            sigma_trial_2d = (lmbda * tr_eps_e * jnp.eye(2)
                              + 2.0 * mu * eps_e_trial_2d)
            sigma_trial_zz = lmbda * tr_eps_e + 2.0 * mu * eps_zz_e_trial

            # Deviatoric trial stress (full 3-D)
            sigma_hydro = (sigma_trial_2d[0, 0] + sigma_trial_2d[1, 1]
                           + sigma_trial_zz) / 3.0

            s_trial_2d = sigma_trial_2d - sigma_hydro * jnp.eye(2)
            s_trial_zz = sigma_trial_zz - sigma_hydro

            # ||s_trial|| = sqrt(s:s) with full 3-D deviatoric
            # s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2
            s_norm_sq = (s_trial_2d[0, 0]**2 + s_trial_2d[1, 1]**2
                         + s_trial_zz**2
                         + 2.0 * s_trial_2d[0, 1]**2)
            s_norm = jnp.sqrt(s_norm_sq + 1e-30)  # regularise

            # Yield function: f = ||s|| - sqrt(2/3) * (sigma_y + H * alpha)
            sqrt_2_3 = jnp.sqrt(2.0 / 3.0)
            f_trial = s_norm - sqrt_2_3 * (sigma_y + H * alpha_old)

            # Plastic multiplier (radial return, closed-form)
            delta_gamma = jnp.maximum(f_trial, 0.0) / (2.0 * mu + 2.0 / 3.0 * H)

            # Flow direction (unit deviatoric tensor)
            n_2d = s_trial_2d / s_norm

            # Updated plastic strain (in-plane; eps_p_zz from incompressibility)
            eps_p_new = eps_p_old + delta_gamma * n_2d

            # Updated accumulated plastic strain
            alpha_new = alpha_old + sqrt_2_3 * delta_gamma

            # Updated in-plane stress
            sigma_2d = sigma_trial_2d - 2.0 * mu * delta_gamma * n_2d

            return sigma_2d, eps_p_new, alpha_new

        def tensor_map(u_grad, E, nu, sigma_y, H,
                       eps_p_old, alpha_old, eps_macro):
            sigma, _, _ = _return_mapping(u_grad, E, nu, sigma_y, H,
                                          eps_p_old, alpha_old, eps_macro)
            return sigma

        def update_int_vars_map(u_grad, E, nu, sigma_y, H,
                                eps_p_old, alpha_old, eps_macro):
            _, eps_p_new, alpha_new = _return_mapping(
                u_grad, E, nu, sigma_y, H,
                eps_p_old, alpha_old, eps_macro)
            return eps_p_new, alpha_new

        return tensor_map, update_int_vars_map

    # ------------------------------------------------------------------
    # Parameter / state management
    # ------------------------------------------------------------------

    def set_params(self, params):
        """Set internal variables from external parameter list.

        Parameters
        ----------
        params : list
            [E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q]
        """
        self.internal_vars = params

    def update_int_vars_gp(self, sol, params):
        """Update plastic state at every quadrature point.

        Parameters
        ----------
        sol : jnp.ndarray (num_nodes, vec)
        params : list  (same layout as internal_vars)

        Returns
        -------
        eps_p_new : (num_cells, num_quads, 2, 2)
        alpha_new : (num_cells, num_quads)
        """
        _, update_int_vars_map = self.get_maps()
        vmap_update = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))

        # Compute displacement gradients at all quad points
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim)
        #   -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = (jnp.take(sol, self.fe.cells, axis=0)[:, None, :, :, None]
                   * self.fe.shape_grads[:, :, :, None, :])
        u_grads = jnp.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)

        eps_p_new, alpha_new = vmap_update(u_grads, *params)
        return eps_p_new, alpha_new

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def compute_avg_stress(self, sol, params):
        """Volume-averaged Cauchy stress over the RVE.

        Must be called *before* ``update_int_vars_gp`` within each load
        step so that the stress is consistent with the current state.

        Parameters
        ----------
        sol : jnp.ndarray (num_nodes, vec)
        params : list  (same layout as internal_vars)

        Returns
        -------
        sigma_avg : jnp.ndarray (2, 2)
            Volume-averaged in-plane Cauchy stress.
        sigma_cell : jnp.ndarray (num_cells, 2, 2)
            Cell-averaged in-plane Cauchy stress (averaged over quads
            within each cell, weighted by JxW).
        """
        tensor_map, _ = self.get_maps()
        vmap_tensor_map = jax.jit(jax.vmap(jax.vmap(tensor_map)))

        u_grads = (jnp.take(sol, self.fe.cells, axis=0)[:, None, :, :, None]
                   * self.fe.shape_grads[:, :, :, None, :])
        u_grads = jnp.sum(u_grads, axis=2)

        # sigma at each quad point: (num_cells, num_quads, 2, 2)
        sigma_qp = vmap_tensor_map(u_grads, *params)

        # Cell average (weighted by JxW)
        # JxW shape: (num_cells, num_quads)
        JxW = self.fe.JxW
        sigma_cell = (jnp.sum(sigma_qp * JxW[:, :, None, None], axis=1)
                      / jnp.sum(JxW, axis=1)[:, None, None])

        # Volume average
        sigma_avg = (jnp.sum(sigma_qp.reshape(-1, 2, 2)
                             * JxW.reshape(-1)[:, None, None], axis=0)
                     / jnp.sum(JxW))

        return sigma_avg, sigma_cell
