"""Plane-strain RVE models for 2-D homogenisation.

Two concrete Problem subclasses for jax-fem:

LinearElasticRVE
    Isotropic linear elasticity (plane strain).
    internal_vars layout: [E_q, nu_q, eps_macro_q]

J2PlasticRVE
    Linear elasticity with J2 isotropic hardening (plane strain).
    internal_vars layout: [E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q]

Both share:
- Macroscopic strain loading via ``set_params``
- ``get_maps()`` returning (tensor_map, update_int_vars_map)
  (update_int_vars_map is None for the linear model)
- ``compute_avg_stress()`` for volume-averaged stress recording

Usage
-----
    from rve_model import LinearElasticRVE, J2PlasticRVE

    # Linear elastic
    problem = LinearElasticRVE(mesh, vec=2, dim=2, ele_type="TRI3",
                               additional_info=(phys_tags, mat_props))
    problem.set_params([E_q, nu_q, eps_macro_q])
    sol_list = solver(problem)
    sigma_avg, sigma_cell = problem.compute_avg_stress(sol_list[0], problem.internal_vars)

    # J2 plasticity
    problem = J2PlasticRVE(mesh, vec=2, dim=2, ele_type="TRI3",
                           additional_info=(phys_tags, mat_props))
    problem.set_params([E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q])
    sol_list = solver(problem)
    sigma_avg, sigma_cell = problem.compute_avg_stress(sol_list[0], problem.internal_vars)
    eps_p_new, alpha_new = problem.update_int_vars_gp(sol_list[0], problem.internal_vars)
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.problem import Problem

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _PlaneStrainRVEBase(Problem):
    """Shared utilities for plane-strain RVE models."""

    def _compute_u_grads(self, sol):
        """Displacement gradients at all quad points.

        Returns
        -------
        u_grads : (num_cells, num_quads, vec, dim)
        """
        u_grads = (jnp.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                   self.fe.shape_grads[:, :, :, None, :])
        return jnp.sum(u_grads, axis=2)

    def set_params(self, params):
        """Replace internal_vars with *params* (list in model-specific order)."""
        self.internal_vars = params

    def compute_avg_strain(self, sol, eps_macro_q):
        """Cell-averaged total strain (symmetric part of grad u + eps_macro).

        Parameters
        ----------
        sol : jnp.ndarray (num_nodes, vec)
        eps_macro_q : jnp.ndarray (num_cells, num_quads, 2, 2)

        Returns
        -------
        eps_cell : jnp.ndarray (num_cells, 2, 2)
        """
        u_grads = self._compute_u_grads(sol)
        eps_qp = 0.5 * (u_grads + jnp.swapaxes(u_grads, -1, -2)) + eps_macro_q
        JxW = self.fe.JxW
        return (jnp.sum(eps_qp * JxW[:, :, None, None], axis=1) /
                jnp.sum(JxW, axis=1)[:, None, None])

    def compute_avg_stress(self, sol, params):
        """Volume-averaged Cauchy stress over the RVE.

        Must be called *before* ``update_int_vars_gp`` within each load step
        so that the stress is consistent with the current state.

        Parameters
        ----------
        sol : jnp.ndarray (num_nodes, vec)
        params : list  (same layout as internal_vars)

        Returns
        -------
        sigma_avg : jnp.ndarray (2, 2)
            Volume-averaged in-plane Cauchy stress.
        sigma_cell : jnp.ndarray (num_cells, 2, 2)
            Cell-averaged in-plane Cauchy stress.
        """
        tensor_map, _ = self.get_maps()
        vmap_tensor_map = jax.jit(jax.vmap(jax.vmap(tensor_map)))

        u_grads = self._compute_u_grads(sol)
        sigma_qp = vmap_tensor_map(u_grads, *params)

        JxW = self.fe.JxW
        sigma_cell = (jnp.sum(sigma_qp * JxW[:, :, None, None], axis=1) /
                      jnp.sum(JxW, axis=1)[:, None, None])
        sigma_avg = (jnp.sum(
            sigma_qp.reshape(-1, 2, 2) * JxW.reshape(-1)[:, None, None],
            axis=0) / jnp.sum(JxW))
        return sigma_avg, sigma_cell


# ---------------------------------------------------------------------------
# Linear elastic
# ---------------------------------------------------------------------------


class LinearElasticRVE(_PlaneStrainRVEBase):
    """2-D plane-strain RVE with isotropic linear elasticity.

    Parameters
    ----------
    mesh, vec, dim, ele_type :
        Standard jax-fem Problem arguments.
    additional_info : tuple (phys_tags, mat_props)
        phys_tags : ndarray (num_cells,)
            Physical group tag per cell (1 = matrix, 2 = aggregate).
        mat_props : dict with keys
            ``E_matrix``, ``nu_matrix``, ``E_aggregate``, ``nu_aggregate``.
    """

    def custom_init(self, phys_tags, mat_props):
        self.fe = self.fes[0]
        nq = self.fe.num_quads
        nc = len(self.fe.cells)

        is_agg = (phys_tags == 2)
        E_cell = np.where(is_agg, mat_props['E_aggregate'],
                          mat_props['E_matrix']).astype(np.float64)
        nu_cell = np.where(is_agg, mat_props['nu_aggregate'],
                           mat_props['nu_matrix']).astype(np.float64)

        E_q = jnp.tile(E_cell[:, None], (1, nq))
        nu_q = jnp.tile(nu_cell[:, None], (1, nq))
        eps_macro_q = jnp.zeros((nc, nq, 2, 2))

        # internal_vars layout:
        #   [0] E_q           (nc, nq)
        #   [1] nu_q          (nc, nq)
        #   [2] eps_macro_q   (nc, nq, 2, 2)
        self.internal_vars = [E_q, nu_q, eps_macro_q]

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        """Return (tensor_map, None).

        tensor_map signature::

            tensor_map(u_grad, E, nu, eps_macro) -> sigma (2, 2)
        """

        def tensor_map(u_grad, E, nu, eps_macro):
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            eps_fluct = 0.5 * (u_grad + u_grad.T)
            eps = eps_fluct + eps_macro  # (2, 2)

            # Plane strain: eps_zz_total = 0, so full 3-D trace = eps_xx + eps_yy
            tr_eps = eps[0, 0] + eps[1, 1]
            sigma = lmbda * tr_eps * jnp.eye(2) + 2.0 * mu * eps
            return sigma

        return tensor_map, None


# ---------------------------------------------------------------------------
# J2 plasticity
# ---------------------------------------------------------------------------


class J2PlasticRVE(_PlaneStrainRVEBase):
    """2-D plane-strain RVE with J2 isotropic hardening.

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
        eps_p_q = jnp.zeros((nc, nq, 2, 2))
        alpha_q = jnp.zeros((nc, nq))
        eps_macro_q = jnp.zeros((nc, nq, 2, 2))

        # internal_vars layout:
        #   [0] E_q           (nc, nq)
        #   [1] nu_q          (nc, nq)
        #   [2] sigma_y_q     (nc, nq)
        #   [3] H_q           (nc, nq)
        #   [4] eps_p_q       (nc, nq, 2, 2)
        #   [5] alpha_q       (nc, nq)
        #   [6] eps_macro_q   (nc, nq, 2, 2)
        self.internal_vars = [
            E_q, nu_q, sigma_y_q, H_q, eps_p_q, alpha_q, eps_macro_q
        ]

    def get_tensor_map(self):
        tensor_map, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        """Return (tensor_map, update_int_vars_map).

        Both functions have the signature::

            f(u_grad, E, nu, sigma_y, H, eps_p_old, alpha_old, eps_macro)

        tensor_map returns sigma (2, 2).
        update_int_vars_map returns (eps_p_new, alpha_new).
        """

        def _return_mapping(u_grad, E, nu, sigma_y, H, eps_p_old, alpha_old,
                            eps_macro):
            """J2 radial return for 2-D plane strain.

            Works in full 3-D stress space (sigma_zz != 0 for plane strain)
            but stores only the 2-D in-plane plastic strain.

            Plane-strain constraint: total eps_zz = 0.
            Plastic incompressibility: eps_p_zz = -(eps_p_xx + eps_p_yy).
            Therefore elastic eps_zz_e = -eps_p_zz = eps_p_xx + eps_p_yy.
            """
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            eps_fluct = 0.5 * (u_grad + u_grad.T)
            eps_total_2d = eps_fluct + eps_macro

            eps_e_trial_2d = eps_total_2d - eps_p_old

            # Out-of-plane elastic strain from plastic incompressibility
            eps_zz_e_trial = eps_p_old[0, 0] + eps_p_old[1, 1]

            tr_eps_e = (eps_e_trial_2d[0, 0] + eps_e_trial_2d[1, 1] +
                        eps_zz_e_trial)

            sigma_trial_2d = (lmbda * tr_eps_e * jnp.eye(2) +
                              2.0 * mu * eps_e_trial_2d)
            sigma_trial_zz = lmbda * tr_eps_e + 2.0 * mu * eps_zz_e_trial

            sigma_hydro = (sigma_trial_2d[0, 0] + sigma_trial_2d[1, 1] +
                           sigma_trial_zz) / 3.0

            s_trial_2d = sigma_trial_2d - sigma_hydro * jnp.eye(2)
            s_trial_zz = sigma_trial_zz - sigma_hydro

            # s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2
            s_norm_sq = (s_trial_2d[0, 0]**2 + s_trial_2d[1, 1]**2 +
                         s_trial_zz**2 + 2.0 * s_trial_2d[0, 1]**2)
            s_norm = jnp.sqrt(s_norm_sq + 1e-30)

            sqrt_2_3 = jnp.sqrt(2.0 / 3.0)
            f_trial = s_norm - sqrt_2_3 * (sigma_y + H * alpha_old)

            delta_gamma = jnp.maximum(f_trial,
                                      0.0) / (2.0 * mu + 2.0 / 3.0 * H)

            n_2d = s_trial_2d / s_norm
            eps_p_new = eps_p_old + delta_gamma * n_2d
            alpha_new = alpha_old + sqrt_2_3 * delta_gamma
            sigma_2d = sigma_trial_2d - 2.0 * mu * delta_gamma * n_2d

            return sigma_2d, eps_p_new, alpha_new

        def tensor_map(u_grad, E, nu, sigma_y, H, eps_p_old, alpha_old,
                       eps_macro):
            sigma, _, _ = _return_mapping(u_grad, E, nu, sigma_y, H, eps_p_old,
                                          alpha_old, eps_macro)
            return sigma

        def update_int_vars_map(u_grad, E, nu, sigma_y, H, eps_p_old,
                                alpha_old, eps_macro):
            _, eps_p_new, alpha_new = _return_mapping(u_grad, E, nu, sigma_y,
                                                      H, eps_p_old, alpha_old,
                                                      eps_macro)
            return eps_p_new, alpha_new

        return tensor_map, update_int_vars_map

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
        u_grads = self._compute_u_grads(sol)
        eps_p_new, alpha_new = vmap_update(u_grads, *params)
        return eps_p_new, alpha_new
