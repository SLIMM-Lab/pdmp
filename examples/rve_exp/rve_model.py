"""Plane-strain RVE model for 2-D homogenisation — linear elastic only.

Concrete Problem subclass for jax-fem:

LinearElasticRVE
    Isotropic linear elasticity (plane strain).
    internal_vars layout: [E_q, nu_q, eps_macro_q]

Usage
-----
    from rve_model import LinearElasticRVE

    problem = LinearElasticRVE(mesh, vec=2, dim=2, ele_type="TRI3",
                               additional_info=(phys_tags, mat_props))
    problem.set_params([E_q, nu_q, eps_macro_q])
    sol_list = solver(problem)
    sigma_avg, sigma_cell = problem.compute_avg_stress(sol_list[0], problem.internal_vars)
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
