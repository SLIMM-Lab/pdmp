"""RVE utility functions for 2-D periodic homogenisation.

Mesh generation, periodic constraint matrices, distance fields, and the
LinearElasticRVE problem class for jax-fem.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import gmsh
import meshio
import scipy.sparse

from jax_fem.problem import Problem
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.basis import get_elements

# ── Validation ───────────────────────────────────────────────────────────


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
                raise ValueError(f"Fibers {i} and {j} overlap or too close: "
                                 f"dist={dist:.4f} < {min_dist:.4f}")


# ── Mesh generation ─────────────────────────────────────────────────────


def generate_multi_fiber_rve_mesh(L,
                                  fibers,
                                  mesh_size,
                                  data_dir,
                                  l_scale=0.1,
                                  ele_type="TRI3"):
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
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)

    square_loop = gmsh.model.geo.addCurveLoop(
        [l_bottom, l_right, l_top, l_left])

    # Create all fiber circles
    circle_loops = []
    fiber_surfaces = []
    all_fiber_curves = []

    for cx, cy, R in fibers:
        pc = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size)
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
    s_matrix = gmsh.model.geo.addPlaneSurface([square_loop] + circle_loops)

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
        1,
        [l_right],
        [l_left],
        [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    )
    gmsh.model.mesh.setPeriodic(
        1,
        [l_top],
        [l_bottom],
        [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1],
    )

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    # Read back with meshio
    meshio_mesh = meshio.read(msh_file)
    points = meshio_mesh.points[:, :2]
    cells = meshio_mesh.cells_dict[cell_type]
    phys_tags = meshio_mesh.cell_data_dict["gmsh:physical"][cell_type]

    return Mesh(points, cells, ele_type=ele_type), phys_tags


# ── Periodic constraint matrix ──────────────────────────────────────────


def build_periodic_pmat(mesh, L, vec):
    """Sparse P_mat that enforces full periodicity on [0, L]^2.

    Nodes on the right / top edges are slaved to their left / bottom
    counterparts.  All four corners are pinned to zero displacement.
    """
    points = mesh.points
    num_nodes = len(points)
    num_dofs = num_nodes * vec
    EPS = 1e-5

    left = np.where(np.abs(points[:, 0]) < EPS)[0]
    right = np.where(np.abs(points[:, 0] - L) < EPS)[0]
    bottom = np.where(np.abs(points[:, 1]) < EPS)[0]
    top = np.where(np.abs(points[:, 1] - L) < EPS)[0]

    bl = np.intersect1d(left, bottom)
    br = np.intersect1d(right, bottom)
    tl = np.intersect1d(left, top)
    tr = np.intersect1d(right, top)
    corners = np.concatenate([bl, br, tl, tr])

    left_int = np.setdiff1d(left, corners)
    right_int = np.setdiff1d(right, corners)
    bottom_int = np.setdiff1d(bottom, corners)
    top_int = np.setdiff1d(top, corners)

    slave_nodes = []
    master_nodes = []

    for ri in right_int:
        dists = np.abs(points[left_int, 1] - points[ri, 1])
        best = np.argmin(dists)
        assert dists[best] < EPS, f"No left match for right node {ri}"
        slave_nodes.append(ri)
        master_nodes.append(left_int[best])

    for ti in top_int:
        dists = np.abs(points[bottom_int, 0] - points[ti, 0])
        best = np.argmin(dists)
        assert dists[best] < EPS, f"No bottom match for top node {ti}"
        slave_nodes.append(ti)
        master_nodes.append(bottom_int[best])

    slave_nodes = np.array(slave_nodes)
    master_nodes = np.array(master_nodes)

    slave_dofs = np.concatenate([slave_nodes * vec + v for v in range(vec)])
    master_dofs = np.concatenate([master_nodes * vec + v for v in range(vec)])

    pinned_dofs = np.concatenate([corners * vec + v for v in range(vec)])

    is_slave = np.zeros(num_dofs, dtype=bool)
    is_pinned = np.zeros(num_dofs, dtype=bool)
    master_of = np.zeros(num_dofs, dtype=int)

    is_slave[slave_dofs] = True
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


# ── Helper: broadcast eps_macro to quad-point shape ─────────────────────


def make_eps_macro_q(eps_macro_voigt, nc, nq):
    """Convert Voigt strain vector to (num_cells, num_quads, 2, 2)."""
    eps_macro = jnp.array([
        [eps_macro_voigt[0], 0.5 * eps_macro_voigt[2]],
        [0.5 * eps_macro_voigt[2], eps_macro_voigt[1]],
    ])
    return jnp.broadcast_to(eps_macro[None, None, :, :], (nc, nq, 2, 2))


# ── Distance field computation ──────────────────────────────────────────


def compute_per_fiber_distance_assignment(quad_points, fibers):
    """Per-fiber distances and Voronoi nearest-fiber index for each quad point.

    Parameters
    ----------
    quad_points : ndarray (nc, nq, 2)
    fibers : list of (cx, cy, R)

    Returns
    -------
    distances_per_fiber : ndarray (nc, nq, n_fibers)
        Distance from each quad point to each fiber surface, clamped at 0.
    nearest_fiber_idx : ndarray (nc, nq), int32
        Index of the closest fiber for each quad point (Voronoi cell id).
    nearest_distance : ndarray (nc, nq)
        Distance to the closest fiber surface, clamped at 0.
    """
    nc, nq, _ = quad_points.shape
    n_fibers = len(fibers)
    per_fiber = np.empty((nc, nq, n_fibers), dtype=np.float64)

    for f, (cx, cy, R) in enumerate(fibers):
        d = np.sqrt((quad_points[:, :, 0] - cx)**2 +
                    (quad_points[:, :, 1] - cy)**2) - R
        per_fiber[:, :, f] = np.maximum(d, 0.0)

    nearest_fiber_idx = np.argmin(per_fiber, axis=2).astype(np.int32)
    nearest_distance = np.take_along_axis(per_fiber,
                                          nearest_fiber_idx[:, :, None],
                                          axis=2).squeeze(-1)
    return per_fiber, nearest_fiber_idx, nearest_distance


def compute_distance_to_nearest_fiber(quad_points, fibers):
    """Compute minimum distance from each quad point to nearest fiber surface.

    Thin wrapper around :func:`compute_per_fiber_distance_assignment` kept for
    backwards compatibility.

    Parameters
    ----------
    quad_points : ndarray (nc, nq, 2)
    fibers : list of (cx, cy, R)

    Returns
    -------
    distances : ndarray (nc, nq)
        Distance to nearest fiber surface, clamped at 0.
    """
    _, _, nearest = compute_per_fiber_distance_assignment(quad_points, fibers)
    return nearest


# ── Post-processing ────────────────────────────────────────────────────


def compute_von_mises_from_cell(sigma_cells, nu_cell):
    """Von Mises stress accounting for plane-strain sigma_zz."""
    s11 = sigma_cells[:, 0, 0]
    s22 = sigma_cells[:, 1, 1]
    s12 = sigma_cells[:, 0, 1]
    s33 = nu_cell * (s11 + s22)
    vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 +
                        (s33 - s11)**2 + 6.0 * s12**2))
    return vm


# ── LinearElasticRVE problem ───────────────────────────────────────────


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
