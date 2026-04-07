import os
import numpy as np
import matplotlib.pyplot as plt

from typing_extensions import override

from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp import logger

import jax
import jax.numpy as jnp


class Model:
    """Base class for the forward model"""

    def __init__(self):
        pass

    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the forward model

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: model evaluation
        """
        raise NotImplementedError

    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the gradient (jacobian) of the forward model outputs with respect to the parameters

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: gradient of model evaluation
        """
        raise NotImplementedError

    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the hessian of the forward model outputs with respect to the parameters

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: hessian of model evaluation
        """
        raise NotImplementedError

    def get_dim_in(self) -> int:
        """Get dimension of the model outputs

        Returns:
            int: dimension of the model
        """
        raise NotImplementedError

    def get_dim_out(self) -> int:
        """Get dimension of the model outputs

        Returns:
            int: dimension of the model outputs
        """
        raise NotImplementedError

    def get_obs_locs(self) -> np.ndarray:
        """Return observation locations as an (m, d) array, or None if unavailable.

        Each row is the spatial coordinate of one *base* observation location.
        For models where each location yields multiple outputs (e.g. 3 DOFs per
        sensor point), this returns the per-point coordinates only — the total
        number of outputs is get_dim_out() = m * n_components.

        Returns:
            (m, d) array of observation coordinates, or None.
        """
        return None

    def get_n_settings(self) -> int:
        """Get number of settings.

        Returns:
            int: Number of settings. Default is 1 if not overridden by derived classes.
        """
        return 1

    @classmethod
    def from_dict(cls, config: dict):
        """Create a model from a dictionary configuration.

        Args:
            config: configuration dictionary

        Returns:
            Model: model
        """
        raise NotImplementedError


class PiecewiseConstantModel(Model):
    """    Forward model for deformation of a 1d bar with piecewise constant Young's modulus."""

    def __init__(self,
                 F: np.ndarray,
                 n_params: int,
                 x_obs: np.ndarray,
                 field=None):
        """Initialize the model

        Args:
            F: collection of prescribed forces for each setting
            n_params: number of parameters (ignored if field provided)
            x_obs: observed x values
            field: Optional GaussianRandomField providing coefficient dimension
        """
        super().__init__()
        self.F_vals = F  # Actual values of F
        self.n_settings_ = self.F_vals.shape[0]  # Number of settings
        self.x_obs_ = x_obs  # Observations
        # store field if provided
        self.field = field
        if self.field is not None:
            self.n_params = int(self.field.dim)
        else:
            self.n_params = n_params  # Number of parameters
        if self.n_params is None:
            raise ValueError(
                "Either n_params or field must define parameter dimension.")

    def _midpoints(self) -> np.ndarray:
        """Return midpoints of the n_params equal sub-intervals of [0,1]."""
        n = self.n_params
        return (np.arange(n) + 0.5) / n

    def _J_params_eff(self) -> np.ndarray:
        """Jacobian of effective piecewise constants with respect to model parameters.

        If a GaussianRandomField is attached, params_eff = Φ(mid) @ params, hence J = Φ(mid).
        Otherwise J is the identity.
        Shape: (n_eff, n_params) where n_eff == n_params == self.n_params.
        """
        n = self.n_params
        if self.field is None:
            return np.eye(n)
        mid = self._midpoints()
        # design_matrix returns shape (len(mid), dim) == (n, n)
        return self.field.design_matrix(mid)

    def _effective_params(self, params: np.ndarray) -> np.ndarray:
        """Return the piecewise constant parameter vector used by the analytic formula.

        If a GaussianRandomField is attached, evaluate it at cell midpoints to obtain
        effective piecewise constant values (these coincide with coefficients
        for a PiecewiseConstantBasis). Otherwise, return params directly.
        """
        if self.field is None:
            return np.asarray(params, dtype=float)
        n = self.n_params
        midpoints = self._midpoints()
        return self.field.evaluate(params, midpoints)

    def eval_E(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate Young's modulus E(x) without sympy.

        For x in ((k-1)/n, k/n] we have E(x) = params[k-1]. For x=0 we assign E(0)=params[0].

        Args:
            params: array of shape (n_params,)
            x: 1D array of points in [0,1]
        Returns:
            1D array of E values with shape (len(x),)
        """
        params = np.asarray(params, dtype=float)
        x = np.asarray(x, dtype=float)
        n = self.n_params
        if params.shape[0] != n:
            raise ValueError(
                f"params must have length {n}, got {params.shape[0]}")
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("All x must be within [0, 1]")
        # Interval index I = ceil(x * n); clip to [1, n]
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        params_eff = self._effective_params(params)
        return params_eff[I - 1]

    def eval(self,
             params: np.ndarray,
             x: np.ndarray = None,
             idx: int = 0) -> np.ndarray:
        """Analytic evaluation of u

        Formula for x in ((I-1)/n, I/n], with I = ceil(x*n), n = number of parameters:
            u(x) = F * [ (1/n) * sum_{k=1}^{I-1} 1/p_k + (x - (I-1)/n)/p_I ].
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError(
                "effective params must be a vector of length n_params for analytic evaluation"
            )
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p = 1.0 / params_eff
        cumsum_inv = np.concatenate(([0.0], np.cumsum(inv_p)))  # length n+1
        base_sum = (1.0 / n) * cumsum_inv[I - 1]
        final_term = (x - (I - 1) / n) / params_eff[I - 1]
        return self.F_vals[idx] * (base_sum + final_term)

    def eval_grad(self,
                  params: np.ndarray,
                  x: np.ndarray = None,
                  idx: int = 0) -> np.ndarray:
        """Analytic gradient w.r.t. model parameters (coefficients).

        Internally, the analytic formulas are expressed w.r.t. the effective piecewise-constant
        values p = params_eff. When a GaussianRandomField is attached, p = Φ(mid) @ a (linear in
        the coefficients a). Using the chain rule: du/da = (du/dp) @ Φ(mid).

        Returns array shape (len(x), n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError(
                "effective params must be a vector of length n_params for analytic gradient"
            )
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p2 = 1.0 / (params_eff**2)
        base_prefix = -(1.0 / n) * inv_p2
        # Gradient w.r.t effective params p
        G_eff = np.zeros((m, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            stop = Ii - 1
            if stop > 0:
                G_eff[row, :stop] = base_prefix[:stop]
            G_eff[row,
                  Ii - 1] = -(x[row] - (Ii - 1) / n) / (params_eff[Ii - 1]**2)
        G_eff *= Fval
        # Chain rule to parameters: J = ∂p/∂a = Φ(mid)
        J = self._J_params_eff()
        return G_eff @ J

    def eval_hessian(self,
                     params: np.ndarray,
                     x: np.ndarray = None,
                     idx: int = 0) -> np.ndarray:
        """Analytic Hessian w.r.t. model parameters (coefficients).

        With p = Φ(mid) @ a linear in a, the Hessian transforms as:
            H_a = J^T H_p J, where J = ∂p/∂a = Φ(mid).
        Returns array shape (len(x), n_params, n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError(
                "effective params must be a vector of length n_params for analytic Hessian"
            )
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p3 = 1.0 / (params_eff**3)
        # Hessian w.r.t effective params p
        H_eff = np.zeros((m, n, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            if Ii - 1 > 0:
                H_eff[row, :Ii - 1, :Ii - 1] = np.diag(2.0 / n *
                                                       inv_p3[:Ii - 1])
            H_eff[row, Ii - 1,
                  Ii - 1] = 2.0 * (x[row] - (Ii - 1) / n) * inv_p3[Ii - 1]
        H_eff *= Fval
        # Transform to parameters via J^T H J
        J = self._J_params_eff()
        # For each output row, compute J^T H_eff[row] J
        H = np.zeros((m, n, n), dtype=float)
        JT = J.T
        for row in range(m):
            H[row] = JT @ H_eff[row] @ J
        return H

    @override
    def get_dim_in(self) -> int:
        return self.n_params  # Return number of parameters

    def get_dim_out(self) -> int:
        return len(self.x_obs_)

    def get_obs_locs(self) -> np.ndarray:
        return self.x_obs_.reshape(-1, 1)

    @override
    def get_n_settings(self):
        return self.n_settings_  # Return number of settings

    @classmethod
    def from_dict(cls, config: dict, field=None):
        """Create a PiecewiseConstantModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary
            field (GaussianRandomField, optional): field providing coefficient dimension

        Returns:
            PiecewiseConstantModel: piecewise constant model
        """
        F = np.array(config['F'])
        if field is not None:
            n_params = field.dim
        else:
            n_params = config['dim']

        if 'x_obs' in config:
            x_obs = np.array(config['x_obs'])
        elif 'n_obs_loc' in config:
            x_obs = np.linspace(0, 1, config['n_obs_loc'] + 1)[1:]
        else:
            raise ValueError("Observation locations not provided")

        return cls(F, n_params, x_obs, field=field)


def get_piececwise_constant_model(
    n_params: int,
    n_obs_loc: int,
    n_obs: int = 1,
    sigma_obs: float = 0.025,
    mean: float = 4.,
    rng: np.random.Generator = np.random.default_rng(0),
    kernel_params: dict[str, float] = None
) -> tuple[PiecewiseConstantModel, np.ndarray, np.ndarray, np.ndarray]:
    """Get a piecewise constant model with noisy observations

    Args:
        n_params: number of parameters
        n_obs_loc: number of observation locations
        n_obs: number of observations
        sigma_obs: observation noise
        mean: mean for the prior
        rng: random number generator
        kernel_params: kernel parameters

    Returns:
        tuple[PiecewiseConstantModel, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            - model
            - observations
            - ground truth
            - prior_cov
    """

    if kernel_params is None:
        kernel_params = {'sigma': 1., 'l': 0.3}

    # get the prior field
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_params, interval)
    prior_cov = compute_coefficients(squared_exponential_kernel,
                                     basis,
                                     interval,
                                     kernel_params=kernel_params)

    # define observation locations
    x_obs = np.linspace(0, 1, n_obs_loc + 1)[1:]

    # set up the forward model
    F = [1.]
    F = np.array([item for item in F for i in range(n_obs)])
    model = PiecewiseConstantModel(F, n_params, x_obs)

    # generate ground truth from a multi-variate normal distribution
    params_gt = rng.multivariate_normal(mean * np.ones(n_params), prior_cov)
    logger.info(f"Ground truth: {params_gt}")

    # generate n_obs noisy observations of u_gt
    u_obs = np.zeros((len(F), len(x_obs)))

    for i in range(len(F)):
        u_gt = model.eval(params_gt, x_obs, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    return model, u_obs, params_gt, prior_cov


class LinearModel(Model):
    """Forward model for a linear system (mainly for testing purposes)"""

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """Initialize the model

        Args:
            A: matrix A
            b: vector b
        """
        super().__init__()
        assert A.shape[0] == b.shape[0], "Dimensions do not match"
        self.dim_in_ = A.shape[1]
        self.dim_out_ = A.shape[0]
        self.A_ = A
        self.b_ = b

    @classmethod
    def from_dict(cls, config: dict):
        """Create a LinearModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary

        Returns:
            LinearModel: linear model
        """
        A = np.array(config['A'])
        b = np.array(config['b'])
        return cls(A, b)

    @override
    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        return self.A_ @ params + self.b_

    @override
    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        return self.A_

    @override
    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        s = self.A_.shape
        return np.zeros((s[0], s[1], s[1]))

    @override
    def get_dim_in(self) -> int:
        return self.dim_in_

    @override
    def get_dim_out(self) -> int:
        return self.dim_out_


def get_model(config: dict, field=None):
    """Get a model from a configuration dictionary

    Args:
        config: configuration dictionary
        field: Optional random field (GaussianRandomField or JaxRandomField) to inject into model

    Returns:
        Model: model
    """
    if config['name'] == 'PiecewiseConstant':
        return PiecewiseConstantModel.from_dict(config, field=field)
    elif config['name'] == 'Linear':
        return LinearModel.from_dict(config)
    elif config['name'] == 'JaxFem':
        return JaxFemModel.from_dict(config, field=field)
    elif config['name'] == 'RVE':
        return RVEModel.from_dict(config)
    else:
        raise ValueError(f"Model {config['name']} not recognized.")


def build_sensor_interpolants(fe, sensors, location_fn_map, tol=1e-8):
    """Build interpolation data for each sensor on a given boundary.

    Each sensor specification in ``sensors`` must include a ``'location_fn'``
    key that identifies which boundary face to search.  It may be either a
    callable ``(point) -> bool`` or a string key looked up in
    ``location_fn_map``.

    Each sensor specification may otherwise be either:

    * Single-point form::

        {"name": "sensor_name", "location_fn": "bottom", "point": [x, y, z]}

    * Multi-point form::

        {"name": "sensor_name", "location_fn": "side_faces",
         "points": [[x1, y1, z1], [x2, y2, z2], ...]}

    Internally, each sensor is represented with arrays of shape
    (n_points, n_face_nodes) for ``nodes`` and ``weights`` so that
    multiple physical points can be associated with a single sensor.

    No aggregation across points is performed here; all point-wise
    displacements are returned by :func:`evaluate_sensor_displacements`.
    """

    interpolants = []
    for spec in sensors:
        if "name" not in spec:
            raise ValueError("Each sensor spec must contain a 'name' key.")

        # Resolve location_fn for this sensor group
        if "location_fn" not in spec:
            raise ValueError(
                f"Sensor '{spec['name']}' must define a 'location_fn' key.")
        loc = spec["location_fn"]
        if callable(loc):
            location_fn = loc
        elif isinstance(loc, str):
            if loc not in location_fn_map:
                raise ValueError(
                    f"Sensor '{spec['name']}': unknown location_fn '{loc}'. "
                    f"Must be one of {list(location_fn_map.keys())} or a callable."
                )
            location_fn = location_fn_map[loc]
        else:
            raise ValueError(
                f"Sensor '{spec['name']}': 'location_fn' must be a string or callable, got {type(loc)}."
            )

        boundary_inds_list = fe.get_boundary_conditions_inds([location_fn])
        if not boundary_inds_list or boundary_inds_list[0].size == 0:
            raise ValueError(
                f"Sensor '{spec['name']}': no boundary faces found for location_fn '{loc}'."
            )
        boundary_inds = boundary_inds_list[0]

        # Resolve list of points for this sensor
        if "points" in spec:
            points = np.asarray(spec["points"], dtype=float)
            if points.ndim == 1:
                points = points[None, :]
        elif "point" in spec:
            points = np.asarray(spec["point"], dtype=float)[None, :]
        else:
            raise ValueError(
                f"Sensor '{spec['name']}' must define 'point' or 'points'.")

        n_points = points.shape[0]
        all_node_ids = []
        all_weights = []

        for p in points:
            weights = None
            node_ids = None
            for cell_idx, face_idx in boundary_inds:
                local_face_nodes = fe.face_inds[face_idx]
                global_node_ids = fe.cells[cell_idx][local_face_nodes]
                coords = fe.points[global_node_ids]
                face_weights = face_interpolation_weights(p, coords, tol)
                if face_weights is not None:
                    weights = face_weights
                    node_ids = global_node_ids
                    break
            if weights is None or node_ids is None:
                raise ValueError(
                    f"Sensor '{spec['name']}' point {p} not located on selected boundary."
                )
            all_node_ids.append(node_ids)
            all_weights.append(weights)

        nodes_arr = np.vstack(all_node_ids)
        weights_arr = np.vstack(all_weights)

        interpolant = {
            "name": spec["name"],
            "points": points,
            "nodes": nodes_arr,
            "weights": weights_arr,
        }

        if n_points == 1:
            interpolant["point"] = points[0]

        interpolants.append(interpolant)

    return interpolants


def face_interpolation_weights(point, coords, tol):
    if len(coords) == 3:
        inside, weights = point_in_triangle(point, coords, tol)
        if inside:
            face_weights = np.zeros(3)
            face_weights[:] = weights
            return face_weights
        return None
    if len(coords) == 4:
        return quad_bilinear_weights(point, coords, tol)
    raise ValueError(
        f"Unsupported face with {len(coords)} nodes for interpolation.")


def quad_bilinear_weights(point, quad_coords, tol):
    """Bilinear interpolation weights for a planar quad face (4 nodes).

    Correctly evaluates all four bilinear shape functions at the query point,
    so that interior face points receive contributions from all four nodes —
    as the FEM bilinear basis requires.

    Unlike a naïve implementation, this does **not** assume a fixed CCW/CW node
    ordering (such as the standard (-1,-1),(+1,-1),(+1,+1),(-1,+1) convention).
    Instead, it projects all four nodes onto the face plane and solves for the
    local (s,t) coordinates that are consistent with the actual node layout.
    This makes it robust to the row-major or otherwise non-standard orderings
    produced by different mesh generators.

    Returns
    -------
    np.ndarray of shape (4,) if the point lies on the face, else None.
    """
    # ── Coplanarity check ────────────────────────────────────────────────────
    v0 = quad_coords[1] - quad_coords[0]
    v1 = quad_coords[3] - quad_coords[0]
    normal = np.cross(v0, v1)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < tol:
        # Fallback: try with the other diagonal
        v1 = quad_coords[2] - quad_coords[0]
        normal = np.cross(v0, v1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < tol:
            return None
    if np.abs(np.dot(normal, point - quad_coords[0])) / normal_norm > tol:
        return None

    # ── Project onto face plane (2D) ─────────────────────────────────────────
    e1 = v0 / np.linalg.norm(v0)
    e2 = np.cross(normal / normal_norm, e1)

    def to_2d(p):
        d = p - quad_coords[0]
        return np.array([np.dot(d, e1), np.dot(d, e2)])

    nodes_2d = np.array([to_2d(quad_coords[i]) for i in range(4)])  # (4, 2)
    point_2d = to_2d(point)  # (2,)

    # ── Determine local (s, t) coordinates for each mesh node ────────────────
    # We fit the bilinear map  x(s,t) = sum_i N_i(s,t) * x_i  by finding the
    # axis-aligned bounding box of the 2D nodes and mapping them to [-1,+1]^2.
    # This correctly handles row-major and column-major orderings.
    mins = nodes_2d.min(axis=0)
    maxs = nodes_2d.max(axis=0)
    span = maxs - mins
    if np.any(span < tol):
        return None

    # Map each node to its local coordinate in [-1, +1]^2
    nodes_st = 2.0 * (nodes_2d - mins) / span - 1.0  # (4, 2)
    # Round to ±1 to avoid floating-point drift at corners
    nodes_st = np.clip(nodes_st, -1.0, 1.0)
    s_nodes = nodes_st[:, 0]
    t_nodes = nodes_st[:, 1]

    # ── Map query point to local coordinates ─────────────────────────────────
    point_st = 2.0 * (point_2d - mins) / span - 1.0

    # ── Check the point is inside [-1,+1]^2 (with tolerance) ─────────────────
    boundary_tol = max(tol, 1e-6)
    if np.abs(point_st[0]) > 1.0 + boundary_tol or np.abs(
            point_st[1]) > 1.0 + boundary_tol:
        return None
    point_st = np.clip(point_st, -1.0, 1.0)
    s, t = point_st

    # ── Evaluate bilinear shape functions ─────────────────────────────────────
    weights = 0.25 * (1.0 + s_nodes * s) * (1.0 + t_nodes * t)
    return weights


def point_in_triangle(point, tri, tol):
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = point - tri[0]

    # Coplanarity check: the point must lie on the plane of the triangle.
    # Compute the face normal and verify the point has no component along it.
    normal = np.cross(v0, v1)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < tol:
        return False, None
    if np.abs(np.dot(normal, v2)) / normal_norm > tol:
        return False, None

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if np.abs(denom) < tol:
        return False, None
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v
    inside = (u >= -tol) and (v >= -tol) and (w >= -tol) and (u + v
                                                              <= 1.0 + tol)
    return inside, (w, u, v)  # weights for tri[0], tri[1], tri[2] respectively


def evaluate_sensor_displacements(sol, interpolants):
    """Evaluate displacements at sensors based on precomputed interpolants.

    For each sensor, one or more physical points may be defined. The function
    returns a list of readings, where each reading is a dict with:

    * ``name``: sensor name
    * ``points``: array of sensor points, shape (n_points, dim)
    * ``u_points``: array of displacements at each point, shape (n_points, vec_dim)
    * ``u``: identical to ``u_points`` (no aggregation across points).

    Any aggregation across points must be done by the caller, if desired.
    """
    readings = []
    for interp in interpolants:
        nodes = jnp.asarray(interp["nodes"], dtype=jnp.int32)
        weights = jnp.asarray(interp["weights"])
        if nodes.ndim == 1:
            nodes = nodes[None, :]
        if weights.ndim == 1:
            weights = weights[None, :]

        # nodal_disp: (n_points, n_face_nodes, vec_dim)
        nodal_disp = sol[nodes]
        # u_points: (n_points, vec_dim)
        u_points = jnp.einsum("pn,pnv->pv", weights, nodal_disp)

        reading = {
            "name": interp["name"],
            "points": jnp.asarray(interp["points"]),
            "u_points": u_points,
            "u": u_points,
        }
        if "point" in interp:
            reading["point"] = jnp.asarray(interp["point"])

        readings.append(reading)

    return readings


class JaxFemModel(Model):
    """JAX-based FEM model wrapper with support for random fields and configurable sensors.

    This model supports efficient likelihood gradient computation via VJP
    (Jacobian-vector product) and can work with JAX-compatible random fields
    to represent spatially-varying material properties.

    The model:
      - Performs FEM solves using JAX for automatic differentiation
      - Returns observed displacement components from sensors
      - Supports J^T v via `linearize` and `eval_vjp`
      - Provides full Jacobian via `eval_grad` and Hessian via `eval_hessian`
      - Can use a random field to map coefficients to material properties
      - Allows flexible sensor placement on mesh boundaries

    Parameters
    ----------
    d_x, d_y, d_z : float
        Domain dimensions in x, y, z directions.
    ele_type : str, optional
        Element type (default 'HEX8').
    nu : float, optional
        Poisson's ratio (default 0.3).
    h : float, optional
        Mesh size parameter (default 0.5).
    indenter_loc: float, optional
        z-coordinate of the indenter location on the top surface (default is mid-height).
    traction : list, optional
        Traction vector on boundary (default [0., 0.015, 0.]).
        Mutually exclusive with ``total_load``.
    total_load : list or array-like, optional
        Total force vector [Fx, Fy, Fz] applied on the loaded boundary
        (location_fns[0]).  The constant traction is derived as
        total_load / surface_area, where surface_area is computed from the
        mesh after Problem construction.  Mutually exclusive with ``traction``.
    n_params : int, optional
        Number of parameters (default 1), overridden if field is provided.
    field : JaxRandomField, optional
        Random field mapping coefficients to material properties.
        If provided, determines n_params from field.dim.
    sensors : list of dict, optional
        List of sensor specifications. Each sensor dict must include:
        - 'name': str, sensor identifier
        - 'location_fn': str or callable, boundary face to place the sensor on.
          Strings are resolved from the built-in map: 'side_faces', 'top_surface', 'bottom'.
          A callable ``(point) -> bool`` may also be provided directly.
        - 'point': array-like, single observation point [x, y, z]
        - 'points': array-like, multiple observation points [[x1,y1,z1], ...]

        Example::

            {"name": "sensor1", "location_fn": "bottom", "point": [0.5, 0.5, 0.0]}
            {"name": "sensor2", "location_fn": "side_faces", "points": [[0.0, 0.5, 1.0], [1.0, 0.5, 1.0]]}

        Default: [{"name": "sensor_left_center", "location_fn": "side_faces",
                   "point": [0, 0.5*d_y, 0.5*d_z]}]
    """

    def __init__(self,
                 d_x: float,
                 d_y: float,
                 d_z: float,
                 ele_type: str = 'HEX8',
                 nu: float = 0.3,
                 h: float = 0.5,
                 indenter_loc: float = None,
                 traction=None,
                 total_load=None,
                 n_params: int = 1,
                 field=None,
                 sensors=None):
        super().__init__()

        if traction is not None and total_load is not None:
            raise ValueError(
                "Cannot specify both 'traction' and 'total_load'. "
                "Use 'traction' for a direct traction vector or "
                "'total_load' for a total force divided by surface area.")

        # Mutable list so the get_surface_maps() closure sees updates
        if traction is not None:
            _traction = list(traction)
        elif total_load is not None:
            _traction = [0., 0., 0.
                         ]  # placeholder; computed after Problem construction
        else:
            _traction = [0., .015, 0.]

        if indenter_loc is None:
            indenter_loc = 0.5 * d_z

        self._total_load = total_load
        self._h = h

        # Store field and determine parameter dimension
        self.field = field
        if self.field is not None:
            self._n_params = int(self.field.dim)
        else:
            self._n_params = n_params

        from jax_fem.problem import Problem
        from jax_fem.solver import ad_wrapper
        from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh

        class LinearElasticity(Problem):

            def custom_init(self):
                self.fe = self.fes[0]

            def get_tensor_map(self):

                def stress(u_grad, E):
                    E_rho = E * 1.
                    mu = E_rho / (2. * (1. + nu))
                    lmbda = E_rho * nu / ((1 + nu) * (1 - 2 * nu))
                    epsilon = 0.5 * (u_grad + u_grad.T)
                    sigma = lmbda * jnp.trace(epsilon) * jnp.eye(
                        self.dim) + 2 * mu * epsilon
                    return sigma

                return stress

            def get_surface_maps(self):

                def surface_map(u, x):
                    return jnp.array(_traction)

                return [surface_map]

            def set_params(self, params):
                E = params[0]
                self.internal_vars = [E]

        def zero_dirichlet_val(point):
            return 0.

        def bottom_face(point):
            return jnp.isclose(point[2], 0., atol=1e-5)

        def indenter(point):
            return (point[2] > indenter_loc) * jnp.isclose(
                point[1], d_y, atol=1e-5)
            # return ((point[0] - d_x / 2.) ** 2 + (point[2] - d_z - 0.75) ** 2 > 1.8
            #         * jnp.isclose(point[1], d_y, atol=1e-5))

        def top_face(point):
            return jnp.isclose(point[2], d_z, atol=1e-5)

        def side_faces(point):
            return (jnp.isclose(point[0], 0., atol=1e-5) +
                    jnp.isclose(point[0], d_x, atol=1e-5) +
                    jnp.isclose(point[1], 0., atol=1e-5) +
                    jnp.isclose(point[1], d_y, atol=1e-5))

        dirichlet_bc_info = [
            [bottom_face, bottom_face, bottom_face],  # location
            [0, 1, 2],  # dof
            [zero_dirichlet_val, zero_dirichlet_val,
             zero_dirichlet_val]  # value
        ]

        location_fns = [indenter]

        # Built-in map of named boundary selectors available to sensor specs
        location_fn_map = {
            'side_faces': side_faces,
            'top_face': top_face,
            'bottom_face': bottom_face,
        }

        # Use provided sensors or default to a single sensor on the side faces todo: remove backwards compatibility
        if sensors is None:
            sensors = [
                {
                    "name": "sensor_left_center",
                    "location_fn": "side_faces",
                    "point": np.array([0, 0.5 * d_y, 0.5 * d_z])
                },
            ]

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        cell_type = get_meshio_cell_type(ele_type)
        n_x = int(d_x / h)
        n_y = int(d_y / h)
        n_z = int(d_z / h)
        meshio_mesh = box_mesh_gmsh(Nx=n_x,
                                    Ny=n_y,
                                    Nz=n_z,
                                    domain_x=d_x,
                                    domain_y=d_y,
                                    domain_z=d_z,
                                    data_dir=data_dir,
                                    ele_type=ele_type)
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

        self.problem = LinearElasticity(mesh,
                                        vec=3,
                                        dim=3,
                                        ele_type=ele_type,
                                        dirichlet_bc_info=dirichlet_bc_info,
                                        location_fns=location_fns)

        # Derive traction from total load and computed surface area
        if total_load is not None:
            total_load = np.asarray(total_load, dtype=float)
            # nanson_scale[0] corresponds to location_fns[0] (loaded boundary)
            # shape: (num_selected_faces, num_vars, num_face_quads)
            surface_area = float(np.sum(self.problem.nanson_scale[0][:, 0, :]))
            if surface_area < 1e-15:
                raise ValueError(
                    f"Computed surface area is effectively zero ({surface_area:.2e}). "
                    "Check that the load location function selects boundary faces."
                )
            derived = total_load / surface_area
            _traction[0] = float(derived[0])
            _traction[1] = float(derived[1])
            _traction[2] = float(derived[2])
            self._surface_area = surface_area
            logger.info(
                f"total_load={total_load.tolist()}, surface_area={surface_area:.6f}, "
                f"traction={list(_traction)}")

        self.sensor_interpolants = build_sensor_interpolants(
            self.problem.fe, sensors, location_fn_map) if sensors else []

        # Use UMFPACK (direct solver) for the adjoint.  The default JAX BiCGStab
        # starts from a zero initial guess and can hit a numerical breakdown
        # (ρ → 0 in the BiCGStab recurrence) for certain RHS vectors, causing
        # sporadic "adjoint solver did not converge" failures near MAP points.
        # UMFPACK is a direct sparse solver and is unconditionally robust here.
        # might need to have two wrappers for problem bc umfpack does not scale well with dofs
        # BiCGStab only caused problems at the MAP so far, expected to work for the rest of domain
        self.fwd_pred = ad_wrapper(
            self.problem, adjoint_solver_options={"umfpack_solver": {}})

        # total observed dofs = total point-wise displacements from all sensors
        sample_sol = jnp.zeros((self.problem.fe.num_total_nodes, 3))
        sample_readings = evaluate_sensor_displacements(
            sample_sol, self.sensor_interpolants)
        self._d_obs = sum(int(r["u"].size) for r in sample_readings)

    @override
    def get_dim_in(self) -> int:
        return self._n_params

    @override
    def get_dim_out(self) -> int:
        return self._d_obs

    def get_obs_locs(self) -> np.ndarray:
        """Return sensor point coordinates as (n_pts_total, 3).

        Each sensor group contributes its points array; all groups are stacked.
        The total number of observations equals n_pts_total * vec_dim (DOFs per
        point), so n_components = get_dim_out() // get_obs_locs().shape[0].
        """
        return np.vstack([s["points"] for s in self.sensor_interpolants])

    def _eval_obs(self, params: jnp.ndarray, idx: int = 0) -> jnp.ndarray:
        """JAX-compatible forward map params -> observation vector.

        This function is purely functional and suitable for jax.vjp, jacrev, etc.
        It performs a FEM solve via `self.fwd_pred` and then extracts the
        concatenated sensor displacements.

        If a random field is attached, params are coefficients that get mapped
        to a spatially-varying material property field. Otherwise, params is
        broadcast directly as a constant field.
        """
        # Map parameters to material property field
        if self.field is None:
            # Simple broadcast: scalar/global params to per-cell/quadrature field
            param_field = params * jnp.ones(
                (self.problem.fe.num_cells, self.problem.fe.num_quads))
        else:
            # Evaluate random field at physical quadrature points
            # physical_quad_points shape: (num_cells, num_quads, spatial_dim)
            physical_quad_points = self.problem.physical_quad_points

            # Flatten to evaluate all quadrature points at once
            # Shape: (num_cells * num_quads, spatial_dim)
            quad_coords_flat = physical_quad_points.reshape(
                -1, physical_quad_points.shape[-1])

            # Evaluate field at all quadrature points
            field_values_flat = self.field.evaluate(
                params, quad_coords_flat)  # (num_cells * num_quads,)

            # Reshape back to (num_cells, num_quads)
            param_field = field_values_flat.reshape(self.problem.fe.num_cells,
                                                    self.problem.fe.num_quads)

        sol_list = self.fwd_pred([param_field])
        sensor_readings = evaluate_sensor_displacements(
            sol_list[0], self.sensor_interpolants)
        if not sensor_readings:
            return jnp.array([])
        # Stack all sensor groups: (n_pts_total, vec_dim)
        u_all = jnp.vstack([reading["u"] for reading in sensor_readings])
        # Transpose and ravel → grouped order: [all ux, all uy, all uz]
        # This matches the block-diagonal covariance structure used by
        # KOGaussianLikelihood (n_components = vec_dim independent GP components).
        return u_all.T.ravel()

    def eval(self,
             params: np.ndarray,
             idx: int = 0,
             save_dir: str = None) -> np.ndarray:  # noqa: D401
        """Evaluate observed displacement components.

        Returns the concatenated vector of per-point sensor displacements.
        """
        from jax_fem.utils import save_sol

        theta = jnp.asarray(params)
        y = self._eval_obs(theta, idx)

        # Optionally still write full-field VTK for debugging
        # Recompute with full field to save; this keeps eval() side-effects
        if self.field is None:
            param_field = theta * jnp.ones(
                (self.problem.fe.num_cells, self.problem.fe.num_quads))
        else:
            # Use field evaluation (same as in _eval_obs)
            physical_quad_points = self.problem.physical_quad_points
            quad_coords_flat = physical_quad_points.reshape(
                -1, physical_quad_points.shape[-1])
            field_values_flat = self.field.evaluate(theta, quad_coords_flat)
            param_field = field_values_flat.reshape(self.problem.fe.num_cells,
                                                    self.problem.fe.num_quads)

        if save_dir:
            sol_list = self.fwd_pred([param_field])
            os.makedirs(save_dir, exist_ok=True)
            vtk_path = os.path.join(save_dir, 'vtk/u.vtu')
            os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
            save_sol(self.problem.fe, sol_list[0], vtk_path)

        return np.asarray(y)

    def linearize(self, params: np.ndarray, idx: int = 0):
        """Single forward pass + VJP closure.

        Parameters
        ----------
        params : np.ndarray, shape (n_params,)
            Current parameter vector.
        idx : int
            Setting index (unused here but kept for API compatibility).

        Returns
        -------
        y : np.ndarray, shape (d_obs,)
            Forward model evaluation at ``params``.
        vjp_fun : callable
            Function that maps a vector ``v`` of shape (d_obs,) to ``J^T v``
            of shape (n_params,), where J is the Jacobian of the observations
            with respect to the parameters.
        """
        theta = jnp.asarray(params)

        def obs_fn(theta_local):
            return self._eval_obs(theta_local, idx)

        y, vjp_handle = jax.vjp(obs_fn, theta)

        def apply_vjp(v: np.ndarray) -> np.ndarray:
            (g, ) = vjp_handle(jnp.asarray(v))
            return np.asarray(g)

        return np.asarray(y), apply_vjp

    def eval_vjp(self, params: np.ndarray, idx: int,
                 v: np.ndarray) -> np.ndarray:
        """Compute J^T v without forming J explicitly.

        Uses a single vjp built at the given ``params``. This is the most
        efficient way to apply the transpose-Jacobian to a vector when only
        a few such products are needed.
        """
        _, vjp_fun = self.linearize(params, idx)
        return vjp_fun(v)

    def eval_grad(self, params: np.ndarray, idx: int = 0) -> np.ndarray:
        """Return full Jacobian d_obs x n_params using JAX.

        This forms the dense Jacobian by looping over output components and
        applying a scalar VJP for each, which avoids batched VJP issues in
        the implicit solver.
        """
        theta = jnp.asarray(params)

        def obs_fn(theta_local):
            return self._eval_obs(theta_local, idx)

        # Build VJP at theta once
        y, vjp_handle = jax.vjp(obs_fn, theta)
        y = jnp.atleast_1d(y)
        d_obs = y.shape[0]
        n_params = theta.shape[0]
        J_rows = []
        # For each output component e_i, compute J^T e_i via VJP and treat it
        # as the i-th row of the Jacobian.
        for i in range(d_obs):
            e_i = jnp.zeros_like(y).at[i].set(1.0)
            (g_i, ) = vjp_handle(e_i)
            J_rows.append(g_i)
        J = jnp.stack(J_rows, axis=0)  # (d_obs, n_params)
        return np.asarray(J)

    def eval_hessian(self,
                     params: np.ndarray,
                     idx: int = 0,
                     h: float = 1e-5) -> np.ndarray:
        """Finite difference approximation of the Hessian using eval_grad.

        Uses central differences: H[:, :, j] ≈ (J(x + h*e_j) - J(x - h*e_j)) / (2h)
        where J is the Jacobian (from eval_grad).

        Parameters
        ----------
        params : np.ndarray, shape (n_params,)
            Current parameter vector.
        idx : int
            Setting index.
        h : float, optional
            Step size for finite differences, default 1e-5.

        Returns
        -------
        np.ndarray, shape (d_obs, n_params, n_params)
            Hessian tensor where H[i, j, k] = ∂²y_i / ∂θ_j ∂θ_k.
        """
        params = np.asarray(params, dtype=float)
        n_params = len(params)
        d_obs = self.get_dim_out()

        hess = np.zeros((d_obs, n_params, n_params))

        for j in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += h
            params_minus[j] -= h

            J_plus = self.eval_grad(params_plus,
                                    idx)  # shape (d_obs, n_params)
            J_minus = self.eval_grad(params_minus,
                                     idx)  # shape (d_obs, n_params)

            # H[:, :, j] = dJ/dθ_j
            hess[:, :, j] = (J_plus - J_minus) / (2 * h)

        # Symmetrize the Hessian for each output component to reduce numerical errors
        for i in range(d_obs):
            hess[i] = 0.5 * (hess[i] + hess[i].T)

        return hess

    @classmethod
    def from_dict(cls, config: dict, field=None):
        """Create a JaxFemModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary with keys like 'd_x', 'd_y', 'd_z',
                          'ele_type', 'nu', 'h', etc.
                          Optional 'sensors' key specifies sensor configurations. Each entry
                          must include 'location_fn' (string key or callable), plus either
                          'point' or 'points':

                              {"name": "s1", "location_fn": "bottom", "point": [x, y, z]}
                              {"name": "s2", "location_fn": "side_faces",
                               "points": [[x1,y1,z1], [x2,y2,z2]]}

            field (JaxRandomField, optional): JAX-compatible random field for spatially-varying properties

        Returns:
            JaxFemModel: JAX FEM model instance
        """
        # Extract configuration with defaults
        d_x = float(config.get('d_x', 1.0))
        d_y = float(config.get('d_y', 1.0))
        d_z = float(config.get('d_z', 2.5))
        ele_type = config.get('ele_type', 'HEX8')
        nu = float(config.get('nu', 0.3))
        h = float(config.get('h', 0.5))
        indenter_loc = config.get('indenter_loc', None)
        traction = config.get('traction', None)
        total_load = config.get('total_load', None)

        # Determine n_params from field if available
        if field is not None:
            n_params = field.dim
        else:
            n_params = int(config.get('n_params', 1))

        # Parse sensors configuration
        sensors = None
        if 'sensors' in config:
            sensors = []
            for sensor_spec in config['sensors']:
                sensor = {"name": sensor_spec["name"]}

                if "location_fn" not in sensor_spec:
                    raise ValueError(
                        f"Sensor '{sensor_spec['name']}' in config must define 'location_fn'."
                    )
                sensor["location_fn"] = sensor_spec["location_fn"]

                if "point" in sensor_spec:
                    sensor["point"] = np.array(sensor_spec["point"])
                elif "points" in sensor_spec:
                    sensor["points"] = np.array(sensor_spec["points"])
                else:
                    raise ValueError(
                        f"Sensor '{sensor_spec['name']}' must define 'point' or 'points'"
                    )

                sensors.append(sensor)

        return cls(
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            indenter_loc=indenter_loc,
            ele_type=ele_type,
            nu=nu,
            h=h,
            traction=traction,
            total_load=total_load,
            n_params=n_params,
            field=field,
            sensors=sensors,
        )


class RVEModel:
    """2D RVE with exponential recovery E-field for forward UQ.

    Not a subclass of Model — designed for propagating parameter samples
    and collecting multiple named output quantities.

    Parameters
    ----------
    fibers : list of (cx, cy, R)
        Fiber center coordinates and radii.
    L : float
        RVE side length.
    mesh_size : float
        Characteristic element length.
    ele_type : str
        Element type (default 'TRI3').
    E_inf : float
        Far-field matrix Young's modulus.
    E_fiber : float
        Fiber Young's modulus.
    nu_matrix : float
        Matrix Poisson ratio.
    nu_fiber : float
        Fiber Poisson ratio.
    eps_macro : tuple of float
        Macroscopic strain (eps_xx, eps_yy, gamma_xy).
    quantities : list of str
        Output quantities to compute.
    msh_file : str or None
        Path to pre-existing .msh file. If None, generates mesh.
    data_dir : str or None
        Directory for mesh generation output.
    """

    SUPPORTED_QUANTITIES = {
        'avg_stress',
        'avg_strain',
        'cell_stresses',
        'cell_strains',
        'displacements',
        'max_von_mises',
        'max_stress',
        'max_strain',
    }
    VOIGT_COMPONENTS = {'xx': (0, 0), 'yy': (1, 1), 'xy': (0, 1)}
    TENSOR_QUANTITIES = {
        'avg_stress', 'avg_strain', 'cell_stresses', 'cell_strains'
    }
    LATEX_LABELS = {
        'avg_stress_xx': r'$\sigma_{xx}^M$',
        'avg_stress_yy': r'$\sigma_{yy}^M$',
        'avg_stress_xy': r'$\sigma_{xy}^M$',
        'avg_strain_xx': r'$\varepsilon_{xx}^M$',
        'avg_strain_yy': r'$\varepsilon_{yy}^M$',
        'avg_strain_xy': r'$\varepsilon_{xy}^M$',
        'max_von_mises': r'$\sigma_\mathrm{VM}^\mathrm{max}$',
        'max_stress': r'$\sigma^\mathrm{max}$',
        'max_strain': r'$\varepsilon^\mathrm{max}$',
    }

    def __init__(self,
                 fibers,
                 L=1.0,
                 mesh_size=0.03,
                 ele_type='TRI3',
                 E_inf=30e3,
                 E_fiber=200e3,
                 nu_matrix=0.35,
                 nu_fiber=0.2,
                 eps_macro=(1e-3, 0.0, 0.0),
                 quantities=None,
                 components=None,
                 msh_file=None,
                 data_dir=None):
        from pdmp.rve_utils import (
            validate_fiber_placement,
            generate_multi_fiber_rve_mesh,
            build_periodic_pmat,
            compute_distance_to_nearest_fiber,
            make_eps_macro_q,
            LinearElasticRVE,
        )
        from jax_fem.solver import ad_wrapper
        import meshio
        from jax_fem.generate_mesh import Mesh, get_meshio_cell_type

        if quantities is None:
            quantities = ['avg_stress', 'max_von_mises']
        unknown = set(quantities) - self.SUPPORTED_QUANTITIES
        if unknown:
            raise ValueError(f"Unknown quantities: {unknown}")

        if components is None:
            components = ['xx', 'yy', 'xy']
        unknown_c = set(components) - set(self.VOIGT_COMPONENTS)
        if unknown_c:
            raise ValueError(
                f"Unknown components: {unknown_c}. Must be 'xx', 'yy', or 'xy'."
            )
        self._components = list(components)
        self._component_indices = [
            self.VOIGT_COMPONENTS[c] for c in components
        ]

        self.fibers = list(fibers)
        self.L = L
        self.E_inf = E_inf
        self.E_fiber = E_fiber
        self.nu_matrix = nu_matrix
        self.nu_fiber = nu_fiber
        self.quantities = list(quantities)
        self.eps_macro_voigt = np.array(eps_macro, dtype=np.float64)

        # Build or load mesh
        if msh_file is not None:
            cell_type = get_meshio_cell_type(ele_type)
            meshio_mesh = meshio.read(msh_file)
            points = meshio_mesh.points[:, :2]
            cells = meshio_mesh.cells_dict[cell_type]
            phys_tags = meshio_mesh.cell_data_dict["gmsh:physical"][cell_type]
            mesh = Mesh(points, cells, ele_type=ele_type)
        else:
            validate_fiber_placement(self.fibers, L, mesh_size)
            if data_dir is None:
                import tempfile
                data_dir = tempfile.mkdtemp(prefix="rve_")
            mesh, phys_tags = generate_multi_fiber_rve_mesh(L,
                                                            self.fibers,
                                                            mesh_size,
                                                            data_dir,
                                                            ele_type=ele_type)

        # Periodic constraints
        P_mat = build_periodic_pmat(mesh, L, vec=2)

        # Create jax-fem problem
        mat_props = dict(
            E_matrix=E_inf,
            nu_matrix=nu_matrix,
            E_aggregate=E_fiber,
            nu_aggregate=nu_fiber,
        )
        self._problem = LinearElasticRVE(
            mesh,
            vec=2,
            dim=2,
            ele_type=ele_type,
            additional_info=(phys_tags, mat_props),
        )
        self._problem.P_mat = P_mat
        self._mesh = mesh

        nc = len(self._problem.fe.cells)
        nq = self._problem.fe.num_quads

        # Precompute distance field and phase masks
        quad_points = np.array(self._problem.physical_quad_points)
        distances = compute_distance_to_nearest_fiber(quad_points, self.fibers)
        self._distances_jnp = jnp.array(distances)

        is_fiber = (phys_tags == 2)
        self._is_fiber_q = jnp.array(
            np.broadcast_to(is_fiber[:, None], (nc, nq)))

        self._nu_q = jnp.array(
            np.where(
                is_fiber[:, None],
                nu_fiber,
                nu_matrix,
            ) * np.ones((nc, nq)))

        self._eps_macro_q = make_eps_macro_q(self.eps_macro_voigt, nc, nq)

        # Precompute cell-averaged nu for von Mises
        JxW = np.array(self._problem.fe.JxW)
        w = JxW / JxW.sum(axis=1, keepdims=True)
        nu_q_np = np.array(self._nu_q)
        self._nu_cell = np.sum(nu_q_np * w, axis=1)

        # AD wrapper for solving
        self._fwd_pred = ad_wrapper(
            self._problem, adjoint_solver_options={"umfpack_solver": {}})

    def eval(self, params):
        """Evaluate the RVE model for given exponential recovery parameters.

        Parameters
        ----------
        params : array-like of shape (2,)
            [rho, l_scale] — recovery ratio and length scale.

        Returns
        -------
        dict
            Requested output quantities as numpy arrays.
        """
        from pdmp.rve_utils import compute_von_mises_from_cell

        params = np.asarray(params, dtype=np.float64)
        rho = jnp.float64(params[0])
        l_scale = jnp.float64(params[1])

        # Compute E field from parameters
        E_matrix_q = self.E_inf * (
            1.0 - (1.0 - rho) * jnp.exp(-self._distances_jnp / l_scale))
        E_q = jnp.where(self._is_fiber_q, self.E_fiber, E_matrix_q)

        fem_params = [E_q, self._nu_q, self._eps_macro_q]
        self._problem.set_params(fem_params)
        sol = self._fwd_pred(fem_params)[0]

        result = {}
        needs_stress = any(
            q in self.quantities for q in
            ['avg_stress', 'cell_stresses', 'max_von_mises', 'max_stress'])
        needs_strain = any(
            q in self.quantities
            for q in ['avg_strain', 'cell_strains', 'max_strain'])

        sigma_avg = sigma_cell = None
        if needs_stress:
            sigma_avg, sigma_cell = self._problem.compute_avg_stress(
                sol, fem_params)

        eps_cell = None
        if needs_strain:
            eps_cell = self._problem.compute_avg_strain(sol, self._eps_macro_q)

        def _extract(arr):
            """Extract selected components from a (..., 2, 2) tensor."""
            return np.stack(
                [arr[..., r, c] for r, c in self._component_indices], axis=-1)

        for q in self.quantities:
            if q == 'avg_stress':
                for comp, val in zip(self._components,
                                     _extract(np.array(sigma_avg))):
                    result[f'avg_stress_{comp}'] = float(val)
            elif q == 'avg_strain':
                eps_qp_avg = self._problem.compute_avg_strain(
                    sol, self._eps_macro_q) if eps_cell is None else eps_cell
                # Volume-average over cells
                JxW = self._problem.fe.JxW
                cell_vols = jnp.sum(JxW, axis=1)
                total_vol = jnp.sum(cell_vols)
                avg = jnp.sum(eps_qp_avg * cell_vols[:, None, None],
                              axis=0) / total_vol
                for comp, val in zip(self._components,
                                     _extract(np.array(avg))):
                    result[f'avg_strain_{comp}'] = float(val)
            elif q == 'cell_stresses':
                result[q] = _extract(np.array(sigma_cell))
            elif q == 'cell_strains':
                result[q] = _extract(np.array(eps_cell))
            elif q == 'displacements':
                result[q] = np.array(sol)
            elif q == 'max_von_mises':
                sigma_cell_np = np.array(sigma_cell)
                vm = compute_von_mises_from_cell(sigma_cell_np, self._nu_cell)
                result[q] = float(np.max(vm))
            elif q == 'max_stress':
                result[q] = float(np.max(np.abs(np.array(sigma_cell))))
            elif q == 'max_strain':
                result[q] = float(np.max(np.abs(np.array(eps_cell))))

        return result

    @staticmethod
    def from_dict(config):
        """Construct an RVEModel from a configuration dictionary."""
        fibers = [tuple(f) for f in config['fibers']]
        eps_macro = tuple(config.get('eps_macro', (1e-3, 0.0, 0.0)))
        return RVEModel(
            fibers=fibers,
            L=config.get('L', 1.0),
            mesh_size=config.get('mesh_size', 0.03),
            ele_type=config.get('ele_type', 'TRI3'),
            E_inf=config.get('E_inf', 30e3),
            E_fiber=config.get('E_fiber', 200e3),
            nu_matrix=config.get('nu_matrix', 0.35),
            nu_fiber=config.get('nu_fiber', 0.2),
            eps_macro=eps_macro,
            quantities=config.get('quantities',
                                  ['avg_stress', 'max_von_mises']),
            components=config.get('components', None),
            msh_file=config.get('msh_file', None),
            data_dir=config.get('data_dir', None),
        )


if __name__ == '__main__':

    # Use non-interactive backend if display may not be available
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:  # pragma: no cover
        pass

    # Example piecewise constant model
    F = np.array([1., 2.])
    x_obs = np.array([0.1, 0.2])
    n_params = 2
    model = PiecewiseConstantModel(F, n_params, x_obs)
    print(
        model.eval_E(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3, 0.6,
                                                     1.1])))

    # x = np.linspace(0, 1, 100)
    x = np.array([0.1, 0.2, 0.3, 0.6, 1.0])
    # params = np.array([0.1, 0.2])
    params_all = np.linspace(1, 5, 100)
    params_all = np.vstack((params_all, np.ones_like(params_all))).T
    # grads = model.eval_grad(np.array(x), np.array([0.1, 0.2]))

    for j in range(len(F)):
        grads = np.zeros((params_all.shape[0], len(x)))
        u = np.zeros((params_all.shape[0], len(x)))
        for i, params in enumerate(params_all):
            grads[i] = model.eval_grad(params, x, j)[:, 1]
            u[i] = model.eval(params, x, j)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # ax.plot(params_all[:, 0], grads[:, 0], label='dE/dp_0')
        # ax.plot(params_all[:, 0], grads[:, 4], label='dE/dp_0')
        for i in range(len(x)):
            ax.plot(params_all[:, 0], u[:, i], label=f'u(x={x[i]})')
        ax.legend()
        plt.show()

    print('Done!')

    # Example linear model
    rng = np.random.default_rng(0)

    n = 2
    m = 3
    A = rng.random((m, n))
    b = rng.random(m)
    model = LinearModel(A, b)
    # model = get_linear_model()
    # n = model.get_dim()
    # m = n

    x_true = rng.random(n)
    y_true = model.eval(x_true)
    print(f"True output: {y_true}")

    # # create a noisy observation
    # noise = 0.1 * np.random.randn(m)
    # y_obs = y_true + noise
    # print(f"Noisy observation: {y_obs}")

    # evaluate the model on a grid and plot contours
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = model.eval(np.array([x[i], y[j]]))[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.contourf(X, Y, Z, 100)
    plt.show()
