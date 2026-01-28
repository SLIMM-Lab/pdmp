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

    def __init__(self, F: np.ndarray, n_params: int, x_obs: np.ndarray, field=None):
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
            raise ValueError("Either n_params or field must define parameter dimension.")

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
            raise ValueError(f"params must have length {n}, got {params.shape[0]}")
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("All x must be within [0, 1]")
        # Interval index I = ceil(x * n); clip to [1, n]
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        params_eff = self._effective_params(params)
        return params_eff[I - 1]

    def eval(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
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
            raise ValueError("effective params must be a vector of length n_params for analytic evaluation")
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p = 1.0 / params_eff
        cumsum_inv = np.concatenate(([0.0], np.cumsum(inv_p)))  # length n+1
        base_sum = (1.0 / n) * cumsum_inv[I - 1]
        final_term = (x - (I - 1) / n) / params_eff[I - 1]
        return self.F_vals[idx] * (base_sum + final_term)

    def eval_grad(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
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
            raise ValueError("effective params must be a vector of length n_params for analytic gradient")
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
            G_eff[row, Ii - 1] = -(x[row] - (Ii - 1) / n) / (params_eff[Ii - 1]**2)
        G_eff *= Fval
        # Chain rule to parameters: J = ∂p/∂a = Φ(mid)
        J = self._J_params_eff()
        return G_eff @ J

    def eval_hessian(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
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
            raise ValueError("effective params must be a vector of length n_params for analytic Hessian")
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p3 = 1.0 / (params_eff**3)
        # Hessian w.r.t effective params p
        H_eff = np.zeros((m, n, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            if Ii - 1 > 0:
                H_eff[row, :Ii - 1, :Ii - 1] = np.diag(2.0 / n * inv_p3[:Ii - 1])
            H_eff[row, Ii - 1, Ii - 1] = 2.0 * (x[row] - (Ii - 1) / n) * inv_p3[Ii - 1]
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
        field: Optional GaussianRandomField to inject into model (if supported)

    Returns:
        Model: model
    """
    if config['name'] == 'PiecewiseConstant':
        return PiecewiseConstantModel.from_dict(config, field=field)
    elif config['name'] == 'Linear':
        return LinearModel.from_dict(config)
    else:
        raise ValueError(f"Model {config['name']} not recognized.")


def build_sensor_interpolants(fe, sensors, location_fn, tol=1e-8):
    """Build interpolation data for each sensor on a given boundary.

    Each sensor specification in ``sensors`` may be either:

    * Legacy single-point form::

        {"name": "sensor_name", "point": [x, y, z]}

    * Multi-point form::

        {"name": "sensor_name", "points": [[x1, y1, z1], [x2, y2, z2], ...]}

    Internally, each sensor is represented with arrays of shape
    (n_points, n_face_nodes) for ``nodes`` and ``weights`` so that
    multiple physical points can be associated with a single sensor.

    No aggregation across points is performed here; all point-wise
    displacements are returned by :func:`evaluate_sensor_displacements`.
    """
    boundary_inds_list = fe.get_boundary_conditions_inds([location_fn])
    if not boundary_inds_list:
        raise ValueError("No boundary faces found for sensor location function.")
    boundary_inds = boundary_inds_list[0]
    if boundary_inds.size == 0:
        raise ValueError("Boundary selector returned zero faces.")

    interpolants = []
    for spec in sensors:
        if "name" not in spec:
            raise ValueError("Each sensor spec must contain a 'name' key.")

        # Resolve list of points for this sensor (legacy or new style)
        if "points" in spec:
            points = np.asarray(spec["points"], dtype=float)
            if points.ndim == 1:
                points = points[None, :]
        elif "point" in spec:
            points = np.asarray(spec["point"], dtype=float)[None, :]
        else:
            raise ValueError(f"Sensor '{spec['name']}' must define 'point' or 'points'.")

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
                    f"Sensor {spec['name']} point {p} not located on selected boundary."
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

        # For backward compatibility, keep a single 'point' entry when applicable
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
        return quad_barycentric_weights(point, coords, tol)
    raise ValueError(f"Unsupported face with {len(coords)} nodes for interpolation.")

def quad_barycentric_weights(point, quad_coords, tol):
    tri_sets = ((0, 1, 2), (0, 2, 3))
    for tri in tri_sets:
        inside, weights = point_in_triangle(point, quad_coords[list(tri)], tol)
        if inside:
            face_weights = np.zeros(len(quad_coords))
            for local_idx, w in zip(tri, weights):
                face_weights[local_idx] = w
            return face_weights
    return None

def point_in_triangle(point, tri, tol):
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = point - tri[0]
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
    inside = (u >= -tol) and (v >= -tol) and (w >= -tol) and (u + v <= 1.0 + tol)
    return inside, (w, v, u)

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
    """Dummy JAX-based FEM model wrapper.

    This is a placeholder illustrating the interface required for efficient
    likelihood gradient computation via VJP (Jacobian-vector product).

    The real implementation should:
      - Perform a single forward FEM solve in `linearize` using jax.vjp.
      - Return observed displacement components (shape (d,)) from `eval`.
      - Support J^T v via the closure returned from `linearize` or via `eval_vjp`.
      - Optionally provide full Jacobian via `eval_grad` and Hessian via `eval_hessian`.

    Parameters
    ----------
    """
    def __init__(self, d_x: float, d_y: float, d_z: float, ele_type: str = 'HEX8', nu: float = 0.3, h: float = 0.5,
                 d_u: float = -0.1, traction=None, obs_loc: np.ndarray = None, n_params: int = 1, d_obs: int = 1):
        super().__init__()

        if traction is None:
            traction = [0., .015, 0.]
        self._h = h
        self._obs_loc = obs_loc
        self._n_params = n_params

        import logging
        logging.disable(logging.INFO)

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
                    sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2 * mu * epsilon
                    return sigma

                return stress

            def get_surface_maps(self):
                def surface_map(u, x):
                    return jnp.array(traction)

                return [surface_map]

            def set_params(self, params):
                E  = params[0]
                self.internal_vars = [E]
                # self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_top()
                # self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)

        def get_dirichlet_top():
            def dirichlet_top(point):
                return d_u
            return dirichlet_top

        def zero_dirichlet_val(point):
            return 0.

        def bottom(point):
            return jnp.isclose(point[2], 0., atol=1e-5)

        def top(point):
            return (point[2] > 2.0) * jnp.isclose(point[1], d_y, atol=1e-5)
            # return ((point[0] - d_x / 2.) ** 2 + (point[2] - d_z - 0.75) ** 2 > 1.8
            #         * jnp.isclose(point[1], d_y, atol=1e-5))

        def top_surface(point):
            return jnp.isclose(point[2], d_z, atol=1e-5)

        def side_faces(point):
            return (jnp.isclose(point[0], 0., atol=1e-5) +
                    jnp.isclose(point[0], d_x, atol=1e-5) +
                    jnp.isclose(point[1], 0., atol=1e-5) +
                    jnp.isclose(point[1], d_y, atol=1e-5))

        dirichlet_bc_info = [
            [bottom,             bottom,             bottom], # location
            [0,                  1,                  2],   # dof
            [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] # value
        ]

        location_fns = [top]

        sensor_specs = [
            {"name": "sensor_left_center", "point": np.array([0, 0.5*d_y, 0.5*d_z])},
        ]

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        cell_type = get_meshio_cell_type(ele_type)
        n_x = int(d_x/h)
        n_y = int(d_y/h)
        n_z = int(d_z/h)
        meshio_mesh = box_mesh_gmsh(
            Nx=n_x, Ny=n_y, Nz=n_z,
            domain_x=d_x, domain_y=d_y, domain_z=d_z,
            data_dir=data_dir, ele_type=ele_type
        )
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

        self.problem = LinearElasticity(
            mesh, vec=3, dim=3, ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns
        )

        self.sensor_interpolants = build_sensor_interpolants(self.problem.fe, sensor_specs, side_faces) if sensor_specs else []

        # rho = 0.5*jnp.ones((self.problem.fe.num_cells, self.problem.fe.num_quads))
        E = 1.e6
        params = [E]

        self.fwd_pred = ad_wrapper(self.problem)

        # infer observed dimension from sensor setup (if any)
        if self.sensor_interpolants:
            # total observed dofs = total point-wise displacements from all sensors
            sample_sol = jnp.zeros((self.problem.fe.num_total_nodes, 3))
            sample_readings = evaluate_sensor_displacements(sample_sol, self.sensor_interpolants)
            self._d_obs = sum(int(r["u"].size) for r in sample_readings)
        else:
            self._d_obs = d_obs

    @override
    def get_dim_in(self) -> int:
        return self._n_params

    @override
    def get_dim_out(self) -> int:
        return self._d_obs

    def _eval_obs(self, params: jnp.ndarray, idx: int = 0) -> jnp.ndarray:
        """JAX-compatible forward map params -> observation vector.

        This function is purely functional and suitable for jax.vjp, jacrev, etc.
        It performs a FEM solve via `self.fwd_pred` and then extracts the
        concatenated sensor displacements.
        """
        # Broadcast scalar/global params to per-cell/quadrature field
        param_field = params * jnp.ones((self.problem.fe.num_cells, self.problem.fe.num_quads))
        sol_list = self.fwd_pred([param_field])
        sensor_readings = evaluate_sensor_displacements(sol_list[0], self.sensor_interpolants)
        u_list = [jnp.ravel(reading["u"]) for reading in sensor_readings]
        return jnp.concatenate(u_list, axis=0) if u_list else jnp.array([])

    def eval(self, params: np.ndarray, idx: int = 0, save: bool = False) -> np.ndarray:  # noqa: D401
        """Evaluate observed displacement components.

        Returns the concatenated vector of per-point sensor displacements.
        """
        from jax_fem.utils import save_sol

        theta = jnp.asarray(params)
        y = self._eval_obs(theta, idx)

        # Optionally still write full-field VTK for debugging
        # Recompute with full field to save; this keeps eval() side-effects
        param_field = theta * jnp.ones((self.problem.fe.num_cells, self.problem.fe.num_quads))
        sol_list = self.fwd_pred([param_field])
        if save:
            data_dir = './data'
            os.makedirs(data_dir, exist_ok=True)
            vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
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
            (g,) = vjp_handle(jnp.asarray(v))
            return np.asarray(g)

        return np.asarray(y), apply_vjp

    def eval_vjp(self, params: np.ndarray, idx: int, v: np.ndarray) -> np.ndarray:
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
            (g_i,) = vjp_handle(e_i)
            J_rows.append(g_i)
        J = jnp.stack(J_rows, axis=0)  # (d_obs, n_params)
        return np.asarray(J)

    def eval_hessian(self, params: np.ndarray, idx: int = 0, h: float = 1e-5) -> np.ndarray:
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

            J_plus = self.eval_grad(params_plus, idx)   # shape (d_obs, n_params)
            J_minus = self.eval_grad(params_minus, idx)  # shape (d_obs, n_params)

            # H[:, :, j] = dJ/dθ_j
            hess[:, :, j] = (J_plus - J_minus) / (2 * h)

        # Symmetrize the Hessian for each output component to reduce numerical errors
        for i in range(d_obs):
            hess[i] = 0.5 * (hess[i] + hess[i].T)

        return hess

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
        model.eval_E(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3, 0.6, 1.1])))

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
