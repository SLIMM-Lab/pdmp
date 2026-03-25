import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh

from pdmp.forward_model import build_sensor_interpolants, evaluate_sensor_displacements

import time

import jax

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
print("Sample array device:", a.device)

TOTAL_FORCE = 40.0  # mN (GPa·µm² consistent units) — total y-force on load face
_traction_y = [0.0]  # computed after problem construction


class LinearElasticity(Problem):

    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):

        def stress(u_grad, E, rho):
            E_rho = E * rho
            nu = 0.3
            mu = E_rho / (2. * (1. + nu))
            lmbda = E_rho * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(
                self.dim) + 2 * mu * epsilon
            return sigma

        return stress

    def get_surface_maps(self):

        def surface_map(u, x):
            return np.array([0., _traction_y[0], 0])

        return [surface_map]

    def set_params(self, params):
        E, rho, scale_d, scale_t = params
        self.internal_vars = [np.ones_like(rho) * E, rho]
        # self.internal_vars = [rho]
        # self.fe.dirichlet_bc_info[-1][-2] = get_dirichlet_bottom(scale_d)
        # self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_top(scale_t)
        # self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)


ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
meshsize = 5
d_x, d_y, d_z = 100., 100., 220.
Nx = int(d_x / meshsize)
Ny = int(d_y / meshsize)
Nz = int(d_z / meshsize)
meshio_mesh = box_mesh_gmsh(Nx=Nx,
                            Ny=Ny,
                            Nz=Nz,
                            domain_x=d_x,
                            domain_y=d_y,
                            domain_z=d_z,
                            data_dir=data_dir,
                            ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


def get_dirichlet_bottom(scale):

    def dirichlet_bottom(point):
        z_disp = scale * d_z
        return z_disp

    return dirichlet_bottom


def get_dirichlet_top(scale):

    def dirichlet_top(point):
        z_disp = scale
        return z_disp

    return dirichlet_top


def zero_dirichlet_val(point):
    return 0.


def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)


def load(point):
    return (point[2] > 110. * np.isclose(point[1], d_y, atol=1e-5))


def top_face(point):
    return np.isclose(point[2], d_z, atol=1e-5)


def side_faces(point):
    return (np.isclose(point[0], 0., atol=1e-5) +
            np.isclose(point[0], d_x, atol=1e-5) +
            np.isclose(point[1], 0., atol=1e-5) +
            np.isclose(point[1], d_y, atol=1e-5))


location_fn_map = {
    'side_faces': side_faces,
    'top_face': top_face,
    'bottom': bottom,
    'load': load,
}

dirichlet_bc_info = [
    [bottom, bottom, bottom],  # location
    [0, 1, 2],  # dof
    [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]  # value
]
location_fns = [load]

# sensor_specs = [
#     {"name": "sensor_midspan", "point": onp.array([0.5*d_x, 0.5*d_y, d_z])},
#     {"name": "sensor_tip", "point": onp.array([0.9*d_x, 0.5*d_y, d_z])}
# ]

sensor_specs = [
    {
        "name": "sensor_left_center",
        "location_fn": "side_faces",
        "point": onp.array([0, 0.5 * d_y, 0.5 * d_z])
    },
    # {"name": "sensor_top_center", "location_fn": "top_face",
    #  "point": onp.array([0.5*d_x, d_y, d_z])}
]

# sensor_specs = [
#     {"name": "sensor_left_center", "location_fn": "side_faces",
#      "points": onp.array([[0, 0.5*d_y, 0.5*d_z],
#                           [0, 0.2*d_y, 0.8*d_z]])},
#     {"name": "sensor_top", "location_fn": "top_face",
#      "point": onp.array([0.4*d_x, 0.3*d_y, d_z])},
# ]

problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)

A_loaded = float(onp.sum(problem.nanson_scale[0][:, 0, :]))
_traction_y[0] = TOTAL_FORCE / A_loaded
print(
    f"Load surface area: {A_loaded:.2f} µm², traction: {_traction_y[0]:.6e} GPa"
)

sensor_interpolants = build_sensor_interpolants(
    problem.fe, sensor_specs, location_fn_map) if sensor_specs else []

rho = 0.5 * np.ones((problem.fe.num_cells, problem.fe.num_quads))
E = 50.
scale_d = 0.
scale_t = -0.1
params = [E, rho, scale_d, scale_t]

# fwd_pred = ad_wrapper(problem, linear=False, use_petsc=False)
fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(params)

vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fe, sol_list[0], vtk_path)

if sensor_interpolants:
    sensor_readings = evaluate_sensor_displacements(sol_list[0],
                                                    sensor_interpolants)
    print("Sensor displacements (ux, uy, uz):")
    for reading in sensor_readings:
        print(f"  {reading['name']} @ {reading['points']} -> {reading['u']}")


def fwd_with_sensors(E):
    params = [E, rho, scale_d, scale_t]
    sol_list = fwd_pred(params)
    sensor_readings = evaluate_sensor_displacements(sol_list[0],
                                                    sensor_interpolants)
    u_list = [jnp.ravel(reading["u"]) for reading in sensor_readings]
    return jnp.concatenate(u_list, axis=0) if u_list else jnp.array([])


# Compute VJP
sensor_vals, vjp_fn = jax.vjp(fwd_with_sensors, E)
print("Sensor values shape:", sensor_vals.shape)
v = np.ones_like(sensor_vals) * 1.
v = v.at[0].set(0.0)  # Only consider the first sensor for gradient computation
v = v.at[2].set(0.0)  # Only consider the first sensor for gradient computation
print("VJP vector shape:", v.shape)
print("VJP vector:", v)
grads = vjp_fn(v)
print("Sensor values:", sensor_vals)
print("Gradients:", grads)


def fd_gradient(E_val, eps=1e-5):
    """Finite difference gradient of sensor output w.r.t. E"""
    # Forward evaluation at E
    sensor_vals_fwd = fwd_with_sensors(E_val + eps)
    # Backward evaluation at E
    sensor_vals_bwd = fwd_with_sensors(E_val - eps)
    # Central difference
    grad_fd = (sensor_vals_fwd - sensor_vals_bwd) / (2 * eps)
    return grad_fd


# Compute FD gradient
E_test = 50.
grad_fd = fd_gradient(E_test)

# Apply same weighting as VJP
v = np.ones(3) * 1.
v = v.at[0].set(0.0)
v = v.at[2].set(0.0)
grad_fd_weighted = np.dot(v, grad_fd)

print("FD gradient (per component):", grad_fd)
print("FD gradient (weighted):", grad_fd_weighted)
print("VJP gradient:", grads[0])
print("Relative error:",
      np.abs(grad_fd_weighted - grads[0]) / np.abs(grads[0]))

# gx, gz = np.meshgrid(
#     np.linspace(0, d_x, Nx + 1),
#     np.linspace(0, d_z, Nz + 1)
# )
# gy = np.ones_like(gx) * d_y
#
# points = np.array([gx.flatten(), gy.flatten(), gz.flatten()])
# bc = np.array(top(points), dtype=float)
#
# fig, ax = plt.subplots()
# ax.contourf(gx, gz, bc.reshape(gx.shape), levels=1, cmap='viridis')
# plt.show()
