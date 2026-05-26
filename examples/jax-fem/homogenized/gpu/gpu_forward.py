"""
JaxFemModel forward solve: GPU vs CPU comparison.

Runs the homogenized ITZ beam forward model on both the ROCm GPU and the CPU,
then prints a side-by-side timing and result comparison.

Usage:
    python gpu_forward.py [--h H] [--no-compare]

    --h           Element size in µm (default: 5).
                  Must divide x_interface=20 to keep the mesh conformal.
    --no-compare  Skip the CPU run; only benchmark the GPU (or CPU if no GPU).

Prerequisites:
    conda activate pdmp-jax-amd   # env with jax-rocm7-plugin installed
    # verify GPU: python -c "import jax; print(jax.devices())"
    # expected:   [RocmDevice(id=0)]

If JAX reports CPU only, try:
    export HSA_OVERRIDE_GFX_VERSION=12.0.0

JAX compilation cache avoids recompiling on subsequent runs:
    export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
"""
import argparse
import os
import time

import numpy as np

# Set compilation cache before jax initialises so it is active from the first
# XLA compilation.
_cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                            os.path.expanduser('~/.cache/jax'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _cache_dir)

import jax          # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pdmp.forward_model import JaxFemModel          # noqa: E402
from pdmp.random_field import JaxExponentialRecoveryField  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--h', type=float, default=10.0,
                    help='element size in µm (default: 10)')
parser.add_argument('--no-compare', action='store_true',
                    help='skip the CPU comparison run')
args = parser.parse_args()

# ── Detect devices ────────────────────────────────────────────────────────────
# jax.devices('rocm') is the reliable way — the .platform attribute on
# individual RocmDevice objects returns 'gpu', not 'rocm'.
try:
    rocm_devices = jax.devices('rocm')
except RuntimeError:
    rocm_devices = []
try:
    cpu_devices = jax.devices('cpu')
except RuntimeError:
    cpu_devices = []

print("Available JAX devices:", jax.devices())
if not rocm_devices:
    print("WARNING: no RocmDevice found.")
    print("  Check: pip show jax-rocm7-plugin")
    print("  Check: export HSA_OVERRIDE_GFX_VERSION=12.0.0")

# Devices to benchmark: GPU first (if present), then CPU (if compare enabled).
devices_to_run = []
if rocm_devices:
    devices_to_run.append(('GPU', rocm_devices[0]))
elif cpu_devices:
    devices_to_run.append(('CPU', cpu_devices[0]))

if not args.no_compare and cpu_devices and rocm_devices:
    devices_to_run.append(('CPU', cpu_devices[0]))

# ── Shared model config ───────────────────────────────────────────────────────
PARAMS = np.array([0.5, 70.0, 50.0])   # rho, l_scale (µm), f_inf (GPa)

FIELD_CONFIG = {
    'name': 'JaxExponentialRecoveryField',
    'infer_f_infinity': True,
    'idx': 2,
    'use_interface': True,
    'x_interface': 20.0,
    'f_constant': 70.0,
    'coefficient_distribution': {
        'name': 'MultivariateNormal',
        'mean': [0.0, 4.042, 0.020],
        'cov': [[1.44, 0.0, 0.0],
                [0.0,  0.41, 0.0],
                [0.0,  0.0,  1.02]],
    },
}

MODEL_BASE = {
    'd_x': 100.0,
    'd_y': 100.0,
    'd_z': 220.0,
    'nu': 0.18,
    'indenter_loc': 110.0,
    'total_load': [0.0, 60.0, 0.0],
    'sensors': [
        {'name': 'load_mid', 'location_fn': 'side_faces',
         'point': [50.0, 100.0, 165.0]},
        {'name': 'top_ctr', 'location_fn': 'top_face',
         'point': [50.0, 50.0, 220.0]},
    ],
}

# Per-device solver configs: GPU uses JAX BiCGStab + Jacobi (GPU-compatible);
# CPU uses PETSc bcgsl + ILU (direct/semi-direct, much faster convergence on CPU).
SOLVER_OPTIONS = {
    'GPU': {'jax_solver':   {'precond': True}},
    'CPU': {'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'ilu'}},
}

OBS_LABELS = [
    'load_mid_ux', 'top_ctr_ux',
    'load_mid_uy', 'top_ctr_uy',
    'load_mid_uz', 'top_ctr_uz',
]

h = args.h
n_x = int(MODEL_BASE['d_x'] / h)
n_y = int(MODEL_BASE['d_y'] / h)
n_z = int(MODEL_BASE['d_z'] / h)
n_nodes = (n_x + 1) * (n_y + 1) * (n_z + 1)
n_dofs  = n_nodes * 3

print(f"\nMesh:  h={h} µm  →  {n_x}×{n_y}×{n_z}  "
      f"({n_x*n_y*n_z} elems, {n_nodes} nodes, {n_dofs} DOFs)")
print(f"Params: rho={PARAMS[0]}, l_scale={PARAMS[1]} µm, f_inf={PARAMS[2]} GPa\n")

# ── Per-device benchmark ──────────────────────────────────────────────────────
results = {}

for label, device in devices_to_run:
    print(f"{'='*60}")
    print(f"Running on {label}: {device}")
    print(f"{'='*60}")

    jax.config.update('jax_default_device', device)

    # Smoke test
    _y = jnp.dot(jnp.ones((4, 4)), jnp.ones((4, 4)))
    _y.block_until_ready()
    print(f"Smoke test passed — compute device: {list(_y.devices())}")

    # Build model (mesh construction is numpy/gmsh; field eval and FEM solve
    # are JAX and will target `device` via the default device setting above).
    t_build = time.perf_counter()
    sol_opts = SOLVER_OPTIONS.get(label, SOLVER_OPTIONS['CPU'])
    field = JaxExponentialRecoveryField.from_dict(FIELD_CONFIG)
    model = JaxFemModel.from_dict(
        {**MODEL_BASE, 'h': h,
         'solver_options': sol_opts,
         'adjoint_solver_options': sol_opts},
        field=field,
    )
    t_build = time.perf_counter() - t_build
    print(f"Model build: {t_build:.2f} s")

    # First solve: includes XLA JIT compilation.
    t0 = time.perf_counter()
    obs = model.eval(PARAMS)
    obs_arr = np.asarray(obs)
    t_first = time.perf_counter() - t0
    print(f"First solve (incl. JIT): {t_first:.2f} s")

    # Second solve: compiled kernel, actual wall-clock cost.
    t0 = time.perf_counter()
    obs2 = model.eval(PARAMS)
    np.asarray(obs2)
    t_second = time.perf_counter() - t0
    print(f"Second solve (compiled): {t_second:.2f} s\n")

    results[label] = {
        'device': device,
        'solver': list(sol_opts.keys())[0],
        'obs': obs_arr,
        't_build': t_build,
        't_first': t_first,
        't_second': t_second,
    }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")

col_w = 14
header = f"{'Observation':<16}" + "".join(
    f"  {lbl:>{col_w}}" for lbl in results
)
print(header)
print("-" * len(header))
for obs_label in OBS_LABELS:
    row = f"{obs_label:<16}"
    for lbl, r in results.items():
        idx = OBS_LABELS.index(obs_label)
        row += f"  {r['obs'][idx]:>{col_w}.6e}"
    print(row)

print()
for lbl, r in results.items():
    print(f"{lbl} ({r['device']}, {r['solver']})")
    print(f"  build:          {r['t_build']:6.2f} s")
    print(f"  first solve:    {r['t_first']:6.2f} s  (incl. JIT)")
    print(f"  second solve:   {r['t_second']:6.2f} s  (compiled kernel)")

if len(results) == 2:
    labels = list(results)
    r0, r1 = results[labels[0]], results[labels[1]]
    speedup = r1['t_second'] / r0['t_second']
    print(f"\nSpeedup ({labels[0]} vs {labels[1]}, compiled): {speedup:.2f}×")
    max_diff = np.max(np.abs(r0['obs'] - r1['obs']))
    print(f"Max obs diff between devices:               {max_diff:.2e}")

print(f"\nCache dir: {_cache_dir}")
