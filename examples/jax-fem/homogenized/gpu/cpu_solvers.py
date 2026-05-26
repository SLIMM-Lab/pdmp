"""
JaxFemModel forward solve: CPU linear-solver comparison.

Times a forward solve of the homogenized ITZ beam model on the CPU for each
of the linear solvers exposed by jax-fem (umfpack, petsc, jax BiCGStab),
including several PETSc ksp_type/pc_type combinations.

For every solver we report:
    build       — model construction (mesh + problem + ad_wrapper)
    first       — first .eval(...) call (includes XLA tracing/compilation)
    second      — second .eval(...) call (compiled kernel, the relevant cost)

Usage:
    python cpu_solvers.py [--h H] [--repeats N] [--solvers a,b,c]

    --h        Element size in µm (default: 10). Must divide x_interface=20.
    --repeats  Compiled solves averaged after the first one (default: 2).
    --solvers  Comma-separated subset of solver tags to run. Default: all.
               Tags: umfpack, petsc_bcgsl_ilu, petsc_gmres_ilu, petsc_cg_ilu,
                     petsc_cg_jacobi, petsc_cg_gamg, petsc_cg_hypre,
                     jax_bicgstab, jax_bicgstab_noprecond

Solvers that fail to construct or solve (e.g. missing PETSc preconditioner
backends such as hypre) are reported as 'failed' and the run continues.
"""
import argparse
import os
import time
import traceback

import numpy as np

# Compilation cache (active from first XLA compile, before jax import).
_cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR',
                            os.path.expanduser('~/.cache/jax'))
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _cache_dir)

import jax          # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pdmp.forward_model import JaxFemModel          # noqa: E402
from pdmp.random_field import JaxExponentialRecoveryField  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
ALL_SOLVERS = {
    'umfpack':                {'umfpack_solver': {}},
    'petsc_bcgsl_ilu':        {'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'ilu'}},
    'petsc_gmres_ilu':        {'petsc_solver': {'ksp_type': 'gmres', 'pc_type': 'ilu'}},
    'petsc_cg_ilu':           {'petsc_solver': {'ksp_type': 'cg',    'pc_type': 'ilu'}},
    'petsc_cg_jacobi':        {'petsc_solver': {'ksp_type': 'cg',    'pc_type': 'jacobi'}},
    'petsc_cg_hypre':         {'petsc_solver': {'ksp_type': 'cg',    'pc_type': 'hypre'}},
    'jax_bicgstab':           {'jax_solver':   {'precond': True}},
    'jax_bicgstab_noprecond': {'jax_solver':   {'precond': False}},
}

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--h', type=float, default=10.0,
                    help='element size in µm (default: 10)')
parser.add_argument('--repeats', type=int, default=2,
                    help='compiled solves after the first one (default: 2)')
parser.add_argument('--solvers', type=str, default='',
                    help='comma-separated subset of solver tags '
                         '(default: all). Available: '
                         + ', '.join(ALL_SOLVERS))
parser.add_argument('--output-dir', type=str,
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help='directory for the summary file '
                         '(default: this script\'s directory)')
parser.add_argument('--no-grad', action='store_true',
                    help='skip the gradient (adjoint) benchmark')
args = parser.parse_args()

if args.solvers:
    selected = [s.strip() for s in args.solvers.split(',') if s.strip()]
    unknown = [s for s in selected if s not in ALL_SOLVERS]
    if unknown:
        raise SystemExit(
            f"Unknown solver tag(s): {unknown}. "
            f"Available: {list(ALL_SOLVERS)}"
        )
    solver_configs = {s: ALL_SOLVERS[s] for s in selected}
else:
    solver_configs = ALL_SOLVERS

# ── Device pinning ────────────────────────────────────────────────────────────
cpu = jax.devices('cpu')[0]
jax.config.update('jax_default_device', cpu)
print(f"JAX devices: {jax.devices()}")
print(f"Pinned to:   {cpu}")

_y = jnp.dot(jnp.ones((4, 4)), jnp.ones((4, 4)))
_y.block_until_ready()
print(f"Smoke test — compute device: {list(_y.devices())}\n")

# ── Shared model config (copied from gpu_forward.py for consistency) ─────────
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

print(f"Mesh:   h={h} µm  →  {n_x}×{n_y}×{n_z}  "
      f"({n_x*n_y*n_z} elems, {n_nodes} nodes, {n_dofs} DOFs)")
print(f"Params: rho={PARAMS[0]}, l_scale={PARAMS[1]} µm, "
      f"f_inf={PARAMS[2]} GPa")
print(f"Repeats per solver: 1 first (JIT) + {args.repeats} compiled\n")

# ── Per-solver benchmark ──────────────────────────────────────────────────────
results = {}

for tag, sol_opts in solver_configs.items():
    print(f"{'='*64}")
    print(f"Solver: {tag}    options: {sol_opts}")
    print(f"{'='*64}")

    try:
        t_build = time.perf_counter()
        field = JaxExponentialRecoveryField.from_dict(FIELD_CONFIG)
        model = JaxFemModel.from_dict(
            {**MODEL_BASE, 'h': h,
             'solver_options': sol_opts,
             'adjoint_solver_options': sol_opts},
            field=field,
        )
        t_build = time.perf_counter() - t_build
        print(f"  build:        {t_build:6.2f} s")

        t0 = time.perf_counter()
        obs = model.eval(PARAMS)
        obs_arr = np.asarray(obs)
        t_first = time.perf_counter() - t0
        print(f"  first solve:  {t_first:6.2f} s  (incl. JIT)")

        compiled_times = []
        for k in range(args.repeats):
            t0 = time.perf_counter()
            obs2 = model.eval(PARAMS)
            np.asarray(obs2)
            dt = time.perf_counter() - t0
            compiled_times.append(dt)
            print(f"  compiled #{k+1}: {dt:6.2f} s")

        t_compiled = float(np.mean(compiled_times))

        entry = {
            'opts': sol_opts,
            'obs': obs_arr,
            't_build': t_build,
            't_first': t_first,
            't_compiled': t_compiled,
            't_compiled_all': compiled_times,
            'status': 'ok',
            'grad_status': 'skipped',
        }

        # Gradient (adjoint) benchmark. One forward + one adjoint solve per
        # call — what each MCMC/PDMP gradient step costs.
        # NB: differentiate _eval_obs, not eval(): the public eval() calls
        # np.asarray on the result, which breaks tracing. _eval_obs is the
        # JAX-compatible forward map (forward_model.py:1040).
        # NB: do NOT wrap in jax.jit. jax-fem's fwd_pred is a @jax.custom_vjp
        # whose primal is host-side Python/SciPy/PETSc — under jit the params
        # become tracers and `onp.array(...)` calls inside `solver()` raise
        # TracerArrayConversionError. jax-fem examples use raw jax.grad
        # (cf. applications/crystal_plasticity/calibration.py:110).
        if not args.no_grad:
            try:
                grad_fn = jax.grad(lambda p: jnp.sum(model._eval_obs(p)))

                t0 = time.perf_counter()
                g = grad_fn(PARAMS)
                g_arr = np.asarray(g)
                t_grad_first = time.perf_counter() - t0
                print(f"  grad first:   {t_grad_first:6.2f} s  (incl. JIT)")

                grad_times = []
                for k in range(args.repeats):
                    t0 = time.perf_counter()
                    g2 = grad_fn(PARAMS)
                    np.asarray(g2)
                    dt = time.perf_counter() - t0
                    grad_times.append(dt)
                    print(f"  grad #{k+1}:     {dt:6.2f} s")

                entry.update({
                    'grad': g_arr,
                    't_grad_first': t_grad_first,
                    't_grad_compiled': float(np.mean(grad_times)),
                    't_grad_compiled_all': grad_times,
                    'grad_status': 'ok',
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  GRAD FAILED: {type(exc).__name__}: {exc}")
                entry.update({
                    'grad_status': 'failed',
                    'grad_error': f"{type(exc).__name__}: {exc}",
                })

        results[tag] = entry
        print()
    except Exception as exc:  # noqa: BLE001 — we want to keep going
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results[tag] = {
            'opts': sol_opts,
            'status': 'failed',
            'error': f"{type(exc).__name__}: {exc}",
        }
        print()

# ── Summary table (tee'd to file) ─────────────────────────────────────────────
summary_lines: list[str] = []

def emit(line: str = '') -> None:
    print(line)
    summary_lines.append(line)

emit(f"{'='*78}")
emit(f"Summary — h={h} µm, {n_x}×{n_y}×{n_z} elements, {n_dofs} DOFs")
emit(f"{'='*78}")

ok_results = {t: r for t, r in results.items() if r['status'] == 'ok'}
failed = {t: r for t, r in results.items() if r['status'] != 'ok'}

def _step_cost(r):
    """fwd + grad compiled cost; falls back to fwd alone if grad missing."""
    if r.get('grad_status') == 'ok':
        return r['t_compiled'] + r['t_grad_compiled']
    return r['t_compiled']

if ok_results:
    fastest = min(ok_results, key=lambda t: _step_cost(ok_results[t]))
    t_ref = _step_cost(ok_results[fastest])

    header = (f"{'solver':<26} {'build':>8} {'fwd':>8} "
              f"{'grad':>8} {'step':>8} {'×fastest':>10}")
    emit(header)
    emit('-' * len(header))
    for tag, r in sorted(ok_results.items(), key=lambda kv: _step_cost(kv[1])):
        marker = '  ←' if tag == fastest else ''
        if r.get('grad_status') == 'ok':
            grad_col = f"{r['t_grad_compiled']:>8.2f}"
            step = _step_cost(r)
            step_col = f"{step:>8.2f}"
            ratio_col = f"{step/t_ref:>10.2f}"
        elif r.get('grad_status') == 'failed':
            grad_col = f"{'fail':>8}"
            step_col = f"{'—':>8}"
            ratio_col = f"{'—':>10}"
        else:  # skipped
            grad_col = f"{'—':>8}"
            step_col = f"{r['t_compiled']:>8.2f}"
            ratio_col = f"{_step_cost(r)/t_ref:>10.2f}"
        emit(f"{tag:<26} {r['t_build']:>8.2f} {r['t_compiled']:>8.2f} "
             f"{grad_col} {step_col} {ratio_col}{marker}")
    emit('(build/fwd/grad/step in seconds; step = fwd + grad, the per-sample cost)')

    # Result consistency vs. fastest solver (sanity check that all solvers
    # converged to the same displacement field).
    emit()
    emit(f"Result agreement vs. '{fastest}' (max |Δobs|):")
    ref_obs = ok_results[fastest]['obs']
    for tag, r in ok_results.items():
        if tag == fastest:
            continue
        max_diff = float(np.max(np.abs(r['obs'] - ref_obs)))
        rel = max_diff / (float(np.max(np.abs(ref_obs))) + 1e-30)
        emit(f"  {tag:<26} {max_diff:.2e}   ({rel:.1e} relative)")

    # Gradient consistency vs. fastest solver that produced a gradient.
    # Picked independently from `fastest` since the step-fastest solver may
    # have grad_status == 'skipped'/'failed'.
    grad_ok = {t: r for t, r in ok_results.items()
               if r.get('grad_status') == 'ok'}
    if len(grad_ok) >= 2:
        grad_ref = min(grad_ok, key=lambda t: grad_ok[t]['t_grad_compiled'])
        ref_grad = grad_ok[grad_ref]['grad']
        emit()
        emit(f"Gradient agreement vs. '{grad_ref}' (max |Δgrad|):")
        for tag, r in grad_ok.items():
            if tag == grad_ref:
                continue
            max_diff = float(np.max(np.abs(r['grad'] - ref_grad)))
            rel = max_diff / (float(np.max(np.abs(ref_grad))) + 1e-30)
            emit(f"  {tag:<26} {max_diff:.2e}   ({rel:.1e} relative)")

    grad_failed = [t for t, r in ok_results.items()
                   if r.get('grad_status') == 'failed']
    if grad_failed:
        emit()
        emit("Gradient (adjoint) failures:")
        for tag in grad_failed:
            emit(f"  {tag:<26} {ok_results[tag]['grad_error']}")
else:
    emit("No solver ran successfully.")

if failed:
    emit()
    emit(f"Failed solvers ({len(failed)}):")
    for tag, r in failed.items():
        emit(f"  {tag:<26} {r['error']}")

# h is a float; format with %g so h=10.0 → "h10", h=2.5 → "h2.5".
out_name = f"cpu_solvers_h{h:g}.txt"
out_path = os.path.join(args.output_dir, out_name)
os.makedirs(args.output_dir, exist_ok=True)
with open(out_path, 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')

print(f"\nSummary written to: {out_path}")
print(f"Cache dir:          {_cache_dir}")
