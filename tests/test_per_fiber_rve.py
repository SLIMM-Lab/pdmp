"""Tests for per-fiber ITZ assembly and the empirical-mixture distribution."""

import os
import tempfile

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from pdmp.distributions import PerFiberEmpiricalMixture
from pdmp.rve_utils import compute_per_fiber_distance_assignment


def test_per_fiber_distance_assignment_basic():
    """Two fibers, hand-checked nearest-fiber assignment and zero-clamp."""
    quad_points = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 10.0], [5.0, 5.0]],
    ])
    fibers = [(0.0, 0.0, 1.0), (10.0, 0.0, 2.0)]
    per_fiber, nearest_idx, nearest = compute_per_fiber_distance_assignment(
        quad_points, fibers)

    assert per_fiber.shape == (2, 2, 2)
    assert nearest_idx.shape == (2, 2)
    assert nearest.shape == (2, 2)

    # (0,0) is on fiber 0 surface → distance 0 to fiber 0, 8 to fiber 1
    assert per_fiber[0, 0, 0] == pytest.approx(0.0)
    assert per_fiber[0, 0, 1] == pytest.approx(8.0)
    assert nearest_idx[0, 0] == 0

    # (10,0) is on fiber 1 surface → distance 9 to fiber 0, 0 to fiber 1
    assert per_fiber[0, 1, 0] == pytest.approx(9.0)
    assert per_fiber[0, 1, 1] == pytest.approx(0.0)
    assert nearest_idx[0, 1] == 1

    # midpoint (5,5) is closer to fiber 0 (sqrt(50)-1) than fiber 1 (sqrt(50)-2)?
    # Actually fiber 1 is closer: sqrt(50)-2 ≈ 5.07 < sqrt(50)-1 ≈ 6.07
    assert nearest_idx[1, 1] == 1


def test_per_fiber_distance_clamps_inside_fiber():
    """Quad points strictly inside a fiber should have zero distance to it."""
    quad_points = np.array([[[0.5,
                              0.0]]])  # inside the unit-radius fiber at origin
    fibers = [(0.0, 0.0, 1.0)]
    per_fiber, _, nearest = compute_per_fiber_distance_assignment(
        quad_points, fibers)
    assert per_fiber[0, 0, 0] == pytest.approx(0.0)
    assert nearest[0, 0] == pytest.approx(0.0)


def test_per_fiber_empirical_mixture_shape_and_indices(tmp_path):
    """get_sample returns the right shape and only pulls the requested columns."""
    chains = []
    for g in range(3):
        # Each chain is 50 rows × 5 cols, with column j = 100*g + j so we can
        # verify that the right geometry's row was taken.
        chain = np.tile(np.arange(5, dtype=float)[None, :], (50, 1))
        chain += 100.0 * g
        path = tmp_path / f"chain_{g}.dat"
        np.savetxt(path, chain)
        chains.append(str(path))

    rng = np.random.default_rng(0)
    dist = PerFiberEmpiricalMixture(sample_files=chains,
                                    n_fibers=4,
                                    param_indices=[0, 2, 3],
                                    rng=rng)

    assert dist.dim == 12  # 4 fibers × 3 columns

    s = dist.get_sample(1)
    assert s.shape == (12, )

    s_batch = dist.get_sample(7)
    assert s_batch.shape == (7, 12)

    # Each fiber-block must look like (col_0, col_2, col_3) of *some* geometry,
    # i.e. (g*100 + 0, g*100 + 2, g*100 + 3) for some g ∈ {0, 1, 2}.
    for row in s_batch:
        for f in range(4):
            block = row[f * 3:(f + 1) * 3]
            g = round(block[0] / 100.0)
            assert 0 <= g <= 2
            assert block == pytest.approx(
                [g * 100 + 0, g * 100 + 2, g * 100 + 3])


def test_per_fiber_empirical_mixture_fixed_assignment(tmp_path):
    """Fixed mode binds each fiber to one geometry for the whole run."""
    chains = []
    for g in range(2):
        chain = np.full((10, 1), float(g))
        path = tmp_path / f"chain_{g}.dat"
        np.savetxt(path, chain)
        chains.append(str(path))

    rng = np.random.default_rng(123)
    dist = PerFiberEmpiricalMixture(sample_files=chains,
                                    n_fibers=5,
                                    param_indices=[0],
                                    assignment_mode='fixed',
                                    rng=rng)
    fixed = dist._fixed_geom_idx
    assert fixed.shape == (5, )

    samples = dist.get_sample(20)
    # Fibre f always sources from chain `fixed[f]`, so column f equals fixed[f].
    for f in range(5):
        assert np.all(samples[:, f] == fixed[f])


def test_per_fiber_empirical_mixture_weight_frequency(tmp_path):
    """Mixture weights must shape the long-run frequency of each geometry."""
    chains = []
    for g in range(3):
        chain = np.full((1, 1), float(g))
        path = tmp_path / f"chain_{g}.dat"
        np.savetxt(path, chain)
        chains.append(str(path))

    rng = np.random.default_rng(7)
    weights = np.array([0.7, 0.2, 0.1])
    dist = PerFiberEmpiricalMixture(sample_files=chains,
                                    n_fibers=1,
                                    param_indices=[0],
                                    weights=weights,
                                    assignment_mode='per_sample',
                                    rng=rng)

    samples = dist.get_sample(20000).ravel()
    counts = np.bincount(samples.astype(int), minlength=3) / samples.size
    assert counts == pytest.approx(weights, abs=0.02)


def test_per_fiber_rve_voronoi_recovers_global_when_uniform():
    """With identical params for every fiber, Voronoi PerFiberRVEModel
    must produce the same matrix-E field as the global RVEModel."""
    pytest.importorskip("gmsh")
    from pdmp.forward_model import RVEModel, PerFiberRVEModel
    import jax.numpy as jnp

    fibers = [(0.3, 0.3, 0.1), (0.7, 0.7, 0.1)]
    common_kwargs = dict(
        fibers=fibers,
        L=1.0,
        mesh_size=0.08,
        E_inf=10.0,
        E_fiber=20.0,
        nu_matrix=0.2,
        nu_fiber=0.2,
        eps_macro=(1e-3, 0.0, 0.0),
        quantities=['avg_stress'],
    )
    base = RVEModel(**common_kwargs)
    pf = PerFiberRVEModel(assembly='voronoi',
                          f_inf_scope='per_fiber',
                          **common_kwargs)

    rho, l_scale, f_inf = 0.5, 0.05, 10.0
    base_E = np.asarray(base._matrix_E_from_params([rho, l_scale]))
    pf_params = np.tile([rho, l_scale, f_inf], len(fibers))
    pf_E = np.asarray(pf._matrix_E_from_params(pf_params))

    np.testing.assert_allclose(pf_E, base_E, rtol=1e-10, atol=1e-10)


def test_per_fiber_rve_voronoi_owns_quad_points():
    """Each Voronoi cell must use *its* fiber's params, not its neighbour's."""
    pytest.importorskip("gmsh")
    from pdmp.forward_model import PerFiberRVEModel
    import jax.numpy as jnp

    fibers = [(0.25, 0.5, 0.1), (0.75, 0.5, 0.1)]
    pf = PerFiberRVEModel(
        fibers=fibers,
        L=1.0,
        mesh_size=0.08,
        E_inf=10.0,
        E_fiber=20.0,
        nu_matrix=0.2,
        nu_fiber=0.2,
        eps_macro=(1e-3, 0.0, 0.0),
        quantities=['avg_stress'],
        assembly='voronoi',
        f_inf_scope='per_fiber',
    )
    # Fiber 0: rho=1 (no recovery deficit) and Einf=1.
    # Fiber 1: rho=0 (full deficit at the surface) and Einf=2.
    params = np.array([1.0, 0.05, 1.0, 0.0, 0.05, 2.0])
    E = np.asarray(pf._matrix_E_from_params(params))
    nearest = np.asarray(pf._nearest_fiber_idx_jnp)

    # In fiber-0 cells: E = Einf_0 * (1 - 0 * exp(...)) = 1 exactly everywhere.
    cell0 = nearest == 0
    cell1 = nearest == 1
    assert cell0.any() and cell1.any()
    assert np.allclose(E[cell0], 1.0, atol=1e-12)
    # In fiber-1 cells: E = 2 * (1 - exp(-d/0.05)). At quad points well outside
    # fiber 1 the value approaches 2; never below 0.
    assert (E[cell1] >= 0.0).all()
    assert (E[cell1] <= 2.0 + 1e-12).all()
