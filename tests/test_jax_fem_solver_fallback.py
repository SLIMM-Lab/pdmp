"""Tests for the UMFPACK fallback in JaxFemModel.

The primary solver is configured to fail by monkey-patching ``fwd_pred`` with
a wrapper that raises ``AssertionError`` — exactly how jax-fem signals a
PETSc convergence failure (``assert err < 0.1`` in ``jax_fem/solver.py``).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pdmp.forward_model import JaxFemModel, ForwardModelFailure


def _broken_wrapper(_):
    """Stand-in wrapper that always raises AssertionError on call."""
    assert False, 'forced solver failure for test'


def _make_adjoint_failing_wrapper(forward_wrapper):
    """Build a jax.custom_vjp closure whose forward succeeds via
    ``forward_wrapper`` but whose adjoint always raises AssertionError.
    """

    @jax.custom_vjp
    def fwd_pred(params):
        return forward_wrapper(params)

    def f_fwd(params):
        sol_list = forward_wrapper(params)
        return sol_list, None

    def f_bwd(_res, _v):
        assert False, 'forced adjoint failure for test'

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred


@pytest.fixture(scope='module')
def model():
    """Small JaxFemModel suitable for fast solver-fallback tests."""
    return JaxFemModel(
        d_x=1.0,
        d_y=1.0,
        d_z=1.0,
        h=0.5,
        n_params=1,
        solver_options={'petsc_solver': {}},
        adjoint_solver_options={'petsc_solver': {}},
    )


def _reset_counters(model):
    model._petsc_fallbacks = 0
    model._umfpack_failures = 0


def test_both_wrappers_built(model):
    """The model carries a primary and a UMFPACK fallback ad_wrapper."""
    assert model.fwd_pred is not None
    assert model.fwd_pred_umfpack is not None
    assert model.fwd_pred is not model.fwd_pred_umfpack


def test_eval_no_fallback_on_success(model):
    """A successful primary solve does not touch the fallback counters."""
    _reset_counters(model)
    y = model.eval(np.array([10.0]))
    assert y.shape[0] > 0
    assert model._petsc_fallbacks == 0
    assert model._umfpack_failures == 0


def test_eval_falls_back_on_primary_failure(model):
    """A failing primary forward solve transparently falls back to UMFPACK."""
    _reset_counters(model)
    real_primary = model.fwd_pred
    try:
        model.fwd_pred = _broken_wrapper
        y_fb = model.eval(np.array([10.0]))
        assert y_fb.shape[0] > 0
        assert model._petsc_fallbacks == 1
        assert model._umfpack_failures == 0
    finally:
        model.fwd_pred = real_primary


def test_linearize_falls_back_on_primary_forward_failure(model):
    """linearize falls back when the primary forward fails; the returned
    vjp_fun also works because it was built over the UMFPACK wrapper.
    """
    _reset_counters(model)
    real_primary = model.fwd_pred
    try:
        model.fwd_pred = _broken_wrapper
        y, vjp_fun = model.linearize(np.array([10.0]))
        assert y.shape[0] > 0
        assert model._petsc_fallbacks == 1
        # vjp through the fallback should also work
        g = vjp_fun(np.ones_like(y))
        assert g.shape == (1, )
        # vjp ran inside the UMFPACK closure, so no additional fallback
        assert model._petsc_fallbacks == 1
        assert model._umfpack_failures == 0
    finally:
        model.fwd_pred = real_primary


def test_linearize_falls_back_on_primary_adjoint_failure(model):
    """When the primary forward succeeds but its adjoint fails, the
    returned vjp_fun rebuilds with UMFPACK and recovers.
    """
    _reset_counters(model)
    real_primary = model.fwd_pred
    try:
        # Forward delegates to the real UMFPACK wrapper so it succeeds, but
        # the custom backward unconditionally raises AssertionError.
        model.fwd_pred = _make_adjoint_failing_wrapper(model.fwd_pred_umfpack)
        y, vjp_fun = model.linearize(np.array([10.0]))
        assert model._petsc_fallbacks == 0  # forward succeeded
        g = vjp_fun(np.ones_like(y))
        assert g.shape == (1, )
        assert model._petsc_fallbacks == 1  # adjoint triggered the rebuild
        assert model._umfpack_failures == 0
    finally:
        model.fwd_pred = real_primary


def test_eval_raises_when_both_wrappers_fail(model):
    """If both primary and UMFPACK fail, ForwardModelFailure is raised."""
    _reset_counters(model)
    real_primary = model.fwd_pred
    real_umfpack = model.fwd_pred_umfpack
    try:
        model.fwd_pred = _broken_wrapper
        model.fwd_pred_umfpack = _broken_wrapper
        with pytest.raises(ForwardModelFailure):
            model.eval(np.array([10.0]))
        assert model._petsc_fallbacks == 1
        assert model._umfpack_failures == 1
    finally:
        model.fwd_pred = real_primary
        model.fwd_pred_umfpack = real_umfpack


def test_linearize_raises_when_both_wrappers_fail_on_forward(model):
    """ForwardModelFailure propagates from linearize if the forward pass
    cannot be completed by either solver.
    """
    _reset_counters(model)
    real_primary = model.fwd_pred
    real_umfpack = model.fwd_pred_umfpack
    try:
        model.fwd_pred = _broken_wrapper
        model.fwd_pred_umfpack = _broken_wrapper
        with pytest.raises(ForwardModelFailure):
            model.linearize(np.array([10.0]))
        assert model._petsc_fallbacks == 1
        assert model._umfpack_failures == 1
    finally:
        model.fwd_pred = real_primary
        model.fwd_pred_umfpack = real_umfpack


def test_eval_grad_uses_fallback(model):
    """eval_grad is built on linearize, so it picks up the fallback too."""
    _reset_counters(model)
    real_primary = model.fwd_pred
    try:
        model.fwd_pred = _broken_wrapper
        J = model.eval_grad(np.array([10.0]))
        assert J.shape == (model.get_dim_out(), 1)
        assert model._petsc_fallbacks == 1
    finally:
        model.fwd_pred = real_primary
