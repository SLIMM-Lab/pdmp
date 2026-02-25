"""Unit tests for the Bayesian-optimisation GP training strategy.

All GP objects are built with minimal settings so the suite stays fast:
  n_restarts=1, lbfgs_steps=2, bo_num_restarts=2, bo_raw_samples=16.
"""

import os

import numpy as np
import pytest
import torch

from pdmp.distributions import MultivariateNormal
from pdmp.surrogates import (
    DerivativeGaussianProcess,
    ExactGPModel,
    GaussianProcess,
    _BoTorchDerivGPWrapper,
    _BoTorchGPWrapper,
    _MaxVarianceAcquisition,
    _WeightedVarianceAcquisition,
)

# ── shared constants ────────────────────────────────────────────────────────────
SEED = 0
DIM = 2
_FAST_GP = dict(
    lbfgs_steps=2,
    n_restarts=1,
    lr=0.3,
    tolerance_grad=1e-3,
    tolerance_change=1e-6,
    print_every=999,
)
_FAST_BO = dict(
    n_bo_init=3,
    n_bo_iter=3,
    bo_num_restarts=2,
    bo_raw_samples=16,
    bo_retrain_interval=3,
    bo_bounds_scale=3.0,
)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _chdir(tmp):
    """Change to *tmp*, return old cwd."""
    prev = os.getcwd()
    os.chdir(tmp)
    return prev


# ── fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def target():
    """Simple 2-D Gaussian target – cheap to evaluate and has hessian_log_density."""
    mean = np.array([1.0, -0.5])
    cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    return MultivariateNormal(mean=mean, cov=cov)


@pytest.fixture(scope='module')
def gp_laplace(target, tmp_path_factory):
    """GaussianProcess trained with the default Laplace strategy (small dataset)."""
    tmp = tmp_path_factory.mktemp('gp_laplace')
    prev = _chdir(tmp)
    try:
        gp = GaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 10),
            training_strategy='laplace',
            n_samples=5,
            **_FAST_GP,
        )
    finally:
        os.chdir(prev)
    return gp


@pytest.fixture(scope='module')
def gp_bo(target, tmp_path_factory):
    """GaussianProcess trained with BO / weighted_variance."""
    tmp = tmp_path_factory.mktemp('gp_bo')
    prev = _chdir(tmp)
    try:
        gp = GaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 1),
            training_strategy='bayesian_optimization',
            acquisition='weighted_variance',
            **_FAST_GP,
            **_FAST_BO,
        )
    finally:
        os.chdir(prev)
    return gp


@pytest.fixture(scope='module')
def gp_bo_max_var(target, tmp_path_factory):
    """GaussianProcess trained with BO / max_variance."""
    tmp = tmp_path_factory.mktemp('gp_bo_max_var')
    prev = _chdir(tmp)
    try:
        gp = GaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 2),
            training_strategy='bayesian_optimization',
            acquisition='max_variance',
            **_FAST_GP,
            **_FAST_BO,
        )
    finally:
        os.chdir(prev)
    return gp


@pytest.fixture(scope='module')
def deriv_gp_bo(target, tmp_path_factory):
    """DerivativeGaussianProcess trained with BO / weighted_variance."""
    tmp = tmp_path_factory.mktemp('deriv_gp_bo')
    prev = _chdir(tmp)
    try:
        gp = DerivativeGaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 3),
            training_strategy='bayesian_optimization',
            acquisition='weighted_variance',
            **_FAST_GP,
            **_FAST_BO,
        )
    finally:
        os.chdir(prev)
    return gp


# ── _get_bo_bounds ──────────────────────────────────────────────────────────────

def test_get_bo_bounds_shape(gp_bo):
    bounds = gp_bo._get_bo_bounds(scale=3.0)
    assert bounds.shape == (2, DIM)


def test_get_bo_bounds_lower_lt_upper(gp_bo):
    bounds = gp_bo._get_bo_bounds(scale=3.0)
    assert (bounds[0] < bounds[1]).all()


def test_get_bo_bounds_width(gp_bo):
    scale = 4.0
    bounds = gp_bo._get_bo_bounds(scale=scale)
    std = torch.tensor(np.sqrt(np.diag(gp_bo._laplace._cov)), dtype=torch.float64)
    expected_width = 2.0 * scale * std
    assert torch.allclose(bounds[1] - bounds[0], expected_width)


# ── _BoTorchGPWrapper ───────────────────────────────────────────────────────────

def test_botorch_gp_wrapper_num_outputs(gp_laplace):
    wrapper = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    assert wrapper.num_outputs == 1


def test_botorch_gp_wrapper_posterior_mean_shape(gp_laplace):
    wrapper = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    X = torch.randn(4, DIM, dtype=torch.float64)
    posterior = wrapper.posterior(X)
    assert posterior.mean.shape == (4, 1)


def test_botorch_gp_wrapper_posterior_variance_shape(gp_laplace):
    wrapper = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    X = torch.randn(4, DIM, dtype=torch.float64)
    posterior = wrapper.posterior(X)
    assert posterior.variance.shape == (4, 1)
    assert (posterior.variance >= 0).all()


# ── _BoTorchDerivGPWrapper ──────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def deriv_gp_laplace(target, tmp_path_factory):
    """DerivativeGaussianProcess trained with the Laplace strategy (small dataset)."""
    tmp = tmp_path_factory.mktemp('deriv_gp_laplace')
    prev = _chdir(tmp)
    try:
        gp = DerivativeGaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 11),
            training_strategy='laplace',
            n_samples=5,
            **_FAST_GP,
        )
    finally:
        os.chdir(prev)
    return gp


def test_botorch_deriv_gp_wrapper_num_outputs(deriv_gp_laplace):
    wrapper = _BoTorchDerivGPWrapper(deriv_gp_laplace._model,
                                     deriv_gp_laplace._likelihood)
    assert wrapper.num_outputs == 1


def test_botorch_deriv_gp_wrapper_posterior_mean_shape(deriv_gp_laplace):
    wrapper = _BoTorchDerivGPWrapper(deriv_gp_laplace._model,
                                     deriv_gp_laplace._likelihood)
    X = torch.randn(4, DIM, dtype=torch.float64)
    posterior = wrapper.posterior(X)
    assert posterior.mean.shape == (4, 1)


def test_botorch_deriv_gp_wrapper_posterior_variance_shape(deriv_gp_laplace):
    wrapper = _BoTorchDerivGPWrapper(deriv_gp_laplace._model,
                                     deriv_gp_laplace._likelihood)
    X = torch.randn(4, DIM, dtype=torch.float64)
    posterior = wrapper.posterior(X)
    assert posterior.variance.shape == (4, 1)
    assert (posterior.variance >= 0).all()


# ── _MaxVarianceAcquisition ─────────────────────────────────────────────────────

def test_max_variance_acquisition_output_shape(gp_laplace):
    wrapper = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    acq = _MaxVarianceAcquisition(model=wrapper)
    X = torch.randn(5, 1, DIM, dtype=torch.float64)  # [batch, q=1, d]
    vals = acq(X)
    assert vals.shape == (5,)


def test_max_variance_acquisition_non_negative(gp_laplace):
    wrapper = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    acq = _MaxVarianceAcquisition(model=wrapper)
    X = torch.randn(5, 1, DIM, dtype=torch.float64)
    assert (acq(X) >= 0).all()


# ── _WeightedVarianceAcquisition ────────────────────────────────────────────────

@pytest.fixture(scope='module')
def weighted_var_acq(gp_laplace):
    """A WeightedVarianceAcquisition built from the Laplace-trained GP."""
    laplace = gp_laplace._laplace
    model = _BoTorchGPWrapper(gp_laplace._model, gp_laplace._likelihood)
    acq = _WeightedVarianceAcquisition(
        model=model,
        laplace_mean=torch.tensor(laplace._mean, dtype=torch.float64),
        laplace_inv_cov=torch.tensor(laplace.gaussian.inv_C, dtype=torch.float64),
        laplace_constant=torch.tensor(laplace.gaussian.constant, dtype=torch.float64),
        laplace_log_det=torch.tensor(laplace.gaussian.log_det, dtype=torch.float64),
        laplace_delta=torch.tensor(laplace._delta, dtype=torch.float64),
    )
    return acq, laplace


def test_weighted_variance_acquisition_output_shape(weighted_var_acq):
    acq, _ = weighted_var_acq
    X = torch.randn(5, 1, DIM, dtype=torch.float64)
    vals = acq(X)
    assert vals.shape == (5,)


def test_weighted_variance_acquisition_non_negative(weighted_var_acq):
    acq, _ = weighted_var_acq
    X = torch.randn(5, 1, DIM, dtype=torch.float64)
    assert (acq(X) >= 0).all()


def test_weighted_variance_laplace_log_density(weighted_var_acq):
    """_laplace_log_density should match laplace.eval(x, delta=True)."""
    acq, laplace = weighted_var_acq
    x = np.array([1.0, -0.5])
    X = torch.tensor(x, dtype=torch.float64)
    ld_acq = acq._laplace_log_density(X).item()
    ld_ref = laplace.eval(x, delta=True)
    assert np.isclose(ld_acq, ld_ref, atol=1e-10)


# ── GaussianProcess BO training ─────────────────────────────────────────────────

def test_bo_gp_total_data_size(gp_bo):
    n_expected = _FAST_BO['n_bo_init'] + _FAST_BO['n_bo_iter']
    assert len(gp_bo._x_data) == n_expected


def test_bo_gp_bo_points_within_bounds(gp_bo):
    """BO-selected points (after the initial Laplace batch) lie within the search box."""
    n_init = _FAST_BO['n_bo_init']
    bounds = gp_bo._get_bo_bounds(scale=_FAST_BO['bo_bounds_scale'])
    bo_pts = gp_bo._x_data[n_init:]  # [n_bo_iter, d]
    tol = 1e-4  # numerical slack from optimize_acqf
    assert (bo_pts >= bounds[0] - tol).all()
    assert (bo_pts <= bounds[1] + tol).all()


def test_bo_gp_eval_finite(gp_bo):
    val = gp_bo.eval(np.array([1.0, -0.5]))
    assert np.isfinite(val)


def test_bo_gp_eval_scalar(gp_bo):
    val = gp_bo.eval(np.array([0.0, 0.0]))
    assert np.ndim(val) == 0


def test_bo_gp_query_point_is_scalar_tensor(gp_bo, target):
    """_bo_query_point for GaussianProcess returns a 0-d tensor."""
    y = gp_bo._bo_query_point(target, np.array([0.5, 0.2]))
    assert y.dim() == 0
    assert y.dtype == torch.float64


def test_bo_gp_query_point_value(gp_bo, target):
    """_bo_query_point equals target.log_density(x) - laplace.eval(x, delta=True)."""
    x = np.array([0.5, 0.2])
    y = gp_bo._bo_query_point(target, x)
    expected = target.log_density(x) - gp_bo._laplace.eval(x, delta=True)
    assert np.isclose(y.item(), expected, atol=1e-10)


def test_bo_max_variance_total_data_size(gp_bo_max_var):
    n_expected = _FAST_BO['n_bo_init'] + _FAST_BO['n_bo_iter']
    assert len(gp_bo_max_var._x_data) == n_expected


def test_bo_max_variance_eval_finite(gp_bo_max_var):
    val = gp_bo_max_var.eval(np.array([1.0, -0.5]))
    assert np.isfinite(val)


def test_bo_invalid_acquisition_raises(target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown acquisition"):
        GaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 99),
            training_strategy='bayesian_optimization',
            acquisition='nonexistent',
            n_bo_init=2,
            n_bo_iter=1,
            bo_num_restarts=1,
            bo_raw_samples=8,
            **_FAST_GP,
        )


def test_bo_invalid_training_strategy_raises(target, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown training_strategy"):
        GaussianProcess(
            target=target,
            rng=np.random.default_rng(SEED + 98),
            training_strategy='gibberish',
            **_FAST_GP,
        )


def test_bo_retrain_interval_zero(target, tmp_path, monkeypatch):
    """bo_retrain_interval=0 skips mid-loop retraining but still converges."""
    monkeypatch.chdir(tmp_path)
    gp = GaussianProcess(
        target=target,
        rng=np.random.default_rng(SEED + 50),
        training_strategy='bayesian_optimization',
        acquisition='weighted_variance',
        bo_retrain_interval=0,
        n_bo_init=3,
        n_bo_iter=3,
        bo_num_restarts=2,
        bo_raw_samples=16,
        bo_bounds_scale=3.0,
        **_FAST_GP,
    )
    assert np.isfinite(gp.eval(np.array([1.0, -0.5])))
    assert len(gp._x_data) == 6


# ── DerivativeGaussianProcess BO training ───────────────────────────────────────

def test_bo_deriv_gp_total_data_size(deriv_gp_bo):
    n_expected = _FAST_BO['n_bo_init'] + _FAST_BO['n_bo_iter']
    assert len(deriv_gp_bo._x_data) == n_expected


def test_bo_deriv_gp_y_data_shape(deriv_gp_bo):
    """y_data for DerivativeGP must have shape [n, d+1]."""
    n_expected = _FAST_BO['n_bo_init'] + _FAST_BO['n_bo_iter']
    assert deriv_gp_bo._y_data.shape == (n_expected, DIM + 1)


def test_bo_deriv_gp_eval_finite(deriv_gp_bo):
    val = deriv_gp_bo.eval(np.array([1.0, -0.5]))
    assert np.isfinite(val)


def test_bo_deriv_gp_query_point_shape(deriv_gp_bo, target):
    """_bo_query_point for DerivativeGP returns a 1-D tensor of length d+1."""
    x = np.array([0.5, 0.2])
    y = deriv_gp_bo._bo_query_point(target, x)
    assert y.shape == (DIM + 1,)
    assert y.dtype == torch.float64


def test_bo_deriv_gp_query_point_function_value(deriv_gp_bo, target):
    """First element of _bo_query_point equals target.log_density - laplace.eval."""
    x = np.array([0.5, 0.2])
    y = deriv_gp_bo._bo_query_point(target, x)
    expected = target.log_density(x) - deriv_gp_bo._laplace.eval(x, delta=True)
    assert np.isclose(y[0].item(), expected, atol=1e-10)


def test_bo_deriv_gp_query_point_gradient(deriv_gp_bo, target):
    """Remaining elements of _bo_query_point equal target.grad - laplace.grad."""
    x = np.array([0.5, 0.2])
    y = deriv_gp_bo._bo_query_point(target, x)
    expected_grad = (target.grad_log_density(x)
                     - deriv_gp_bo._laplace.grad(x))
    assert np.allclose(y[1:].numpy(), expected_grad, atol=1e-10)
