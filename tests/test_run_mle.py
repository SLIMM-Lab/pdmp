"""Tests for the run_mle.py driver script."""

import os
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock

from pdmp.distributions import Posterior
from run_mle import get_likelihood_target, run_optimization


class _QuadraticTarget:
    """log p(x) = -0.5 ||x - mu||^2, optimum at mu."""

    def __init__(self, mu):
        self._mu = np.asarray(mu, dtype=float)

    def log_density(self, x):
        diff = x - self._mu
        return -0.5 * float(diff @ diff)

    def grad_log_density(self, x):
        return -(x - self._mu)


# ---------------------------------------------------------------------------
# get_likelihood_target
# ---------------------------------------------------------------------------

def test_get_likelihood_target_strips_prior():
    mock_likelihood = MagicMock()
    mock_prior = MagicMock()
    posterior = Posterior(mock_prior, mock_likelihood)
    assert get_likelihood_target(posterior) is mock_likelihood


def test_get_likelihood_target_returns_non_posterior_unchanged():
    target = _QuadraticTarget([0.0])
    assert get_likelihood_target(target) is target


# ---------------------------------------------------------------------------
# run_optimization
# ---------------------------------------------------------------------------

def test_run_optimization_finds_quadratic_minimum():
    mu = np.array([3.0, -2.0])
    target = _QuadraticTarget(mu)
    result = run_optimization(target, np.zeros(2), method='L-BFGS-B', options={})
    assert result.success
    assert np.allclose(result.x, mu, atol=1e-5)


def test_run_optimization_returns_lower_value_from_closer_start():
    mu = np.array([1.0, 1.0])
    target = _QuadraticTarget(mu)
    res_near = run_optimization(target, np.array([0.9, 0.9]), method='L-BFGS-B', options={})
    res_far = run_optimization(target, np.array([10.0, 10.0]), method='L-BFGS-B', options={})
    assert np.allclose(res_near.x, mu, atol=1e-5)
    assert np.allclose(res_far.x, mu, atol=1e-5)


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

_GAUSSIAN_CONFIG = """\
problem:
  name: Gaussian
  mean: [1.0, 2.0]
  cov: [[1.0, 0.0], [0.0, 1.0]]
optimizer:
  x_0: [0.0, 0.0]
  method: L-BFGS-B
  n_restarts: 0
seed: 42
output:
  dir: {out_dir}
  logging:
    log_file: run.log
    level: WARNING
"""


def test_main_writes_mle_dat(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_mle.py', '--config', str(config_path)])

    from run_mle import main
    main()

    assert (out_dir / 'mle.dat').exists()


def test_main_mle_converges_to_mean(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_mle.py', '--config', str(config_path)])

    from run_mle import main
    main()

    xi_star = np.loadtxt(out_dir / 'mle.dat')
    assert np.allclose(xi_star, [1.0, 2.0], atol=1e-4)


def test_main_writes_config_and_restart_objectives(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_mle.py', '--config', str(config_path)])

    from run_mle import main
    main()

    assert (out_dir / 'restart_objectives.dat').exists()
    assert (out_dir / 'config_used.yml').exists()
