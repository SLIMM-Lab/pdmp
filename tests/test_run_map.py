"""Tests for the run_map.py driver script."""

import sys

import numpy as np
import pytest
from unittest.mock import MagicMock

from pdmp.distributions import Posterior, TransformedLikelihood
from run_map import get_physical_transformation, run_optimization


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
# get_physical_transformation
# ---------------------------------------------------------------------------

def test_get_physical_transformation_returns_none_without_likelihood():
    """Non-Posterior targets have no _likelihood attribute."""
    target = _QuadraticTarget([0.0])
    assert get_physical_transformation(target) is None


def test_get_physical_transformation_returns_none_for_plain_likelihood():
    """Posterior whose _likelihood is not a TransformedLikelihood → None."""
    mock_likelihood = MagicMock(spec=[])  # not a TransformedLikelihood
    mock_prior = MagicMock()
    posterior = Posterior(mock_prior, mock_likelihood)
    assert get_physical_transformation(posterior) is None


def test_get_physical_transformation_extracts_transformation():
    """Posterior with TransformedLikelihood exposes its _transformation."""
    mock_transformation = MagicMock()
    mock_transformed_llh = MagicMock(spec=TransformedLikelihood)
    mock_transformed_llh._transformation = mock_transformation
    mock_prior = MagicMock()
    posterior = Posterior(mock_prior, mock_transformed_llh)
    assert get_physical_transformation(posterior) is mock_transformation


# ---------------------------------------------------------------------------
# run_optimization
# ---------------------------------------------------------------------------

def test_run_optimization_finds_quadratic_minimum():
    mu = np.array([3.0, -2.0])
    target = _QuadraticTarget(mu)
    result = run_optimization(target, np.zeros(2), method='L-BFGS-B', options={})
    assert result.success
    assert np.allclose(result.x, mu, atol=1e-5)


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


def test_main_writes_map_dat(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_map.py', '--config', str(config_path)])

    from run_map import main
    main()

    assert (out_dir / 'map.dat').exists()


def test_main_map_converges_to_mean(tmp_path, monkeypatch):
    """For a symmetric Gaussian prior, MAP == mean of the prior."""
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_map.py', '--config', str(config_path)])

    from run_map import main
    main()

    xi_star = np.loadtxt(out_dir / 'map.dat')
    assert np.allclose(xi_star, [1.0, 2.0], atol=1e-4)


def test_main_writes_config_and_restart_objectives(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv', ['run_map.py', '--config', str(config_path)])

    from run_map import main
    main()

    assert (out_dir / 'restart_objectives.dat').exists()
    assert (out_dir / 'config_used.yml').exists()
