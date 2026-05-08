"""Tests for the run_laplace.py driver script."""

import sys

import numpy as np
import pytest

from pdmp.distributions import MultivariateNormal, find_mean, find_curvature


# ---------------------------------------------------------------------------
# Unit: find_mean / find_curvature on a Gaussian (Laplace is exact)
# ---------------------------------------------------------------------------

def test_laplace_of_gaussian_exact_mean():
    """find_mean recovers the mean of a MultivariateNormal exactly."""
    rng = np.random.default_rng(0)
    mu = np.array([1.0, 2.0])
    cov = np.diag([1.0, 0.5])
    mvn = MultivariateNormal(mean=mu, cov=cov, rng=rng)
    result = find_mean(mvn, x_0=np.zeros(2))
    assert np.allclose(result, mu, atol=1e-4)


def test_laplace_of_gaussian_exact_cov():
    """find_curvature recovers the covariance of a MultivariateNormal exactly."""
    rng = np.random.default_rng(0)
    mu = np.array([1.0, 2.0])
    cov = np.diag([1.0, 0.5])
    mvn = MultivariateNormal(mean=mu, cov=cov, rng=rng)
    mean = find_mean(mvn, x_0=np.zeros(2))
    result = find_curvature(mvn, mean=mean)
    assert np.allclose(result, cov, atol=1e-4)


# ---------------------------------------------------------------------------
# Integration: main() on a Gaussian problem
# ---------------------------------------------------------------------------

_GAUSSIAN_CONFIG = """\
problem:
  name: Gaussian
  mean: [1.0, 2.0]
  cov: [[1.0, 0.0], [0.0, 1.0]]
laplace:
  n_samples: 200
seed: 0
output:
  dir: {out_dir}
  logging:
    log_file: run.log
    level: WARNING
"""

_TRANSFORMED_CONFIG = """\
problem:
  name: Transformed
  transformation: Identity
  distribution:
    name: Gaussian
    mean: [1.0, 2.0]
    cov: [[1.0, 0.0], [0.0, 1.0]]
laplace:
  n_samples: 200
seed: 0
output:
  dir: {out_dir}
  logging:
    log_file: run.log
    level: WARNING
"""


@pytest.fixture
def gaussian_run(tmp_path, monkeypatch):
    """Run main() on a Gaussian config; return the output directory path."""
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_GAUSSIAN_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv',
                        ['run_laplace.py', '--config', str(config_path)])
    from run_laplace import main
    main()
    return out_dir


def test_main_writes_output_files(gaussian_run):
    """main() writes mean.dat, cov.dat, samples.dat, config_used.yaml."""
    out_dir = gaussian_run
    assert (out_dir / 'mean.dat').exists()
    assert (out_dir / 'cov.dat').exists()
    assert (out_dir / 'samples.dat').exists()
    assert (out_dir / 'config_used.yaml').exists()


def test_main_laplace_converges_to_gaussian_mean(gaussian_run):
    """mean.dat contains the correct Gaussian mean [1.0, 2.0]."""
    mean = np.loadtxt(gaussian_run / 'mean.dat')
    assert np.allclose(mean, [1.0, 2.0], atol=1e-3)


def test_main_laplace_recovers_gaussian_cov(gaussian_run):
    """cov.dat contains the correct Gaussian covariance diag([1, 1])."""
    cov = np.loadtxt(gaussian_run / 'cov.dat')
    assert np.allclose(np.diag(cov), [1.0, 1.0], atol=1e-3)


def test_main_samples_shape(gaussian_run):
    """samples.dat has shape (n_samples, dim)."""
    samples = np.loadtxt(gaussian_run / 'samples.dat')
    assert samples.shape == (200, 2)


# ---------------------------------------------------------------------------
# Integration: outer Transformed wrapper is stripped
# ---------------------------------------------------------------------------

def test_main_strips_transformed_wrapper(tmp_path, monkeypatch):
    """main() strips an outer Transformed wrapper and still produces mean.dat."""
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "results"
    config_path.write_text(_TRANSFORMED_CONFIG.format(out_dir=out_dir))
    monkeypatch.setattr(sys, 'argv',
                        ['run_laplace.py', '--config', str(config_path)])
    from run_laplace import main
    main()
    assert (out_dir / 'mean.dat').exists()
    mean = np.loadtxt(out_dir / 'mean.dat')
    assert np.allclose(mean, [1.0, 2.0], atol=1e-3)
