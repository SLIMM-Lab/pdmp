"""Tests for the forward_uq_moment.py driver script."""

import sys

import numpy as np
import pytest

from forward_uq_moment import unscented_sigma_points, fd_jacobian, build_latent_to_physical

# ---------------------------------------------------------------------------
# Unit: unscented_sigma_points
# ---------------------------------------------------------------------------


def test_sigma_points_count():
    """unscented_sigma_points returns 2d+1 points for d-dimensional input."""
    mu = np.zeros(3)
    cov = np.eye(3)
    points, wm, wc = unscented_sigma_points(mu, cov)
    assert points.shape == (7, 3)
    assert wm.shape == (7, )
    assert wc.shape == (7, )


def test_sigma_points_central_is_mean():
    """First sigma point equals the input mean."""
    mu = np.array([1.0, -2.0, 0.5])
    cov = np.diag([2.0, 0.5, 1.0])
    points, _, _ = unscented_sigma_points(mu, cov)
    assert np.allclose(points[0], mu)


def test_sigma_points_mean_weights_sum_to_one():
    """Mean weights must sum to exactly 1."""
    mu = np.zeros(3)
    cov = np.eye(3)
    _, wm, _ = unscented_sigma_points(mu, cov, alpha=1.0, beta=2.0, kappa=0.0)
    assert np.isclose(wm.sum(), 1.0)


def test_ut_linear_exact_mean():
    """UT recovers the pushforward mean exactly for a linear function."""
    np.random.seed(0)
    d_in, d_out = 3, 2
    A = np.random.randn(d_out, d_in)
    b = np.random.randn(d_out)
    mu = np.array([0.5, 1.0, -0.3])
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.8]])

    points, wm, wc = unscented_sigma_points(mu, cov)
    ys = np.array([A @ p + b for p in points])
    mu_ut = (wm[:, None] * ys).sum(axis=0)

    assert np.allclose(mu_ut, A @ mu + b, atol=1e-12)


def test_ut_linear_exact_cov():
    """UT recovers the pushforward covariance exactly for a linear function."""
    np.random.seed(0)
    d_in, d_out = 3, 2
    A = np.random.randn(d_out, d_in)
    b = np.random.randn(d_out)
    mu = np.array([0.5, 1.0, -0.3])
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.8]])

    points, wm, wc = unscented_sigma_points(mu, cov)
    ys = np.array([A @ p + b for p in points])
    mu_ut = (wm[:, None] * ys).sum(axis=0)
    diffs = ys - mu_ut
    cov_ut = (wc[:, None, None] * diffs[:, :, None] *
              diffs[:, None, :]).sum(axis=0)

    assert np.allclose(cov_ut, A @ cov @ A.T, atol=1e-12)


def test_ut_quadratic_mean_correction():
    """UT captures the variance correction E[X²] = μ² + σ² for quadratic f."""
    mu = np.array([2.0])
    cov = np.array([[0.5]])

    points, wm, _ = unscented_sigma_points(mu, cov)
    ys = np.array([(p**2) for p in points])
    mu_ut = (wm[:, None] * ys).sum(axis=0)

    expected = mu**2 + np.diag(cov)
    assert np.allclose(mu_ut, expected, atol=1e-12)

    # Linearization is biased: f(mu) = mu², missing the sigma² correction
    mu_lin = mu**2
    assert not np.allclose(mu_lin, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Unit: fd_jacobian
# ---------------------------------------------------------------------------


def test_fd_jacobian_linear():
    """fd_jacobian matches the analytical Jacobian for a linear function."""
    np.random.seed(1)
    d_in, d_out = 4, 3
    A = np.random.randn(d_out, d_in)
    b = np.random.randn(d_out)
    x = np.random.randn(d_in)

    J, _ = fd_jacobian(lambda v: A @ v + b, x, h=1e-3)
    assert np.allclose(J, A, atol=1e-8)


def test_fd_jacobian_f0():
    """fd_jacobian returns f(x) as the second output."""
    np.random.seed(2)
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    x = np.array([0.5, -0.5, 1.5])

    _, f0 = fd_jacobian(lambda v: A @ v + b, x, h=1e-3)
    assert np.allclose(f0, A @ x + b, atol=1e-10)


# ---------------------------------------------------------------------------
# Unit: build_latent_to_physical
# ---------------------------------------------------------------------------


def test_build_latent_to_physical_no_transform():
    """Returns None when likelihood is not a TransformedLikelihood."""
    cfg = {
        'problem': {
            'name': 'BayesianInverse',
            'likelihood': {
                'name': 'GaussianLikelihood'
            },
        }
    }
    assert build_latent_to_physical(cfg) is None


def test_build_latent_to_physical_with_composite():
    """Composite Sigmoid+Exponential maps [0,0] → [0.5, 1.0]."""
    cfg = {
        'problem': {
            'name': 'BayesianInverse',
            'likelihood': {
                'name':
                'TransformedLikelihood',
                'transformation':
                'Composite',
                'transformations': [
                    {
                        'type': 'Sigmoid',
                        'a': 0.0,
                        'b': 1.0
                    },
                    'Exponential',
                ],
                'indices': [[0], [1]],
            },
        }
    }
    T = build_latent_to_physical(cfg)
    assert T is not None
    x_phys = T(np.array([0.0, 0.0]))
    assert np.allclose(x_phys, [0.5, 1.0], atol=1e-10)


def test_build_strips_outer_transformed():
    """Outer Transformed wrappers are stripped before finding the likelihood."""
    cfg = {
        'problem': {
            'name': 'Transformed',
            'transformation': 'Identity',
            'distribution': {
                'name': 'BayesianInverse',
                'likelihood': {
                    'name': 'GaussianLikelihood'
                },
            },
        }
    }
    # Should not raise; inner GaussianLikelihood → returns None
    result = build_latent_to_physical(cfg)
    assert result is None


def test_build_raises_for_non_bayesian_inner():
    """Raises ValueError if inner problem is not BayesianInverse."""
    cfg = {
        'problem': {
            'name': 'Gaussian',
            'mean': [0.0],
            'cov': [[1.0]],
        }
    }
    with pytest.raises(ValueError, match="BayesianInverse"):
        build_latent_to_physical(cfg)


# ---------------------------------------------------------------------------
# Integration: main() with LinearModel (no FEM required)
# ---------------------------------------------------------------------------

_INFERENCE_CONFIG = """\
problem:
  name: BayesianInverse
  likelihood:
    name: GaussianLikelihood
"""

_MOMENT_CONFIG = """\
model:
  name: Linear
  A: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
  b: [0.0, 0.0]
forward_uq_moment:
  posterior_dir: {posterior_dir}
  inference_config: {inference_config}
  param_indices: [0, 1, 2]
  n_synthetic_samples: 50
  seed: 0
  fd_step: 1.0e-3
  ut:
    alpha: 1.0
    beta: 2.0
    kappa: 0.0
output:
  dir: {out_dir}
"""


@pytest.fixture
def linear_run(tmp_path, monkeypatch):
    """Run main() with A=diag(1,2), mu=0, Sigma=I; return the output dir."""
    # Write Laplace posterior files
    posterior_dir = tmp_path / "posterior"
    posterior_dir.mkdir()
    np.savetxt(posterior_dir / "mean.dat", np.zeros(3))
    np.savetxt(posterior_dir / "cov.dat", np.eye(3))

    # Write minimal inference config (no TransformedLikelihood → identity map)
    inf_cfg = tmp_path / "inference_config.yaml"
    inf_cfg.write_text(_INFERENCE_CONFIG)

    # Write moment-matching config
    out_dir = tmp_path / "results"
    moment_cfg = tmp_path / "moment_config.yaml"
    moment_cfg.write_text(
        _MOMENT_CONFIG.format(
            posterior_dir=str(posterior_dir),
            inference_config=str(inf_cfg),
            out_dir=str(out_dir),
        ))

    monkeypatch.setattr(sys, 'argv', ['forward_uq_moment.py', str(moment_cfg)])
    from forward_uq_moment import main
    main()
    return out_dir


def test_main_writes_output_files(linear_run):
    """main() writes all expected output files."""
    expected = [
        'mean_ut.dat',
        'cov_ut.dat',
        'mean_lin.dat',
        'cov_lin.dat',
        'samples_ut.dat',
        'samples_lin.dat',
        'jacobian.dat',
        'sigma_points.dat',
        'sigma_outputs.dat',
    ]
    for fname in expected:
        assert (linear_run / fname).exists(), f"Missing: {fname}"


def test_main_linear_model_ut_exact(linear_run):
    """UT gives exact moments for a linear model: mean=0, cov=diag(1,4)."""
    mu_ut = np.loadtxt(linear_run / 'mean_ut.dat')
    cov_ut = np.loadtxt(linear_run / 'cov_ut.dat')

    assert np.allclose(mu_ut, [0.0, 0.0], atol=1e-10)
    assert np.allclose(cov_ut, [[1.0, 0.0], [0.0, 4.0]], atol=1e-10)


def test_main_linear_model_lin_exact(linear_run):
    """Linearization gives exact moments for a linear model: mean=0, cov=diag(1,4)."""
    mu_lin = np.loadtxt(linear_run / 'mean_lin.dat')
    cov_lin = np.loadtxt(linear_run / 'cov_lin.dat')

    assert np.allclose(mu_lin, [0.0, 0.0], atol=1e-10)
    assert np.allclose(cov_lin, [[1.0, 0.0], [0.0, 4.0]], atol=1e-10)
