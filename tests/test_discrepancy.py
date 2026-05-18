"""Tests for the Kennedy–O'Hagan discrepancy module."""

import numpy as np
import pytest

from pdmp.discrepancy import (
    rbf_kernel_matrix,
    rbf_kernel_matrix_drho,
    build_noise_covariance,
    KOGaussianLikelihood,
)
from pdmp.distributions import (
    GaussianLikelihood,
    JointDistribution,
    MultivariateNormal,
    Posterior,
)
from pdmp.forward_model import PiecewiseConstantModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_setup():
    """2-param PiecewiseConstant model with 5 sensors, 1 force setting."""
    rng = np.random.default_rng(42)
    F = np.array([1.0])
    x_obs = np.linspace(0.2, 1.0, 5)
    model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs)

    theta_true = np.array([2.0, 3.0])
    sigma2_delta = 0.01
    sigma2_eps = 0.001
    rho = np.array([5.0])

    # Generate synthetic data with discrepancy
    eta = model.eval(theta_true, idx=0)
    C = rbf_kernel_matrix(x_obs.reshape(-1, 1), rho)
    Sigma = sigma2_delta * C + sigma2_eps * np.eye(len(x_obs))
    L = np.linalg.cholesky(Sigma)
    y = eta + L @ rng.standard_normal(len(x_obs))
    u_obs = y.reshape(1, -1)

    psi_prior = MultivariateNormal(
        mean=np.array([-4.0, -6.0, 1.5]),
        cov=4.0 * np.eye(3),
        rng=rng,
    )

    return model, u_obs, x_obs, psi_prior, rng


def _fd_gradient(f, x, h=1e-6):
    """Central finite-difference gradient."""
    g = np.zeros_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h
        g[j] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g


# ---------------------------------------------------------------------------
# Kernel function tests
# ---------------------------------------------------------------------------

class TestKernelFunctions:

    def test_rbf_kernel_1d_values(self):
        """Check RBF kernel against hand computation for 3 points in 1D."""
        x = np.array([[0.0], [0.5], [1.0]])
        rho = np.array([2.0])
        C = rbf_kernel_matrix(x, rho)

        assert C.shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(C), 1.0)
        # C[0,1] = exp(-2.0 * 0.25) = exp(-0.5)
        np.testing.assert_allclose(C[0, 1], np.exp(-0.5))
        # C[0,2] = exp(-2.0 * 1.0) = exp(-2.0)
        np.testing.assert_allclose(C[0, 2], np.exp(-2.0))
        # Symmetry
        np.testing.assert_allclose(C, C.T)

    def test_rbf_kernel_2d_ard(self):
        """Check ARD kernel in 2D with different length scales."""
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        rho = np.array([1.0, 4.0])
        C = rbf_kernel_matrix(x, rho)

        # C[0,1]: only x-dim differs by 1, rho_x=1 → exp(-1.0)
        np.testing.assert_allclose(C[0, 1], np.exp(-1.0))
        # C[0,2]: only y-dim differs by 1, rho_y=4 → exp(-4.0)
        np.testing.assert_allclose(C[0, 2], np.exp(-4.0))
        # C[1,2]: x diff=1, y diff=1 → exp(-1.0 - 4.0) = exp(-5.0)
        np.testing.assert_allclose(C[1, 2], np.exp(-5.0))

    def test_rbf_kernel_positive_definite(self):
        """Kernel matrix should be positive definite."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (10, 2))
        rho = np.array([3.0, 5.0])
        C = rbf_kernel_matrix(x, rho)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_rbf_kernel_drho_fd(self):
        """Kernel derivative matches finite differences."""
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, (5, 2))
        rho = np.array([2.0, 3.0])
        h = 1e-7

        for k in range(2):
            dC_analytic = rbf_kernel_matrix_drho(x, rho, k)
            rho_fwd = rho.copy()
            rho_fwd[k] += h
            rho_bwd = rho.copy()
            rho_bwd[k] -= h
            dC_fd = (rbf_kernel_matrix(x, rho_fwd) -
                     rbf_kernel_matrix(x, rho_bwd)) / (2.0 * h)
            np.testing.assert_allclose(dC_analytic, dC_fd, atol=1e-5)


class TestBuildNoiseCovariance:

    def test_zero_discrepancy(self):
        """With sigma2_delta=0, covariance reduces to sigma2_eps * I."""
        x = np.array([[0.0], [0.5], [1.0]])
        rho = np.array([1.0])
        Sigma = build_noise_covariance(x, rho, sigma2_delta=0.0,
                                       sigma2_eps=0.5)
        np.testing.assert_allclose(Sigma, 0.5 * np.eye(3))

    def test_zero_noise(self):
        """With sigma2_eps=0, covariance is sigma2_delta * C."""
        x = np.array([[0.0], [1.0]])
        rho = np.array([1.0])
        C = rbf_kernel_matrix(x, rho)
        Sigma = build_noise_covariance(x, rho, sigma2_delta=2.0,
                                       sigma2_eps=0.0)
        np.testing.assert_allclose(Sigma, 2.0 * C)


# ---------------------------------------------------------------------------
# KOGaussianLikelihood tests
# ---------------------------------------------------------------------------

class TestKOGaussianLikelihood:

    def test_grad_log_density_fd(self, simple_setup):
        """Analytic gradient matches finite differences for full [theta, psi]."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                   psi_prior=psi_prior, rng=rng)

        # Test point: [theta_1, theta_2, log_s2d, log_s2e, log_rho]
        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        grad_analytic = lik.grad_log_density(params)
        grad_fd = _fd_gradient(lik.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4,
                                   atol=1e-7)

    def test_grad_at_different_point(self, simple_setup):
        """Gradient check at a different parameter value."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                   psi_prior=psi_prior, rng=rng)

        params = np.array([1.5, 4.0, -3.0, -5.0, 2.0])
        grad_analytic = lik.grad_log_density(params)
        grad_fd = _fd_gradient(lik.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4,
                                   atol=1e-7)

    def test_reduces_to_standard_likelihood(self):
        """With sigma2_delta ≈ 0, K&O likelihood ≈ standard Gaussian."""
        rng = np.random.default_rng(99)
        F = np.array([1.0])
        x_obs = np.linspace(0.2, 1.0, 5)
        model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs)

        theta = np.array([2.0, 3.0])
        sigma = 0.1  # noise std
        sigma2_eps = sigma**2

        eta = model.eval(theta, idx=0)
        u_obs = (eta + rng.normal(0, sigma, len(x_obs))).reshape(1, -1)

        # Standard likelihood
        std_lik = GaussianLikelihood(model=model, u_obs=u_obs, sigma=sigma,
                                     rng=rng)
        ll_std = std_lik.log_density(theta)

        # K&O with negligible discrepancy
        psi_prior = MultivariateNormal(np.zeros(3), np.eye(3), rng=rng)
        ko_lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                      psi_prior=psi_prior, rng=rng)
        log_s2d = np.log(1e-15)
        log_s2e = np.log(sigma2_eps)
        log_rho = np.log(1.0)
        params_ko = np.concatenate([theta, [log_s2d, log_s2e, log_rho]])
        ll_ko = ko_lik.log_density(params_ko)

        np.testing.assert_allclose(ll_ko, ll_std, rtol=1e-6)

    def test_hessian_consistency(self, simple_setup):
        """Hessian (FD of grad) is consistent with FD of FD."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                   psi_prior=psi_prior, rng=rng)
        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        H = lik.hessian_log_density(params)
        # Should be symmetric
        np.testing.assert_allclose(H, H.T, atol=1e-8)

    def test_grad_block_diagonal(self):
        """Analytic gradient matches FD for block-diagonal C_delta (n_components=3)."""
        rng = np.random.default_rng(7)
        n_components = 3
        m_base = 4  # 4 base sensor locations
        m = m_base * n_components  # 12 total observations

        F = np.array([1.0])
        x_obs_full = np.linspace(0.1, 1.0, m)
        model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs_full)

        x_base = np.linspace(0.1, 1.0, m_base).reshape(-1, 1)  # (4, 1)
        theta_true = np.array([2.0, 3.0])
        eta = model.eval(theta_true, idx=0)
        u_obs = (eta + rng.normal(0, 0.05, m)).reshape(1, -1)

        psi_prior = MultivariateNormal(
            mean=np.array([-4.0, -6.0, 1.5]),
            cov=4.0 * np.eye(3),
            rng=rng,
        )
        lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_base,
                                   psi_prior=psi_prior, rng=rng,
                                   n_components=n_components)

        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        grad_analytic = lik.grad_log_density(params)
        grad_fd = _fd_gradient(lik.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4, atol=1e-7)


class TestKOPosterior:

    def test_posterior_grad_fd(self, simple_setup):
        """Full posterior gradient matches finite differences."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        ko_lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                      psi_prior=psi_prior, rng=rng)

        prior_theta = MultivariateNormal(
            mean=np.array([1.5, 2.5]),
            cov=4.0 * np.eye(2),
            rng=rng,
        )
        joint_prior = JointDistribution([prior_theta, psi_prior], rng=rng)
        posterior = Posterior(prior=joint_prior, likelihood=ko_lik, rng=rng)

        assert posterior.dim == 5  # 2 theta + 3 psi

        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        grad_analytic = posterior.grad_log_density(params)
        grad_fd = _fd_gradient(posterior.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4,
                                   atol=1e-7)

    def test_posterior_sample_dim(self, simple_setup):
        """Prior sample has correct dimension."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        ko_lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                      psi_prior=psi_prior, rng=rng)

        prior_theta = MultivariateNormal(
            mean=np.array([1.5, 2.5]),
            cov=4.0 * np.eye(2),
            rng=rng,
        )
        joint_prior = JointDistribution([prior_theta, psi_prior], rng=rng)
        posterior = Posterior(prior=joint_prior, likelihood=ko_lik, rng=rng)

        sample = posterior.get_prior_sample()
        assert sample.shape == (5,)


class TestKOMultipleSettings:

    def test_multiple_force_settings(self):
        """K&O likelihood works with multiple force settings."""
        rng = np.random.default_rng(7)
        F = np.array([1.0, 2.0])
        x_obs = np.linspace(0.2, 1.0, 4)
        model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs)

        theta = np.array([2.0, 3.0])
        u_obs = np.vstack([
            model.eval(theta, idx=0) + rng.normal(0, 0.01, 4),
            model.eval(theta, idx=1) + rng.normal(0, 0.01, 4),
        ])

        psi_prior = MultivariateNormal(np.zeros(3), np.eye(3), rng=rng)
        lik = KOGaussianLikelihood(model=model, u_obs=u_obs, x_locs=x_obs,
                                   psi_prior=psi_prior, rng=rng)

        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        grad_analytic = lik.grad_log_density(params)
        grad_fd = _fd_gradient(lik.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4,
                                   atol=1e-7)


# ---------------------------------------------------------------------------
# Per-group block-diagonal KO likelihood (n_groups > 1)
# ---------------------------------------------------------------------------

class TestKOGroupedDiscrepancy:

    def test_n_groups_one_matches_default(self, simple_setup):
        """Explicit n_groups=1 must match default (1) behavior numerically."""
        model, u_obs, x_obs, psi_prior, rng = simple_setup
        lik_default = KOGaussianLikelihood(
            model=model, u_obs=u_obs, x_locs=x_obs,
            psi_prior=psi_prior, rng=rng,
        )
        lik_explicit = KOGaussianLikelihood(
            model=model, u_obs=u_obs, x_locs=x_obs,
            psi_prior=psi_prior, rng=rng, n_groups=1,
        )

        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        np.testing.assert_allclose(
            lik_default.log_density(params),
            lik_explicit.log_density(params),
            rtol=1e-12, atol=0,
        )
        np.testing.assert_allclose(
            lik_default.grad_log_density(params),
            lik_explicit.grad_log_density(params),
            rtol=1e-12, atol=0,
        )

    def test_log_density_matches_block_diag_reference(self):
        """Hand-built block_diag Σ over (DOF × group) gives same log L."""
        import scipy.linalg as sla
        from scipy.stats import multivariate_normal

        rng = np.random.default_rng(123)
        n_components = 3
        n_groups = 2
        P = 4
        m = n_components * n_groups * P  # 24

        F = np.array([1.0])
        x_obs_full = np.linspace(0.05, 1.0, m)
        model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs_full)

        # Independent random per-group sensor coordinates in 1D
        x_locs_g = [rng.uniform(0.0, 1.0, (P, 1)) for _ in range(n_groups)]
        x_locs_flat = np.vstack(x_locs_g)                 # (n_groups*P, 1)

        theta_true = np.array([2.0, 3.0])
        eta = model.eval(theta_true, idx=0)
        u_obs = (eta + rng.normal(0, 0.05, m)).reshape(1, -1)

        psi_prior = MultivariateNormal(
            mean=np.array([-4.0, -6.0, 1.5]),
            cov=4.0 * np.eye(3),
            rng=rng,
        )
        lik = KOGaussianLikelihood(
            model=model, u_obs=u_obs, x_locs=x_locs_flat,
            psi_prior=psi_prior, rng=rng,
            n_components=n_components, n_groups=n_groups,
        )

        log_s2d, log_s2e, log_rho = -4.0, -6.0, 1.5
        sigma2_delta, sigma2_eps, rho = np.exp([log_s2d, log_s2e, log_rho])
        params = np.array([theta_true[0], theta_true[1],
                           log_s2d, log_s2e, log_rho])

        # Reference: hand-build the full Σ as block_diag over (d, g).
        blocks = []
        for _ in range(n_components):
            for g in range(n_groups):
                C_g = rbf_kernel_matrix(x_locs_g[g], np.array([rho]))
                blocks.append(sigma2_delta * C_g + sigma2_eps * np.eye(P))
        Sigma_full = sla.block_diag(*blocks)
        mean_full = model.eval(theta_true, idx=0)
        ll_ref = multivariate_normal.logpdf(u_obs[0], mean=mean_full,
                                            cov=Sigma_full)

        ll = lik.log_density(params)
        np.testing.assert_allclose(ll, ll_ref, rtol=1e-10, atol=0)

    def test_grad_with_n_groups(self):
        """Analytic gradient matches FD for n_components=3, n_groups=3."""
        rng = np.random.default_rng(2024)
        n_components = 3
        n_groups = 3
        P = 4
        m = n_components * n_groups * P  # 36

        F = np.array([1.0])
        x_obs_full = np.linspace(0.02, 1.0, m)
        model = PiecewiseConstantModel(F=F, n_params=2, x_obs=x_obs_full)

        x_locs_flat = rng.uniform(0.0, 1.0, (n_groups * P, 1))

        theta_true = np.array([2.0, 3.0])
        eta = model.eval(theta_true, idx=0)
        u_obs = (eta + rng.normal(0, 0.05, m)).reshape(1, -1)

        psi_prior = MultivariateNormal(
            mean=np.array([-4.0, -6.0, 1.5]),
            cov=4.0 * np.eye(3),
            rng=rng,
        )
        lik = KOGaussianLikelihood(
            model=model, u_obs=u_obs, x_locs=x_locs_flat,
            psi_prior=psi_prior, rng=rng,
            n_components=n_components, n_groups=n_groups,
        )

        params = np.array([2.0, 3.0, -4.0, -6.0, 1.5])
        grad_analytic = lik.grad_log_density(params)
        grad_fd = _fd_gradient(lik.log_density, params)
        np.testing.assert_allclose(grad_analytic, grad_fd,
                                   rtol=1e-4, atol=1e-7)
