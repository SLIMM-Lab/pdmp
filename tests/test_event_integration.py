"""Tests for the BPS event-time integrator (fixed-step vs adaptive ODE).

These run the Bouncy Particle Sampler with **no surrogate**, so the event-time
generator integrates the *true* target rate and the only error sources are the
integration scheme and finite-run Monte-Carlo noise. The target is a mild
2-component Gaussian mixture -- a small, smooth deviation from a Gaussian whose
mean/covariance are known in closed form, but whose log-gradient is nonlinear so
the integration step actually matters (a single Gaussian has a piecewise-linear
rate the trapezoid integrates almost exactly).
"""

import numpy as np
import pytest

from pdmp.bouncy_particle import BouncyParticleSampler
from pdmp.zigzag import ZigZagSampler
from pdmp.distributions import GaussianMixture, MultivariateNormal
from pdmp.utils import running_mean, running_variance


def make_target(sep: float = 1.5) -> GaussianMixture:
    """Mild unimodal 2-component Gaussian mixture in 2D (sep < 2 -> unimodal)."""
    means = np.array([[-sep / 2, 0.0], [sep / 2, 0.0]])
    covs = np.array([np.eye(2), np.eye(2)])
    weights = np.array([0.5, 0.5])
    return GaussianMixture(means, covs, weights)


def run_bps(target, integrator, t_max=2000.0, seed=0, **kw):
    bps = BouncyParticleSampler(target=target,
                                t_max=t_max,
                                refresh_rate=0.1,
                                integrator=integrator,
                                seed=seed,
                                **kw)
    bps.run()
    n = bps._iter
    t, x, v = bps.times[:n + 1], bps.positions[:n + 1], bps.velocities[:n]
    est_mean = running_mean(t, x, v)[-1]
    est_var = running_variance(t, x, v)[-1]
    evals_per_event = float(np.mean(bps._rate_evals_per_event))
    return est_mean, est_var, evals_per_event


@pytest.mark.parametrize("integrator,kw", [
    ("ode", {}),
    ("fixed", {"int_dt": 0.01}),
])
def test_recovers_mixture_moments(integrator, kw):
    """Both integrators recover the analytic mixture mean/variance."""
    target = make_target()
    true_mean = target.mean
    true_var = np.diag(target.cov)

    est_mean, est_var, _ = run_bps(target, integrator, **kw)

    # Generous tolerances: this guards against a broken integrator (wrong
    # invariant measure), not against Monte-Carlo noise at finite horizon.
    assert np.linalg.norm(est_mean - true_mean) < 0.2, \
        f"{integrator}: mean {est_mean} far from {true_mean}"
    assert np.linalg.norm(est_var - true_var) < 0.4, \
        f"{integrator}: var {est_var} far from {true_var}"


def test_ode_agrees_with_fixed_step():
    """The adaptive ODE path samples the same target as the fixed-step march."""
    target = make_target()
    mean_ode, var_ode, _ = run_bps(target, "ode")
    mean_fix, var_fix, _ = run_bps(target, "fixed", int_dt=0.01)

    assert np.linalg.norm(mean_ode - mean_fix) < 0.25
    assert np.linalg.norm(var_ode - var_fix) < 0.4


def test_ode_cheaper_than_fine_fixed_step():
    """The ODE integrator matches accuracy at far fewer rate evaluations than
    the legacy fixed-step default (int_dt=0.01)."""
    target = make_target()
    _, _, evals_ode = run_bps(target, "ode")
    _, _, evals_fixed = run_bps(target, "fixed", int_dt=0.01)

    # Fixed dt=0.01 spends ~200 evals/event; the adaptive ODE spends far fewer.
    assert evals_ode < 0.5 * evals_fixed, \
        f"ODE evals/event {evals_ode:.1f} not < half of fixed {evals_fixed:.1f}"


def test_default_integrator_is_ode():
    """The ODE integrator is the default for the general (no-surrogate) path."""
    bps = BouncyParticleSampler(target=make_target(), n_max=10, seed=0)
    assert bps._generate_event_times == bps._inverse_cdf_ode


def test_bounce_rate_excludes_refresh_rate():
    """The bounce intensity is the canonical (v.gradU)_+ (+gamma floor), with
    refresh_rate NOT folded in -- refresh is handled by the separate clock."""
    target = MultivariateNormal(mean=np.zeros(2), cov=np.eye(2))
    refresh_rate = 0.3
    bps = BouncyParticleSampler(target=target, n_max=10,
                                refresh_rate=refresh_rate, seed=0)
    # Position with a non-zero canonical rate so we can read off the offset.
    bps.positions[0] = np.array([2.0, 0.0])
    bps.velocities[0] = np.array([1.0, 0.0])  # moving away from the mode
    # grad log p = -x = [-2, 0]; -(grad . v) = 2 > 0  ->  rate = 2 + gamma.
    rate = bps._target_rates(bps.positions[0])
    assert rate == pytest.approx(2.0 + bps._gamma, abs=1e-9)
    # refresh_rate must not appear in the bounce rate.
    assert abs(rate - (2.0 + refresh_rate)) > 0.1


def test_fixed_step_hang_safe_on_zero_bounce_rate():
    """With refresh_rate removed from the rate, the canonical rate can be ~0 over
    a stretch. The fixed-step integrator must still terminate (capped at
    refresh_time) instead of looping ~1/gamma times."""
    target = MultivariateNormal(mean=np.zeros(2), cov=np.eye(2))
    bps = BouncyParticleSampler(target=target, n_max=10, refresh_rate=0.1,
                                integrator='fixed', int_dt=0.01, seed=1)
    # Move straight toward the mode: -(grad . v) = (x.v) < 0 for the whole ray
    # until the mode is passed, so the bounce rate is ~0 (just gamma).
    bps.positions[0] = np.array([6.0, 0.0])
    bps.velocities[0] = np.array([-1.0, 0.0])

    for _ in range(5):
        tau, event_type = bps._inverse_cdf()
        assert np.isfinite(tau) and tau >= 0
        assert event_type in (0, 1)
        # Bounded by ~refresh_time / int_dt; a missing cap (rate ~ gamma=1e-6)
        # would need ~1e8 iterations.
        assert bps._rate_evals_per_event[-1] < 50_000


# --------------------------------------------------------------------------- #
# ZigZag carries the same adaptive-ODE event-time generator (multi-dimensional:
# one terminal event per coordinate). ZigZag has no velocity-refresh mechanism,
# so its rate is already the canonical (-grad . v)_+ + gamma -- no refresh term.
# --------------------------------------------------------------------------- #


def run_zigzag(target, integrator, t_max=500.0, seed=0, **kw):
    zz = ZigZagSampler(target=target,
                       t_max=t_max,
                       integrator=integrator,
                       seed=seed,
                       **kw)
    zz.run()
    n = zz._iter
    t, x, v = zz.times[:n + 1], zz.positions[:n + 1], zz.velocities[:n]
    est_mean = running_mean(t, x, v)[-1]
    est_var = running_variance(t, x, v)[-1]
    evals_per_event = float(np.mean(zz._rate_evals_per_event))
    return est_mean, est_var, evals_per_event


@pytest.mark.parametrize("integrator,kw", [
    ("ode", {}),
    ("fixed", {"dt": 0.01}),
])
def test_zigzag_recovers_mixture_moments(integrator, kw):
    """Both ZigZag integrators recover the analytic mixture mean/variance."""
    target = make_target()
    est_mean, est_var, _ = run_zigzag(target, integrator, **kw)

    assert np.linalg.norm(est_mean - target.mean) < 0.2, \
        f"{integrator}: mean {est_mean} far from {target.mean}"
    assert np.linalg.norm(est_var - np.diag(target.cov)) < 0.4, \
        f"{integrator}: var {est_var} far from {np.diag(target.cov)}"


def test_zigzag_ode_agrees_with_fixed_step():
    """ZigZag's adaptive ODE path samples the same target as the fixed march."""
    target = make_target()
    mean_ode, var_ode, _ = run_zigzag(target, "ode")
    mean_fix, var_fix, _ = run_zigzag(target, "fixed", dt=0.01)

    assert np.linalg.norm(mean_ode - mean_fix) < 0.25
    assert np.linalg.norm(var_ode - var_fix) < 0.4


def test_zigzag_default_integrator_is_ode():
    """The ODE integrator is the default for ZigZag's general path."""
    zz = ZigZagSampler(target=make_target(), n_max=10, seed=0)
    assert zz._generate_event_times == zz._inverse_cdf_ode
