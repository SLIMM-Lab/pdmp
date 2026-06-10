"""Bit-identical checkpoint/resume tests for the PDMP samplers.

A run that is checkpointed mid-way, reconstructed from the same config and
resumed must produce *exactly* the same skeleton (positions/times/velocities)
and final rng state as an uninterrupted run. We use a cheap 2-D Gaussian target
(no FEM, no surrogate) so the tests run in a fraction of a second.
"""

import numpy as np
import pytest

from pdmp.distributions import MultivariateNormal
from pdmp.zigzag import ZigZagSampler
from pdmp.bouncy_particle import BouncyParticleSampler


SEED = 1234
N_MAX = 200
STOP_AT = 100  # emulate an interruption after this many events


def make_target():
    """A small, fixed 2-D Gaussian — fast and fully analytic."""
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    return MultivariateNormal(mean=mean, cov=cov)


@pytest.mark.parametrize("SamplerCls", [ZigZagSampler, BouncyParticleSampler])
def test_resume_is_bit_identical(SamplerCls, tmp_path):
    """Checkpoint after STOP_AT events, resume in a fresh sampler, and assert
    the completed skeleton matches an uninterrupted run exactly."""

    # --- reference: one uninterrupted run ----------------------------------
    ref = SamplerCls(make_target(), n_max=N_MAX,
                     rng=np.random.default_rng(SEED))
    ref.run()

    # --- interrupted run: step partway, then checkpoint --------------------
    s1 = SamplerCls(make_target(), n_max=N_MAX,
                    rng=np.random.default_rng(SEED))
    for _ in range(STOP_AT):
        s1._step()  # exactly what the run loop does, without checkpoint cadence
    assert s1._iter == STOP_AT

    ckpt = str(tmp_path / "checkpoint.pkl")
    s1.save_checkpoint(ckpt)

    # --- resume: rebuild from the same config, load, finish ----------------
    s2 = SamplerCls(make_target(), n_max=N_MAX,
                    rng=np.random.default_rng(SEED))
    resumed_iter = s2.load_checkpoint(ckpt)
    assert resumed_iter == STOP_AT
    s2.run()

    # --- the resumed run must equal the reference exactly ------------------
    assert s2._iter == ref._iter
    assert np.array_equal(s2.positions, ref.positions)
    assert np.array_equal(s2.times, ref.times)
    assert np.array_equal(s2.velocities, ref.velocities)
    # The rng must end in the identical internal state.
    assert s2._rng.bit_generator.state == ref._rng.bit_generator.state


def test_gp_surrogate_reload_is_bit_identical(tmp_path):
    """A GP surrogate trained once and reloaded from disk must reproduce its
    predictions (and hence the sample path) bit-for-bit.

    This mirrors the fresh-vs-resume flow in ``run_inference.py``: each launch
    rebuilds the target + surrogate from the same seed, so the (rebuilt) Laplace
    base is identical; the reloaded GP restores the exact trained state instead
    of retraining. Kept small so it runs in a couple of seconds.
    """
    import torch
    from pdmp.surrogates import GaussianProcess

    torch.set_default_dtype(torch.float64)

    def build(train_on_init, reload_path=None):
        # One shared rng for target + surrogate, freshly seeded per "launch".
        rng = np.random.default_rng(SEED)
        gp = GaussianProcess(make_target(), rng=rng, n_samples=20,
                             n_restarts=3, lbfgs_steps=20,
                             train_on_init=train_on_init)
        if reload_path is not None:
            gp.load_model(reload_path)
        return gp

    save_dir = str(tmp_path / "surrogate")

    # "Fresh run": trains, then persists (train() also auto-saves, but be explicit).
    gp_fresh = build(train_on_init=True)
    gp_fresh.save_model(save_dir)
    # train_data.pt is the exact binary round-trip that guarantees the match.
    assert (tmp_path / "surrogate" / "train_data.pt").exists()

    # "Resumed run": same seed, no retraining, reload from disk.
    gp_resumed = build(train_on_init=False, reload_path=save_dir)

    xs = np.random.default_rng(SEED + 7).normal(size=(8, 2))
    grad_fresh = np.array([gp_fresh.grad(x) for x in xs])
    grad_resumed = np.array([gp_resumed.grad(x) for x in xs])
    eval_fresh = np.array([gp_fresh.eval(x) for x in xs])
    eval_resumed = np.array([gp_resumed.eval(x) for x in xs])

    assert np.array_equal(grad_fresh, grad_resumed)
    assert np.array_equal(eval_fresh, eval_resumed)


@pytest.mark.parametrize("SamplerCls", [ZigZagSampler, BouncyParticleSampler])
def test_load_state_dict_rejects_mismatched_config(SamplerCls):
    """A checkpoint from a differently-sized run must not be silently loaded."""

    src = SamplerCls(make_target(), n_max=N_MAX,
                     rng=np.random.default_rng(SEED))
    state = src.state_dict()

    # Different budget -> dim/n_max guard must trip.
    dst = SamplerCls(make_target(), n_max=N_MAX + 1,
                     rng=np.random.default_rng(SEED))
    with pytest.raises(ValueError):
        dst.load_state_dict(state)
