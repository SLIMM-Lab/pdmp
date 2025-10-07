import numpy as np

from pdmp.loader import get_target
from pdmp.random_field import get_field
from pdmp.forward_model import get_model
from pdmp.distributions import get_prior, Posterior


def test_random_field_loader_integration():
    rng = np.random.default_rng(0)
    field_cfg = {
        'name': 'GaussianRandomField',
        'dim': 3,
        'mean': 1.5,
        'interval': [0.0, 1.0],
        'kernel_params': {'sigma': 1.0, 'l': 0.3},
    }
    config = {
        'name': 'BayesianInverse',
        'model': {
            'name': 'PiecewiseConstant',
            'field': field_cfg,
            'n_obs_loc': 3,
            'F': [1.0],
        },
        'prior': {'name': 'FromField'},
        'likelihood': {'name': 'FlatLikelihood'},
    }

    posterior = get_target(config, rng=rng)
    assert isinstance(posterior, Posterior)

    # Reconstruct field to compare expected mean/cov (loader currently doesn't expose it)
    field = get_field(field_cfg, rng=rng)
    assert posterior.dim == field.dim
    # Check mean/cov numerical equality
    assert np.allclose(posterior.prior.mean, field.mean)
    assert np.allclose(posterior.prior.cov, field.cov)


def test_random_field_direct_usage():
    rng = np.random.default_rng(1)
    field_cfg = {
        'name': 'GaussianRandomField',
        'dim': 3,
        'mean': 0.7,
        'interval': [0.0, 1.0],
        'kernel_params': {'sigma': 1.0, 'l': 0.25},
    }
    field = get_field(field_cfg, rng=rng)

    # Build model directly with field
    model_cfg = {
        'name': 'PiecewiseConstant',
        'n_obs_loc': 3,
        'F': [1.0, 2.0],
        'dim': field.dim,  # fallback value, should be overridden by field
    }
    model = get_model(model_cfg, field=field)
    assert model.get_dim_in() == field.dim

    # Prior from field
    prior_cfg = {'name': 'FromField'}
    prior = get_prior(prior_cfg, rng=rng, field=field)
    assert np.allclose(prior.mean, field.mean)
    assert np.allclose(prior.cov, field.cov)

    # Simple evaluation round-trip: sample coeffs, evaluate model (uses params directly)
    coeff_sample = prior.get_sample()
    # Model expects parameter vector of same dimension
    u_val = model.eval(coeff_sample)
    assert u_val.shape[0] == model.get_dim_out()

