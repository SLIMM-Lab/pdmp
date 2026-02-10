#!/usr/bin/env python3
"""Example of using JaxFemModel in inverse problem."""

import numpy as np
import os

from scipy.stats import gaussian_kde

from pdmp.distributions import Transformation
from pdmp.loader import get_target, get_sampler, get_surrogate
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp.forward_model import JaxFemModel
from pdmp.utils import central_moment_from_skeleton, sample_equidistant_along_path

save_fig = False
fig_path = "./figures"
generate_observations = False

config = {
    'name': 'BayesianInverse',
    'model': {
        'name': 'JaxFem',
        'd_x': 1.0,
        'd_y': 1.0,
        'd_z': 2.5,
        'h': 0.25,
        'nu': 0.3,
        'field': {
            'name': 'JaxConstantField',
            'mean': 1.0,
            'std': 0.25
        }
    },
    'prior': {
        'name': 'FromField'
    },
    'likelihood': {
        'name': 'TransformedLikelihood',
        'transformation': 'Exponential',
        'likelihood': {
            'name': 'GaussianLikelihood',
            'sigma': 0.001,
            'observation_file': 'observations.dat'  # to be created
        }
    }
}

surrogate_config = {
    'name': 'Laplace',
    'mean': [0.0],
    'cov': [[1.0]]
}

zig_zag_config = {
    'name': 'ZigZag',
    't_max': 20,
    'dt': 0.01,
    'offset_shrinkage': 0.01,
    'x_0': np.array([0.0])
}

transformed_config = {
    'name': 'Transformed',
    'transformation': 'Affine',
    'distribution': config
}

if __name__ == "__main__":

    rng = np.random.default_rng(seed=42)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Create a temporary observation file
    theta_true = np.array([1.1])

    # check if observation file exists, if not generate it
    if generate_observations and not os.path.exists(config['likelihood']['likelihood']['observation_file']):

        # First, generate synthetic observations
        print("=" * 70)
        print("Generating synthetic observations...")
        print("=" * 70)

        model = JaxFemModel.from_dict(config['model'])

        # Create a simple model to generate observations
        y_obs = model.eval(np.exp(theta_true)) + rng.standard_normal(model.get_dim_out()) * config['likelihood']['likelihood']['sigma']

        # Write observations to file
        file_name = 'observations.dat'
        np.savetxt(file_name, y_obs.reshape(1, -1))

        print(f"  Created observation file: {file_name}")
        print(f"  Observations shape: {y_obs.shape}")
        print(f"  Observations: {y_obs}")

    # Now load the full problem using the loader
    print("=" * 70)
    print("Loading full problem using loader...")
    print("=" * 70)
    target = get_target(config, rng=rng)
    print("Loaded target distribution:")

    fig, ax = get_2d_despined_figure(figsize=(5, 3), equal_axes=False, keep_ticks=True)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$p(\theta)$')

    M = np.array([[0.04336]])
    b = np.array([1.08057])
    transformed_config['M'] = M**2
    transformed_config['b'] = b

    trans_target = get_target(transformed_config, rng=rng)
    transform: Transformation = trans_target._transformation

    surrogate = get_surrogate(surrogate_config, target=trans_target, rng=rng)
    zig_zag = get_sampler(zig_zag_config, target=trans_target, rng=rng, surrogate=surrogate)
    zig_zag.run()
    zig_zag.write_data(folder='zig-zag_data')
    pos = zig_zag.positions
    vel = zig_zag.velocities
    times = zig_zag.times

    pos = np.loadtxt('zig-zag_data/positions.dat').reshape(-1, 1)
    vel = np.loadtxt('zig-zag_data/velocities.dat').reshape(-1, 1)
    times = np.loadtxt('zig-zag_data/times.dat')

    mean = central_moment_from_skeleton(times, pos, vel, degree=1)
    samples_zz = sample_equidistant_along_path(pos, vel, times, N=500)
    samples_zz_orig = transform.transform(samples_zz)

    sampler_config = {
        'name': 'RandomWalkMetropolis',
        'sigma': 0.1,
        'x_0': np.array([1.0]),
        'n_samples': 300
    }

    sampler = get_sampler(sampler_config, target=target, rng=rng)
    sampler.run()
    samples = sampler.chain


    x_min = 0.9
    x_max = 1.25
    x = np.linspace(x_min, x_max, 10).reshape(-1, 1)
    # log_pdf_surrogate = np.zeros_like(x)
    log_pdf_target = np.zeros_like(x)
    log_pdf_prior = np.zeros_like(x)
    for i, xi in enumerate(x):
        # log_pdf_surrogate[i] = surrogate.eval(xi)
        # log_pdf_target[i] = trans_target.log_density(xi)
        log_pdf_target[i] = trans_target._base_distribution.log_density(xi)
        log_pdf_prior[i] = target.prior.log_density(xi)

    # log_pdf_surrogate = log_pdf_surrogate - np.max(log_pdf_surrogate)
    # log_pdf_target = log_pdf_target - np.max(log_pdf_target) + np.log(0.39)
    log_pdf_target = log_pdf_target - np.max(log_pdf_target) + np.log(8.8)

    # ax.plot(x, np.exp(log_pdf_surrogate), '-', label='Surrogate', c='C0')
    ax.plot(x, np.exp(log_pdf_prior), '-', label='Prior', c='C0')
    ax.plot(x, np.exp(log_pdf_target), '-', label='Posterior', c='C1')

    # perform kde and plot
    x_grid = np.linspace(x_min, x_max, 100)
    kde = gaussian_kde(samples_zz_orig.flatten())
    kde_values = kde(x_grid)
    ax.plot(x_grid, kde_values, label='KDE of ZZS', c='C2', ls='-')

    kde = gaussian_kde(samples.flatten())
    kde_values = kde(x_grid)
    ax.plot(x_grid, kde_values, label='KDE of RWM', c='C3', ls='-')

    ax.axvline(theta_true[0], c='k', ls='--', label='True value')

    ax.legend()

    if save_fig:
        fig.savefig(os.path.join(fig_path, 'zz_rwm_post.pdf'), bbox_inches='tight')

    fig.show()
