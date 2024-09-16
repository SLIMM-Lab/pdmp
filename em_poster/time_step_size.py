import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from src.samplers import ZigZagSampler
from src.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from src.utils import plot_pdf_contours, get_2d_despined_figure
from src.forward_model import ForwardModel
from src.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis

sns.set_style('white')
rng = np.random.default_rng(0)

save_fig = False
fig_path = '/home/leon/ownCloud/Documents/presentations/em/symposium_25/figures'

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # get the prior field
    kernel_params = {'sigma': 1., 'l': 0.3}
    n_b = 2
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_b, interval)
    prior_cov = compute_coefficients(squared_exponential_kernel, basis, interval, kernel_params=kernel_params)
    mean = 4.
    prior = MultivariateNormal(mean * np.ones(n_b), prior_cov, rng=rng)

    # set up the forward model
    n_obs = 2
    F = [1.]
    F = [item for item in F for i in range(n_obs)]
    n_params = n_b
    model = ForwardModel(F, n_params)

    # generate ground truth from a multi-variate normal distribution
    mean = 3 * np.ones(n_params)
    cov = np.eye(n_params)
    params_gt = rng.multivariate_normal(mean, cov)
    print(f"Ground truth: \n {params_gt}")
    margins = 1.5
    plot_limits = ([2.2, 4.2], [2.1, 4.2])

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    # generate n_obs noisy obersevations of u_gt
    sigma_obs = 0.025
    # sigma_obs = 0.05

    # evaluate the ground truth at the observation and test locations
    u_gt = model.eval(x_obs, params_gt)
    u_gt_test = model.eval(x_test, params_gt)
    u_obs = np.zeros((len(F), u_gt.shape[0]))

    for i in range(len(F)):
        u_gt = model.eval(x_obs, params_gt, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    # define likelihood and posterior
    likelihood = GaussianLikelihood(model, x_obs, u_obs, sigma_obs)
    target = Posterior(prior, likelihood)

    # set up the plot
    scaling = 0.9
    fig, ax = get_2d_despined_figure(plot_limits=plot_limits, keep_ticks=False, figsize=(scaling * 3, scaling *3.2))
    plot_pdf_contours(target, ax, plot_limits)

    # set zig-zag parameters
    n_events = 200
    # dts = [0.05, 0.005]
    dts = [0.1, 0.01]

    # run the zig-zag sampler for different time
    for dt in dts:
        sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False, dt=dt, x0=params_gt)
        sampler.run()

        # extract events and path and plot
        time = sampler.times_
        positions = sampler.positions_
        velocities = sampler.velocities_
        path = ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)[0]

        # save figure and remove path
        if save_fig:
            fig.savefig(join(fig_path, f'zig_zag_dt-{dt}.pdf'))
        path.remove()

    plt.show()
