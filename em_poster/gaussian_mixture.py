import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from pdmp.zigzag import ZigZagSampler
from pdmp.distributions import Posterior, FlatLikelihood, GaussianMixture
from pdmp.plotting import plot_pdf_contours
from pdmp.plotting_utils import get_2d_despined_figure

sns.set_style('white')

rng = np.random.default_rng(0)

save_fig = False
fig_path = '/home/leon/ownCloud/Documents/presentations/em/symposium_25/figures'


if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    rng = np.random.default_rng(0)

    # define p_x params
    scale = 0.71
    means = np.sqrt(scale) * np.array([[-2.5, 0.], [2.5, 0.]])
    covs = np.array(
         [np.array([[2., 0.7],
                   [0.7, 0.6]]),
          np.array([[1.5, -0.3],
                    [-0.3, 0.8]])])
    covs = covs @ np.diag([1, scale])
    weights = np.array([0.5, 0.5])

    # Define the target distribution
    prior = GaussianMixture(means, covs, weights, rng=rng)
    likelihood = FlatLikelihood(2)
    target = Posterior(prior, likelihood)
    mixture_cov = prior.get_cov()

    plot_limits = ([-6., 6.], [-2.2, 2.2])
    scale = 0.35
    figsize = scale * np.abs(plot_limits[0][1] - plot_limits[0][0]), scale * np.abs(plot_limits[1][1] - plot_limits[1][0])

    # set up the plot
    fig, ax = get_2d_despined_figure(plot_limits=plot_limits, figsize=figsize, axes_label='E')

    # plot the contours of the p_x
    plot_pdf_contours(prior, ax, plot_limits, n_levels=20)

    # set zig-zag parameters
    n_events = 1000

    sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False, dt=0.01)
    sampler.run()

    # extract events and path and plot
    time = sampler.times_
    positions = sampler.positions_
    velocities = sampler.velocities_
    path = ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)[0]

    # save figure and remove path
    if save_fig:
        fig.savefig(join(fig_path, f'zig_zag_gaussian_mixture.pdf'))

    plt.show()
