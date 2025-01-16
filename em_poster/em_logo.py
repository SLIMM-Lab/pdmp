import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from pdmp.zigzag import ZigZagSampler
from pdmp.distributions import Posterior, FlatLikelihood, Distribution
from pdmp.utils import grad_fd
from pdmp.plotting import get_2d_despined_figure


class EMLogo(Distribution):

    def __init__(self, rng:np.random.Generator =None):
        super().__init__(rng=rng)
        self.a = 1.7
        self.b = 1.5
        self.c = 0.1
        self.d = 7.
        self.e = 1.
        self.f = 3.5
        self.g = 0.25
        self.h = 5.

    def get_dim(self) -> int:
        return 2

    def log_density(self, x: np.ndarray) -> float:
        return 4 * (- self.a * (self.b*x[1]  + np.sin((x[0] + self.d) * (self.e - self.c*x[0])))**2.
                + 0.35 * np.exp(- (x[0] + self.f)**2)
                + 0.35 * np.exp(- self.g * (x[0] - self.h)**2))
                # + np.exp(- np.abs(0.25 * (x[0])**2))

    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        return grad_fd(self.log_density, x)

    def hessian_log_density(self, x: np.ndarray) -> np.ndarray:
        return -np.eye(2)

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
    means = np.array([[-3., 0.], [3., 0.]])
    covs = np.array([np.diag([2.5, 1.5]), np.diag([2., 1.])])
    weights = np.array([0.5, 0.5])

    # Define the target distribution
    prior = EMLogo(rng=rng)
    likelihood = FlatLikelihood(2)
    target = Posterior(prior, likelihood)

    plot_limits = ([-5., 5.], [-1.4, 1.4])

    # set up the plot
    scale = 1.
    fig, ax = get_2d_despined_figure(plot_limits=plot_limits, figsize=(scale * np.sum(np.abs(plot_limits), axis=1)))
    ax.axis('off')

    # # plot the contours of the p_x
    # plot_pdf_contours(prior, ax, plot_limits, n_levels=20)

    # set zig-zag parameters and run sampler
    n_events = 4000
    sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False, dt=0.01, x0=np.array([0., 0.]))
    sampler.run()

    # extract events and path and plot
    time = sampler.times_
    positions = sampler.positions_
    velocities = sampler.velocities_
    path = ax.plot(positions[:, 0], positions[:, 1], c='C1', alpha=0.75, linewidth=1.)[0]

    # save figure and remove path
    if save_fig:
        fig.savefig(join(fig_path, f'zig_zag_em_logo_orange.pdf'))

    plt.show()
