import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from src.samplers import MetropolisHastingsSampler, LangevinDynamicsSampler, ZigZagSampler
from src.distributions import MultivariateNormal, Likelihood, Posterior, FlatLikelihood
from src.utils import plot_pdf_contours, plot_samples, central_moment_from_skeleton

sns.set_style('white')

rng = np.random.default_rng(0)

save_fig = False
fig_path = 'figures'

if __name__ == '__main__':

    # define p_x params
    mean = np.zeros(2)
    cov = np.diag([2., 1.])

    # Define the target distribution
    prior = MultivariateNormal(mean, cov)
    likelihood = FlatLikelihood(2)
    target = Posterior(prior, likelihood)

    # set up the plot
    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    ax.grid(False)

    # format plot
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    plot_limits = [[-4.5, 4.5], [-3, 3]]
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.axis('equal')
    ax.autoscale(enable=False)
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    # plot the contours of the p_x
    plot_pdf_contours(target, ax, plot_limits)

    # set zig-zag parameters
    n_events = 200
    dts = [0.1, 0.01]

    # run the zig-zag sampler for different time
    for dt in dts:
        sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False, dt=dt)
        sampler.run()

        # extract events and path and plot
        time = sampler.times_
        positions = sampler.positions_
        velocities = sampler.velocities_
        path = ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)[0]

        # save figure and remove path
        if save_fig:
            fig.savefig(join(fig_path, f'zig_zag_normal_dt-{dt}.pdf'))
        path.remove()

    plt.show()
