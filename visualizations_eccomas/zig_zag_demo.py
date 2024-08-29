import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from src.samplers import ZigZagSampler
from src.distributions import MultivariateNormal, Posterior, FlatLikelihood
from src.utils import plot_pdf_contours

sns.set_style('white')

rng = np.random.default_rng(0)

fig_name = 'non_rev_demo'

if __name__ == '__main__':

    mean = np.zeros(2)
    cov = np.diag([2., 1.])
    plot_limits = [[-4.5, 4.5], [-3, 3]]

    # Define the target distribution
    prior = MultivariateNormal(mean, cov)
    likelihood = FlatLikelihood(2)
    target = Posterior(prior, likelihood)

    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    ax.grid(False)

    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.axis('equal')
    ax.autoscale(enable=False)

    # despine the plot
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    plot_pdf_contours(target, ax, plot_limits)

    n_events = 200
    sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False)
    sampler.run()

    time = sampler.times_
    positions = sampler.positions_
    velocities = sampler.velocities_

    steps = [2,3,4,20,200]

    ax.scatter(*positions[0], c='C0', marker='o', s=10)

    for i, step in enumerate(steps):
        path = ax.plot(positions[:step, 0], positions[:step, 1], c='C0', alpha=0.75, linewidth=1.)[0]
        # if i == len(steps) - 1:
        #     ax.scatter(*positions[steps[-1] - 1], c='C0', marker='o', s=10)
        fig.savefig(join('figures', f'zig_zag_normal_{step}.pdf'))
        path.remove()

    print(positions[-1])
    plt.show()
