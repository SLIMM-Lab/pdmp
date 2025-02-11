import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import join

from pdmp.mcmc import RandomWalkMetropolisSampler, LangevinDynamicsSampler, ZigZagSampler
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from pdmp.utils import plot_pdf_contours, central_moment_from_skeleton
from pdmp.plotting import plot_samples

sns.set_style('white')

save_fig = True
fig_path = './figures'
fig_name = 'non_rev_demo'

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    mean = np.zeros(2)
    cov = np.diag([2., 1.])
    plot_limits = [[-4.5, 4.5], [-3, 3]]

    # Define the target distribution
    target = MultivariateNormal(mean, cov)

    # create and formate the plot
    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    ax.grid(False)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.axis('equal')
    sns.despine()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    # plot pdf contours
    plot_pdf_contours(target, ax, plot_limits)

    # define events and velocities
    x0 = np.array([-2., -2.])
    degrees = [np.pi/6, 2*np.pi/3, - 3.5 * np.pi/5]
    vs = np.array([[np.cos(d), np.sin(d)] for d in degrees])
    steps = [4, 2, 3]
    x = np.zeros((np.sum(steps) + 1, 2))
    x[0] = x0

    # plot the initial point
    ax.scatter(x[0, 0], x[0, 1], c='C2', s=20, zorder=5)
    if save_fig:
        fig.savefig(join(fig_path, fig_name + f"_{0}_green.pdf"))

    # loop over the events and velocities
    idx = 1
    for i, v in enumerate(vs):
        print(f"i: {i}")
        for j in range(steps[i]):
            print(f"j: {j}")
            x[idx] = x[idx - 1] + v
            ax.plot(x[idx - 1:idx + 1, 0], x[idx - 1:idx + 1, 1], c='C0')
            dot = ax.scatter(x[idx, 0], x[idx, 1], c='C0', s=20, zorder=5)
            fig.savefig(join('./figures', fig_name + f"_{idx}_blue.pdf"))
            dot.remove()

            # check if turning points is reached
            if j == steps[i] - 1:
                ax.scatter(x[idx, 0], x[idx, 1], c='C3', s=20, zorder=5)
                if save_fig:
                    fig.savefig(join('./figures', fig_name + f"_{idx}_red.pdf"))
            else:
                ax.scatter(x[idx, 0], x[idx, 1], c='C2', s=20, zorder=5)
                if save_fig:
                    fig.savefig(join('./figures', fig_name + f"_{idx}_green.pdf"))

            idx += 1

    fig.show()
