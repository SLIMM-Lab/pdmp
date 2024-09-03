import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import binom

from src.distributions import MultivariateNormal


def get_2d_despined_figure(plot_limits, nrows=1, ncols=1, figsize=(3., 4.), constrained_layout=True):

    # create figure
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=constrained_layout)

    # format the plot
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.axis('equal')
    ax.grid(False)
    ax.autoscale(enable=False)

    # set labels
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    # despine the plot
    sns.despine()

    # get rid of the ticks and tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    return fig, ax


def plot_pdf_contours(distribution, ax, plot_limits, n_grid=100, alpha=0.6, n_levels=20,
                      cmap=sns.color_palette('rocket', as_cmap=True)):
    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = np.exp(distribution.logDensity(np.array([X[i, j], Y[i, j]])))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax


def plot_samples(samples, ax, color_code=True, n_vis=500):
    samples_plot = samples[0::samples.shape[0] // n_vis]

    if color_code:
        ax.scatter(*samples_plot.transpose(), s=3, zorder=2, c=np.linspace(0, 1, samples_plot.shape[0]))
    else:
        ax.scatter(*samples_plot.transpose(), s=3, zorder=2, c='C0')

    return ax


def central_moment_from_skeleton(t, x, v, degree):

    # Compute the mean of the curve
    # mean = np.zeros(x.shape[1])
    if degree != 1:
        mean = central_moment_from_skeleton(t, x, v, 1)
    else:
        mean = np.zeros(x.shape[1])

    n_events, d = x.shape

    n_segments = n_events - 1
    total_integral = np.zeros_like(mean)

    for i in range(n_segments):
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        v0 = v[i]

        def parametric_point(s):
            return x0 + s * v0

        def integrand(k):
            return (t1 - t0)**(k + 1) / (k + 1)

        a = x0 - mean
        for k in range(degree + 1):
            # binomial_coeff = binom(degree, k)
            # integral = np.zeros_like(mean)
            total_integral += binom(degree, k) * a ** (degree - k) * v0 ** k * integrand(k)

    return total_integral / t[-1]


if __name__ == '__main__':

    # ---------------------------- test pd curve moments ----------------------------
    # Define times, positions and velocities
    t = np.array([0, 1, 2, 5])
    x = np.array([[0, 0], [-1, -1], [-2, 0], [1, 3]])
    v = np.array([[-1, -1], [-1, 1], [1, 1]])

    # Define the power n for the statistical moment
    n = 1

    # Compute the integral
    mean = central_moment_from_skeleton(t, x, v, n)
    variance = central_moment_from_skeleton(t, x, v, 2)
    print("Mean along the piecewise linear curve:", mean)
    print("Std along the piecewise linear curve:", np.sqrt(variance))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(*x.T, 'o-')
    ax.scatter(*mean, c='r')
    ax.axis('equal')
    plt.show()


    # ---------------------------- test visualization ----------------------------
    # normal 2d
    rng = np.random.default_rng(0)
    mean, cov = np.array([0, 0]), np.array([[1, 0.3], [0.3, 1.]])
    posterior = MultivariateNormal(mean, cov, rng=rng)
    plot_limits = ([-3, 3], [-3, 3])

    fig, ax = get_2d_despined_figure(plot_limits, figsize=(5, 3.5))
    plot_pdf_contours(posterior, ax, plot_limits)

    n_samples = 5000

    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        samples[i] = posterior.getSample()

    plot_samples(samples, ax, color_code=True)

    plt.show()



