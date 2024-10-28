import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import binom
from typing import Union, List

from src.distributions import Distribution, MultivariateNormal


def get_2d_despined_figure(plot_limits: tuple[list[float], list[float]] = None,
                           nrows: int = 1,
                           ncols: int = 1,
                           figsize: tuple[float, float] = (3., 4.),
                           constrained_layout: bool = True,
                           keep_ticks: bool = False,
                           axes_label: Union[tuple[str, ...], str] = '\\theta',
                           equal_axes = True)\
        -> tuple[plt.Figure, plt.Axes]:
    """
    Create a 2D despined figure with specified plot limits and formatting.

    Parameters:
    plot_limits (tuple): A tuple containing two lists, each specifying the x and y axis limits respectively.
    nrows (int, optional): Number of rows in the subplot grid. Default is 1.
    ncols (int, optional): Number of columns in the subplot grid. Default is 1.
    figsize (tuple, optional): Size of the figure in inches. Default is (3., 4.).
    constrained_layout (bool, optional): Whether to use constrained layout for the figure. Default is True.
    keep_ticks (bool, optional): Whether to keep the ticks. Mainly for debugging purpose. Default is True.

    Returns:
    tuple: A tuple containing the figure and axes objects.
    """

    # create figure
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=constrained_layout)

    # format the plot
    if plot_limits is not None:
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])

    if equal_axes:
        ax.axis('equal')
        ax.autoscale(enable=False)

    ax.grid(False)

    # set labels
    if isinstance(axes_label, tuple):
        ax.set_xlabel(rf'${axes_label[0]}$')
        ax.set_ylabel(rf'${axes_label[1]}$')
    else:
        ax.set_xlabel(rf'${axes_label}_1$')
        ax.set_ylabel(rf'${axes_label}_2$')

    # despine the plot
    sns.despine()

    # get rid of the ticks and tick labels
    if not keep_ticks:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    return fig, ax


def plot_pdf_contours(
        distribution: Distribution,
        ax: plt.Axes,
        plot_limits: tuple[list[float], list[float]],
        n_grid: int = 100,
        alpha: float = 0.6,
        n_levels: int = 20,
        cmap: matplotlib.colors.ListedColormap = sns.color_palette('rocket', as_cmap=True)
) -> plt.Axes:
    """
    Plot the probability density function (PDF) contours of a multivariate normal distribution.

    Parameters:
    distribution (MultivariateNormal): The multivariate normal distribution to plot.
    ax (plt.Axes): The matplotlib axes object to plot on.
    plot_limits (tuple): A tuple containing two lists, each specifying the x and y axis limits respectively.
    n_grid (int, optional): Number of grid points for the x and y axes. Default is 100.
    alpha (float, optional): Transparency level of the contour plot. Default is 0.6.
    n_levels (int, optional): Number of contour levels to plot. Default is 20.
    cmap (sns.palettes._ColorPalette, optional): Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
    plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = np.exp(distribution.log_density(np.array([X[i, j], Y[i, j]])))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax


def plot_pfd_contour_slice(
        distribution: Distribution,
        ax: plt.Axes,
        plot_limits: tuple[list[float], list[float]],
        slice_loc: np.ndarray,
        idcs_plane: tuple[int, int] = (0, 1),
        n_grid: int = 100,
        alpha: float = 0.6,
        n_levels: int = 20,
        cmap: matplotlib.colors.ListedColormap = sns.color_palette('rocket', as_cmap=True)
) -> plt.Axes:
    """
    Plot the probability density function (PDF) contours of a multivariate normal distribution.

    Parameters:
    distribution (MultivariateNormal): The multivariate normal distribution to plot.
    ax (plt.Axes): The matplotlib axes object to plot on.
    plot_limits (tuple): A tuple containing two lists, each specifying the x and y axis limits respectively.
    n_grid (int, optional): Number of grid points for the x and y axes. Default is 100.
    alpha (float, optional): Transparency level of the contour plot. Default is 0.6.
    n_levels (int, optional): Number of contour levels to plot. Default is 20.
    cmap (sns.palettes._ColorPalette, optional): Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
    plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            point = slice_loc.copy()
            point[idcs_plane[0]] = X[i, j]
            point[idcs_plane[1]] = Y[i, j]
            Z[i, j] = np.exp(distribution.log_density(point))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax

def plot_samples(samples: np.ndarray,
                 ax: plt.Axes,
                 color_code: bool = True,
                 n_vis: int = 500,
                 size = 3) -> plt.Axes:
    """
    Plot samples on a given matplotlib axes object.

    Parameters:
    samples (np.ndarray): The samples to plot, expected to be a 2D array.
    ax (plt.Axes): The matplotlib axes object to plot on.
    color_code (bool, optional): Whether to color code the samples. Default is True.
    n_vis (int, optional): Number of samples to visualize. Default is 500.

    Returns:
    plt.Axes: The matplotlib axes object with the samples plotted.
    """

    n_vis = np.min([n_vis, samples.shape[0]])
    samples_plot = samples[0::samples.shape[0] // n_vis]

    if color_code:
        ax.scatter(*samples_plot.transpose(), s=size, zorder=2, c=np.linspace(0, 1, samples_plot.shape[0]))
    else:
        ax.scatter(*samples_plot.transpose(), s=size, zorder=2, c='C0')

    return ax


def central_moment_from_skeleton(t: np.ndarray, x: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute the central moment of a piecewise linear curve defined by its skeleton.

    Parameters:
    t (np.ndarray): 1D array of time points.
    x (np.ndarray): 2D array of positions corresponding to the time points.
    v (np.ndarray): 2D array of velocities corresponding to the segments between time points.
    degree (int): The degree of the moment to compute.

    Returns:
    np.ndarray: The computed central moment of the specified degree.
    """

    # Compute the mean of the curve
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

        def integrand(k):
            return (t1 - t0)**(k + 1) / (k + 1)

        a = x0 - mean
        for k in range(degree + 1):
            # binomial_coeff = binom(degree, k)
            # integral = np.zeros_like(mean)
            total_integral += binom(degree, k) * a ** (degree - k) * v0 ** k * integrand(k)

    return total_integral / t[-1]


def grad_fd(f: callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute the gradient of a function using finite differences.

    Parameters:
    f (callable): The function for which the gradient is to be computed. It should take a numpy array as input and return a scalar.
    x (np.ndarray): The point at which the gradient is to be computed. It should be a 1D numpy array.
    h (float, optional): The step size for the finite difference approximation. Default is 1e-5.

    Returns:
    np.ndarray: The gradient of the function at the point x. It will be a 1D numpy array of the same length as x.
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = (f(x + h * np.eye(n)[i]) - f(x - h * np.eye(n)[i])) / (2 * h)
    return grad


def hessian_fd(f: callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute the Hessian matrix of a function using finite differences.

    Parameters:
    f (callable): The function for which the Hessian is to be computed. It should take a numpy array as input and return a scalar.
    x (np.ndarray): The point at which the Hessian is to be computed. It should be a 1D numpy array.
    h (float, optional): The step size for the finite difference approximation. Default is 1e-5.

    Returns:
    np.ndarray: The Hessian matrix of the function at the point x. It will be a 2D numpy array of shape (n, n) where n is the length of x.
    """
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hess[i, j] = (f(x + h * np.eye(n)[i] + h * np.eye(n)[j]) -
                          f(x - h * np.eye(n)[i] + h * np.eye(n)[j]) -
                          f(x + h * np.eye(n)[i] - h * np.eye(n)[j]) +
                          f(x - h * np.eye(n)[i] - h * np.eye(n)[j])) / (4 * h**2)
    return hess


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
        samples[i] = posterior.get_sample()

    plot_samples(samples, ax, color_code=True)

    plt.show()



