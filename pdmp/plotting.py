from typing import Union, cast

import matplotlib.colors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pdmp.distributions import Distribution, Transformation, AffineTransformtion
from pdmp.surrogates import SurrogateModel


def plot_samples(samples: np.ndarray,
                 ax: plt.Axes,
                 color_code: bool = True,
                 n_vis: int = 500,
                 size=3,
                 **kwargs) -> plt.Axes:
    """Plot samples on a given matplotlib axes object.

    Args:
        samples: The samples to plot, expected to be a 2D array.
        ax: The matplotlib axes object to plot on.
        color_code: Whether to color code the samples. Default is True.
        n_vis: Number of samples to visualize. Default is 500.

    Returns:
        plt.Axes: The matplotlib axes object with the samples plotted.
    """

    n_vis = np.min([n_vis, samples.shape[0]])
    samples_plot = samples[0::samples.shape[0] // n_vis]

    if color_code:
        ax.scatter(*samples_plot.transpose(),
                   s=size,
                   zorder=2,
                   c=np.linspace(0, 1, samples_plot.shape[0]))
    else:
        ax.scatter(*samples_plot.transpose(), s=size, zorder=2, **kwargs)

    return ax


def plot_trace(samples: np.ndarray,
               components: list[int] = None,
               keep_ticks: bool = True,
               axis_label='\\theta') -> tuple[plt.Figure, plt.Axes]:
    """Plot the trace of samples

    Args:
        samples: The samples to plot, expected to be a 2D array.
        components: The indices of the components to plot. Default is None.
        keep_ticks: Whether to keep the ticks on the axes. Default is True.
        axis_label: The label for the y-axis. Default is '\\theta'.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """

    def plot_one_trace(ax: plt.Axes, samples: np.ndarray, component: int,
                       axis_label: str):
        """Plot one trace of the samples.

        Args:
            ax: The matplotlib axes object to plot on.
            samples: The samples to plot, expected to be a 2D array.
            component: The index of the component to plot.
            axis_label: The label for the y-axis.
        """

        ax.grid(False)

        # get rid of the ticks and tick labels
        if not keep_ticks:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_ylabel(rf'${axis_label}_{component}$')

        # make the axes linewidths bigger
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.)

        ax.plot(samples[:, component])
        ax.set_ylabel(rf'${axis_label}_{component}$')

    n_params = samples.shape[1]
    if components is None:
        components = list(range(n_params))

    # get figure
    fig, ax = plt.subplots(len(components),
                           1,
                           figsize=(4, n_params * 2.5),
                           constrained_layout=True,
                           sharex=True)

    # despine the plot
    sns.despine()

    if len(components) == 1:
        plot_one_trace(ax, samples, components[0], axis_label)
        ax.set_xlabel(f'Sample index')
    else:
        for i, comp in enumerate(components):
            plot_one_trace(ax[i], samples, comp, axis_label)
        ax[-1].set_xlabel(f'Sample index')

    plt.show()

    return fig, ax


def plot_pdf_grad_contours(target: Union[Distribution, SurrogateModel],
                           ax: plt.Axes,
                           plot_limits: tuple[list[float], list[float]],
                           idx: int = 0,
                           n_grid: int = 100,
                           alpha: float = 0.6,
                           levels: Union[int, np.ndarray] = None,
                           cmap: matplotlib.colors.Colormap = sns.color_palette(
                               'rocket', as_cmap=True),
                           transformation: Transformation = None) -> plt.Axes:
    """Plot the probability density function (PDF) contours of a distribution.

    Args:
        taget: The target distribution to plot.
        ax: The matplotlib axes object to plot on.
        plot_limits: A tuple containing two lists, each specifying the x and y axis limits respectively.
        n_grid: Number of grid points for the x and y axes. Default is 100.
        alpha: Transparency level of the contour plot. Default is 0.6.
        n_levels: Number of contour levels to plot. Default is 20.
        cmap: Colormap to use for the contour plot. Default is 'rocket' colormap.
        transformation: The transformation to apply to the target distribution. Default is None.

    Returns:
        plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    # check if levels are given or already defined
    if levels is None:
        for child in ax.get_children():
            if isinstance(child, matplotlib.contour.QuadContourSet):
                levels = child.levels

    if levels is None:
        levels = 20

    if transformation is None:
        transformation = AffineTransformtion(M=np.eye(2), b=np.zeros(2))

    if isinstance(target, SurrogateModel):
        f_eval = lambda x: target.grad(x, idx=idx)
    else:
        f_eval = lambda x: target.grad_log_density(x)[idx]

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):

            x_i = np.array([X[i, j], Y[i, j]])
            xi_i = transformation.inverse_transform(x_i)

            Z[i,
              j] = np.exp(f_eval(xi_i) - transformation.log_det_jacobian(x_i))

    ax.contour(X, Y, Z, levels=levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax


def plot_pdf_contours(target: Union[Distribution, SurrogateModel],
                      ax: plt.Axes,
                      plot_limits: tuple[list[float], list[float]] = None,
                      n_grid: int = 100,
                      alpha: float = 0.6,
                      levels: Union[int, np.ndarray] = None,
                      log=False,
                      cmap: matplotlib.colors.Colormap = sns.color_palette(
                          'rocket', as_cmap=True),
                      transformation: Transformation = None) -> plt.Axes:
    """Plot the probability density function (PDF) contours of a distribution.

    Args:
        target: The target distribution to plot.
        ax: The matplotlib axes object to plot on.
        plot_limits: A tuple containing two lists, each specifying the x and y axis limits respectively.
        n_grid: Number of grid points for the x and y axes. Default is 100.
        alpha: Transparency level of the contour plot. Default is 0.6.
        levels: Number of contour levels to plot. Default is 20.
        cmap: Colormap to use for the contour plot. Default is 'rocket' colormap.
        transformation: The transformation to apply to the target distribution. Default is None.

    Returns:
        plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    # check if levels are given or already defined
    if levels is None:
        for child in ax.get_children():
            if isinstance(child, matplotlib.contour.QuadContourSet):
                levels = child.levels

    if levels is None:
        levels = 20

    # set identity transformation if none is given
    if transformation is None:
        transformation = AffineTransformtion(M=np.eye(2), b=np.zeros(2))

    if isinstance(target, SurrogateModel):
        f_eval = lambda x: target.eval(x, delta=True)
    else:
        f_eval = target.log_density

    if plot_limits is None:
        bounds = ax.dataLim.bounds
        # Check if any bounds are infinite and use viewLim as fallback
        if any(np.isinf(bounds)):
            bounds = ax._viewLim.bounds
        plot_limits = ([bounds[0], bounds[0] + bounds[2]],
                       [bounds[1], bounds[1] + bounds[3]])

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):

            x_i = np.array([X[i, j], Y[i, j]])
            xi_i = transformation.inverse_transform(x_i)

            if log:
                Z[i, j] = f_eval(xi_i) - transformation.log_det_jacobian(x_i)
            else:
                Z[i, j] = np.exp(
                    f_eval(xi_i) - transformation.log_det_jacobian(x_i))

    ax.contour(X, Y, Z, levels=levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax


def plot_pfd_contour_conditional(
    target: [Distribution, SurrogateModel],
    ax: plt.Axes,
    plot_limits: tuple[list[float], list[float]],
    slice: np.ndarray,
    idcs_plane: tuple[int, int] = (0, 1),
    n_grid: int = 100,
    alpha: float = 0.6,
    n_levels: int = 20,
    cmap: matplotlib.colors.ListedColormap = sns.color_palette('rocket',
                                                               as_cmap=True)
) -> plt.Axes:
    """Plot the conditional probability density function (PDF) contours of a distribution.

    Args:
        target: The target distribution to plot.
        ax: The matplotlib axes object to plot on.
        plot_limits: A tuple containing two lists, each specifying the x and y axis limits respectively.
        slice: The coordinates to condition on.
        idcs_plane: The indices of the plane to condition on. Default is (0, 1).
        n_grid: Number of grid points for the x and y axes. Default is 100.
        alpha: Transparency level of the contour plot. Default is 0.6.
        n_levels: Number of contour levels to plot. Default is 20.
        cmap: Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
        plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """
    if isinstance(target, SurrogateModel):
        f_eval = lambda x: target.eval(x, delta=True)
    else:
        f_eval = target.log_density

    x = np.linspace(*plot_limits[0], n_grid)
    y = np.linspace(*plot_limits[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            point = slice.copy()
            point[idcs_plane[0]] = X[i, j]
            point[idcs_plane[1]] = Y[i, j]
            Z[i, j] = np.exp(f_eval(point))

    ax.contour(X, Y, Z, levels=n_levels, zorder=1, alpha=alpha, cmap=cmap)

    return ax


def plot_pfd_contour_marginal(samples: np.ndarray,
                              ax: plt.Axes,
                              idcs: tuple[int, int] = (0, 1),
                              alpha: float = 0.6,
                              n_levels: int = 15,
                              cmap: matplotlib.colors.ListedColormap = sns.
                              color_palette('rocket', as_cmap=True),
                              **kde_kwargs) -> plt.Axes:
    """Plot the probability density function (PDF) contours of a multivariate normal distribution.

    Args:
        samples: The samples to plot, expected to be a 2D array.
        ax: The matplotlib axes object to plot on.
        idcs: The indices of the dimensions to plot. Default is (0, 1).
        alpha: Transparency level of the contour plot. Default is 0.6.
        n_levels: Number of contour levels to plot. Default is 20.
        cmap: Colormap to use for the contour plot. Default is 'rocket' colormap.

    Returns:
        plt.Axes: The matplotlib axes object with the PDF contours plotted.
    """

    sns.kdeplot(x=samples[:, idcs[0]],
                y=samples[:, idcs[1]],
                ax=ax,
                cmap=cmap,
                levels=n_levels,
                alpha=alpha,
                zorder=1,
                **kde_kwargs)

    return ax
