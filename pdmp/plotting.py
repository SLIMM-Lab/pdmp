from typing import Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def get_2d_despined_figure(
        plot_limits: tuple[list[float], list[float]] = None,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (3., 4.),
        constrained_layout: bool = True,
        keep_ticks: bool = False,
        axes_label: Union[tuple[str, ...], str] = '\\theta',
        equal_axes = True
) -> tuple[plt.Figure, plt.Axes]:
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

def plot_samples(
        samples: np.ndarray,
        ax: plt.Axes,
        color_code: bool = True,
        n_vis: int = 500,
        size = 3,
        **kwargs
) -> plt.Axes:
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
        ax.scatter(*samples_plot.transpose(), s=size, zorder=2, **kwargs)

    return ax


def plot_trace(
        samples: np.ndarray,
        components: list[int] = None,
        keep_ticks: bool = True,
        axis_label = '\\theta'
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the trace of samples

    Parameters:
    samples (np.ndarray): The samples to plot, expected to be a 2D array.
    components (list[int], optional): The components to plot. Default is None.
    """

    def plot_one_trace(
            ax: plt.Axes,
            samples: np.ndarray,
            component: int,
            axis_label: str
    ):

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
    fig, ax = plt.subplots(len(components), 1, figsize=(4, n_params * 2.5), constrained_layout=True,
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
