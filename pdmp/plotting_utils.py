import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union


def get_2d_despined_figure(plot_limits: tuple[list[float], list[float]] = None,
                           nrows: int = 1,
                           ncols: int = 1,
                           figsize: tuple[float, float] = (3., 4.),
                           constrained_layout: bool = True,
                           keep_ticks: bool = False,
                           axes_label: Union[tuple[str, ...], str] = '\\theta',
                           equal_axes=True) -> tuple[plt.Figure, plt.Axes]:
    """Create a 2D despined figure with specified plot limits and formatting.

    Args:
        plot_limits: A tuple containing two lists, each specifying the x and y axis limits respectively.
        nrows: Number of rows in the subplot grid. Default is 1.
        ncols: Number of columns in the subplot grid. Default is 1.
        figsize: Size of the figure in inches. Default is (3., 4.).
        constrained_layout: Whether to use constrained layout for the figure. Default is True.
        keep_ticks: Whether to keep the ticks. Mainly for debugging purpose. Default is True.
        axes_label: Axes label. Default is '\\theta'.

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """

    # create figure
    fig, ax = plt.subplots(nrows,
                           ncols,
                           figsize=figsize,
                           constrained_layout=constrained_layout)

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
