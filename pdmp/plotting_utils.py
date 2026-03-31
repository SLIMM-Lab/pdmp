import numpy as np
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
    """Create a despined figure with specified plot limits and formatting.

    Args:
        plot_limits: A tuple containing two lists, each specifying the x and y axis limits respectively.
            Only applied when nrows == ncols == 1.
        nrows: Number of rows in the subplot grid. Default is 1.
        ncols: Number of columns in the subplot grid. Default is 1.
        figsize: Size of the figure in inches. Default is (3., 4.).
        constrained_layout: Whether to use constrained layout for the figure. Default is True.
        keep_ticks: Whether to keep the ticks. Mainly for debugging purpose. Default is True.
        axes_label: Axes label. Only applied when nrows == ncols == 1. Default is '\\theta'.
        equal_axes: Whether to set equal axes. Only applied when nrows == ncols == 1. Default is True.

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """

    # create figure
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=figsize,
                             constrained_layout=constrained_layout)

    # single-axis-only operations: limits, equal axes, labels
    if nrows == 1 and ncols == 1:
        if plot_limits is not None:
            axes.set_xlim(plot_limits[0])
            axes.set_ylim(plot_limits[1])

        if equal_axes:
            axes.axis('equal')
            axes.autoscale(enable=False)

        if isinstance(axes_label, tuple):
            axes.set_xlabel(rf'${axes_label[0]}$')
            axes.set_ylabel(rf'${axes_label[1]}$')
        else:
            axes.set_xlabel(rf'${axes_label}_1$')
            axes.set_ylabel(rf'${axes_label}_2$')

    # per-axis style applied to all subplots
    for ax in np.array(axes).ravel():
        ax.grid(False)

        if not keep_ticks:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.)

    sns.despine()

    return fig, axes
