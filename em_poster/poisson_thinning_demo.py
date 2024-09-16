from typing import Callable

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

from numpy import ndarray, dtype, floating, float_

sns.set_style('white')

lambda_max_homogeneous = 1.5  # Upper bound for the homogeneous Poisson process
# Define the rate function for the non-homogeneous Poisson process
def rate_function(t):
    return 0.6 * np.sin(t) + 0.7  # Adding 1.5 to ensure the rate is always positive

# Define the tighter bound for the homogeneous Poisson process
def lambda_max_function(t):
    return 0.4 * np.sin(t) + 1.1

def lambda_const_function(t):
    return lambda_max_homogeneous * np.ones_like(t)

def plot_rates_and_events(times: ndarray,
                          rates: ndarray,
                          f_lambda_max: Callable[[ndarray], ndarray],
                          events: ndarray,
                          events_accepted: ndarray,
                          upper_bound: str,
                          plot_path: str,
                          legend: bool = True,
                          save_fig: bool = False) -> None:

    # Plot the rate functions and the events
    fig, ax = plt.subplots(figsize=(2.7, 2.), constrained_layout=True)
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'Rate $\lambda(t)$')
    ax.grid(False)
    ax.set_xlim([0, T])
    ax.set_ylim([0, 1.55])

    # despine the plot
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    # Generate time values for plotting the rate functions
    times = np.linspace(0, T, 1000)

    ax.plot(times, f_lambda_max(times), label=r'Upper Bound $\tilde{\lambda}(t)$', c='C1', linestyle='--')
    ax.fill_between(times, np.zeros_like(rates), f_lambda_max(times), color='C1', alpha=0.15)
    if legend:
        ax.legend(**legend_params)
    if save_fig:
        fig.savefig(join(plot_path, f'poisson_thinning_{upper_bound}_upper_bound.pdf'))

    vlines = ax.vlines(events, ymin=0, ymax=f_lambda_max(events), color='C1', alpha=0.7,
                       label='Proposed Events', linewidth=1.5)
    if legend:
        ax.legend(**legend_params)
    if save_fig:
        fig.savefig(join(plot_path, f'poisson_thinning_{upper_bound}_upper_bound_events.pdf'))
    vlines.remove()

    ax.plot(times, rate_function(times), label=r'True Rate $\lambda(t)$', c='C0', linestyle='--')
    vlines = ax.vlines(events, ymin=0, ymax=f_lambda_max(events), color='C1', alpha=0.7,
                       label='Proposed Events', linewidth=1.5)
    ax.fill_between(times, np.zeros_like(rates), rates, color='C0', alpha=0.15)
    if legend:
        ax.legend(**legend_params)
    if save_fig:
        fig.savefig(join(plot_path, f'poisson_thinning_{upper_bound}_rate_pre.pdf'))

    # find elements in events that are not in events_accepted
    events_rejected = np.setdiff1d(events, events_accepted)

    vlines.remove()
    ax.vlines(events_rejected, ymin=0, ymax=f_lambda_max(events_rejected), color='C1',
              alpha=0.7, label='Proposed Events', linewidth=1.5)
    ax.vlines(events_accepted, ymin=0, ymax=rate_function(events_accepted), color='C0', alpha=0.7,
              label='Accepted Events', linewidth=1.5)
    if legend:
        ax.legend(**legend_params)
    if save_fig:
        fig.savefig(join(fig_path, f'poisson_thinning_{upper_bound}_rate_post.pdf'))

    plt.show()

# Parameters
T = 10  # Total time

# Generate homogeneous Poisson process events with a constant rate
np.random.seed(1)  # For reproducibility

setting = 0

if setting == 0:
    upper_bound = 'const'
else:
    upper_bound = 'sine'

legend_params = {'loc': 'lower right'}

fig_path = '/home/leon/ownCloud/Documents/presentations/em/symposium_25/figures'
save_fig = False

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Generate homogeneous Poisson process events with a constant rate
    events_hom = []
    time = 0
    while time < T:
        time += np.random.exponential(1 / lambda_max_homogeneous)
        if time < T:
            events_hom.append(time)

    events_hom = np.array(events_hom)

    # Apply thinning algorithm to generate events from the tighter bound process
    events_sine = []
    events_not_sine = []
    for t in events_hom:
        if np.random.uniform(0, 1) < lambda_max_function(t) / lambda_const_function(t):
            events_sine.append(t)
        else:
            events_not_sine.append(t)

    events_sine = np.array(events_sine)
    events_not_sine = np.array(events_not_sine)

    # Apply thinning algorithm again to generate non-homogeneous Poisson process events
    events_non_hom = []
    events_not_non_hom = []
    for t in events_sine:
        if np.random.uniform(0, 1) < rate_function(t) / lambda_max_function(t):
            events_non_hom.append(t)
        else:
            events_not_non_hom.append(t)

    # not_non_homogenous_events = [x for x in events_sine if x not in events_non_hom]

    events_non_hom = np.array(events_non_hom)
    events_not_non_hom = np.array(events_not_non_hom)

    events_not_non_not_sine = np.hstack((events_not_non_hom, events_not_sine))

    # Generate time values for plotting the rate functions
    times = np.linspace(0, T, 1000)
    rates = rate_function(times)
    lambda_max_values = lambda_max_function(times)
    lambda_const_values = np.ones_like(times) * lambda_const_function(times)

    plot_rates_and_events(times, rates, lambda_const_function, events_hom, events_non_hom, 'const',
                          fig_path, save_fig=save_fig, legend=False)
    plot_rates_and_events(times, rates, lambda_max_function, events_sine, events_non_hom, 'sine',
                          fig_path, save_fig=save_fig, legend=False)
