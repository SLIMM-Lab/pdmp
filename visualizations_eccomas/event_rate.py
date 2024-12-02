import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os

import matplotlib.patches as patches
import matplotlib.lines as lines

sns.set_style('white')

# parameter for the rate functions
a = 0.04
b = 8.
c = 1.5

# pdf and rate functions
def normal_pdf(x, mu, sigma):
    # return sp.stats.multivariate_normal.p_x(x, mean, cov)
    return np.exp(-(a * (x - mu) ** 4 + x ** 2) / sigma ** 2)


def log_normal_pdf(x, mu, cov):
    # return sp.stats.multivariate_normal.logpdf(x, mean, cov)
    return -(a * (x - mu) ** 4 + x ** 2) / cov ** 2


def grad_log_pdf(x, mu, sigma):
    # return -1/(sigma**2) * (x - mean)
    return -(3 * a * (x - mu) ** 3 + 2 * x) / sigma ** 2

def tight_upper_bound(x, mu, sigma):
    return -(b * a * (x - mu) ** 2 + 2 * x) / sigma ** 2 - 0.5

def const_upper_bound(x):
    return 10.879999999999999* np.ones_like(x)

def set_legend_pdf_rate(ax: plt.Axes):
    # handle for the pdf
    pdf = lines.Line2D([], [], color='C0', label=r'$p(\theta | \mathcal{D})$')
    # handle for the rate function
    rate = lines.Line2D([], [], color='C1', label=r'$\lambda(\theta_t, v_t)$')
    # handle for the integral
    integral = patches.Patch(color='C1', alpha=0.3, label=r'$\int_0^t \lambda(\theta_s, v_s)ds$')

    # add the legend
    ax.legend(handles=[pdf, rate, integral], loc='upper center')

def set_legend_thinning(ax: plt.Axes):
    # handle for the rate function
    rate = lines.Line2D([], [], color='C1', label=r'True rate       $\lambda(\theta_t, v_t)$')
    # handle for the upper bound
    bound = lines.Line2D([], [], color='C2', alpha=0.7, label=r'Upper bound $\tilde{\lambda}(\theta_t, v_t)$')

    # add the legend
    ax.legend(handles=[rate, bound], loc='upper center', ncol=1)


# define settings
settings = [{'v': 1,
             'x_0': -1.,
             'x_T': 2.2,
             'y_arr': 1,
             'dx_arr': 0.3},
            {'v': -1,
             'x_0': 2.2,
             'x_T': -1.5,
             'y_arr': 2.2,
             'dx_arr': 0.2}]

save_fig = False
# fig_path = './figures'
fig_path = '/home/leon/ownCloud/Documents/presentations/conferences/24-ducoms/latex/includes/poisson_thinning'

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    x = np.linspace(-3, 3, 200)

    p_x = sp.stats.norm.pdf(x)
    n_log_pdf = - sp.stats.norm.logpdf(x)


    # read settings

    setting = settings[0]
    v = setting['v']
    x_0 = setting['x_0']
    x_T = setting['x_T']
    y_arr = setting['y_arr']
    dx_arr = setting['dx_arr']

    # compute the rate function
    x_to_T = np.linspace(x_0, x_T, 200)
    rate_to_T = np.maximum(0, - v * grad_log_pdf(x_to_T, 0, 1))
    rate = np.maximum(0, - v * grad_log_pdf(x, 0, 1))
    u_b_t = np.maximum(0, - v * tight_upper_bound(x, 0, 1)) + c
    print(u_b_t[-1])
    u_b_c = const_upper_bound(x)
    f_rate = lambda x: np.maximum(0, - v * grad_log_pdf(x, 0, 1))
    f_ub = lambda x: np.maximum(0, - v * tight_upper_bound(x, 0, 1)) + c

    # create and format the plot
    fig, ax = plt.subplots(figsize=(4., 3), constrained_layout=True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 20)
    ax.autoscale(enable=False)
    ax.grid(False)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\lambda$')
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    ax.scatter(x_0, 0, c='C3', s=50, zorder=10, marker='x')
    ax.set_xticks([x_0])
    ax.set_xticklabels([rf'$\theta_{0}$'])

    ax.plot(x, rate, label=r'$\lambda(\theta_t, v_t)$)', c='C1', zorder=4, linewidth=1.5)
    ax.fill_between(x, 0, rate, color='C1', alpha=0.25, zorder=1)
    set_legend_thinning(ax)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_true_rate.pdf'))

    # ax.plot(x, u_b_t, label=r'$\lambda(\theta_t, v_t)$)', c='C0')
    ax.plot(x, u_b_c, label=r'$\lambda(\theta_t, v_t)$)', c='C2', zorder=4, linewidth=1.5)
    ax.fill_between(x, 0, u_b_c, color='C2', alpha=0.2, zorder=0)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_const_rate.pdf'))

    e_const = []

    s = 3.
    x_now = x_0
    x_max = x[-1]
    rate_c = u_b_c[0]
    rng = np.random.default_rng(1)

    while x_now < x_max:
        x_now += rng.exponential(s / rate_c)
        if x_now < x_max:
            e_const.append(x_now)
    e_const = np.array(e_const)

    vlines = ax.vlines(e_const, ymin=0, ymax=const_upper_bound(e_const), color='C2', alpha=0.7, label='Proposed Events',
                       linewidth=2.0, zorder=2)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_const_proposed.pdf'))

    vlines.remove()

    e_c_accept = []
    e_c_reject = []

    for e in e_const:
        if rng.uniform() < f_rate(e) / rate_c:
            e_c_accept.append(e)
        else:
            e_c_reject.append(e)

    e_c_accept = np.array(e_c_accept)
    e_c_reject = np.array(e_c_reject)

    vlines_a = ax.vlines(e_c_accept, ymin=0, ymax=f_rate(e_c_accept), color='C1', alpha=1.0, label='Accepted Events',
                         linewidth=2.0, zorder=3)
    vlines_r = ax.vlines(e_c_reject, ymin=0, ymax=const_upper_bound(e_c_reject), color='C2', alpha=0.3, label='Rejected Events',
                         linewidth=1., zorder=2)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_const_accepted.pdf'))

    vlines_a.remove()

    x_T = e_c_accept[0]
    ax.vlines(x_T, ymin=0, ymax=f_rate(x_T), color='C1', alpha=1.0, label='Accepted Events', linewidth=2.0, zorder=3)
    ax.vlines(e_c_accept[1:], ymin=0, ymax=f_rate(e_c_accept[1:]), color='C1', alpha=0.4, label='Accepted Events',
              linewidth=1., zorder=3)
    ax.scatter(x_T, 0, c='C3', s=50, zorder=10, marker='x')
    ax.set_xticks([x_0, x_T])
    ax.set_xticklabels([rf'$\theta_{0}$', rf'$\theta_{1}$'])

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_const_accepted_first.pdf'))

    plt.show()


    # ------------------------------------------- non-const rate -------------------------------------------
    # create and format the plot
    fig, ax = plt.subplots(figsize=(4., 3), constrained_layout=True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 20)
    ax.autoscale(enable=False)
    ax.grid(False)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\lambda$')
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    ax.scatter(x_0, 0, c='C3', s=50, zorder=10, marker='x')
    ax.set_xticks([x_0])
    ax.set_xticklabels([rf'$\theta_{0}$'])

    ax.plot(x, rate, label=r'$\lambda(\theta_t, v_t)$)', c='C1', zorder=4, linewidth=1.5)
    ax.fill_between(x, 0, rate, color='C1', alpha=0.25, zorder=1)

    ax.plot(x, u_b_t, label=r'$\lambda(\theta_t, v_t)$)', c='C2')
    ax.fill_between(x, 0, u_b_t, color='C2', alpha=0.2, zorder=0)
    set_legend_thinning(ax)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_tight_rate.pdf'))

    plt.show()

    e_const = []

    s = 3.
    x_now = x_0
    x_max = x[-1]
    rate_c = u_b_c[0]

    while x_now < x_max:
        x_now += rng.exponential(s / rate_c)
        if x_now < x_max:
            e_const.append(x_now)
    e_const = np.array(e_const)

    rng = np.random.default_rng(1)
    e_tight = []

    for e in e_const:
        if rng.uniform() < f_ub(e) / rate_c:
            e_tight.append(e)

    e_tight = np.array(e_tight)

    vlines = ax.vlines(e_tight, ymin=0, ymax=f_ub(e_tight), color='C2', alpha=0.7, label='Proposed Events',
                       linewidth=2.0, zorder=2)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_tight_proposed.pdf'))

    vlines.remove()

    e_t_accept = []
    e_t_reject = []

    for e in e_tight:
        if rng.uniform() < f_rate(e) / f_ub(e):
            e_t_accept.append(e)
        else:
            e_t_reject.append(e)

    e_t_accept = np.array(e_t_accept)
    e_t_reject = np.array(e_t_reject)

    vlines_a = ax.vlines(e_t_accept, ymin=0, ymax=f_rate(e_t_accept), color='C1', alpha=1.0, label='Accepted Events',
                         linewidth=2.0, zorder=3)
    vlines_r = ax.vlines(e_t_reject, ymin=0, ymax=f_ub(e_t_reject), color='C2', alpha=0.3, label='Rejected Events',
                         linewidth=1., zorder=2)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_tight_accepted.pdf'))

    vlines_a.remove()

    x_T = e_t_accept[0]
    ax.vlines(x_T, ymin=0, ymax=f_rate(x_T), color='C1', alpha=1.0, label='Accepted Events', linewidth=2.0, zorder=3)
    ax.vlines(e_t_accept[1:], ymin=0, ymax=f_rate(e_t_accept[1:]), color='C1', alpha=0.4, label='Accepted Events',
              linewidth=1., zorder=3)
    ax.scatter(x_T, 0, c='C3', s=50, zorder=10, marker='x')
    ax.set_xticks([x_0, x_T])
    ax.set_xticklabels([rf'$\theta_{0}$', rf'$\theta_{1}$'])

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'thinning_tight_accepted_first.pdf'))

    plt.show()
