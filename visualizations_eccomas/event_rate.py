import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os

sns.set_style('white')

# parameter for the rate functions
a = 0.1

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

save_fig = True
fig_path = './figures'


if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    x = np.linspace(-3, 3, 200)

    p_x = sp.stats.norm.pdf(x)
    n_log_pdf = - sp.stats.norm.logpdf(x)

    for i, setting in enumerate(settings):

        # read settings
        v = setting['v']
        x_0 = setting['x_0']
        x_T = setting['x_T']
        y_arr = setting['y_arr']
        dx_arr = setting['dx_arr']

        # compute the rate function
        x_to_T = np.linspace(x_0, x_T, 200)
        rate_to_T = np.maximum(0, - v * grad_log_pdf(x_to_T, 0, 1))
        rate = np.maximum(0, - v * grad_log_pdf(x, 0, 1))

        # create and format the plot
        fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 20)
        ax.grid(False)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$p(\theta)$')
        sns.despine()

        # get rid of the ticks labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # make the axes linewidths bigger
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.)

        # plot the pdf and rate function, and the starting point
        ax.plot(x, 15 * p_x, label=r'$p(\theta)$', c='C0')
        ax.plot(x, rate, label=r'$\lambda(\theta_t, v_t)$)', c='C1')
        ax.legend(loc='upper center')
        ax.scatter(x_0, 0, c='C2', s=50, zorder=10, marker='x')

        # plot an arrow at x_0 to x_0 + t * v
        ax.arrow(x_0, y_arr, 0.5 * v, 0, head_width=0.5, head_length=0.1, fc='C2', ec='C2')
        ax.text(x_0 + dx_arr * v, y_arr + 0.6, rf'$v_{i}$', fontsize=12, color='C2', ha='center')
        ax.set_xticks([x_0])
        ax.set_xticklabels([rf'$\theta_{i}$'])
        if save_fig:
            fig.savefig(os.path.join(fig_path, f'pdf_rate_{i}.pdf'))

        # plot event and fill area between 0 and rate function till this point
        ax.fill_between(x_to_T, 0, rate_to_T, color='C1', alpha=0.3, label=r'$\int_0^t \lambda(\theta_s, v_s)ds$')
        ax.legend(loc='upper center')
        ax.scatter(x_0, 0, c='C2', s=50, zorder=10, marker='x')
        ax.scatter(x_T, -v * grad_log_pdf(x_T, 0, 1), c='C3', s=50, zorder=10, marker='x')
        ax.set_xticks([x_0, x_T])
        ax.set_xticklabels([rf'$\theta_{i}$', rf'$\theta_{i + 1}$'])
        if save_fig:
            fig.savefig(os.path.join(fig_path, f'pdf_rate_event_{i}.pdf'))

        plt.show()
