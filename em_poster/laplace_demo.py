import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from pdmp.mcmc import MetropolisHastingsSampler
from pdmp.zigzag import ZigZagSampler
from pdmp.plotting import plot_samples, plot_pdf_contours
from pdmp.plotting_utils import get_2d_despined_figure

sns.set_style('white')
rng = np.random.default_rng(0)

save_fig = False
fig_path = '/home/leon/ownCloud/Documents/presentations/em/symposium_25/figures'

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # get the prior field
    kernel_params = {'sigma': 1., 'l': 0.3}
    n_b = 2
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_b, interval)
    prior_cov = compute_coefficients(squared_exponential_kernel, basis, interval, kernel_params=kernel_params)
    mean = 4.
    prior = MultivariateNormal(mean * np.ones(n_b), prior_cov, rng=rng)

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    # set up the forward model
    n_obs = 2
    F = [1.]
    F = np.array([item for item in F for i in range(n_obs)])
    n_params = n_b
    model = PiecewiseConstantModel(F, n_params, x_obs)

    # generate ground truth from a multi-variate normal distribution
    mean = 3 * np.ones(n_params)
    cov = np.eye(n_params)
    params_gt = rng.multivariate_normal(mean, cov)
    print(f"Ground truth: \n {params_gt}")
    margins = 1.5
    plot_limits = ([2.3, 4.1], [2.05, 4.15])

    # generate n_obs noisy obersevations of u_gt
    sigma_obs = 0.025

    # evaluate the ground truth at the observation and test locations
    u_gt = model.eval(params_gt, x_obs)
    u_gt_test = model.eval(params_gt, x_test)
    u_obs = np.zeros((len(F), u_gt.shape[0]))

    for i in range(len(F)):
        u_gt = model.eval(params_gt, x_obs, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    # define likelihood and posterior
    likelihood = GaussianLikelihood(model, u_obs, sigma_obs)
    target = Posterior(prior, likelihood)

    # ---------------------------------- laplace approx ----------------------------------------------
    # define negative log posterior and gradient for optimization
    log_post = lambda x: target.log_density(x)
    n_log_post = lambda x: - log_post(x)
    n_grad_log_post = lambda x: - target.grad_log_density(x)

    # find the minimum of the function with bfgs
    x0 = np.array([4., 4.])
    opt = minimize(n_log_post, x0, method='BFGS', jac=n_grad_log_post)
    x_map = opt.x
    print(opt)

    hess_an = target.hessian_log_density(x_map)
    print(f"Hess an: \n{hess_an}")

    cov_map = np.linalg.inv(- hess_an)
    laplace_approx = MultivariateNormal(x_map, cov_map, rng=rng)
    # get figure and plot pdfs
    scale = 1.2
    figsize = scale * np.abs(plot_limits[0][1] - plot_limits[0][0]), scale * np.abs(plot_limits[1][1] - plot_limits[1][0])
    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, axes_label='E')
    ax.plot(x_map[0], x_map[1], 'ro')
    plot_pdf_contours(target, ax, plot_limits, alpha=0.3)
    plot_pdf_contours(laplace_approx, ax, plot_limits, n_levels=10, alpha=1.0, cmap=sns.color_palette('mako', as_cmap=True))
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])

    if save_fig:
        fig.savefig(os.path.join(fig_path, 'laplace_approx_contours.pdf'))
    plt.show()

    # ---------------------------------- zig zag sampler ----------------------------------------------
    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, keep_ticks=False, axes_label='E')
    plot_pdf_contours(target, ax, plot_limits)
    if save_fig:
        fig.savefig(os.path.join(fig_path, 'target_contours.pdf'))

    n_events = 4000
    n_accepted = 200
    approx = {'mean': x_map, 'inv_cov': - hess_an}
    # x_0 = np.array([3.5, 2.5])
    x_0 = np.array([2.8, 3.5])
    x_0 = x_map
    sampler = ZigZagSampler(target, n_max=n_events, rng=rng, approximation=approx, x_0=x_0,
                            n_events_accepted=n_accepted, gamma=0.02)
    sampler.run()

    positions = sampler.positions_
    ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)

    # if save_fig:
    #     fig.savefig(os.path.join(fig_path, f'zig_zag_thinning_{n_accepted}.pdf'))

    plt.show()

    # ---------------------------------- zig zag int ----------------------------------------------

    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, keep_ticks=False, axes_label='E')
    plot_pdf_contours(target, ax, plot_limits)

    x_0 = np.array([2.8, 3.5])
    n_events = 200
    sampler = ZigZagSampler(target, n_max=n_events, rng=rng, x_0=x_0)
    sampler.run()

    positions = sampler.positions_
    ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    if save_fig:
        fig.savefig(os.path.join(fig_path, f'zig_zag_dt-{0.01}.pdf'))
    plt.show()

    # ------------------------- RWMH -------------------------
    # get 'true' mean and variance from mcmc
    n_samples = 2000
    mh_sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(.5), rng=rng,
                                           prec=cov, cov_factor=0.5, x_0=x_0)
    mh_sampler.run()
    mh_samples = mh_sampler.chain_

    # get figure and plot pdfs
    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, axes_label='E')
    plot_pdf_contours(target, ax, plot_limits)
    plot_samples(mh_samples, ax, color_code=False, n_vis=2000, size=1.5)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'random_walk_{n_samples}.pdf'))

    plt.show()

    # MH with 200 samples
    n_samples = 200
    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, axes_label='E')
    plot_pdf_contours(target, ax, plot_limits)
    plot_samples(mh_samples[:n_samples], ax, color_code=False, n_vis=500, size=1.5)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'random_walk_{n_samples}.pdf'))

    plt.show()
