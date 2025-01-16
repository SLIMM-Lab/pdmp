import os.path
import numpy as np
import seaborn as sns

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior, plot_pdf_contours
from pdmp.mcmc import MetropolisHastingsSampler
from pdmp.plotting import get_2d_despined_figure, plot_samples
from bisect import bisect_right

sns.set_style('white')
rng = np.random.default_rng(0)

save_fig = False
format = 'pdf'
dpi = 400
# fig_path = './figures/animation_zig_zag'
# base_path = '/home/leon/Downloads/zig-zag/rwm'
base_path = '/home/leon/ownCloud/Documents/presentations/conferences/24-ducoms/latex/includes/rwm_animation'
fig_path = os.path.join(base_path, format)

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
    plot_limits = ([2.2, 4.2], [1.95, 4.25])

    # generate n_obs noisy observations of u_gt
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

    # ---------------------------------- zig zag int ----------------------------------------------

    scale = 1.2
    figsize = (scale * np.abs(plot_limits[0][1] - plot_limits[0][0]),
               scale * np.abs(plot_limits[1][1] - plot_limits[1][0]))

    fig, ax = get_2d_despined_figure(plot_limits, figsize=figsize, keep_ticks=False, axes_label='\\theta')
    plot_pdf_contours(target, ax, plot_limits, n_levels=15, alpha=0.5)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'posterior_0.{format}'), dpi=dpi)

    # x_0 = np.array([2.8, 3.5])
    x_0 = np.array([3.5, 3.0])
    n_samples = 1000
    sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(.125), rng=rng, prec=cov, cov_factor=0.5, x_0=x_0)
    sampler.run()

    samples = sampler.chain_
    proposals = sampler.proposals_
    accepted = sampler.accepted_

    # plot_samples(samples, ax, color_code=False, n_vis=2000, size=1.5)
    # plot_samples(proposals, ax, color_code=False, n_vis=2000, size=1.5, color='C1')
    # plt.show()

    iter_slow = 15
    iter_fast = 1000

    p_g = None
    p_b = None

    idx = 0
    markersize = 2.0

    for i in range(1, iter_slow - 1):
        if p_g is not None:
            p_g.remove()
        if p_b is not None:
            p_b.remove()

        p_g = ax.scatter(samples[:i, 0], samples[:i, 1], s=markersize, alpha=0.75, c='C2')
        if save_fig:
            fig.savefig(os.path.join(fig_path, f's-{idx}.{format}'), dpi=dpi)
        idx += 1

        p_b = ax.scatter(proposals[i, 0], proposals[i, 1], s=markersize, alpha=0.75, c='C0')
        if save_fig:
            fig.savefig(os.path.join(fig_path, f's-{idx}.{format}'), dpi=dpi)
        idx += 1
        p_b.remove()
        p_b = None

        if not accepted[i]:
            p_r = ax.scatter(proposals[i, 0], proposals[i, 1], s=markersize, alpha=0.75, c='C3')
            if save_fig:
                fig.savefig(os.path.join(fig_path, f's-{idx}.{format}'), dpi=dpi)
            idx += 1
            p_r.remove()

    print("Total number of slow images: ", idx)

    idx = 0
    markersize = 2.0

    for i in range(iter_slow, iter_fast - 1):
        if p_g is not None:
            p_g.remove()
        if p_b is not None:
            p_b.remove()

        p_g = ax.scatter(samples[:i, 0], samples[:i, 1], s=markersize, alpha=0.75, c='C2')
        if save_fig:
            fig.savefig(os.path.join(fig_path, f'f-{idx}.{format}'), dpi=dpi)
        idx += 1

        p_b = ax.scatter(proposals[i, 0], proposals[i, 1], s=markersize, alpha=0.75, c='C0')
        if save_fig:
            fig.savefig(os.path.join(fig_path, f'f-{idx}.{format}'), dpi=dpi)
        idx += 1
        p_b.remove()
        p_b = None

        if not accepted[i]:
            p_r = ax.scatter(proposals[i, 0], proposals[i, 1], s=markersize, alpha=0.75, c='C3')
            if save_fig:
                fig.savefig(os.path.join(fig_path, f'f-{idx}.{format}'), dpi=dpi)
            idx += 1
            p_r.remove()


    print("Total number of fast images: ", idx)

    p_g.remove()

    ax.scatter(samples[:, 0], samples[:, 1], s=markersize, alpha=0.75, c='C2', zorder=10)
    ax.scatter(proposals[accepted==False, 0], proposals[accepted==False, 1], s=markersize, alpha=0.75, c='C3', zorder=9)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'samples_proposals-0.{format}'), dpi=dpi)
