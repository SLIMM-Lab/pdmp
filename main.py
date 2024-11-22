import os.path

from pdmp.forward_model import ForwardModel
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp.mcmc import MetropolisHastingsSampler, LangevinDynamicsSampler, ZigZagSampler
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from pdmp.utils import plot_pdf_contours, plot_samples, central_moment_from_skeleton

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

rng = np.random.default_rng(1)

save_fig = False

if __name__ == '__main__':

    # get the prior field
    kernel_params = {'sigma': 1., 'l': 0.3}
    n_b = 2
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_b, interval)
    prior_cov = compute_coefficients(squared_exponential_kernel, basis, interval, kernel_params=kernel_params)
    mean = 4.
    prior = MultivariateNormal(mean * np.ones(n_b), prior_cov, rng=rng)

    # set up the forward model
    n_obs = 2
    F = [1.]
    F = [item for item in F for i in range(n_obs)]
    n_params = 2
    model = ForwardModel(F, n_params)

    # generate ground truth from a multi-variate normal distribution
    mean = 3 * np.ones(n_params)
    cov = np.eye(n_params)
    params_gt = rng.multivariate_normal(mean, cov)
    print(f"Ground truth: \n {params_gt}")
    margins = 1.5
    # plot_limits = ([params_gt[0] - margins, params_gt[0] + margins], [params_gt[1] - margins, params_gt[1] + margins])
    plot_limits = ([2.3, 4.6], [2.8, 5.9])

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    # # plot the ground truth and the obsevations
    # u_gt_test_1 = model.eval(x_test, params_gt, idx=0)
    # u_gt_test_2 = model.eval(x_test, params_gt, idx=1)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    # ax.plot(x_test, u_gt_test_1, label=r'$u_{\mathrm{gt}}$')
    # ax.plot(x_test, u_gt_test_2, label=r'$u_{\mathrm{gt}}$')

    # plt.show()

    # generate n_obs noisy obersevations of u_gt
    sigma_obs = 0.025
    # sigma_obs = 0.05

    # evaluate the ground truth at the observation and test locations
    u_gt = model.eval(x_obs, params_gt)
    u_gt_test = model.eval(x_test, params_gt)
    u_obs = np.zeros((len(F), u_gt.shape[0]))

    for i in range(len(F)):
        u_gt = model.eval(x_obs, params_gt, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    # # plot the ground truth and the observations
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    # ax.plot(x_test, u_gt_test, label=r'$u_{\mathrm{gt}}$')
    # for i in range(len(F)):
    #     if i == 0:
    #         ax.scatter(x_obs, u_obs[i], c='C1', alpha=0.5, label=r'$u_{\mathrm{obs}}$')
    #     else:
    #         ax.scatter(x_obs, u_obs[i], c='C1', alpha=0.5)
    # ax.legend()
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$u$')
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', 'ground_truth_and_observations.png'))
    # plt.show()

    # define likelihood and posterior
    likelihood = GaussianLikelihood(model, x_obs, u_obs, sigma_obs)
    target = Posterior(prior, likelihood)

    # plot the posterior
    fig, ax = plt.subplots(1, 1, figsize=(3, 4.0), constrained_layout=True)
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.axis('equal')
    ax.grid(False)
    ax.autoscale(enable=False)

    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    # despine the plot
    sns.despine()

    # get rid of the ticks labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # make the axes linewidths bigger
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.)

    plot_pdf_contours(target, ax, plot_limits)


    # # ---------------------------------- Cinlar ----------------------------------------------
    # n_events = 200
    # # approx = {'mean': map_point, 'inv_cov': np.linalg.inv(map_cov)}
    # # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx)
    # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False)
    # sampler.run()
    # #
    # time = sampler.times_
    # positions = sampler.positions_
    # velocities = sampler.velocities_
    #
    # ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', 'zig_zag_cinlar.p_x'))
    #     # fig.savefig(os.path.join('./figures', 'posterior.png'))
    # plt.show()
    #
    # pdmp_mean = central_moment_from_skeleton(time, positions, velocities, 1)
    # pdmp_var = central_moment_from_skeleton(time, positions, velocities, 2)
    #
    # print(pdmp_mean)
    # print(pdmp_var)
    # print(np.linalg.inv(pdmp_var))
    # # ax.scatter(*positions.T, c='C0')

    # # ---------------------------------- MH ----------------------------------------------
    # # set up sampling algorithm
    # n_samples = 20000
    # Sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
    #                                     prec=cov)
    # Sampler.run()
    # samples = Sampler.chain_
    # n_samples = 2000
    # samples = samples[-n_samples:]
    #
    # n_vis = 200
    #
    # samples_plot = samples[0::n_samples//n_vis]
    # ax.scatter(*samples_plot.transpose(), s=3, zorder=2, c='C0', alpha=0.75)
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', f'mh_{n_samples}.p_x'))
    # fig.show()


    # ---------------------------------- laplace approx ----------------------------------------------

    log_post = lambda x: target.log_density(x)
    n_log_post = lambda x: - log_post(x)
    # # f = lambda x: model.eval(x_obs, x)[0]

    # find the minimum of the function with bfgs
    from scipy.optimize import minimize
    x0 = np.array([4., 4.])
    x_map = minimize(n_log_post, x0, method='BFGS', jac=False).x
    print(x_map)

    # ax.plot(x_map[0], x_map[1], 'ro')

    model.eval_hessian(x_obs, x_map)

    def grad(f, x, h):
        grad = np.zeros_like(x)
        for i in range(grad.shape[0]):
            grad[i] = (f(x + h * np.eye(x.shape[0])[i]) - f(x - h * np.eye(x.shape[0])[i])) / (2 * h)
        return grad

    # numerically approximate hessian
    def hess(f, x, h):
        hess = np.zeros((x.shape[0], x.shape[0]))
        for i in range(hess.shape[0]):
            for j in range(hess.shape[1]):
                hess[i, j] = (f(x + h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
                              f(x - h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
                              f(x + h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j]) +
                              f(x - h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j])) / (4 * h**2)
        return hess

    # hess_an = target.hessianLogDensity(x_map)
    # print(f"Hess an: \n{hess_an}")

    hess_ap = hess(log_post, x_map, 1e-5)
    print(f"Hess ap: \n{hess_ap}")

    cov_map = np.linalg.inv(- hess_ap)
    laplace_approx = MultivariateNormal(x_map, cov_map, rng=rng)
    plot_pdf_contours(target, ax, plot_limits, alpha=0.3)
    # plot_pdf_contours(target, ax, plot_limits, alpha=0.3)
    # plot_pdf_contours(laplace_approx, ax, plot_limits, n_levels=10, alpha=1.0, cmap=sns.color_palette('mako', as_cmap=True))
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])

    if save_fig:
        # fig.savefig(os.path.join('./figures', 'laplace_approximation.p_x'))
        fig.savefig(os.path.join('visualizations_eccomas/figures', 'posterior.p_x'))

    # n_events = 2000
    # approx = {'mean': x_map, 'inv_cov': - hess_ap}
    # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx, x0=np.array([4.,4.]),
    #                         n_events_accepted=200)
    # # sampler = ZigZagSampler(target, n_events=n_events, rng=rng)
    # sampler.run()
    #
    # print(f"Times:\n{sampler.times_}\n\n")
    # print(f"Positions:\n{sampler.positions_}\n\n")
    # print(f"Velocities:\n{sampler.velocities_}\n\n")

    # plot_pdf_contours(target, ax, plot_limits)
    # positions = sampler.positions_
    # ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    # ax.set_xlabel(r'$\theta_1$')
    # ax.set_ylabel(r'$\theta_2$')
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', 'zig_zag_thinning_200.p_x'))

    plt.show()
    #
    # x0 = np.array([4., 4.])
    # # x0 = x_map
    # print(f"scipy: ")
    # grad0 = grad(n_log_post, x0, 1e-8)
    # [print(f" {g:.10f}") for g in grad0]
    #
    # print(f"analytic:")
    # grad_an = - target.gradLogDensity(x0)
    # # grad_an = model.eval_grad(x_obs, x0)[0]
    # [print(f" {g:.10f}") for g in grad_an]
    #
    # n_test_g = 100
    # x_test_g = np.vstack((np.ones(n_test_g) * x0[0], np.linspace(x0[1] - 1, x0[1] + 1, n_test_g))).T
    # f_test_g = np.zeros(n_test_g)
    #
    # for i in range(n_test_g):
    #     f_test_g[i] = n_log_post(x_test_g[i])
    #
    # f_test_0 = n_log_post(x0)
    #
    # # plot a straight line through x0[0] with the gradient at x0[0]
    # y_test = f_test_0 + grad0[1] * (x_test_g[:, 1] - x0[1])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # ax.plot(x_test_g[:, 1], f_test_g)
    # ax.plot(x_test_g[:, 1], y_test)
    # plt.show()
