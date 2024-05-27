import os.path

from src.forward_model import ForwardModel
from src.project_field import compute_coefficients, squared_exponential_kernel
from src.samplers import MetropolisHastingsSampler, LangevinDynamicsSampler, ZigZagSampler
from src.distributions import MultivariateNormal, Likelihood, Posterior
from src.utils import plot_pdf_contours, plot_samples, central_moment_from_skeleton

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

rng = np.random.default_rng(1)

save_fig = False

if __name__ == '__main__':

    # set up random field
    kernel = squared_exponential_kernel
    n_b = 2
    interval = [0, 1]
    mean = 4.
    basis = []

    for i in range(n_b):
        basis.append(lambda x, i=i: np.piecewise(x,
                                                 [x < (i / n_b),
                                                  x == (i / n_b),
                                                  ((i / n_b) < x) & (x < ((i + 1) / n_b)),
                                                  x == ((i + 1) / n_b),
                                                  x > ((i + 1) / n_b)], [0, 0.5, 1, 0.5, 0]))

    prior_cov = compute_coefficients(n_b, squared_exponential_kernel, basis, interval)
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
    plot_limits = ([2.3, 4.5], [2.7, 5.8])

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    # u_gt_test_1 = model.eval(x_test, params_gt, idx=0)
    # u_gt_test_2 = model.eval(x_test, params_gt, idx=1)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    # ax.plot(x_test, u_gt_test_1, label=r'$u_{\mathrm{gt}}$')
    # ax.plot(x_test, u_gt_test_2, label=r'$u_{\mathrm{gt}}$')

    # plt.show()

    # generate n_obs noisy obersevations of u_gt
    sigma_obs = 0.025
    sigma_obs = 0.05

    # evaluate the ground truth at the observation and test locations
    u_gt = model.eval(x_obs, params_gt)
    u_gt_test = model.eval(x_test, params_gt)
    u_obs = np.zeros((len(F), u_gt.shape[0]))

    for i in range(len(F)):
        u_gt = model.eval(x_obs, params_gt, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    # plot the ground truth and the observations
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    ax.plot(x_test, u_gt_test, label=r'$u_{\mathrm{gt}}$')
    for i in range(len(F)):
        if i == 0:
            ax.scatter(x_obs, u_obs[i], c='C1', alpha=0.5, label=r'$u_{\mathrm{obs}}$')
        else:
            ax.scatter(x_obs, u_obs[i], c='C1', alpha=0.5)
    ax.legend()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    if save_fig:
        fig.savefig(os.path.join('./figures', 'ground_truth_and_observations.png'))
    plt.show()

    # define likelihood and posterior
    likelihood = Likelihood(model, x_obs, u_obs, sigma_obs)
    target = Posterior(prior, likelihood)

    # plot the posterior
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.5), constrained_layout=True)
    ax.axis('equal')
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])

    plot_pdf_contours(target, ax, plot_limits)

    # map_point = np.array([3.36691724, 4.09563908])
    # map_cov = np.array([[ 0.09886727, -0.07772097],
    #                 [-0.07772097,  0.29766597]])
    # laplace_approx = MultivariateNormal(map_point, map_cov, rng=rng)
    # plot_pdf_contours(laplace_approx, ax, plot_limits)

    # plt.show()

    n_events = 200
    # approx = {'mean': map_point, 'inv_cov': np.linalg.inv(map_cov)}
    # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx)
    sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False)
    sampler.run()
    #
    time = sampler.times_
    positions = sampler.positions_
    velocities = sampler.velocities_

    ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    if save_fig:
        fig.savefig(os.path.join('./figures', 'zig_zag_cinlar.png'))
        # fig.savefig(os.path.join('./figures', 'posterior.png'))
    plt.show()

    pdmp_mean = central_moment_from_skeleton(time, positions, velocities, 1)
    pdmp_var = central_moment_from_skeleton(time, positions, velocities, 2)

    print(pdmp_mean)
    print(pdmp_var)
    # ax.scatter(*positions.T, c='C0')

    # ---------------------------------- MALA ----------------------------------------------
    # # set up sampling algorithm
    # n_samples = 20000
    # # MALA = LangevinDynamicsSampler(posterior, n_samples=n_samples, sigma=np.sqrt(1.5), adjusted=True, rng=rng,
    # #                                prec=cov)
    # Sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
    #                                     prec=cov)
    # Sampler.run()
    # samples = Sampler.chain_
    # n_vis = 500
    #
    # # plot the samples
    # x = np.linspace(plot_limits[0][0], plot_limits[0][1], 100)
    # y = np.linspace(plot_limits[1][0], plot_limits[1][1], 70)
    # gx, gy = np.meshgrid(x, y)
    # gz = np.zeros_like(gx)
    # grad_z = np.zeros((gx.shape[0], gx.shape[1], 2))
    #
    # for i in range(gx.shape[0]):
    #     for j in range(gx.shape[1]):
    #         gz[i,j] = np.exp(target.logDensity(np.array([gx[i, j], gy[i, j]])))
    #
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # ax.set_xlim(plot_limits[0][0],plot_limits[0][1])
    # ax.set_ylim(plot_limits[1][0],plot_limits[1][1])
    #
    # ax.contour(gx, gy, gz, levels=20, zorder=1, alpha=0.6)
    # # ax.contour(gx, gy, gz, levels=levels, zorder=1)
    # samples_plot = samples[0::n_samples//n_vis]
    # # ax.scatter(*samples[2000::n_samples//n_vis].transpose(), s=3, zorder=2, c=np.linspace(0, 1, n_vis))
    # ax.scatter(*samples_plot.transpose(), s=3, zorder=2, c=np.linspace(0,1, samples_plot.shape[0]))
    # # ax.scatter(*SPN.transpose(), s=5)
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(samples[:4000, 0])
    # fig.show()
    #
    # print(np.mean(samples, axis=0))
    # print(np.cov(samples.transpose()))


    # ---------------------------------- gradient test ----------------------------------------------

    # f = lambda x: - target.logDensity(x)
    # # # f = lambda x: model.eval(x_obs, x)[0]
    #
    # # find the minimum of the function with bfgs
    # from scipy.optimize import minimize
    # x0 = np.array([4., 4.])
    # x_map = minimize(f, x0, method='BFGS', jac=False).x
    # print(x_map)
    #
    # ax.plot(x_map[0], x_map[1], 'ro')
    #
    # model.eval_hessian(x_obs, x_map)
    #
    # def grad(f, x, h):
    #     grad = np.zeros_like(x)
    #     for i in range(grad.shape[0]):
    #         grad[i] = (f(x + h * np.eye(x.shape[0])[i]) - f(x - h * np.eye(x.shape[0])[i])) / (2 * h)
    #     return grad
    #
    # # numerically approximate hessian
    # def hess(f, x, h):
    #     hess = np.zeros((x.shape[0], x.shape[0]))
    #     for i in range(hess.shape[0]):
    #         for j in range(hess.shape[1]):
    #             hess[i, j] = (f(x + h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
    #                           f(x - h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
    #                           f(x + h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j]) +
    #                           f(x - h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j])) / (4 * h**2)
    #     return hess
    #
    # # hess_an = target.hessianLogDensity(x_map)
    # # print(f"Hess an: \n{hess_an}")
    #
    # hess_ap = hess(f, x_map, 1e-5)
    # print(f"Hess ap: \n{hess_ap}")
    #
    # map_point = np.array([3.36691724, 4.09563908])
    # map_cov = np.array([[ 0.09886727, -0.07772097],
    #                     [-0.07772097,  0.29766597]])
    # # cov_map = np.linalg.inv(hess(f, x_map, 1e-6))
    # cov_map = np.linalg.inv(hess_ap)
    # laplace_approx = MultivariateNormal(x_map, cov_map, rng=rng)
    # plot_pdf_contours(laplace_approx, ax, plot_limits, n_levels=10, alpha=0.5, cmap=sns.color_palette('mako', as_cmap=True))
    #
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', 'laplace_approximation.png'))
    #
    # n_events = 100
    # approx = {'mean': map_point, 'inv_cov': cov_map}
    # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx)
    # # sampler = ZigZagSampler(target, n_events=n_events, rng=rng)
    # sampler.run()
    # #
    # positions = sampler.positions_
    # ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    # ax.set_xlabel(r'$\theta_1$')
    # ax.set_ylabel(r'$\theta_2$')
    # if save_fig:
    #     fig.savefig(os.path.join('./figures', 'zig_zag_cinlar.png'))
    #     # fig.savefig(os.path.join('./figures', 'posterior.png'))
    #
    # plt.show()

    # # x0 = np.array([4., 4.])
    # print(f"scipy: ")
    # grad0 = grad(f, x_map, 1e-8)
    # [print(f" {g:.10f}") for g in grad0]
    #
    # print(f"analytic:")
    # grad_an = target.gradLogDensity(x_map)
    # # grad_an = model.eval_grad(x_obs, x0)[0]
    # [print(f" {g:.10f}") for g in grad_an]
    #
    # n_test_g = 100
    # # x_test_g = np.vstack((np.linspace(3, 5, n_test_g), np.ones(n_test_g) * 4.)).T
    # x_test_g = np.vstack((np.ones(n_test_g) * 4., np.linspace(3, 5, n_test_g))).T
    # f_test_g = np.zeros(n_test_g)
    #
    # for i in range(n_test_g):
    #     f_test_g[i] = f(x_test_g[i])
    #
    # f_test_0 = f(x0)
    #
    # # plot a straight line through x0[0] with the gradient at x0[0]
    # y_test = f_test_0 + grad0[1] * (x_test_g[:, 1] - x0[1])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # ax.plot(x_test_g[:, 1], f_test_g)
    # ax.plot(x_test_g[:, 1], y_test)
    # plt.show()
