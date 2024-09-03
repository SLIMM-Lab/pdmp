import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from src.forward_model import ForwardModel
from src.project_field import compute_coefficients, squared_exponential_kernel
from src.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from src.samplers import ZigZagSampler
from src.utils import plot_pdf_contours, plot_samples, central_moment_from_skeleton, get_2d_despined_figure
from visualizations_eccomas.non_rev_demo import fig_path


def grad_fd(f, x, h=1e-5):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = (f(x + h * np.eye(n)[i]) - f(x - h * np.eye(n)[i])) / (2 * h)
    return grad

def hessian_fd(f, x, h=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hess[i, j] = (f(x + h * np.eye(n)[i] + h * np.eye(n)[j]) -
                          f(x - h * np.eye(n)[i] + h * np.eye(n)[j]) -
                          f(x + h * np.eye(n)[i] - h * np.eye(n)[j]) +
                          f(x - h * np.eye(n)[i] - h * np.eye(n)[j])) / (4 * h**2)
    return hess


sns.set_style('white')
rng = np.random.default_rng(1)

save_fig = False
fig_path = 'laplace_approx/figures'

if __name__ == '__main__':

    # check if the directory exists, otherwise create
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

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
    n_params = n_b
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

    # define likelihood and posterior
    likelihood = GaussianLikelihood(model, x_obs, u_obs, sigma_obs)
    target = Posterior(prior, likelihood)

    # # ------------------------- Test 1d ------------------------------
    # # # hessian of model
    # # f = lambda theta: model.eval(x_obs, theta)[0]
    # # df = lambda theta: model.eval_grad(x_obs, theta)[0]
    # # ddf = lambda theta: model.eval_hessian(x_obs, theta)[0,0]
    #
    # # # hessian of likelihood
    # # f = lambda theta: likelihood.log_density(theta)
    # # df = lambda theta: likelihood.grad_log_density(theta)
    # # ddf = lambda theta: likelihood.hessian_log_density(theta)[0,0]
    #
    # # hessian of posterior
    # f = lambda theta: target.log_density(theta)
    # df = lambda theta: target.grad_log_density(theta)
    # ddf = lambda theta: target.hessian_log_density(theta)[0,0]
    #
    # x0 = np.array([2.])
    # f0 = f(x0)
    # df0 = df(x0)
    # ddf0 = ddf(x0)
    #
    # n_plot = 100
    # x_plot = np.linspace(0.5, 5.5, n_plot)
    # f_plot = np.zeros(n_plot)
    # for i in range(n_plot):
    #     f_plot[i] = f(x_plot[[i]])
    #
    # # plot straight line through x0
    # f_lin = f0 + df0 * (x_plot - x0)
    #
    # # plot quadratic function trough x0
    # f_quad = f0 + df0 * (x_plot - x0) + 0.5 * ddf0 * (x_plot - x0)**2
    #
    # fig, ax = plt.subplots()
    # ax.plot(x_plot, f_plot, c='C0')
    # ax.plot(x_plot, f_lin, c='C1')
    # ax.plot(x_plot, f_quad, c='C2', ls='--')
    # plt.show()


    # # ------------------------- Test 2d ------------------------------
    # # hessian of model
    # f = lambda theta: model.eval(x_obs, theta)[0]
    # df = lambda theta: model.eval_grad(x_obs, theta)[0]
    # ddf = lambda theta: model.eval_hessian(x_obs, theta)[0]

    # # hessian of likelihood
    # f = lambda theta: likelihood.log_density(theta)
    # df = lambda theta: likelihood.grad_log_density(theta)
    # ddf = lambda theta: likelihood.hessian_log_density(theta)

    # hessian of posterior
    f = lambda theta: target.log_density(theta)
    df = lambda theta: target.grad_log_density(theta)
    ddf = lambda theta: target.hessian_log_density(theta)

    x0 = np.array([2., 2.5])
    f0 = f(x0)
    df0 = df(x0)
    ddf0 = ddf(x0)

    df0_fd = grad_fd(f, x0)
    ddf0_fd = hessian_fd(f, x0, h=1e-6)

    print(f"df0:\n{df0}")
    print(f"\ndf0_fd:\n{df0_fd}")
    print(f"\nddf0:\n{ddf0}")
    print(f"\nddf0_fd:\n{ddf0_fd}")

    # evalute f
    idx = 1
    n_plot = 100
    x_plot = np.ones((n_plot, n_params)) * x0
    x_plot[:, idx] = np.linspace(1.0, 4.5, n_plot)
    # x_plot = np.linspace(1., 5.5, n_plot)
    f_plot = np.zeros(n_plot)
    for i in range(n_plot):
        f_plot[i] = f(x_plot[i])

    # plot straight line through x0
    f_lin = f0 + df0 @ (x_plot - x0).T

    # # plot quadratic function trough x0
    f_quad = f0 + (x_plot - x0) @ df0 + 0.5 * np.einsum('ij, jk, ik -> i', (x_plot - x0), ddf0, (x_plot - x0))

    fig, ax = plt.subplots()
    ax.plot(x_plot[:,idx], f_plot, c='C0')
    ax.plot(x_plot[:,idx], f_lin, c='C1')
    ax.plot(x_plot[:,idx], f_quad, c='C2')

    if save_fig:
        fig.savefig(os.path.join(fig_path, 'test_hessian.pdf'))

    plt.show()

    # ---------------------------------- laplace approx ----------------------------------------------
    # plot the posterior
    fig, ax = get_2d_despined_figure(plot_limits)
    plot_pdf_contours(target, ax, plot_limits)
    plt.show()

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
    fig, ax = get_2d_despined_figure(plot_limits)
    ax.plot(x_map[0], x_map[1], 'ro')
    plot_pdf_contours(target, ax, plot_limits, alpha=0.3)
    plot_pdf_contours(laplace_approx, ax, plot_limits, n_levels=10, alpha=1.0, cmap=sns.color_palette('mako', as_cmap=True))
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])

    if save_fig:
        fig.savefig(os.path.join(fig_path, 'laplace_approx_contours.pdf'))

    plt.show()

    # ---------------------------------- zig zag sampler ----------------------------------------------
    fig, ax = get_2d_despined_figure(plot_limits)
    plot_pdf_contours(target, ax, plot_limits)

    n_events = 2000
    approx = {'mean': x_map, 'inv_cov': - hess_an}
    sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx, x0=np.array([4.,4.]),
                            n_events_accepted=200)
    # sampler = ZigZagSampler(target, n_events=n_events, rng=rng)
    sampler.run()

    # print(f"Times:\n{sampler.times_}\n\n")
    # print(f"Positions:\n{sampler.positions_}\n\n")
    # print(f"Velocities:\n{sampler.velocities_}\n\n")

    plot_pdf_contours(target, ax, plot_limits)
    positions = sampler.positions_
    ax.plot(positions[:, 0], positions[:, 1], c='C0', alpha=0.75, linewidth=1.)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    if save_fig:
        fig.savefig(os.path.join(fig_path, 'zig_zag_thinning_200.pdf'))

    plt.show()
