import os.path

from src.forward_model import ForwardModel
from src.project_field import compute_coefficients, squared_exponential_kernel
from src.samplers import MetropolisHastingsSampler, LangevinDynamicsSampler, ZigZagSampler
from src.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from src.utils import plot_pdf_contours, plot_samples, central_moment_from_skeleton

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

sns.set_style('darkgrid')

# numerically approximate hessian
def hess(f, x, h):
    hess = np.zeros((x.shape[0], x.shape[0]))
    for i in range(hess.shape[0]):
        for j in range(hess.shape[1]):
            hess[i, j] = (f(x + h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
                          f(x - h * np.eye(x.shape[0])[i] + h * np.eye(x.shape[0])[j]) -
                          f(x + h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j]) +
                          f(x - h * np.eye(x.shape[0])[i] - h * np.eye(x.shape[0])[j])) / (4 * h ** 2)
    return hess


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
    plot_limits = ([2.3, 4.5], [2.7, 5.8])

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    sigma_obs = 0.025

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

    # get laplace approximation
    log_post = lambda x: target.logDensity(x)
    n_log_post = lambda x: - log_post(x)

    x0 = np.array([4., 4.])
    x_map = minimize(n_log_post, x0, method='BFGS', jac=False).x
    print(f"Map: {x_map}")

    hess_ap = hess(log_post, x_map, 1e-5)
    print(f"Hess ap: \n{hess_ap}")
    approx = {'mean': x_map, 'inv_cov': - hess_ap}

    # get 'true' mean and variance from mcmc
    # n_samples = 100
    n_samples = 100000
    Sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                        prec=cov)
    Sampler.run()
    samples = Sampler.chain_
    samples = samples[5000:]
    true_mean = np.mean(samples, axis=0)
    # true_var = np.cov(samples.transpose())
    # true_var = np.var(samples, axis=0)

    n_runs = 20
    means_zig_zag = np.zeros((n_runs, n_b))
    means_mh = np.zeros((n_runs, n_b))
    variances = np.zeros((n_runs, n_b))

    for i in range(n_runs):
        n_events = 5000
        # approx = {'mean': map_point, 'inv_cov': np.linalg.inv(map_cov)}
        sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx, n_events_accepted=200)
        # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False)
        sampler.run()
        #
        time = sampler.times_
        positions = sampler.positions_
        velocities = sampler.velocities_

        means_zig_zag[i] = central_moment_from_skeleton(time, positions, velocities, 1)
        # variances[i] = central_moment_from_skeleton(time, positions, velocities, 2)

    for i in range(n_runs):
        # get 'true' mean and variance from mcmc
        n_samples = 20000
        # n_samples = 200
        Sampler = MetropolisHastingsSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                            prec=cov)
        Sampler.run()
        samples = Sampler.chain_
        samples = samples[-200:]
        # samples = samples[-10:]
        means_mh[i] = np.mean(samples, axis=0)

    print(f"True mean: {true_mean}")
    print(f"Means ZZ:\n{means_zig_zag}")
    print(f"Means MH:\n{means_mh}")
    # print(variances)

    # compute mse between true mean and the means_zig_zag from the zig-zag
    mse_zig_zag = np.mean((means_zig_zag - true_mean) ** 2)
    mse_mh = np.mean((means_mh - true_mean) ** 2)
    print(f"MSE ZZ: {mse_zig_zag}")
    print(f"MSE MH: {mse_mh}")

    #
    # print(pdmp_mean)
    # print(pdmp_var)
    # print(np.linalg.inv(pdmp_var))
    # ax.scatter(*positions.T, c='C0')