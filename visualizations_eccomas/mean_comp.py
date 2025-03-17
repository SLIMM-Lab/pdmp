import os.path

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp.mcmc import RandomWalkMetropolisSampler, LangevinDynamicsSampler
from pdmp.zigzag import ZigZagSampler
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior
from pdmp.utils import central_moment_from_skeleton

import numpy as np
import seaborn as sns
sns.set_style('darkgrid')

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

    # define observation and test locations
    x_obs = np.linspace(0, 1, 3)[1:]
    x_test = np.linspace(0, 1, 100)

    # set up the forward model
    n_obs = 2
    F = [1.]
    F = np.array([item for item in F for i in range(n_obs)])
    n_params = 2
    model = PiecewiseConstantModel(F, n_params, x_obs)

    # generate ground truth from a multi-variate normal distribution
    mean = 3 * np.ones(n_params)
    cov = np.eye(n_params)
    params_gt = rng.multivariate_normal(mean, cov)
    print(f"Ground truth: \n {params_gt}")
    margins = 1.5
    # plot_limits = ([params_gt[0] - margins, params_gt[0] + margins], [params_gt[1] - margins, params_gt[1] + margins])
    plot_limits = ([2.3, 4.5], [2.7, 5.8])

    sigma_obs = 0.025
    # sigma_obs = 0.05

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

    # get 'true' mean and variance from mcmc
    n_samples = 100000
    Sampler = RandomWalkMetropolisSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                          prec=cov)
    Sampler.run()
    samples = Sampler.chain_
    true_mean = np.mean(samples, axis=0)
    # true_var = np.cov(samples.transpose())
    # true_var = np.var(samples, axis=0)

    n_runs = 20
    means_zig_zag = np.zeros((n_runs, n_b))
    means_mh = np.zeros((n_runs, n_b))
    variances = np.zeros((n_runs, n_b))

    for i in range(n_runs):
        n_events = 200
        # approx = {'mean': map_point, 'inv_cov': np.linalg.inv(map_cov)}
        # sampler = ZigZagSampler(target, n_events=n_events, rng=rng, approximation=approx)
        sampler = ZigZagSampler(target, n_events=n_events, rng=rng, sub_sampling=False)
        sampler.run()
        #
        time = sampler.times
        positions = sampler.positions
        velocities = sampler.velocities

        means_zig_zag[i] = central_moment_from_skeleton(time, positions, velocities, 1)
        # variances[i] = central_moment_from_skeleton(time, positions, velocities, 2)

    for i in range(n_runs):
        # get 'true' mean and variance from mcmc
        n_samples = 200
        Sampler = RandomWalkMetropolisSampler(target, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                              prec=cov)
        Sampler.run()
        samples = Sampler.chain_
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