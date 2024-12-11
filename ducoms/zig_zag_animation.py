import os.path
import numpy as np
import seaborn as sns

from pdmp.forward_model import PiecewiseConstantModel
from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp.distributions import MultivariateNormal, GaussianLikelihood, Posterior, plot_pdf_contours
from pdmp.zigzag import ZigZagSampler
from pdmp.utils import get_2d_despined_figure
from bisect import bisect_right

sns.set_style('white')
rng = np.random.default_rng(0)

save_fig = False
format = 'pdf'
dpi = 400
base_path = '/home/leon/ownCloud/Documents/presentations/conferences/24-ducoms/latex/includes/zig-zag_animation'
# base_path = '/home/leon/Downloads/zig-zag/zig_zag'
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
    plot_pdf_contours(target, ax, plot_limits, n_levels=15)

    x_0 = np.array([3.8, 3.2])
    v_0 = np.array([-1., -1.])
    n_events = 600
    sampler = ZigZagSampler(target, n_max=n_events, rng=rng, x_0=x_0, v_0=v_0)
    sampler.run()

    positions = sampler.positions_
    times = sampler.times_
    velocities = sampler.velocities_

    t_switch = 1.
    t_final = 150.

    assert times[-1] > t_final, f"Times[-1] is {times[-1]} and should be greater than {t_final}"

    t_slow = np.linspace(0, t_switch, 400)
    t_fast = np.linspace(t_switch, t_final, 1000)
    plot = None
    idx = 0

    for i, t in enumerate(t_slow):

        if i%100 == 0:
            print(f"Slow: {i}/{len(t_slow)}")

        if plot is not None:
            plot[0].remove()

        d_idx = bisect_right(times[idx:], t) - 1
        idx += d_idx
        p = positions[:idx+1]
        p = np.vstack((p, p[-1] + velocities[idx] * (t - times[idx])))
        plot = ax.plot(p[:, 0], p[:, 1], c='C0', alpha=0.75, linewidth=1.)

        if save_fig:
            fig.savefig(os.path.join(fig_path, f's-{i}.{format}'), dpi=dpi)

    for i, t in enumerate(t_fast):
        if plot is not None:
            plot[0].remove()

        d_idx = bisect_right(times[idx:], t) - 1
        idx += d_idx
        p = positions[:idx+1]
        p = np.vstack((p, p[-1] + velocities[idx] * (t - times[idx])))
        plot = ax.plot(p[:, 0], p[:, 1], c='C0', alpha=0.75, linewidth=1.)

        if save_fig:
            fig.savefig(os.path.join(fig_path, f'f-{i}.{format}'), dpi=dpi)
