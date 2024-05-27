import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from typing import Tuple
from src.distributions import MultivariateNormal
from src.distributions import getSample

import seaborn as sns


class MetropolisHastingsSampler:

    def __init__(self, target, sigma=0.5, n_samples=10000, rng=None, seed=None, prec=None):
        self.target_ = target
        self.dim_ = self.target_.getDim()
        self.sigma_ = sigma
        self.n_samples_ = n_samples
        self.n_accept_ = 0
        self.n_accept_last_ = 0
        self.rescale_interval_ = 100

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        self.proposalDist_ = MultivariateNormal(np.zeros(self.dim_),
                                                np.eye(self.dim_),
                                                rng=self.rng_)

        if prec is None:
            self.prec_ = np.eye(self.dim_)
        else:
            assert np.any(prec.shape != self.dim_) and (len(prec.shape) == 2)
            self.prec_ = prec

        self.precL_ = np.linalg.cholesky(self.prec_)

        self.reset()

    def reset(self):
        self.iter_ = 0
        # check if self.target_ has a method getPriorSample, and if so, use it
        if hasattr(self.target_, 'getPriorSample'):
            self.state_ = self.target_.getPriorSample()
        else:
            self.state_ = getSample(self.target_.getDim(), rng=self.rng_)
        self.chain_ = np.zeros((self.n_samples_, self.dim_))
        self.chain_[0, :] = self.state_

    def get_sample_covariance(self):
        return np.cov(self.chain_.transpose())

    def set_preconditioner(self, cov):
        self.prec_ = cov
        self.precL_ = np.linalg.cholesky(self.prec_)

    def step(self):
        self.iter_ += 1
        proposal = self.state_ + self.sigma_ * self.precL_ @ self.proposalDist_.getSample()
        logDensityProp = self.target_.logDensity(proposal)
        logDensityCurrent = self.target_.logDensity(self.state_)

        if (logDensityProp - logDensityCurrent) > np.log(self.rng_.uniform()):
            self.state_ = proposal
            self.n_accept_ += 1
            self.n_accept_last_ += 1

        self.chain_[self.iter_, :] = self.state_

    def run(self):
        for i in range(1, self.n_samples_):

            if i == 1000:
                self.set_preconditioner(20 * self.get_sample_covariance())

            if i % self.rescale_interval_ == 0:
                print(f"Iteration: {i}")
                print(f"Acceptance rate: {self.n_accept_last_ / self.rescale_interval_}")
                if self.n_accept_last_ / self.rescale_interval_ < 0.2:
                    self.set_preconditioner(0.9 * self.prec_)
                    print("Decrease")
                elif self.n_accept_last_ / self.rescale_interval_ > 0.25:
                    self.set_preconditioner(1.1 * self.prec_)
                    print("Increase")
                self.n_accept_last_ = 0
            self.step()
        print(f"Total acceptance rate: {self.n_accept_ / self.n_samples_}")


class LangevinDynamicsSampler:

    def __init__(self, target, sigma=0.5, n_samples=10000, adjusted=True,
                 prec=None, rng=None, seed=None):
        self.target_ = target
        self.dim_ = self.target_.getDim()
        self.sigma_ = sigma
        self.n_samples_ = n_samples
        self.adjusted_ = adjusted
        self.n_accept_ = 0
        self.n_accept_last_ = 0
        self.rescale_interval_ = 100

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        self.proposalDist_ = MultivariateNormal(np.zeros(self.dim_),
                                                np.eye(self.dim_),
                                                rng=self.rng_)

        if prec is None:
            self.prec_ = np.eye(self.dim_)
        else:
            assert np.any(prec.shape != self.dim_) and (len(prec.shape) == 2)
            self.prec_ = prec

        self.precL_ = np.linalg.cholesky(self.prec_)

        self.reset()

    def reset(self):
        self.iter_ = 1
        self.state_ = getSample(self.target_.getDim(), rng=self.rng_)
        self.state_ = np.zeros_like(self.state_)
        self.state_ = np.array([0.5, 0.4])
        self.chain_ = np.zeros((self.n_samples_, self.dim_))
        self.chain_[0, :] = self.state_
        self.logDensity_ = self.target_.logDensity(self.state_)
        self.gradLogDensity_ = self.target_.gradLogDensity(self.state_)

    def logProposalDensity(self, y, x, grad_x):
        diff = y - x - 0.5 * self.sigma_ ** 2 * self.prec_ @ grad_x
        return - 0.5 * diff @ sp.linalg.solve(self.sigma_ ** 2 * self.prec_, diff)

    def step(self):
        self.iter_ += 1
        self.randn_ = self.proposalDist_.getSample()
        # self.randn_ = np.array([0.8037, -1.715])
        prop = (self.state_ + self.sigma_ * self.precL_ @ self.randn_
                + 0.5 * self.sigma_ ** 2 * self.prec_ @ self.gradLogDensity_)
        logDensityProp = self.target_.logDensity(prop)
        gradLogDensityProp = self.target_.gradLogDensity(prop)
        logNumerator = logDensityProp + self.logProposalDensity(self.state_, prop, gradLogDensityProp)
        logDenominator = self.logDensity_ + self.logProposalDensity(prop, self.state_, self.gradLogDensity_)

        if not self.adjusted_ or ((logNumerator - logDenominator) > np.log(self.rng_.uniform())):
            self.state_ = prop
            self.chain_[self.iter_, :] = prop
            self.logDensity_ = logDensityProp
            self.gradLogDensity_ = gradLogDensityProp
            self.n_accept_ += 1
            self.n_accept_last_ += 1
        else:
            self.chain_[self.iter_, :] = self.state_

    def get_sample_covariance(self):
        return np.cov(self.chain_.transpose())

    def set_preconditioner(self, cov):
        self.prec_ = cov
        self.precL_ = np.linalg.cholesky(self.prec_)

    def run(self):
        for i in range(1, self.n_samples_ - 1):
            if i == 1000:
                self.set_preconditioner(20 * self.get_sample_covariance())

            if i % self.rescale_interval_ == 0:
                print(f"Iteration: {i}")
                print(f"Acceptance rate: {self.n_accept_last_ / self.rescale_interval_}")
                if self.n_accept_last_ / self.rescale_interval_ < 0.5:
                    self.set_preconditioner(0.9 * self.prec_)
                    print("Decrease")
                elif self.n_accept_last_ / self.rescale_interval_ > 0.6:
                    self.set_preconditioner(1.1 * self.prec_)
                    print("Increase")
                self.n_accept_last_ = 0
            self.step()
            # if i % 1000 == 0:
            #     print(f"Iteration: {i}")
        print(f"Acceptance rate: {self.n_accept_ / self.n_samples_}")


class HamiltonianMonteCarlo:

    def __init__(self, target, step_scale=0.1, leap_frog_steps=20, n_samples=10000, scaling=None,
                 path_length=2.0, rng=None, seed=None, plot=False, plot_limits=None, plot_joint=False):
        self.iter_ = None
        self.chain_ = None
        self.state_ = None
        self.target_ = target
        self.dim_ = self.target_.getDim()
        self.step_scale_ = step_scale
        self.leap_frog_steps_ = leap_frog_steps
        self.n_samples_ = n_samples
        self.path_length_ = path_length
        self.n_accepted_ = 1
        self.p_old_ = 0.0

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        self.iter_ = None
        self.chain_ = None
        self.state_ = None

        self.reset()

        if scaling is None:
            self.scaling_ = MultivariateNormal(np.zeros(self.dim_), np.eye(self.dim_), rng=self.rng_)
        elif (len(scaling.shape) == 2) and np.any(scaling.shape != self.dim_):
            self.scaling_ = MultivariateNormal(np.zeros(self.dim_), scaling, rng=self.rng_)
        elif len(scaling.shape) == 1 and len(scaling) == self.dim_:
            self.scaling_ = MultivariateNormal(np.zeros(self.dim_), np.diagonal(scaling), rng=self.rng_)
        else:
            raise Exception("No valid scaling specified.")

        self.M_ = self.scaling_.getCov()
        self.M_inv_ = np.linalg.inv(self.M_)

        if plot and (self.dim_ == 1 or self.dim_ == 2):
            self.plot_ = True
            # self.fig_, self.ax_ = plt.subplots()
            self.fig_ = sns.JointGrid()
            self.ax_ = self.fig_.ax_joint

            if plot_limits is None:
                x_lim = [-2, 2]
                y_lim = [-2, 2]
            else:
                x_lim = plot_limits[0]
                y_lim = plot_limits[1]
            x = np.linspace(x_lim[0], x_lim[1], 100)
            y = np.linspace(y_lim[0], y_lim[1], 100)
            self.gx_, self.gy_ = np.meshgrid(x, y)
            self.gz_ = np.zeros_like(self.gx_)

            for i in range(self.gx_.shape[0]):
                for j in range(self.gx_.shape[1]):
                    if self.dim_ == 2:
                        self.gz_[i, j] = np.exp(self.target_.logDensity(np.array([self.gx_[i, j],
                                                                                  self.gy_[i, j]])))
                    else:
                        self.gz_[i, j] = np.exp(self.target_.logDensity(np.array([self.gx_[i, j]])) +
                                                self.scaling_.logDensity(np.array([self.gy_[i, j]])))

            # self.ax_.axis('equal')
            # self.ax_.set_xlim([x_lim[0], x_lim[1]])
            # self.ax_.set_ylim([y_lim[0], y_lim[1]])

            self.ax_.contour(self.gx_, self.gy_, self.gz_, levels=20, zorder=1, alpha=0.3, linewidths=1.5)
            self.fig_.ax_marg_x.plot(self.gx_[0], self.gz_[0])
            self.fig_.ax_marg_y.plot(self.gz_[:, 0], self.gy_[:, 0])

            if self.dim_ == 1:
                self.ax_.set_xlabel(r"$p(\theta)$")
                self.ax_.set_ylabel(r"$p(\mathrm{p})$")
            else:
                self.ax_.set_xlabel(r"$p(\theta_1)$")
                self.ax_.set_ylabel(r"$p(\theta_2)$")
            self.fig_.fig.tight_layout()
            self.fig_.fig.show()

        else:
            self.plot_ = False

    def reset(self):
        self.iter_ = 1
        self.state_ = np.zeros(self.dim_) + 0.5
        self.state_ = np.array([1.])
        self.chain_ = np.empty((self.n_samples_, self.dim_))
        self.chain_[0, :] = self.state_

    def leap_frog_step_(self, p0: np.ndarray, q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = p0 + self.step_scale_ * 0.5 * self.target_.gradLogDensity(q0)
        q = q0 + self.step_scale_ * self.M_inv_ @ p
        p = p + self.step_scale_ * 0.5 * self.target_.gradLogDensity(q)
        return p, q

    def get_hamiltonian(self, p: np.ndarray, q: np.ndarray) -> float:
        potential = - self.target_.logDensity(q)
        kinetic = 0.5 * p @ self.M_inv_ @ p
        kinetic = kinetic + 0.5 * np.log((2. * np.pi) ** self.dim_ * np.linalg.det(self.M_))

        return potential + kinetic

    def step_(self):

        p_hist = np.empty((self.leap_frog_steps_ + 1, self.dim_))
        q_hist = np.empty((self.leap_frog_steps_ + 1, self.dim_))

        p_hist[0, :] = p0 = self.scaling_.getSample()
        q_hist[0, :] = q0 = self.state_

        hamiltonian0 = self.get_hamiltonian(p0, q0)

        for i in range(1, self.leap_frog_steps_ + 1):
            p_hist[i, :], q_hist[i, :] = self.leap_frog_step_(p_hist[i - 1, :], q_hist[i - 1, :])

        hamiltonian = self.get_hamiltonian(p_hist[-1, :], q_hist[-1, :])

        accept = (- hamiltonian + hamiltonian0) > np.log(self.rng_.uniform())
        if accept:
            self.state_ = q_hist[-1, :]
            self.n_accepted_ += 1
            color = 'g'
        else:
            color = 'r'

        if self.plot_:
            if self.dim_ == 2:
                # self.ax_.plot(*q_hist.transpose(), marker=".", color=color)
                self.ax_.plot(*q_hist.transpose(), marker=".")
            else:
                # self.ax_.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", color=color, markersize=2.)
                aw = 0.01
                # self.ax_.arrow(q0[0], self.p_old_, 0, p0[0] - self.p_old_, alpha=0.5,
                #                color='k', width=aw, length_includes_head=True, head_width=7*aw)
                # self.ax_.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4.)
                self.ax_.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4., color=color)

                if accept:
                    self.p_old_ = p_hist[-1, 0]

        self.chain_[self.iter_, :] = self.state_
        self.iter_ += 1

    def run_(self):
        for i in range(1, self.n_samples_):
            self.step_()
            if self.plot_:
                print(f"Iteration: {i}")

                # self.fig_.fig.show()
            elif i % 1000 == 0:
                print(f"Iteration: {i}")
        print(f"Acceptance rate: {self.n_accepted_ / self.n_samples_}")

        # if self.plot_:
        # self.fig_.fig.show()
        # self.fig_.savefig(f"plots/hmc_2d_l2.pdf")


class ZigZagSampler:

    def __init__(self, target, n_events=1000, gamma=0.01, rng=None, seed=None, approximation=None,
                 sub_sampling=False, **kwargs):
        self.target_ = target
        self.dim_ = self.target_.getDim()
        self.n_obs_ = self.target_.get_n_obs()
        self.n_events_ = n_events
        self.times_ = np.zeros(self.n_events_)
        self.positions_ = np.zeros((self.n_events_, self.dim_))
        self.velocities_ = np.zeros((self.n_events_, self.dim_))
        self.iter_ = 0
        self.gamma_ = gamma
        self.offset_ = 0.
        self.plot_ = False
        self.approximation_ = approximation
        self.thinning_ = False
        self.sub_sampling_ = sub_sampling

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        if 'x0' in kwargs:
            self.positions_[0] = kwargs['x0']
        elif hasattr(self.target_, 'getPriorSample'):
            self.positions_[0] = self.target_.getPriorSample()

        # draw initial velocity from binomial distribution
        self.velocities_[0] = 2 * self.rng_.binomial(1, 0.5, self.dim_) - 1

        if hasattr(self.target_, 'getBounds'):
            pass
        elif self.approximation_ is not None:
            self.thinning_ = True
            self.generate_event_times = self.cinlars_method_linear
        else:
            self.generate_event_times = self.cinlars_method
        if 'dt' in kwargs:
            self.dt_ = kwargs['dt']
        else:
            self.dt_ = 0.01

        if 'plot' in kwargs:
            self.plot_ = kwargs['plot']
            self.ax_ = kwargs['ax']

    def rates(self, x, idx_d=None, idx_n=None):
        if idx_d is None:
            return np.maximum(-self.target_.gradLogDensity(x, idx_n) * self.velocities_[self.iter_], 0) + self.gamma_
        else:
            return np.maximum(0, -self.target_.gradLogDensity(x, idx_n, sub_sampling=self.sub_sampling_)[idx_d] * self.velocities_[self.iter_, idx_d]) + self.gamma_

    def cinlars_method(self):
        s = -np.log(self.rng_.uniform(0, 1, self.dim_))
        taus = np.zeros(self.dim_)

        j = None

        if self.sub_sampling_:
            j = self.rng_.integers(self.n_obs_)

        # print(f"Sampling likelihood component {j}")

        for i in range(self.dim_):

            # print(f"Sampling dimension {i}")
            integral = 0.

            rate_t0 = self.rates(self.positions_[self.iter_], idx_d=i, idx_n=j)

            while integral < s[i]:
                # rate_t0 = self.rates(self.positions_[self.iter_] + taus[i] * self.velocities_[self.iter_],
                #                      idx_d=i, idx_n=j)
                rate_t1 = self.rates(self.positions_[self.iter_] + (taus[i] + self.dt_) * self.velocities_[self.iter_],
                                     idx_d=i, idx_n=j)

                # rate_t0_full = self.rates(self.positions_[self.iter_] + taus[i] * self.velocities_[self.iter_],
                #                      idx_d=i)

                # print(f"  rate_t0: {rate_t0:.5f}   rate_t0 full:{rate_t0_full:.5f}")
                integral += np.trapz(np.array([rate_t0, rate_t1]), dx=self.dt_)
                taus[i] += self.dt_
                # print(f"      tau: {taus[i]:.5f},  integral: {integral:.5f}")
                rate_t0 = rate_t1

        # print(f"Initial s: {s}")
        # print(f"Final taus: {taus}")
        return taus

    def cinlars_method_linear(self):
        s = -np.log(self.rng_.uniform(0, 1, self.dim_))

        a = (np.abs(self.approximation_['inv_cov'])
             @ np.abs((self.positions_[self.iter_] - self.approximation_['mean'])) + self.gamma_ + self.offset_)
        b = np.sum(np.abs(self.approximation_['inv_cov']), axis=1)

        return np.divide((np.sqrt(a ** 2 + 2 * b * s) - a), b, where=b != 0)

    def approximate_rates(self, x, idx=None):

        if idx is None:
            rates = self.velocities_[self.iter_] * (self.approximation_['inv_cov']
                                                    @ (x - self.approximation_['mean']))
            return np.maximum(rates, 0) + self.gamma_ + self.offset_
        else:
            rate = self.velocities_[self.iter_, idx] * (self.approximation_['inv_cov'][idx]
                                                        @ (x - self.approximation_['mean'][idx]))
            return np.maximum(rate, 0) + self.gamma_ + self.offset_

    def poisson_thinning(self, j, T):
        m = self.rates(self.positions_[self.iter_] + T * self.velocities_[self.iter_], idx=j)
        M = self.approximate_rates(self.positions_[self.iter_] + T * self.velocities_[self.iter_], idx=j)
        u = self.rng_.uniform(0, 1)
        if u < m / M:
            self.velocities_[self.iter_ + 1, j] = -self.velocities_[self.iter_, j]

        if m > M:
            print(f"upper bound too tight, m: {m:.5f}, M: {M:.5f}"
                   "\n ...increasing offset")
            self.offset_ += m - M

    def step(self):
        taus = self.generate_event_times()
        j = np.argmin(taus)
        T = taus[j]
        self.times_[self.iter_ + 1] = self.times_[self.iter_] + T
        self.positions_[self.iter_ + 1] = self.positions_[self.iter_] + T * self.velocities_[self.iter_]
        self.velocities_[self.iter_ + 1] = self.velocities_[self.iter_]
        self.velocities_[self.iter_ + 1, j] = - self.velocities_[self.iter_, j]

        if self.thinning_:
            self.poisson_thinning(j, T)

        self.iter_ += 1

    def run(self):
        for i in range(1, self.n_events_):
            if i % 50 == 0:
                print(f"Sampling event {i}")
            self.step()

        print(f"Done")


if __name__ == '__main__':

    #  # log normal 2d
    #  rng = np.random.default_rng(0)
    #  mean, cov = np.array([0, 0]), np.diag([1,1.5])
    #  posterior = MultivariateLogNormal(mean, cov, rng=rng)
    #  plot_limits = ([0.05, 4], [0.05, 4])

    # normal 2d
    rng = np.random.default_rng(0)
    mean, cov = np.array([0, 0]), np.array([[1, 0.3], [0.3, 1.]])
    posterior = MultivariateNormal(mean, cov, rng=rng)
    plot_limits = ([-3, 3], [-3, 3])

    # # MALA
    n_samples = 20000
    # MALA = LangevinDynamicsSampler(posterior, n_samples=n_samples, sigma=np.sqrt(1.5), adjusted=True, rng=rng,
    #                                prec=cov)
    Sampler = MetropolisHastingsSampler(posterior, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                        prec=cov)
    Sampler.run()
    samples = Sampler.chain_
    n_vis = 500

    x = np.linspace(plot_limits[0][0], plot_limits[0][1], 100)
    y = np.linspace(plot_limits[1][0], plot_limits[1][1], 70)
    gx, gy = np.meshgrid(x, y)
    gz = np.zeros_like(gx)
    grad_z = np.zeros((gx.shape[0], gx.shape[1], 2))

    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            gz[i, j] = np.exp(posterior.logDensity(np.array([gx[i, j], gy[i, j]])))
            grad_z[i, j, :] = posterior.gradLogDensity(np.array([gx[i, j], gy[i, j]]))

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlim(plot_limits[0][0], plot_limits[0][1])
    ax.set_ylim(plot_limits[1][0], plot_limits[1][1])
    # ax.quiver(gx.flatten(),gy.flatten(),grad_z[:,:,0].flatten(), grad_z[:,:,1].flatten())
    # fig.show()

    ax.contour(gx, gy, gz, levels=20, zorder=1, alpha=0.6)
    # ax.contour(gx, gy, gz, levels=levels, zorder=1)
    samples_plot = samples[0::n_samples // n_vis]
    # ax.scatter(*samples[2000::n_samples//n_vis].transpose(), s=3, zorder=2, c=np.linspace(0, 1, n_vis))
    ax.scatter(*samples_plot.transpose(), s=3, zorder=2, c=np.linspace(0, 1, samples_plot.shape[0]))
    # ax.scatter(*SPN.transpose(), s=5)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(samples[:4000, 0])
    fig.show()

    print(np.mean(samples, axis=0))
    print(np.cov(samples.transpose()))
