import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from pdmp.distributions import Distribution, MultivariateNormal
from pdmp import logger

from pdmp.utils import get_2d_despined_figure, plot_pdf_contours


class ZigZagSampler:
    """
    ZigZagSampler class for sampling from a target distribution using the ZigZag process.
    """

    def __init__(self,
                 target: Distribution,
                 n_max: int = None,
                 t_max: float = None,
                 gamma: float = 1e-6,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 approximation: dict = None,
                 sub_sampling: bool = False,
                 n_events_accepted: int = None,
                 verbose: int = 1,
                 print_every: int = 100,
                 **kwargs):
        """
        Initialize the ZigZagSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        n_events (int, optional): The number of events to sample. Default is 1000.
        gamma (float, optional): The refresh rate parameter for the ZigZag process. Default is 1e-6 to avoid zero
            division.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        approximation (dict, optional): Approximation parameters. Default is None.
        sub_sampling (bool, optional): Whether to use sub-sampling. Default is False.
        n_events_accepted (int, optional): Number of accepted events. Default is None.
        verbose (int, optional): Level of verbosity: 0 is none, 1 is major, 2 is all outputs.
        print_every (int, optional): Interval to print outputs.
        kwargs: Additional keyword arguments.
        """

        self.target_ = target
        self.dim_ = self.target_.get_dim()
        # self.n_obs_ = self.target_.get_n_obs()

        if n_max is not None:
            self.n_max_ = n_max
            self.run = self.run_budget

        # make very large skeleton if algorithm is run with time limit
        if t_max is not None:
            self.t_max_ = float(t_max)
            self.n_max_ = 100000
            n_events_accepted = self.n_max_
            self.run = self.run_time

        if n_max is None and t_max is None:
            self.n_max_ = 1000
            self.run = self.run_budget

        self.times_ = np.zeros(self.n_max_)
        self.positions_ = np.zeros((self.n_max_, self.dim_))
        self.velocities_ = np.zeros((self.n_max_, self.dim_))
        self.iter_ = 0
        self.gamma_ = gamma
        self.offset_ = 0.
        self.plot_ = False
        self.approximation_ = approximation
        self.thinning_ = False
        self.n_accepted_ = 0
        self.n_accepted_0_ = n_events_accepted
        self.accepted_iters_ = np.zeros(self.n_accepted_0_, dtype=int)
        self.sub_sampling_ = sub_sampling
        self.verbose_ = verbose
        self.print_every_ = print_every

        # if 'ss' in kwargs:
        #     self.ss_ = kwargs['ss']
        #
        # if 'us' in kwargs:
        #     self.us_ = kwargs['us']

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        if 'x_0' in kwargs:
            self.positions_[0] = kwargs['x_0']
        elif hasattr(self.target_, 'get_prior_sample'):
            self.positions_[0] = self.target_.get_prior_sample()

        # draw initial velocity from binomial distribution
        if 'v_0' in kwargs:
            self.velocities_[0] = kwargs['v_0']
        else:
            self.velocities_[0] = 2 * self.rng_.binomial(1, 0.5, self.dim_) - 1

        if hasattr(self.target_, 'get_bounds'):
            pass
        elif self.approximation_ is not None:
            self.thinning_ = True
            self.generate_event_times = self.inverse_cdf_linear
        else:
            self.generate_event_times = self.inverse_cdf
        if 'dt' in kwargs:
            self.dt_ = kwargs['dt']
        else:
            self.dt_ = 0.01

        if 'plot' in kwargs:
            self.plot_ = kwargs['plot']
            self.ax_ = kwargs['ax']

        logger.info("ZigZagSampler initialized.")

    def rates(self, x: np.ndarray, idx_d: int = None, idx_n: int = None) -> np.ndarray:
        """
        Calculate the rates for the ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        idx_d (int, optional): Dimension index. Default is None.
        idx_n (int, optional): Observation index. Default is None.

        Returns:
        np.ndarray: The calculated rates.
        """
        if idx_d is None:
            return np.maximum(-self.target_.grad_log_density(x) * self.velocities_[self.iter_], 0) + self.gamma_
        else:
            return np.maximum(0, -self.target_.grad_log_density(x)[idx_d] * self.velocities_[self.iter_, idx_d]) + self.gamma_

    def inverse_cdf(self) -> tuple[np.ndarray, int]:
        """
        Generate event times using the inverse cdf method.

        Returns:
        np.ndarray: The generated event times.
        """
        s = -np.log(self.rng_.uniform(0, 1, self.dim_))
        taus = np.zeros(self.dim_)

        j = None

        # if self.sub_sampling_:
        #     j = self.rng_.integers(self.n_obs_)
        #
        # print(f"Sampling likelihood component {j}")

        integral = np.zeros(self.dim_)
        rate_t0 = self.rates(self.positions_[self.iter_], idx_n=j)
        rate_t1 = np.zeros_like(rate_t0)

        # advance all process until one reaches s
        while np.all(integral < s):
            rate_t1 = self.rates(self.positions_[self.iter_] + (taus + self.dt_) * self.velocities_[self.iter_], idx_n=j)
            integral += np.trapezoid(np.array([rate_t0, rate_t1]), dx=self.dt_, axis=0)
            taus += self.dt_
            rate_t0 = rate_t1

        # linear correction to last step
        taus -= (integral - s) / rate_t1

        logger.debug(f"S    : {s}")
        logger.debug(f"taus : {taus}")

        i = np.argmin(taus)
        return taus[i], i

    def inverse_cdf_linear(self) -> tuple[np.ndarray, int]:
        """
        Generate event times using the inverse cdf method assuming b linear rate function.

        Returns:
        np.ndarray: The generated event times.
        """

        # get samples from the CDF
        S = -np.log(self.rng_.uniform(0, 1, self.dim_))
        # S = self.ss_[self.iter_]
        logger.debug(f"S:    {S}")

        # get the linear approximation of the rates
        a = self.velocities_[self.iter_] * (self.approximation_['inv_cov'] @ self.velocities_[self.iter_])
        b = (self.velocities_[self.iter_] *
             (self.approximation_['inv_cov'] @ (self.positions_[self.iter_] - self.approximation_['mean']))
             + self.offset_)

        # init variables
        s = np.zeros(self.dim_)
        taus = np.zeros(self.dim_)

        # compute root
        taus_0 = - b / a

        # check for each component where the intersection with the x-axis is and compute integral accordingly
        for i in range(self.dim_):

            if (a[i] > 0) and (b[i] > 0):
                b_i = b[i] + self.gamma_
                taus[i] = (np.sqrt(b_i ** 2 + 2 * a[i] * S[i]) - b_i) / a[i]
            elif (a[i] > 0) and (b[i] < 0):
                taus_const = S[i] / self.gamma_
                if taus_const < taus_0[i]:
                    taus[i] = taus_const
                else:
                    s[i] += taus_0[i] * self.gamma_
                    d_s = S[i] - s[i]
                    b_i = b[i] + self.gamma_
                    taus[i] = taus_0[i] + (np.sqrt((b_i + a[i] * taus_0[i]) ** 2 + 2 * a[i] * d_s)
                                           - (b_i + a[i] * taus_0[i])) / a[i]
            elif (a[i] < 0) and (b[i] > 0):
                s_0 = (0.5 * b[i] + self.gamma_) * taus_0[i]
                if S[i] > s_0:
                    taus[i] = taus_0[i] + (S[i] - s_0) / self.gamma_
                else:
                    taus[i] = (np.sqrt((b[i] + self.gamma_) ** 2 + 2 * a[i] * S[i])
                               - (b[i] + self.gamma_)) / a[i]
            else:
                taus[i] = S[i] / self.gamma_

        logger.debug(f"taus: {taus}")

        j = np.argmin(taus)
        return taus[j], j

    def approximate_rates(self, x: np.ndarray, idx=None) -> np.ndarray:
        """
        Calculate the approximate rates for the ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        idx (int, optional): Dimension index. Default is None.

        Returns:
        np.ndarray: The calculated approximate rates.
        """
        if idx is None:
            rates = self.velocities_[self.iter_] * (self.approximation_['inv_cov']
                                                    @ (x - self.approximation_['mean'])) + self.offset_
            # return np.maximum(rates, 0) + self.gamma_ + self.offset_
            return np.maximum(rates, 0) + self.gamma_
        else:
            rate = self.velocities_[self.iter_, idx] * (self.approximation_['inv_cov'][idx]
                                                        @ (x - self.approximation_['mean'])) + self.offset_
            # return np.maximum(rate, 0) + self.gamma_ + self.offset_
            return np.maximum(rate, 0) + self.gamma_

    def approximate_bounds(self, x: np.ndarray, T: np.ndarray, idx=None) -> np.ndarray:
        """
        Calculate the approximate bounds for the ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        T (np.ndarray): The time increment.
        idx (int, optional): Dimension index. Default is None.

        Returns:
        np.ndarray: The calculated approximate bounds.
        """

        a = self.velocities_[self.iter_] * (self.approximation_['inv_cov'] @ self.velocities_[self.iter_])
        b = (self.velocities_[self.iter_] *
             (self.approximation_['inv_cov'] @ (self.positions_[self.iter_] - self.approximation_['mean']))
             + self.offset_)

        rates = a * T + b
        rates = np.maximum(rates, 0) + self.gamma_

        if idx is None:
            return rates
        else:
            return rates[idx]

    def poisson_thinning(self, j: int, T: np.ndarray):
        """
        Perform Poisson thinning for the ZigZag process.

        Parameters:
        j (int): Dimension index.
        T (np.ndarray): Time increment.
        """
        m = self.rates(self.positions_[self.iter_] + T * self.velocities_[self.iter_], idx_d=j)
        # m = self.approximate_rates(self.positions_[self.iter_] + T * self.velocities_[self.iter_], idx=j)
        M = self.approximate_bounds(self.positions_[self.iter_], T, idx=j)
        u = self.rng_.uniform(0, 1)
        # u = self.us_[self.iter_]
        logger.debug(f"      u:    {u}")
        if u < (m / M):
            self.velocities_[self.iter_ + 1, j] = -self.velocities_[self.iter_, j]
            self.n_accepted_ += 1
            self.accepted_iters_[self.n_accepted_] = self.iter_ + 1
        else:
            self.velocities_[self.iter_ + 1, j] = self.velocities_[self.iter_, j]

        logger.debug(f"      ratio: {m/M}")

        if m > M:
            self.offset_ += m - M
            self.revert_step()
            logger.info(f"  Action at time {self.times_[self.iter_]:.2f}; current position: {self.positions_[self.iter_]}")
            logger.info(f"     upper bound too tight, m: {m:.4f}, M: {M:.4f}")
            logger.info(f"      ...increasing offset to: {self.offset_:.4f}")

    def step(self):
        """
        Perform a single ZigZag step.
        """
        T, j = self.generate_event_times()
        self.times_[self.iter_ + 1] = self.times_[self.iter_] + T
        self.positions_[self.iter_ + 1] = self.positions_[self.iter_] + T * self.velocities_[self.iter_]
        self.velocities_[self.iter_ + 1] = self.velocities_[self.iter_]
        self.velocities_[self.iter_ + 1, j] = - self.velocities_[self.iter_, j]

        if self.thinning_:
            self.poisson_thinning(j, T)

        self.iter_ += 1

    def revert_step(self):
        """
        Revert the last ZigZag step.
        """
        self.times_[self.iter_ + 1] = 0.
        self.positions_[self.iter_ + 1] = 0.
        self.velocities_[self.iter_ + 1] = 0.
        self.accepted_iters_[self.n_accepted_] = 0
        self.n_accepted_ -= 1
        self.iter_ -= 1


    def shutdown(self):
        """
        Shutdown the ZigZag sampler.
        """

        logger.info("Shutting down ZigZag sampler. Summary:")
        if self.thinning_:
            logger.info(f"    Acceptance rate : {self.n_accepted_ / self.iter_:.3f}")
            logger.info(f"    Final offset    : {self.offset_:.3f}")

            idx = self.n_accepted_ + 1
            self.positions_ = self.positions_[self.accepted_iters_[:idx]]
            self.times_ = self.times_[self.accepted_iters_[:idx]]
            self.velocities_ = self.velocities_[self.accepted_iters_[:idx]]

        logger.info("Run successfully completed.")

    def run_budget(self):
        """
        Run the ZigZag sampler.
        """

        with tqdm(total=self.n_max_) as pbar:
            for i in range(1, self.n_max_):
                if i % self.print_every_ == 0:
                    pbar.clear()
                    logger.debug(f"Sampling event {i}")
                    pbar.refresh()
                self.step()
                pbar.update()
                if self.thinning_:
                    if self.n_accepted_ == self.n_accepted_0_ - 1:
                        break

        self.shutdown()

    def run_time(self):
        """
        Run the ZigZag sampler.
        """

        time = 0.
        with tqdm(total=self.t_max_, leave=True, bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:

            while self.times_[self.iter_] < self.t_max_:
                if self.iter_ % self.print_every_ == 0:
                    pbar.clear()
                    logger.debug(f"Sampling event {self.iter_}")
                    pbar.refresh()
                self.step()
                incr = np.min((self.t_max_, self.times_[max(0, self.iter_)])) - time
                time = self.times_[max(0, self.iter_)]
                pbar.update(incr)

        # remove empty skeleton
        self.times_ = self.times_[:self.iter_ + 1]
        self.positions_ = self.positions_[:self.iter_ + 1]
        self.velocities_ = self.velocities_[:self.iter_ + 1]

        self.shutdown()

    def write_data(self, folder: str):
        """
        Write the chain data to a file.

        Parameters:
        filename (str): The name of the folder to write the data to.
        """

        data = {}
        if self.thinning_:
            data['acceptance_rate'] = self.n_accepted_ / self.n_max_

        with open(os.path.join(folder, 'other.pkl'), 'wb') as f:
            pickle.dump(data, f)

        np.savetxt(os.path.join(folder, 'positions.dat'), self.positions_)
        np.savetxt(os.path.join(folder, 'times.dat'), self.times_)
        np.savetxt(os.path.join(folder, 'velocities.dat'), self.velocities_)

if __name__ == '__main__':

    # normal 2d
    rng = np.random.default_rng(0)
    mean, cov = np.array([0, 0]), np.array([[1, 0.3], [0.3, 1.]])
    posterior = MultivariateNormal(mean, cov, rng=rng)
    plot_limits = ([-3.5, 3.5], [-3.5, 3.5])

    # zig-zag
    t_max = 500
    zig_zag = ZigZagSampler(posterior, t_max=500)
    zig_zag.run()

    positions = zig_zag.positions_

    fig, ax = get_2d_despined_figure(plot_limits=plot_limits, figsize=(4, 4), keep_ticks=True)
    plot_pdf_contours(posterior, ax, plot_limits=plot_limits)
    ax.plot(*positions.T)
    plt.show()
