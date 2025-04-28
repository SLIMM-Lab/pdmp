import pickle
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np

from typing import cast, Any

from tqdm import tqdm

from pdmp import logger
from pdmp.sampler import Sampler
from pdmp.distributions import Distribution, MultivariateNormal
from pdmp.surrogates import (
    SurrogateModel, LaplaceSurrogate, NeuralNetwork, GaussianProcess, DerivativeGaussianProcess, ConstantSurrogate,
    RandomConstantSurrogate
)
from pdmp.plotting import plot_pdf_contours
from pdmp.plotting_utils import get_2d_despined_figure


class ZigZagSampler(Sampler):
    """
    ZigZagSampler class for sampling from a target distribution using the ZigZag process.
    """

    def __init__(
            self,
            target: Distribution, *,
            surrogate: SurrogateModel = None,
            n_max: int = None,
            t_max: float = None,
            gamma: float = 1e-6,
            rng: np.random.Generator = None,
            seed: int = None,
            sub_sampling: bool = False,
            n_events_accepted: int = None,
            print_every: int = 100,
            update_bar_every: int = 10,
            offset_shrinkage: float = 0.0,
            **kwargs
    ):
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
        print_every (int, optional): Interval to print outputs.
        kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.target = target
        self._dim = self.target.get_dim()
        # self.n_obs_ = self.target.get_n_obs()

        if n_max is not None:
            self._n_max = n_max
            self.run = self._run_budget

        # make very large skeleton if algorithm is run with time limit
        if t_max is not None:
            self._t_max = float(t_max)
            self._n_max = 10000000 #TODO this might be too large, find a better way to handle this
            n_events_accepted = self._n_max
            self.run = self._run_time

        if n_max is None and t_max is None:
            self._n_max = 1000
            self.run = self._run_budget

        self.times = np.zeros(self._n_max)
        self.positions = np.zeros((self._n_max, self._dim))
        self.velocities = np.zeros((self._n_max, self._dim))
        self._iter = 0
        self._gamma = gamma
        self._offset = np.zeros(self._dim)
        self._offset_history = np.zeros((self._n_max, self._dim))
        self._thinning = False
        self._n_accepted = 0
        self._n_accepted_0 = n_events_accepted
        self._accepted_iters = np.zeros(self._n_accepted_0, dtype=int)
        self._sub_sampling = sub_sampling
        self._print_every = print_every
        self._update_bar_every = update_bar_every
        self._offset_shrinkage = offset_shrinkage

        # if 'ss' in kwargs:
        #     self.ss_ = kwargs['ss']
        #
        # if 'us' in kwargs:
        #     self.us_ = kwargs['us']

        if rng is None and seed is None:
            self._rng = np.random.default_rng(0)
        elif rng is None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = rng

        if 'x_0' in kwargs:
            self.positions[0] = kwargs['x_0']
        else:
            self.positions[0] = self._rng.normal(0, 1, self._dim)

        # draw initial velocity from binomial distribution
        if 'v_0' in kwargs:
            self.velocities[0] = kwargs['v_0']
        else:
            self.velocities[0] = 2 * self._rng.binomial(1, 0.5, self._dim) - 1

        if hasattr(self.target, 'get_bounds'):
            pass
        elif surrogate is not None:
            self._thinning = True
            self._s = None

            if isinstance(surrogate, LaplaceSurrogate):
                self.surrogate = cast(LaplaceSurrogate, surrogate)
                self._generate_event_times = self._inverse_cdf_linear

            if isinstance(surrogate, NeuralNetwork):
                self.surrogate = cast(NeuralNetwork, surrogate)
                self._generate_event_times =  self._inverse_cdf

            if isinstance(surrogate, GaussianProcess):
                self.surrogate = cast(GaussianProcess, surrogate)
                self._generate_event_times = self._inverse_cdf

            if isinstance(surrogate, DerivativeGaussianProcess):
                self.surrogate = cast(DerivativeGaussianProcess, surrogate)
                self._generate_event_times = self._inverse_cdf

            if isinstance(surrogate, ConstantSurrogate):
                self.surrogate = cast(ConstantSurrogate, surrogate)
                self._generate_event_times = self._inverse_cdf
                self._offset = np.ones_like(self._offset)

            if isinstance(surrogate, RandomConstantSurrogate):
                self.surrogate = cast(RandomConstantSurrogate, surrogate)
                self._generate_event_times = self._inverse_cdf
                self._offset = np.ones_like(self._offset)

            self._cdf_rates = self._surrogate_rates

            # for later use
            self._times_all = None
            self._eval_times = []

        else:
            self._generate_event_times = self._inverse_cdf
            self._cdf_rates = self._target_rates

        if 'dt' in kwargs:
            self._dt = kwargs['dt']
        else:
            self._dt = 0.001

        logger.info("ZigZagSampler initialized.")

    @classmethod
    def from_dict(
            cls,
            config: dict[str, Any],
            target: Distribution,
            surrogate: SurrogateModel = None,
            rng: np.random.Generator = None
    ):
        """
        Initialize the ZigZagSampler class from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        target (Distribution): The target distribution to sample from.
        rng (np.random.Generator, optional): Random number generator.
        surrogate (SurrogateModel, optional): The surrogate model. Default is None.

        Returns:
        ZigZagSampler: The initialized ZigZagSampler class.
        """

        return cls(target, surrogate=surrogate, rng=rng, **config)

    def _target_rates(self, x: np.ndarray, idx_d: int = None, idx_n: int = None) -> np.ndarray:
        """
        Calculate the rates for the target ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        idx_d (int, optional): Dimension index. Default is None.
        idx_n (int, optional): Observation index. Default is None.

        Returns:
        np.ndarray: The calculated rates.
        """

        grad = self.target.grad_log_density(x)
        log_p = self.target.log_density(x)

        if self._thinning:
            self.surrogate.add_data(x=x, y=log_p, dy_dx=grad)

        rates = np.maximum(- grad * self.velocities[self._iter], 0) + self._gamma

        if idx_d is None:
            return rates
        else:
            return rates[idx_d]

    def _surrogate_rates(self, x: np.ndarray, idx_d: int = None, idx_n: int = None) -> np.ndarray:
        """
        Calculate the surrogate rates for the surrogate ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        idx_d (int, optional): Dimension index. Default is None.
        idx_n (int, optional): Observation index. Default is None.

        Returns:
        np.ndarray: The calculated surrogate rates.
        """
        if idx_d is None:
            rates = - self.surrogate.grad(x) * self.velocities[self._iter] + self._offset
            return np.maximum(rates, 0) + self._gamma
        else:
            rate = - self.surrogate.grad(x, idx_d) * self.velocities[self._iter, idx_d] + self._offset[idx_d]
            return np.maximum(rate, 0) + self._gamma

    def _inverse_cdf(self) -> tuple[np.ndarray, int]:
        """
        Generate event times using the inverse cdf method.

        Returns:
        np.ndarray: The generated event times.
        """
        # recover rng from previous iteration in case of rejection
        if self._s is None:
            self._s = -np.log(self._rng.uniform(0, 1, self._dim))
        s = self._s

        taus = np.zeros(self._dim)

        j = None

        # if self._sub_sampling:
        #     j = self._rng.integers(self.n_obs_)
        #
        # print(f"Sampling likelihood component {j}")

        integral = np.zeros(self._dim)
        rate_t0 = self._cdf_rates(self.positions[self._iter], idx_n=j)
        rate_t1 = np.zeros_like(rate_t0)

        # advance all process until one reaches s
        while np.all(integral < s):
            rate_t1 = self._cdf_rates(
                self.positions[self._iter] + (taus + self._dt) * self.velocities[self._iter],
                idx_n=j
            )
            integral += np.trapezoid(np.array([rate_t0, rate_t1]), dx=self._dt, axis=0)
            taus += self._dt
            rate_t0 = rate_t1

        # linear correction to last step
        taus -= (integral - s) / rate_t1

        # logger.debug(f"S    : {s}")
        # logger.debug(f"taus : {taus}")

        i = np.argmin(taus)
        return taus[i], i

    def _inverse_cdf_linear(self) -> tuple[np.ndarray, int]:
        """
        Generate event times using the inverse cdf method assuming b linear rate function.

        Returns:
        np.ndarray: The generated event times.
        """

        # # get samples from the CDF
        # S = -np.log(self._rng.uniform(0, 1, self._dim))
        # recover rng from previous iteration in case of rejection
        if self._s is None:
            self._s = -np.log(self._rng.uniform(0, 1, self._dim))
        S = self._s

        # S = self.ss_[self._iter]
        # logger.debug(f"S:    {S}")

        # init variables
        s = np.zeros(self._dim)
        taus = S / (self._gamma + self._offset)

        # get the linear approximation of the rates
        a = self.velocities[self._iter] * (self.surrogate.gaussian.get_inv_cov() @ self.velocities[self._iter])
        b = (self.velocities[self._iter] *
             (self.surrogate.gaussian.get_inv_cov()
              @ (self.positions[self._iter] - self.surrogate.gaussian.get_mean()))
             + self._offset)

        # compute root
        taus_0 = - b / a

        # check for each component where the intersection with the x-axis is and compute integral accordingly
        for i in range(self._dim):

            if (a[i] >= 0) and (b[i] >= 0):
                b_i = b[i] + self._gamma
                taus[i] = (np.sqrt(b_i ** 2 + 2 * a[i] * S[i]) - b_i) / a[i]
            elif (a[i] >= 0) and (b[i] < 0):
                taus_const = S[i] / self._gamma
                if taus_const < taus_0[i]:
                    taus[i] = taus_const
                else:
                    s[i] += taus_0[i] * self._gamma
                    d_s = S[i] - s[i]
                    b_i = b[i] + self._gamma
                    taus[i] = taus_0[i] + (np.sqrt((b_i + a[i] * taus_0[i]) ** 2 + 2 * a[i] * d_s)
                                           - (b_i + a[i] * taus_0[i])) / a[i]
            elif (a[i] < 0) and (b[i] >= 0):
                s_0 = (0.5 * b[i] + self._gamma) * taus_0[i]
                if S[i] > s_0:
                    taus[i] = taus_0[i] + (S[i] - s_0) / self._gamma
                else:
                    taus[i] = (np.sqrt((b[i] + self._gamma) ** 2 + 2 * a[i] * S[i])
                               - (b[i] + self._gamma)) / a[i]
            else:
                taus[i] = S[i] / self._gamma

            # logger.debug(f"taus: {taus}")

        j = np.argmin(taus)
        return taus[j], j

    def _approximate_rates(self, x: np.ndarray, idx=None) -> np.ndarray:
        """
        Calculate the approximate rates for the ZigZag process.

        Parameters:
        x (np.ndarray): The current position.
        idx (int, optional): Dimension index. Default is None.

        Returns:
        np.ndarray: The calculated approximate rates.
        """

        if idx is None:
            rates = - self.velocities[self._iter] * self.surrogate.grad(x) + self._offset
            return np.maximum(rates, 0) + self._gamma
        else:
            rate = - self.velocities[self._iter, idx] * self.surrogate.grad(x, idx) + self._offset[idx]
            return np.maximum(rate, 0) + self._gamma

    def _poisson_thinning(self, j: int, T: np.ndarray):
        """
        Perform Poisson thinning for the ZigZag process.

        Parameters:
        j (int): Dimension index.
        T (np.ndarray): Time increment.
        """
        pos = self.positions[self._iter] + T * self.velocities[self._iter]
        m = self._target_rates(pos, idx_d=j)
        M = self._approximate_rates(pos, idx=j)
        u = self._rng.uniform(0, 1)
        # u = self.us_[self._iter]
        # logger.debug(f"      u:    {u}")

        # logger.debug(f"      ratio: {m/M}")

        if m > M:
            delta_offset = 1.01 * (m - M) + 1e-3
            self._offset[j] += delta_offset
            self._revert_step()
            logger.info(f"  Action at time {self.times[self._iter]:.2f}; current position: {self.positions[self._iter]}")
            logger.info(f"     upper bound too tight, m: {m:.4f}, M: {M:.4f}")
            logger.info(f"      ...increasing offset {j} by {delta_offset:.4e} to: {self._offset[j]:.4f}")
        elif u < (m / M):
            self.velocities[self._iter + 1, j] = -self.velocities[self._iter, j]
            self._n_accepted += 1
            self._accepted_iters[self._n_accepted] = self._iter + 1
            self._s = None
        else:
            self.velocities[self._iter + 1, j] = self.velocities[self._iter, j]
            self._s = None

    def _step(self):
        """
        Perform a single ZigZag step.
        """
        T, j = self._generate_event_times()
        self.times[self._iter + 1] = self.times[self._iter] + T
        self.positions[self._iter + 1] = self.positions[self._iter] + T * self.velocities[self._iter]
        self.velocities[self._iter + 1] = self.velocities[self._iter]
        self.velocities[self._iter + 1, j] = - self.velocities[self._iter, j]

        if self._thinning:
            self._eval_times.append(self.times[self._iter + 1])
            self._poisson_thinning(j, T)

        dt = self.times[self._iter + 1] - self.times[self._iter]
        self._offset *= np.exp(- self._offset_shrinkage * dt)
        self._offset_history[self._iter + 1] = self._offset
        self._iter += 1

    def _revert_step(self):
        """
        Revert the last ZigZag step.
        """
        self.times[self._iter + 1] = 0.
        self.positions[self._iter + 1] = 0.
        self.velocities[self._iter + 1] = 0
        self._offset_history[self._iter + 1] = 0.

        dt = self.times[self._iter] - self.times[self._iter - 1]
        self._offset *= np.exp(self._offset_shrinkage * dt)
        self._iter -= 1


    def _shutdown(self):
        """
        Shutdown the ZigZag sampler.
        """

        logger.info("Shutting down ZigZag sampler. Summary:")
        if self._thinning:
            logger.info(f"    Acceptance rate : {self.acceptance_rate:.3f}")
            logger.info(f"    Final offsets    : {self._offset}")

            # this is kept so that model evaluations could be tracked
            self._times_all = self.times

            idx = self._n_accepted + 1
            self.positions = self.positions[self._accepted_iters[:idx]]
            self.times = self.times[self._accepted_iters[:idx]]
            self.velocities = self.velocities[self._accepted_iters[:idx]]
            self._offset_history = self._offset_history[self._accepted_iters[:idx]]

        logger.info("Run successfully completed.")

    def _run_budget(self):
        """
        Run the ZigZag sampler.
        """
        logger.warning(f"Running ZigZag sampler with budget n_max={self._n_max}")

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=self._n_max, file=sys.stdout, dynamic_ncols=True, disable=disable_tqdm) as pbar:
            for i in range(1, self._n_max):
                if i % self._print_every == 0:
                    pbar.clear()
                    logger.debug(f"Sampling event {i}")
                    pbar.refresh()
                self._step()
                if self._thinning:
                    if self._n_accepted == self._n_accepted_0 - 1:
                        break

        self._shutdown()

    def _run_time(self):
        """
        Run the ZigZag sampler.
        """

        logger.warning(f"Running ZigZag sampler with time limit T={self._t_max}")
        time = 0.

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(
            total=self._t_max,
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            disable=disable_tqdm
        ) as pbar:

            while self.times[self._iter] < self._t_max:
                if self._iter % self._print_every == 0:
                    pbar.clear()
                    logger.debug(f"Sampling event {self._iter}")
                    pbar.refresh()
                self._step()
                if self._iter % self._update_bar_every == 0:
                    incr = np.min((self._t_max, self.times[max(0, self._iter)])) - time
                    time = self.times[max(0, self._iter)]
                    pbar.update(incr)

        # remove empty skeleton
        self.times = self.times[:self._iter + 1]
        self.positions = self.positions[:self._iter + 1]
        self.velocities = self.velocities[:self._iter + 1]

        self._shutdown()

    def write_data(self, folder: str, precision: int = 6):
        """
        Write the chain data to a file.

        Parameters:
        filename (str): The name of the folder to write the data to.
        precision (int, optional): The precision of the data. Default is 6.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        data = {}
        if self._thinning:
            data['acceptance_rate'] = self.acceptance_rate
            data['offset'] = self._offset

        with open(os.path.join(folder, 'other.pkl'), 'wb') as f:
            pickle.dump(data, f)

        np.savetxt(os.path.join(folder, 'positions.dat'), self.positions, fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'times.dat'), self.times, fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'velocities.dat'), self.velocities, fmt='%d')
        np.savetxt(os.path.join(folder, 'times_all.dat'), self._times_all, fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'offset_history.dat'), self._offset_history, fmt=f'%.{precision}e')

        self._eval_times.sort()
        np.savetxt(
            os.path.join(folder, 'eval_times.dat'),
            np.array(self._eval_times),
            fmt=f'%.{precision}e'
        )

    @property
    def acceptance_rate(self) -> float:
        """
        Get the acceptance rate.

        Returns:
        float: The acceptance rate.
        """
        return self._n_accepted / len(self._eval_times)

    @property
    def offset(self):
        """
        Get the offset.

        Returns:
        float: The offset.
        """
        return self._offset

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

    positions = zig_zag.positions

    fig, ax = get_2d_despined_figure(plot_limits=plot_limits, figsize=(4, 4), keep_ticks=True)
    plot_pdf_contours(posterior, ax, plot_limits=plot_limits)
    ax.plot(*positions.T)
    plt.show()
