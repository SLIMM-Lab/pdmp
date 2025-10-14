import pickle
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np

from typing import cast, Any, Union, override

from tqdm import tqdm

from pdmp import logger
from pdmp.sampler import Sampler, SAMPLER_REGISTRY, register_sampler
from pdmp.distributions import Distribution, MultivariateNormal
from pdmp.surrogates import (SurrogateModel, LaplaceSurrogate, NeuralNetwork,
                             GaussianProcess, DerivativeGaussianProcess,
                             ConstantSurrogate, RandomConstantSurrogate)
from pdmp.plotting import plot_pdf_contours
from pdmp.plotting_utils import get_2d_despined_figure


@register_sampler('ZigZag')
class ZigZagSampler(Sampler):
    """ZigZagSampler class for sampling from a target distribution using the ZigZag process."""

    def __init__(self,
                 target: Distribution,
                 *,
                 surrogate: SurrogateModel = None,
                 n_max: int = None,
                 t_max: float = None,
                 gamma: float = 1e-6,
                 dt: float = 0.001,
                 n_events_accepted: int = None,
                 offset_shrinkage: float = 0.0,
                 x_0: np.ndarray = None,
                 x_0_lap: bool = False,
                 v_0: np.ndarray = None,
                 sub_sampling: bool = False,
                 print_every: int = 100,
                 update_bar_every: int = 10,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 **kwargs):
        """Initialize the ZigZagSampler class.

        Args:
            target: The target distribution to sample from.
            surrogate: The surrogate model to use. Default is None.
            n_max: The number of events to sample. Default is 1000.
            t_max: The maximum time to sample. Default is None.
            gamma: The refresh rate parameter for the ZigZag process. Default is 1e-6 to avoid zero division.
            dt: The time step for the ZigZag process. Default is 0.001.
            n_events_accepted: Number of accepted events. Default is None.
            offset_shrinkage: The shrinkage parameter for the offset. Default is 0.0.
            x_0: Initial position. Default is None.
            v_0: Initial velocity. Default is None.
            sub_sampling: Whether to use sub-sampling. Default is False.
            print_every: Interval to print outputs.
            rng: Random number generator. Default is None.
            seed: Seed for the random number generator. Default is None.
            kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.target = target
        self._dim = self.target.dim

        if n_max is not None:
            self._n_max = n_max
            self.run = self._run_budget

        # make very large skeleton if algorithm is run with time limit
        if t_max is not None:
            self._t_max = float(t_max)
            self._n_max = 10000000  #TODO this might be too large, find a better way to handle this
            n_events_accepted = self._n_max
            self.run = self._run_time

        if n_max is None and t_max is None:
            self._n_max = 1000
            self.run = self._run_budget

        # init all variables
        self.times = np.zeros(self._n_max)
        self.positions = np.zeros((self._n_max, self._dim))
        self.velocities = np.zeros((self._n_max, self._dim))
        self._iter = 0
        self._gamma = gamma
        self._dt = dt
        self.offset = np.zeros(self._dim)
        self._offset_history = np.zeros((self._n_max, self._dim))
        self._n_accepted = 0
        self._n_accepted_0 = n_events_accepted
        self._accepted_iters = np.zeros(self._n_accepted_0, dtype=int)
        self._offset_shrinkage = offset_shrinkage
        self._sub_sampling = sub_sampling
        self._thinning = False

        # init logging related variables
        self._print_every = print_every
        self._update_bar_every = update_bar_every

        # get rng
        if rng is None and seed is None:
            self._rng = np.random.default_rng(0)
        elif rng is None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = rng

        # if not specified draw iid standard normal samples as inital position
        if x_0 is None:
            if x_0_lap:
                lap = LaplaceSurrogate.from_dict(target=self.target, rng=self._rng)
                self.positions[0] = lap.get_samples(1)
            else:
                self.positions[0] = self._rng.normal(0, 1, self._dim)
        else:
            self.positions[0] = x_0

        # if not specified draw initial velocity from binomial distribution
        if v_0 is None:
            self.velocities[0] = 2 * self._rng.binomial(1, 0.5, self._dim) - 1
        else:
            if len(v_0) != self._dim:
                raise ValueError(
                    f"Initial velocity v_0 must have the same dimension as target distribution, expected {self._dim}, got {len(v_0)}"
                )
            if not np.all((v_0 == -1) + (v_0 == 1)):
                raise ValueError(
                    f"All components of initial velocity v_0 must be either -1 or +1, got {v_0}"
                )
            self.velocities[0] = v_0

        self._s = None
        # check how events from pdmp should be generated
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
                self._generate_event_times = self._inverse_cdf

            if isinstance(surrogate, GaussianProcess):
                self.surrogate = cast(GaussianProcess, surrogate)
                self._generate_event_times = self._inverse_cdf

            if isinstance(surrogate, DerivativeGaussianProcess):
                self.surrogate = cast(DerivativeGaussianProcess, surrogate)
                self._generate_event_times = self._inverse_cdf

            if isinstance(surrogate, ConstantSurrogate):
                self.surrogate = cast(ConstantSurrogate, surrogate)
                self._generate_event_times = self._inverse_cdf_constant
                self.offset = np.ones_like(self.offset)

            if isinstance(surrogate, RandomConstantSurrogate):
                self.surrogate = cast(RandomConstantSurrogate, surrogate)
                self._generate_event_times = self._inverse_cdf_constant
                self.offset = np.ones_like(self.offset)

            self._cdf_rates = self._surrogate_rates

            # for later use
            self._times_all = None
            self._eval_times = []

        else:
            self._generate_event_times = self._inverse_cdf
            self._cdf_rates = self._target_rates

        # TODO: either implement sub-sampling or remove it
        # self.n_obs_ = self.target.n_obs
        # if 'ss' in kwargs:
        #     self.ss_ = kwargs['ss']
        #
        # if 'us' in kwargs:
        #     self.us_ = kwargs['us']

        if kwargs is not None:
            logger.warning(f'Unused kwargs: \n{kwargs}')

        logger.info(f"{self.__class__.__name__} initialized.")

    def _target_rates(self,
                      x: np.ndarray,
                      idx_d: int = None,
                      idx_n: int = None) -> np.ndarray:
        """Calculate the rates for the target ZigZag process.

        Args:
            x: The current position.
            idx_d: Dimension index. Default is None.
            idx_n: Observation index. Default is None.

        Returns:
            np.ndarray: The calculated rates.
        """

        grad = self.target.grad_log_density(x)
        log_p = self.target.log_density(x)

        if self._thinning:
            self.surrogate.add_data(x=x, y=log_p, dy_dx=grad)

        rates = np.maximum(-grad * self.velocities[self._iter], 0) + self._gamma

        if idx_d is None:
            return rates
        else:
            return rates[idx_d]

    def _surrogate_rates(self,
                         x: np.ndarray,
                         idx_d: int = None,
                         idx_n: int = None) -> np.ndarray:
        """Calculate the surrogate rates for the surrogate ZigZag process.

        Args:
            x: The current position.
            idx_d: Dimension index. Default is None.
            idx_n: Observation index. Default is None.

        Returns:
            np.ndarray: The calculated surrogate rates.
        """
        if idx_d is None:
            rates = -self.surrogate.grad(x) * self.velocities[
                self._iter] + self.offset
            return np.maximum(rates, 0) + self._gamma
        else:
            rate = -self.surrogate.grad(x, idx_d) * self.velocities[
                self._iter, idx_d] + self.offset[idx_d]
            return np.maximum(rate, 0) + self._gamma

    def _inverse_cdf(self) -> tuple[np.floating, np.integer]:
        """Generate event times using the inverse cdf method.

        Returns:
            tuple[float, int]: A tuple containing
                - The generated event time.
                - The index of the dimension where the event occurred.
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
                self.positions[self._iter] +
                (taus + self._dt) * self.velocities[self._iter],
                idx_n=j)
            integral += np.trapezoid(np.array([rate_t0, rate_t1]),
                                     dx=self._dt,
                                     axis=0)
            taus += self._dt
            rate_t0 = rate_t1

        # find component that reached s first
        i = np.argmax((integral - s) / rate_t1)

        # linear correction to last step
        tau = taus[i] - (integral[i] - s[i]) / rate_t1[i]

        # logger.debug(f"S    : {s}")
        # logger.debug(f"taus : {taus}")

        return tau, i

    def _inverse_cdf_linear(self) -> tuple[np.floating, np.integer]:
        """Generate event times using the inverse cdf method assuming b linear rate function.

        Returns:
            tuple[float, int]: A tuple containing
                - The generated event time.
                - The index of the dimension where the event occurred.
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
        taus = S / (self._gamma + self.offset)

        # get the linear approximation of the rates
        a = self.velocities[self._iter] * (
            self.surrogate.gaussian.inv_C @ self.velocities[self._iter])
        b = (self.velocities[self._iter] * (self.surrogate.gaussian.inv_C @ (
            self.positions[self._iter] - self.surrogate.gaussian.mean)) +
             self.offset)

        # compute root
        taus_0 = -b / a

        # check for each component where the intersection with the x-axis is and compute integral accordingly
        for i in range(self._dim):

            if (a[i] >= 0) and (b[i] >= 0):
                b_i = b[i] + self._gamma
                taus[i] = (np.sqrt(b_i**2 + 2 * a[i] * S[i]) - b_i) / a[i]
            elif (a[i] >= 0) and (b[i] < 0):
                taus_const = S[i] / self._gamma
                if taus_const < taus_0[i]:
                    taus[i] = taus_const
                else:
                    s[i] += taus_0[i] * self._gamma
                    d_s = S[i] - s[i]
                    b_i = b[i] + self._gamma
                    taus[i] = taus_0[i] + (np.sqrt(
                        (b_i + a[i] * taus_0[i])**2 + 2 * a[i] * d_s) -
                                           (b_i + a[i] * taus_0[i])) / a[i]
            elif (a[i] < 0) and (b[i] >= 0):
                s_0 = (0.5 * b[i] + self._gamma) * taus_0[i]
                if S[i] > s_0:
                    taus[i] = taus_0[i] + (S[i] - s_0) / self._gamma
                else:
                    taus[i] = (np.sqrt(
                        (b[i] + self._gamma)**2 + 2 * a[i] * S[i]) -
                               (b[i] + self._gamma)) / a[i]
            else:
                taus[i] = S[i] / self._gamma

            # logger.debug(f"taus: {taus}")

        j = np.argmin(taus)
        return taus[j], j

    def _inverse_cdf_constant(self) -> tuple[np.floating , np.integer]:
        """Generate event times using analytical formula for constant surrogate rates.

        For constant surrogate models, the rates are constant (offset + gamma),
        so we can use the analytical inverse CDF: tau = S / rate.

        Returns:
            tuple[float, int]: A tuple containing
                - The generated event time.
                - The index of the dimension where the event occurred.
        """
        # recover rng from previous iteration in case of rejection
        if self._s is None:
            self._s = -np.log(self._rng.uniform(0, 1, self._dim))
        s = self._s

        # For constant surrogate, rates are just offset + gamma
        # (since surrogate gradient is zero everywhere)
        rates = self.offset + self._gamma

        # Analytical inverse CDF for constant rates: tau = S / rate
        taus = s / rates

        # Find component that reaches s first (minimum tau)
        j = np.argmin(taus)

        return taus[j], j

    def _approximate_rates(self, x: np.ndarray, idx=None) -> np.ndarray:
        """Calculate the approximate rates for the ZigZag process.

        Args:
            x: The current position.
            idx: Dimension index. Default is None.

        Returns:
            np.ndarray: The calculated approximate rates.
        """

        if idx is None:
            rates = -self.velocities[self._iter] * self.surrogate.grad(
                x) + self.offset
            return np.maximum(rates, 0) + self._gamma
        else:
            rate = -self.velocities[self._iter, idx] * self.surrogate.grad(
                x, idx) + self.offset[idx]
            return np.maximum(rate, 0) + self._gamma

    def _poisson_thinning(self, j: int, T: np.ndarray):
        """Perform Poisson thinning for the ZigZag process.

        Args:
            j: Dimension index.
            T: Time increment.
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
            self.offset[j] += delta_offset
            self._revert_step()
            logger.info(
                f"  Action at time {self.times[self._iter]:.2f}; current position: {self.positions[self._iter]}"
            )
            logger.info(f"     upper bound too tight, m: {m:.4f}, M: {M:.4f}")
            logger.info(
                f"      ...increasing offset {j} by {delta_offset:.4e} to: {self.offset[j]:.4f}"
            )
        elif u < (m / M):
            self.velocities[self._iter + 1, j] = -self.velocities[self._iter, j]
            self._n_accepted += 1
            self._accepted_iters[self._n_accepted] = self._iter + 1
            self._s = None
        else:
            self.velocities[self._iter + 1, j] = self.velocities[self._iter, j]
            self._s = None

    def _step(self):
        """Perform a single ZigZag step."""
        T, j = self._generate_event_times()
        self.times[self._iter + 1] = self.times[self._iter] + T
        self.positions[
            self._iter +
            1] = self.positions[self._iter] + T * self.velocities[self._iter]
        self.velocities[self._iter + 1] = self.velocities[self._iter]
        self.velocities[self._iter + 1, j] = -self.velocities[self._iter, j]

        if self._thinning:
            self._eval_times.append(self.times[self._iter + 1])
            self._poisson_thinning(j, T)
        else:
            self._s = None

        dt = self.times[self._iter + 1] - self.times[self._iter]
        self.offset *= np.exp(-self._offset_shrinkage * dt)
        self._offset_history[self._iter + 1] = self.offset
        self._iter += 1

    def _revert_step(self):
        """Revert the last ZigZag step."""
        self.times[self._iter + 1] = 0.
        self.positions[self._iter + 1] = 0.
        self.velocities[self._iter + 1] = 0
        self._offset_history[self._iter + 1] = 0.

        dt = self.times[self._iter] - self.times[self._iter - 1]
        self.offset *= np.exp(self._offset_shrinkage * dt)
        self._iter -= 1

    def _shutdown(self):
        """Shutdown the ZigZag sampler."""

        logger.info("Shutting down ZigZag sampler. Summary:")
        if self._thinning:
            logger.info(f"    Acceptance rate : {self.acceptance_rate:.3f}")
            logger.info(f"    Final offsets    : {self.offset}")

            # this is kept so that model evaluations could be tracked
            self._times_all = self.times

            idx = self._n_accepted + 1
            self.positions = self.positions[self._accepted_iters[:idx]]
            self.times = self.times[self._accepted_iters[:idx]]
            self.velocities = self.velocities[self._accepted_iters[:idx]]
            self._offset_history = self._offset_history[
                self._accepted_iters[:idx]]

        logger.info("Run successfully completed.")

    def _run_budget(self):
        """Run the ZigZag sampler."""

        logger.warning(
            f"Running ZigZag sampler with budget n_max={self._n_max}")

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=self._n_max,
                  file=sys.stdout,
                  dynamic_ncols=True,
                  disable=disable_tqdm) as pbar:
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
        """Run the ZigZag sampler."""

        logger.warning(
            f"Running ZigZag sampler with time limit T={self._t_max}")
        time = 0.

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(
                total=self._t_max,
                leave=True,
                file=sys.stdout,
                dynamic_ncols=True,
                bar_format=
                '{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                disable=disable_tqdm) as pbar:

            while self.times[self._iter] < self._t_max:
                if self._iter % self._print_every == 0:
                    pbar.clear()
                    logger.debug(f"Sampling event {self._iter}")
                    pbar.refresh()
                self._step()
                if self._iter % self._update_bar_every == 0:
                    incr = np.min(
                        (self._t_max, self.times[max(0, self._iter)])) - time
                    time = self.times[max(0, self._iter)]
                    pbar.update(incr)

        # remove empty skeleton
        self.times = self.times[:self._iter + 1]
        self.positions = self.positions[:self._iter + 1]
        self.velocities = self.velocities[:self._iter + 1]

        self._shutdown()

    @override
    def write_data(self, folder: str, precision: int = 6):

        if not os.path.exists(folder):
            os.makedirs(folder)

        data = {}
        if self._thinning:
            data['acceptance_rate'] = self.acceptance_rate
            data['offset'] = self.offset

        with open(os.path.join(folder, 'other.pkl'), 'wb') as f:
            pickle.dump(data, f)

        np.savetxt(os.path.join(folder, 'positions.dat'),
                   self.positions,
                   fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'times.dat'),
                   self.times,
                   fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'velocities.dat'),
                   self.velocities,
                   fmt='%d')
        np.savetxt(os.path.join(folder, 'times_all.dat'),
                   self._times_all,
                   fmt=f'%.{precision}e')
        np.savetxt(os.path.join(folder, 'offset_history.dat'),
                   self._offset_history,
                   fmt=f'%.{precision}e')

        self._eval_times.sort()
        np.savetxt(os.path.join(folder, 'eval_times.dat'),
                   np.array(self._eval_times),
                   fmt=f'%.{precision}e')

    @property
    def acceptance_rate(self) -> float:
        """Get the acceptance rate.

        Returns:
            float: The acceptance rate.
        """
        return self._n_accepted / len(self._eval_times)


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

    fig, ax = get_2d_despined_figure(plot_limits=plot_limits,
                                     figsize=(4, 4),
                                     keep_ticks=True)
    plot_pdf_contours(posterior, ax, plot_limits=plot_limits)
    ax.plot(*positions.T)
    plt.show()



























