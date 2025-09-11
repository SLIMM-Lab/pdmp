import pickle
import os
import sys

import numpy as np

from typing import cast, override

from tqdm import tqdm

from pdmp import logger
from pdmp.sampler import Sampler, SAMPLER_REGISTRY, register_sampler
from pdmp.distributions import Distribution
from pdmp.surrogates import (SurrogateModel, LaplaceSurrogate, NeuralNetwork,
                             GaussianProcess, DerivativeGaussianProcess,
                             ConstantSurrogate, RandomConstantSurrogate)

@register_sampler('BouncyParticle')
class BouncyParticleSampler(Sampler):
    """
    Bouncy Particle Sampler class for sampling from a target distribution.

    This sampler implements the Bouncy Particle Sampler which is a piecewise deterministic
    Markov process that, unlike ZigZag, reflects the velocity vector off the gradient of
    the target density rather than flipping individual components.
    """

    def __init__(self,
                 target: Distribution,
                 *,
                 surrogate: SurrogateModel = None,
                 n_max: int = None,
                 t_max: float = None,
                 refresh_rate: float = 0.1,
                 gamma: float = 1e-6,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 sub_sampling: bool = False,
                 n_events_accepted: int = None,
                 print_every: int = 100,
                 update_bar_every: int = 10,
                 offset_shrinkage: float = 0.0,
                 x_0: np.ndarray = None,
                 x_0_lap: bool = False,
                 v_0: np.ndarray = None,
                 **kwargs):
        """
        Initialize the BouncyParticleSampler class.

        Parameters:
        target (Distribution): Target distribution to sample from.
        surrogate (SurrogateModel, optional): Surrogate model for thinning. Default is None.
        n_max (int, optional): Maximum number of events. Default is None.
        t_max (float, optional): Maximum simulation time. Default is None.
        refresh_rate (float, optional): Rate for refreshing velocities. Default is 0.1.
        gamma (float, optional): Small constant to avoid division by zero. Default is 1e-6.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        sub_sampling (bool, optional): Whether to use sub-sampling. Default is False.
        n_events_accepted (int, optional): Number of accepted events. Default is None.
        print_every (int, optional): Interval to print outputs. Default is 100.
        update_bar_every (int, optional): Update progress bar interval. Default is 10.
        offset_shrinkage (float, optional): Shrinkage rate for offset. Default is 0.0.
        """
        super().__init__()

        self.target = target
        self._dim = self.target.dim
        self._refresh_rate = refresh_rate

        if n_max is not None:
            self._n_max = n_max
            self.run = self._run_budget

        if t_max is not None:
            self._t_max = float(t_max)
            self._n_max = 10000000
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
        self._offset = 0.0
        self._offset_history = np.zeros((self._n_max, self._dim))
        self._thinning = False
        self._n_accepted = 0
        self._n_accepted_0 = n_events_accepted
        self._accepted_iters = np.zeros(
            self._n_accepted_0, dtype=int) if n_events_accepted else None
        self._sub_sampling = sub_sampling
        self._print_every = print_every
        self._update_bar_every = update_bar_every
        self._offset_shrinkage = offset_shrinkage

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
            v = self._rng.normal(0, 1, self._dim)
            self.velocities[0] = v / np.linalg.norm(v)
        else:
            if len(v_0) != self._dim:
                raise ValueError(
                    f"Initial velocity v_0 must have the same dimension as target distribution, expected {self._dim}, got {len(v_0)}"
                )
            self.velocities[0] = v_0

        # Setup surrogate model if provided
        if surrogate is not None:
            self._thinning = True
            self._s = None
            self.surrogate = surrogate
            # Select the correct event time generation method based on surrogate type
            if isinstance(surrogate, LaplaceSurrogate):
                self._generate_event_times = self._inverse_cdf_linear
            elif isinstance(surrogate, (ConstantSurrogate, RandomConstantSurrogate)):
                self._generate_event_times = self._inverse_cdf_constant
            else:
                self._generate_event_times = self._inverse_cdf
            self._cdf_rates = self._surrogate_rates
            self._eval_times = []
            self._times_all = None
        else:
            self._generate_event_times = self._inverse_cdf
            self._cdf_rates = self._target_rates

        logger.info("BouncyParticleSampler initialized.")

    def _target_rates(self,
                      x: np.ndarray,
                      idx_d: int = None,
                      idx_n: int = None) -> float:
        """
        Calculate the rate for the Bouncy Particle Sampler.

        Parameters:
        x (np.ndarray): Current position.
        idx_d (int, optional): Not used in BPS. Default is None.
        idx_n (int, optional): Observation index for sub-sampling. Default is None.

        Returns:
        float: The calculated rate.
        """
        grad = self.target.grad_log_density(x)

        if self._thinning:
            log_p = self.target.log_density(x)
            self.surrogate.add_data(x=x, y=log_p, dy_dx=grad)

        # BPS uses the inner product of gradient and velocity
        rate = np.maximum(-np.dot(grad, self.velocities[self._iter]),
                          0) + self._refresh_rate

        return rate

    def _surrogate_rates(self,
                         x: np.ndarray,
                         idx_d: int = None,
                         idx_n: int = None) -> float:
        """
        Calculate the surrogate rate for thinning.

        Parameters:
        x (np.ndarray): Current position.
        idx_d (int, optional): Not used in BPS. Default is None.
        idx_n (int, optional): Not used. Default is None.

        Returns:
        float: The calculated surrogate rate.
        """
        grad_surrogate = self.surrogate.grad(x)
        rate = np.maximum(
            -np.dot(grad_surrogate, self.velocities[self._iter]) + self._offset,
            0) + self._refresh_rate
        return rate

    def _inverse_cdf(self) -> tuple[float, int]:
        """
        Generate event time using the inverse CDF method.

        Returns:
        tuple[float, int]: The generated event time and event type (0 for bounce, 1 for refresh)
        """
        # Sample exponential time for next event
        s = -np.log(self._rng.uniform())

        # Sample exponential time for next refresh
        refresh_time = self._rng.exponential(scale=1.0 / self._refresh_rate)

        # Calculate next bounce time
        tau = 0.
        integral = 0.
        rate_t0 = self._cdf_rates(self.positions[self._iter])
        rate_t1 = 0.
        dt = 0.01  # Step size for numerical integration

        while integral < s:
            next_pos = self.positions[
                self._iter] + (tau + dt) * self.velocities[self._iter]
            rate_t1 = self._cdf_rates(next_pos)
            integral += np.trapezoid([rate_t0, rate_t1], dx=dt)
            tau += dt
            rate_t0 = rate_t1

        # Linear interpolation for better accuracy
        tau -= (integral - s) / rate_t1

        # Return the smaller of bounce time and refresh time, with event type
        if tau < refresh_time:
            return tau, 0  # Bounce event
        else:
            return refresh_time, 1  # Refresh event

    def _inverse_cdf_linear(self) -> tuple[float, int]:
        """
        Generate event time using the inverse CDF method with a linear surrogate rate (Laplace surrogate).

        Returns:
            tuple[float, int]: The generated event time and event type (0 for bounce, 1 for refresh)
        """
        # Sample exponential time for next event
        S = -np.log(self._rng.uniform())

        # Sample exponential time for next refresh
        refresh_time = self._rng.exponential(scale=1.0 / self._refresh_rate)

        v = self.velocities[self._iter]
        x = self.positions[self._iter]
        lap = self.surrogate  # Should be LaplaceSurrogate
        inv_C = lap.gaussian.inv_C
        mean = lap.gaussian.mean
        offset = self._offset if hasattr(self, '_offset') else 0.0
        gamma = self._gamma

        a = float(v @ inv_C @ v)
        b = float(v @ inv_C @ (x - mean)) + offset

        # Solve for tau: integral_0^tau (a t + b + gamma) dt = S
        # That is: (a/2) * tau^2 + b * tau + gamma * tau = S
        # (a/2) * tau^2 + (b + gamma) * tau - S = 0
        A = 0.5 * a
        B = b + gamma
        C = -S
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            tau = float('inf')  # No real solution, treat as infinite
        else:
            tau = (-B + np.sqrt(discriminant)) / (2*A) if A != 0 else -C/B
            if tau < 0:
                tau = (-B - np.sqrt(discriminant)) / (2*A) if A != 0 else -C/B
            if tau < 0:
                tau = float('inf')

        # Return the smaller of bounce time and refresh time, with event type
        if tau < refresh_time:
            return tau, 0  # Bounce event
        else:
            return refresh_time, 1  # Refresh event

    def _inverse_cdf_constant(self) -> tuple[float, int]:
        """
        Generate event time using analytical formula for constant surrogate rates.

        Returns:
            tuple[float, int]: The generated event time and event type (0 for bounce, 1 for refresh)
        """
        # Sample exponential time for next event
        S = -np.log(self._rng.uniform())

        # Sample exponential time for next refresh
        refresh_time = self._rng.exponential(scale=1.0 / self._refresh_rate)

        rate = self._offset + self._gamma
        if rate <= 0:
            tau = float('inf')
        else:
            tau = S / rate

        if tau < refresh_time:
            return tau, 0  # Bounce event
        else:
            return refresh_time, 1  # Refresh event


    def _step(self):
        """
        Perform a single Bouncy Particle Sampler step.
        """
        T, event_type = self._generate_event_times()

        # Update position
        new_pos = self.positions[self._iter] + T * self.velocities[self._iter]
        self.positions[self._iter + 1] = new_pos
        self.times[self._iter + 1] = self.times[self._iter] + T

        # Handle event based on type
        if event_type == 0:  # Bounce
            # Get gradient at new position
            grad = self.target.grad_log_density(new_pos)

            # Reflection formula: v' = v - 2(v·∇U/|∇U|²)∇U
            grad_norm_sq = np.sum(grad**2)
            if grad_norm_sq > 1e-10:  # Avoid division by zero
                reflection = self.velocities[self._iter] - 2 * np.dot(
                    self.velocities[self._iter], grad) * grad / grad_norm_sq
                self.velocities[self._iter + 1] = reflection
            else:
                # If gradient is too small, keep velocity unchanged
                self.velocities[self._iter + 1] = self.velocities[self._iter]

            if self._thinning:
                self._eval_times.append(self.times[self._iter + 1])
                self._poisson_thinning(new_pos, T)

        else:  # Refresh
            # Sample new velocity from unit sphere
            v = self._rng.normal(0, 1, self._dim)
            self.velocities[self._iter + 1] = v / np.linalg.norm(v)
            self._n_accepted += 1
            if self._accepted_iters is not None:
                self._accepted_iters[self._n_accepted] = self._iter + 1

        # Update offset for thinning
        dt = self.times[self._iter + 1] - self.times[self._iter]
        self._offset *= np.exp(-self._offset_shrinkage * dt)
        self._offset_history[self._iter + 1] = self._offset
        self._iter += 1

    def _poisson_thinning(self, pos: np.ndarray, T: float):
        """
        Perform Poisson thinning for the Bouncy Particle Sampler.

        Parameters:
        pos (np.ndarray): New position after bounce attempt.
        T (float): Time increment.
        """
        m = self._target_rates(pos)
        M = self._surrogate_rates(pos)
        u = self._rng.uniform()

        if m > M:
            # Upper bound was too tight, increase offset
            delta_offset = 1.01 * (m - M) + 1e-3
            self._offset += delta_offset
            self._revert_step()
            logger.info(
                f"  Action at time {self.times[self._iter]:.2f}; current position: {self.positions[self._iter]}"
            )
            logger.info(f"     upper bound too tight, m: {m:.4f}, M: {M:.4f}")
            logger.info(
                f"      ...increasing offset by {delta_offset:.4e} to: {self._offset:.4f}"
            )
        elif u < (m / M):
            # Accept bounce
            self._n_accepted += 1
            if self._accepted_iters is not None:
                self._accepted_iters[self._n_accepted] = self._iter + 1
        else:
            # Reject bounce, revert to previous velocity
            self.velocities[self._iter + 1] = self.velocities[self._iter]

    def _revert_step(self):
        """
        Revert the last step.
        """
        self.times[self._iter + 1] = 0.
        self.positions[self._iter + 1] = 0.
        self.velocities[self._iter + 1] = 0
        self._offset_history[self._iter + 1] = 0.

        dt = self.times[self._iter] - self.times[self._iter - 1]
        self._offset *= np.exp(self._offset_shrinkage * dt)
        self._iter -= 1

    def _shutdown(self):
        """Shutdown the ZigZag sampler."""

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
            self._offset_history = self._offset_history[
                self._accepted_iters[:idx]]

    def _run_budget(self):
        """Run the BouncyParticle sampler."""

        logger.warning(
            f"Running Bouncy Particle Sampler sampler with budget n_max={self._n_max}")

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
            f"Running Bouncy Particle Sampler with time limit T={self._t_max}")
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
            data['offset'] = self._offset

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
                   fmt=f'%.{precision}e')
        if self._thinning:
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
