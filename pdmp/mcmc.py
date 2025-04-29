import pickle
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Tuple, override
from tqdm import tqdm

from pdmp.sampler import Sampler, SAMPLER_REGISTRY, register_sampler
from pdmp.distributions import Distribution, MultivariateNormal
from pdmp import logger

class StepSampler(Sampler):

    def __init__(
            self,
            target: Distribution,
            n_samples: int = 10000,
            rng: np.random.Generator = None,
            seed: int = None,
            prec: np.ndarray = None,
            cov_factor: float = 1.,
            **kwargs
    ):
        """
        Initialize the StepSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        prec (np.ndarray, optional): Preconditioner matrix. Default is None.
        """
        super().__init__()

        self.target = target
        self._dim = self.target.get_dim()
        self._n_samples = n_samples
        self._state = np.zeros(self._dim, dtype=np.float64)
        self.chain = np.zeros((self._n_samples, self._dim), dtype=np.float64)
        self._iter = 1
        self._n_accept = 1
        self._n_accept_last = 1
        self._rescale_interval = 100
        self._cov_factor = cov_factor

        if rng is None and seed is None:
            self._rng = np.random.default_rng(0)
        elif rng is None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = rng

        self._proposal_dist = MultivariateNormal(
            np.zeros(self._dim),
            np.eye(self._dim),
            rng=self._rng
        )

        if prec is None:
            self._prec = np.eye(self._dim, dtype=np.float64)
        elif (prec.shape[0] == prec.shape[1] == self._dim) and (len(prec.shape) == 2):
            self._prec = prec
        elif len(prec.shape) == 1 and len(prec) == self._dim:
            self._prec = np.diagonal(prec)
        else:
            raise Exception("No valid preconditioner specified.")

        self._prec_L = np.linalg.cholesky(self._prec)

    # @classmethod
    # def from_dict(cls, config: dict, target: Distribution, rng: np.random.Generator = None, **kwargs):
    #     """
    #     Initialize the StepSampler-derived class from a dictionary.
    #
    #     Parameters:
    #     config (dict): The configuration dictionary.
    #     target (Distribution): The target distribution to sample from.
    #     rng (np.random.Generator, optional): Random number generator. Default is None.
    #
    #     Returns:
    #     StepSampler: The StepSampler instance.
    #     """
    #     return cls(target=target, rng=rng, **config)

    def _reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        self._iter = 1
        self._n_accept = 1
        self._n_accept_last = 1

        # check if self.target has a method get_prior_sample, and if so, use it
        if x_0 is not None:
            self._state = x_0
        else:
            self._state = self._rng.random(self._dim)

        self.chain = np.zeros((self._n_samples, self._dim))
        self.chain[0, :] = self._state

    def _get_sample_covariance(self) -> np.ndarray:
        """
        Get the sample covariance of the chain.

        Returns:
        np.ndarray: The sample covariance matrix.
        """
        if self._dim == 1:
            return np.array([[np.var(self.chain[:self._iter])]])
        else:
            return np.cov(self.chain[:self._iter], rowvar=False)

    def _set_preconditioner(self, cov: np.ndarray):
        """
        Set the preconditioner matrix.

        Parameters:
        cov (np.ndarray): The covariance matrix to set as the preconditioner.
        """
        self._prec = cov
        self._prec_L = np.linalg.cholesky(self._prec)

    def _step(self):
        """
        Perform a single step.
        """
        raise NotImplementedError("The step method must be implemented in a subclass.")

    def run(self):
        """
        Run the StepSampler.
        """
        raise NotImplementedError("The step method must be implemented in a subclass.")

    def write_data(self, folder: str, precision: int = 6):
        """
        Write the chain data to a file.

        Parameters:
        filename (str): The name of the folder to write the data to.
        precision (int, optional): The precision of the output. Default is 6.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        data = {
            'acceptance_rate': self._n_accept / self._n_samples,
            'preconditioner': self._prec
        }

        with open(os.path.join(folder, 'other.pkl'), 'wb') as f:
            pickle.dump(data, f)

        np.savetxt(os.path.join(folder, 'samples.dat'), self.chain, fmt=f'%.{precision}e')


@register_sampler('RandomWalkMetropolis')
class RandomWalkMetropolisSampler(StepSampler):
    """
    A class to perform sampling with the Random-Walk Metropolis algorithm.
    """

    def __init__(
            self,
            target: Distribution,
            sigma: float = 0.5,
            n_samples: int = 10000,
            rng: np.random.Generator = None,
            seed: int = None,
            prec: np.ndarray = None,
            cov_factor: float = 1.,
            x_0: np.ndarray = None,
            **kwargs
    ):
        """
        Initialize the RandomWalkMetropolisSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        sigma (float, optional): The standard deviation of the proposal distribution. Default is 0.5.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        prec (np.ndarray, optional): Preconditioner matrix. Default is None.
        """
        super().__init__(target=target, n_samples=n_samples, rng=rng, seed=seed, prec=prec, cov_factor=cov_factor)
        self._sigma = sigma
        self._log_density_old = 0.
        self._reset(x_0)
        self._proposals = np.zeros_like(self.chain)
        self._proposals[0, :] = self._state
        self._accepted = np.zeros(self._n_samples, dtype=bool)
        self._accepted[0] = True

    @classmethod
    def from_dict(cls, config: dict, target: Distribution, rng: np.random.Generator = None, **kwargs):
        """
        Initialize the RandomWalkMetropolisSampler class from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        target (Distribution): The target distribution to sample from.
        rng (np.random.Generator, optional): Random number generator. Default is None.

        Returns:
        RandomWalkMetropolisSampler: The RandomWalkMetropolisSampler instance.
        """
        return cls(target=target, rng=rng, **config)

    def _reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        super()._reset(x_0)
        self._log_density_old = self.target.log_density(self._state)

    def _step(self):
        """
        Perform a single RWM step.
        """
        proposal = self._state + self._sigma * self._prec_L @ self._proposal_dist.get_sample()
        self._proposals[self._iter, :] = proposal
        log_density_new = self.target.log_density(proposal)

        if (log_density_new - self._log_density_old) > np.log(self._rng.uniform()):
            self._state = proposal
            self._log_density_old = log_density_new
            self._n_accept += 1
            self._n_accept_last += 1
            self._accepted[self._iter] = True

        self.chain[self._iter, :] = self._state
        self._iter += 1

    def run(self):
        """
        Run the RWM sampler.
        """
        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=self._n_samples, file=sys.stdout, dynamic_ncols=False, disable=disable_tqdm) as pbar:
            for i in range(1, self._n_samples):

                if i == 1000:
                    self._set_preconditioner(self._cov_factor * 2.38 ** 2 / self._dim * self._get_sample_covariance())

                if i % self._rescale_interval == 0:
                    logger.info(f"Iteration: {i}")
                    logger.info(f"Acceptance rate: {self._n_accept_last / self._rescale_interval}")
                    if self._n_accept_last / self._rescale_interval < 0.2:
                        self._set_preconditioner(0.9 * self._prec)
                        logger.info("Decrease")
                    elif self._n_accept_last / self._rescale_interval > 0.25:
                        self._set_preconditioner(1.1 * self._prec)
                        logger.info("Increase")
                    self._n_accept_last = 0
                self._step()
                pbar.update()
        logger.info(f"Total acceptance rate: {self._n_accept / self._n_samples}")


@register_sampler('LangevinDynamics')
class LangevinDynamicsSampler(StepSampler):
    """
    Langevin Dynamics Sampler class for sampling from a target distribution using Langevin dynamics.
    """

    def __init__(self,
                 target: Distribution,
                 sigma: float = 0.5,
                 n_samples: int = 10000,
                 adjusted: bool = True,
                 prec: np.ndarray = None,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 cov_factor: float = 1.,
                 x_0: np.ndarray = None,
                 **kwargs):
        """
        Initialize the LangevinDynamicsSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        sigma (float, optional): The standard deviation of the proposal distribution. Default is 0.5.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        adjusted (bool, optional): Whether to use the adjusted Langevin dynamics. Default is True.
        prec (np.ndarray, optional): Preconditioner matrix. Default is None.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        """
        super().__init__(target=target, n_samples=n_samples, rng=rng, seed=seed, prec=prec, cov_factor=cov_factor)
        self.sigma_ = sigma
        self.log_density_ = 0.
        self.grad_log_density_ = np.zeros(self._dim, dtype=np.float64)
        self.adjusted_ = adjusted
        self._reset(x_0)

    def _reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        super()._reset(x_0)
        self.log_density_ = self.target.log_density(self._state)
        self.grad_log_density_ = self.target.grad_log_density(self._state)

    def _log_proposal_density(self, y: np.ndarray, x: np.ndarray, grad_x: np.ndarray) -> float:
        """
        Calculate the log proposal density.

        Parameters:
        y (np.ndarray): The proposed state.
        x (np.ndarray): The current state.
        grad_x (np.ndarray): The gradient of the log density at the current state.

        Returns:
        float: The log proposal density.
        """
        diff = y - x - 0.5 * self.sigma_ ** 2 * self._prec @ grad_x
        return - 0.5 * np.linalg.norm(np.linalg.solve(self.sigma_ * self._prec_L, diff))**2

    def _step(self):
        """
        Perform a single Langevin dynamics step.
        """
        self.randn_ = self._proposal_dist.get_sample()
        # self.randn_ = np.array([0.8037, -1.715])
        prop = (self._state + self.sigma_ * self._prec_L @ self.randn_
                + 0.5 * self.sigma_ ** 2 * self._prec @ self.grad_log_density_)
        log_density_prop = self.target.log_density(prop)
        grad_log_density_prop = self.target.grad_log_density(prop)
        log_numerator = log_density_prop + self._log_proposal_density(self._state, prop, grad_log_density_prop)
        log_denominator = self.log_density_ + self._log_proposal_density(prop, self._state, self.grad_log_density_)

        if not self.adjusted_ or ((log_numerator - log_denominator) > np.log(self._rng.uniform())):
            self._state = prop
            self.chain[self._iter, :] = prop
            self.log_density_ = log_density_prop
            self.grad_log_density_ = grad_log_density_prop
            self._n_accept += 1
            self._n_accept_last += 1
        else:
            self.chain[self._iter, :] = self._state

        self._iter += 1

    def run(self):
        """
        Run the Langevin dynamics sampler.
        """
        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=self._n_samples, file=sys.stdout, dynamic_ncols=False, disable=disable_tqdm) as pbar:
            for i in range(1, self._n_samples):
                if i == 1000:
                    self._set_preconditioner(
                        self._cov_factor * np.power(self._dim, -1. / 3.) * self._get_sample_covariance())

                if i % self._rescale_interval == 0:
                    logger.info(f"Iteration: {i}")
                    logger.info(f"Acceptance rate: {self._n_accept_last / self._rescale_interval}")
                    if self._n_accept_last / self._rescale_interval < 0.5:
                        self._set_preconditioner(0.9 * self._prec)
                        logger.info("Decrease")
                    elif self._n_accept_last / self._rescale_interval > 0.6:
                        self._set_preconditioner(1.1 * self._prec)
                        logger.info("Increase")
                    self._n_accept_last = 0
                self._step()
                pbar.update()
        logger.info(f"Acceptance rate: {self._n_accept / self._n_samples}")


@register_sampler('HamiltonianMonteCarlo')
class HamiltonianMonteCarlo(StepSampler):
    """
    Hamiltonian Monte Carlo (HMC) sampler class for sampling from a target distribution using HMC.
    """

    def __init__(
            self,
            target: Distribution,
            step_scale: float = 0.1,
            leap_frog_steps: int = 20,
            n_samples: int = 10000,
            prec: np.ndarray = None,
            rng: np.random.Generator = None,
            seed: int = None,
            plot: bool = False,
            x_0: np.ndarray = None,
            plot_limits: Tuple[float, float] = None,
            **kwargs
    ):
        """
        Initialize the HamiltonianMonteCarlo class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        step_scale (float, optional): The step size for the leapfrog integrator. Default is 0.1.
        leap_frog_steps (int, optional): The number of leapfrog steps to perform. Default is 20.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        prec (np.ndarray, optional): Preconditioner matrix for the momentum distribution. Default is None.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        plot (bool, optional): Whether to plot the sampling process. Default is False.
        plot_limits (Tuple[float, float], optional): Limits for the plot. Default is None.
        plot_joint (bool, optional): Whether to plot the joint distribution. Default is False.
        """
        super().__init__(target=target, n_samples=n_samples, rng=rng, seed=seed, prec=prec)

        self._step_scale = step_scale
        self._leap_frog_steps = leap_frog_steps
        self._n_accepted = 1
        self._p_old = 0.0

        self._prec_inv = np.linalg.inv(self._prec)
        self._prec_det = np.linalg.det(self._prec)

        super()._reset(x_0)

        if plot and (self._dim == 1 or self._dim == 2):
            self._plot = True
            self._init_plot(plot_limits)
        else:
            self._plot = False

    def _leap_frog_step(self, p0: np.ndarray, q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single leapfrog step.

        Parameters:
        p0 (np.ndarray): Initial momentum.
        q0 (np.ndarray): Initial position.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Updated momentum and position.
        """
        p = p0 + self._step_scale * 0.5 * self.target.grad_log_density(q0)
        q = q0 + self._step_scale * self._prec_inv @ p
        p = p + self._step_scale * 0.5 * self.target.grad_log_density(q)
        return p, q

    def _get_hamiltonian(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Hamiltonian.

        Parameters:
        p (np.ndarray): Momentum.
        q (np.ndarray): Position.

        Returns:
        float: The Hamiltonian value.
        """
        potential = - self.target.log_density(q)
        kinetic = 0.5 * p @ self._prec_inv @ p
        kinetic = kinetic + 0.5 * np.log((2. * np.pi) ** self._dim * self._prec_det)

        return potential + kinetic

    def _step(self):
        """
        Perform a single HMC step.
        """
        p_hist = np.empty((self._leap_frog_steps + 1, self._dim))
        q_hist = np.empty((self._leap_frog_steps + 1, self._dim))

        p_hist[0, :] = p_0 = self._prec_L @ self._proposal_dist.get_sample()
        q_hist[0, :] = q_0 = self._state

        hamiltonian_0 = self._get_hamiltonian(p_0, q_0)

        for i in range(1, self._leap_frog_steps + 1):
            p_hist[i, :], q_hist[i, :] = self._leap_frog_step(p_hist[i - 1, :], q_hist[i - 1, :])

        hamiltonian = self._get_hamiltonian(p_hist[-1, :], q_hist[-1, :])

        accept = (- hamiltonian + hamiltonian_0) > np.log(self._rng.uniform())
        if accept:
            self._state = q_hist[-1, :]
            self._n_accepted += 1
            color = 'g'
        else:
            color = 'r'

        if self._plot:
            if self._dim == 2:
                # self._ax.plot(*q_hist.transpose(), marker=".", color=color)
                self._ax.plot(*q_hist.transpose(), marker=".")
            else:
                # self._ax.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", color=color, markersize=2.)
                aw = 0.01
                # self._ax.arrow(q_0[0], self._p_old, 0, p_0[0] - self._p_old, alpha=0.5,
                #                color='k', width=aw, length_includes_head=True, head_width=7*aw)
                # self._ax.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4.)
                self._ax.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4., color=color)

                if accept:
                    self._p_old = p_hist[-1, 0]

        self.chain[self._iter, :] = self._state
        self._iter += 1

    def run(self):
        """
        Run the HMC sampler.
        """
        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=self._n_samples, file=sys.stdout, dynamic_ncols=False, disable=disable_tqdm) as pbar:
            for i in range(1, self._n_samples):
                self._step()
                pbar.update()
                if self._plot:
                    logger.info(f"Iteration: {i}")

                    # self._fig.fig.show()
                elif i % 1000 == 0:
                    logger.info(f"Iteration: {i}")
        logger.info(f"Acceptance rate: {self._n_accepted / self._n_samples}")

        if self._plot:
            self._fig.fig.show()
            self._fig.savefig(f"plots/hmc_2d_l2.p_x")

    def _init_plot(self, plot_limits: Tuple[float, float] = None):
        """
        Initialize the plot for the Hamiltonian Monte Carlo sampler.

        This method sets up the plotting environment, including the figure and axes,
        and configures the plot limits and labels based on the dimensionality of the
        target distribution.

        Note:
            This method is only applicable if the dimensionality of the target distribution
            is 1 or 2 and the `plot` parameter is set to True during initialization.
        """

        self._fig = sns.JointGrid()
        self._ax = self._fig.ax_joint

        if plot_limits is None:
            x_lim = [-2, 2]
            y_lim = [-2, 2]
        else:
            x_lim = plot_limits[0]
            y_lim = plot_limits[1]
        x = np.linspace(x_lim[0], x_lim[1], 100)
        y = np.linspace(y_lim[0], y_lim[1], 100)
        self._gx, self._gy = np.meshgrid(x, y)
        self._gz = np.zeros_like(self._gx)

        for i in range(self._gx.shape[0]):
            for j in range(self._gx.shape[1]):
                if self._dim == 2:
                    self._gz[i, j] = np.exp(self.target.log_density(np.array(
                        [self._gx[i, j],
                         self._gy[i, j]]
                    )))
                else:
                    self._gz[i, j] = np.exp(self.target.log_density(np.array([self._gx[i, j]])) +
                                            self._proposal_dist.log_density(np.array([self._gy[i, j]])))

        self._ax.contour(self._gx, self._gy, self._gz, levels=20, zorder=1, alpha=0.3, linewidths=1.5)
        self._fig.ax_marg_x.plot(self._gx[0], self._gz[0])
        self._fig.ax_marg_y.plot(self._gz[:, 0], self._gy[:, 0])

        if self._dim == 1:
            self._ax.set_xlabel(r"$p(\theta)$")
            self._ax.set_ylabel(r"$p(\mathrm{p})$")
        else:
            self._ax.set_xlabel(r"$p(\theta_1)$")
            self._ax.set_ylabel(r"$p(\theta_2)$")
        self._fig.fig.tight_layout()
        self._fig.fig.show()


@register_sampler('NaiveNUTS')
class NaiveNUTS(StepSampler):
    """
    Naive version of the No-U-Turn sampler (NUTS)
    """
    def __init__(
            self,
            target: Distribution,
            epsilon: float = 0.1,
            n_samples: int = 10000,
            prec: np.ndarray = None,
            rng: np.random.Generator = None,
            seed: int = None,
            x_0: np.ndarray = None,
            plot: bool = False,
            plot_limits: Tuple[float, float] = None,
            delta_max: float = 1000.,
            **kwargs
    ):
        """Naive version of the No-U-Turn sampler (NUTS).

        Args:
            target (Distribution): The target distribution to sample from.
            epsilon (float, optional): The step size for the leapfrog integrator. Defaults to 0.1.
            n_samples (int, optional): The number of samples to generate. Defaults to 10000.
            prec (np.ndarray, optional): Preconditioner matrix for the momentum distribution. Defaults to None.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            seed (int, optional): Seed for the random number generator. Defaults to None.
            x_0 (np.ndarray, optional): Initial position. Defaults to None.
            plot (bool, optional): Whether to plot the sampling process. Defaults to False.
            plot_limits (Tuple[float, float], optional): Limits for the plot. Defaults to None.
            delta_max (float, optional): Maximum energy difference threshold. Defaults to 1000.
        """
        super().__init__(target=target, n_samples=n_samples, rng=rng, seed=seed, prec=prec)

        self._n_evals = np.zeros(n_samples, dtype=int)

        self._step_scale = epsilon
        self._delta_max = delta_max
        self._n_accepted = 1
        self._p_old = 0.0

        self._prec_inv = np.linalg.inv(self._prec)
        self._prec_det = np.linalg.det(self._prec)

        super()._reset(x_0)

        if plot and (self._dim == 1 or self._dim == 2):
            self._plot = True
            self._init_plot(plot_limits)
        else:
            self._plot = False

    def _init_plot(self, plot_limits: Tuple[float, float] = None):
        """Initialize the plot for the Hamiltonian Monte Carlo sampler.

        Args:
            plot_limits (Tuple[float, float], optional): Limits for the plot. Defaults to None.

        This method sets up the plotting environment, including the figure and axes,
        and configures the plot limits and labels based on the dimensionality of the
        target distribution.

        Note:
            This method is only applicable if the dimensionality of the target distribution
            is 1 or 2 and the `plot` parameter is set to True during initialization.
        """
        sns.set_theme(style="ticks", rc={"axes.spines.right": False,"axes.spines.top": False})

        if plot_limits is None:
            x_lim = [-2, 2]
            y_lim = [-2, 2]
        else:
            x_lim = plot_limits[0]
            y_lim = plot_limits[1]

        ratio = abs((x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0]))
        size = 5.

        self._fig, self._ax = plt.subplots(figsize=(size + 1, size/ratio + 1))

        x = np.linspace(x_lim[0], x_lim[1], 100)
        y = np.linspace(y_lim[0], y_lim[1], 100)
        self._gx, self._gy = np.meshgrid(x, y)
        self._gz = np.zeros_like(self._gx)

        for i in range(self._gx.shape[0]):
            for j in range(self._gx.shape[1]):
                if self._dim == 2:
                    self._gz[i, j] = np.exp(self.target.log_density(np.array(
                        [self._gx[i, j],
                         self._gy[i, j]]
                    )))
                else:
                    self._gz[i, j] = np.exp(self.target.log_density(np.array([self._gx[i, j]])) +
                                            self._proposal_dist.log_density(np.array([self._gy[i, j]])))

        self._ax.contour(self._gx, self._gy, self._gz, levels=20, zorder=1, alpha=0.3, linewidths=1.5)

        if self._dim == 1:
            self._ax.set_xlabel(r"$p(\theta)$")
            self._ax.set_ylabel(r"$p(\mathrm{p})$")
        else:
            self._ax.set_xlabel(r"$p(\theta_1)$")
            self._ax.set_ylabel(r"$p(\theta_2)$")
        self._fig.tight_layout()
        self._fig.show()

    def _leap_frog_step(
            self,
            p0: np.ndarray,
            q0: np.ndarray,
            v: int,
            epsilon: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a single leapfrog step.

        Args:
            p0 (np.ndarray): Initial momentum.
            q0 (np.ndarray): Initial position.
            v (int): Direction of the leapfrog step.
            epsilon (float): Step size.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - p (np.ndarray): Updated momentum.
                - q (np.ndarray): Updated position.
        """
        p = p0 + v * epsilon * 0.5 * self.target.grad_log_density(q0)
        q = q0 + v * epsilon * self._prec_inv @ p
        p = p +  v * epsilon * 0.5 * self.target.grad_log_density(q)
        return p, q

    def _get_hamiltonian(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate the Hamiltonian.

        Args:
            p (np.ndarray): Momentum.
            q (np.ndarray): Position.

        Returns:
            float: The Hamiltonian value.
        """
        potential = - self.target.log_density(q)
        kinetic = 0.5 * p @ self._prec_inv @ p
        kinetic = kinetic + 0.5 * np.log((2. * np.pi) ** self._dim * self._prec_det)

        return potential + kinetic

    def _plot_trajectory(self, trajectory: list[dict]):
        """Plot the trajectory of an individual NUTS step.

        Args:
            trajectory (list[dict]): List of dictionaries containing trajectory information.
                every point has a type (left, right, accept, reject), and the latter two from/to coordinates.
        """
        color = 'C0'
        for i, segment in enumerate(trajectory):
            if segment['type'] == "left":
                color = 'C0'
                continue
            if segment['type'] == "right":
                color = 'C1'
                continue
            if segment['type'] == "reject":
                color = 'C3'
            self._ax.plot(*np.array([segment["from"], segment["to"]]).T, color=color, alpha=0.8)
            self._ax.scatter(*segment["to"], color=color, marker='.', alpha=0.8)

        self._ax.scatter(*self._state, color='C5', marker='.', s=100, alpha=1., zorder=10)

    def _step(self):
        """Perform a single NUTS step.
        """
        trajectory = []

        def build_tree(p, q, u, v, j):
            """Build a tree of leap frog steps.

            Args:
                p (np.ndarray): Momentum.
                q (np.ndarray): Position.
                u (np.ndarray): Random variable for slice sampling.
                v (int): Direction of the leapfrog step.
                j (int): Current depth of the tree.

            Returns:
                p_minus (np.ndarray): Momentum of leftmost leaf.
                q_minus (np.ndarray): Position of leftmost leaf.
                p_plus (np.ndarray): Momentum of rightmost leaf.
                q_plus (np.ndarray): Position of rightmost leaf.
                C_prime (list[np.ndarray]): Set of candidate points C'.
                s_prime (int): Whether conditions for continuation are satisfied (0 or 1).
            """
            # base case: take a single leap-frog step
            if j == 0:
                C_prime = []
                p_prime, q_prime = self._leap_frog_step(p, q, v, self._step_scale)
                ham = self._get_hamiltonian(p, q)

                if u < np.exp(- ham):
                    C_prime.append((p_prime, q_prime))
                    trajectory.append({"type": "accept", "from": q, "to": q_prime})
                else:
                    trajectory.append({"type": "reject", "from": q, "to": q_prime})

                s_prime = int(np.log(u) < self._delta_max - ham)

                return p_prime, q_prime, p_prime, q_prime, C_prime, s_prime


            # build left and right subtrees recursively
            # (each subtree has a left and right subtree, if not a node)
            else:
                p_minus, q_minus, p_plus, q_plus, C_prime, s_prime = build_tree(p, q, u, v, j - 1)

                if v == -1:
                    p_minus, q_minus, _, _, C_prime_prime, s_prime_prime = build_tree(p_minus, q_minus, u, v, j - 1)
                else:
                    _, _, p_plus, q_plus, C_prime_prime, s_prime_prime = build_tree(p_plus, q_plus, u, v, j - 1)

                # update stopping condition
                s_prime = (
                        s_prime
                        * s_prime_prime
                        * int(np.dot((q_plus - q_minus), p_minus) > 0.0)
                        * int(np.dot((q_plus - q_minus), p_plus) > 0.0)
                )

                # update candidate set C'
                C_prime.extend(C_prime_prime)

            return p_minus, q_minus, p_plus, q_plus, C_prime, s_prime

        # sample momentum and compute hamiltonian
        p_0 = self._prec_L @ self._proposal_dist.get_sample()
        hamiltonian_0 = self._get_hamiltonian(p_0, self._state)

        # sample u for slice sampling
        u = self._rng.random() * np.exp(- hamiltonian_0)

        # init the position and momemtum variables for this iteration
        q_plus = q_minus = self._state
        p_minus = p_plus = p_0

        # init the stopping criterion, the tree depth, and the number of nodes in C (q_0 is in C)
        C = [(p_minus, q_minus)]
        s = 1
        j = 0

        # build the tree by doubling its size until stopping criterion is met
        while s == 1:
            v = self._rng.choice([-1, 1])
            if v == -1:
                trajectory.append({"type": "left"})
                p_minus, q_minus, _, _, C_prime, s_prime = build_tree(p_minus, q_minus, u, v, j)
            else:
                trajectory.append({"type": "right"})
                _, _, p_plus, q_plus, C_prime, s_prime = build_tree(p_plus, q_plus, u, v, j)

            if s_prime == 1:
                C.extend(C_prime)

            s = (
                    s_prime *
                    int(np.dot((q_plus - q_minus), p_minus) > 0.0) *
                    int(np.dot((q_plus - q_minus), p_plus) > 0.0)
            )
            j += 1

        # select a random point from C and update chain and state
        index = self._rng.integers(0, len(C))
        p, q = C[index][0], C[index][1]
        self._state = q
        self.chain[self._iter, :] = q
        self._iter += 1

        # plot the latest trajectory
        if self._plot:
            self._plot_trajectory(trajectory)

    def run(self):
        """Run the NaiveNUTS sampler.
        """
        # disable tqdm if running on a cluster or in pycharm
        disable_tqdm = (
                'PBS_ENVIRONMENT' in os.environ
                or 'SLURM_JOB_ID' in os.environ
                or 'GIO_LAUNCHED_DESKTOP_FILE' in os.environ
        )

        with tqdm(total=self._n_samples, file=sys.stdout, dynamic_ncols=False, disable=disable_tqdm) as pbar:
            for i in range(1, self._n_samples):
                self._step()
                pbar.update()
                if self._plot:
                    self._fig.show()
                elif i % 1000 == 0:
                    logger.info(f"Iteration: {i}")
        # logger.info(f"Acceptance rate: {}")

    @override
    def write_data(self, folder: str, precision: int = 6):
        super().write_data(folder, precision)
        np.savetxt(
            os.path.join(folder, 'n_evals.dat'),
            np.cumsum(self._n_evals),
            fmt=f'%d'
        )


@register_sampler('EfficientNUTS')
class EfficientNUTS(NaiveNUTS):
    """
    Efficient NUTS sampler class for sampling from a target distribution using the base version of the algorithm.
    """
    def __init__(
            self,
            target: Distribution,
            **kwargs
    ):
        """ Initialize the HamiltonianMonteCarlo class.

        Args:
            target (Distribution): The target distribution to sample from.
            step_scale (float, optional): The step size for the leapfrog integrator. Defaults to 0.1.
            n_samples (int, optional): The number of samples to generate. Defaults to 10000.
            prec (np.ndarray, optional): Preconditioner matrix for the momentum distribution. Defaults to None.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            seed (int, optional): Seed for the random number generator. Defaults to None.
            plot (bool, optional): Whether to plot the sampling process. Defaults to False.
            plot_limits (Tuple[float, float], optional): Limits for the plot. Defaults to None.
        """
        super().__init__(target=target, **kwargs)

    @override
    def _step(self):
        trajectory = []

        def build_tree(
                p: np.ndarray,
                q: np.ndarray,
                u: np.ndarray,
                v: int,
                j: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
            """Build a tree of leap frog steps.

            Args:
                p (np.ndarray): Momentum.
                q (np.ndarray): Position.
                u (np.ndarray): Random variable for slice sampling.
                v (int): Direction of the leapfrog step.
                j (int): Current depth of the tree.

            Returns:
                p_minus (np.ndarray): Momentum of leftmost leaf.
                q_minus (np.ndarray): Position of leftmost leaf.
                p_plus (np.ndarray): Momentum of rightmost leaf.
                q_plus (np.ndarray): Position of rightmost leaf.
                q_prime (np.ndarray): Position of the candidate leaf.
                n_prime (int): Number of candidate leaves in C'.
                s_prime (int): Whether conditions for continuation are satisfied (0 or 1).
            """
            # base case: take a single leap-frog step
            if j == 0:
                p_prime, q_prime = self._leap_frog_step(p, q, v, self._step_scale)
                ham = self._get_hamiltonian(p_prime, q_prime)
                n_prime = int(u < np.exp(- ham))
                s_prime = int(np.log(u) < self._delta_max - ham)

                if self._plot:
                    if n_prime:
                        trajectory.append({"type": "accept", "from": q, "to": q_prime})
                    else:
                        trajectory.append({"type": "reject", "from": q, "to": q_prime})


                return p_prime, q_prime, p_prime, q_prime, q_prime, n_prime, s_prime

            # build left and right subtrees recursively
            # (each subtree has a left and right subtree, if not a node)
            else:
                p_minus, q_minus, p_plus, q_plus, q_prime, n_prime, s_prime = build_tree(p, q, u, v, j - 1)

                if v == -1:
                    p_minus, q_minus, _, _, q_prime_prime, n_prime_prime, s_prime_prime = build_tree(
                        p_minus, q_minus, u, v, j - 1
                    )
                else:
                    _, _, p_plus, q_plus, q_prime_prime, n_prime_prime, s_prime_prime = build_tree(
                        p_plus, q_plus, u, v, j - 1
                    )

                # draw move from q' to q'' and update number leaves in candidate set C'
                n_sum = n_prime + n_prime_prime
                if n_sum > 0:
                    if self._rng.binomial(1, n_prime_prime / (n_sum)):
                        q_prime = q_prime_prime
                n_prime = n_sum

                # update stopping condition
                s_prime = (
                        s_prime
                        * s_prime_prime
                        * int(np.dot((q_plus - q_minus), p_minus) > 0.0)
                        * int(np.dot((q_plus - q_minus), p_plus) > 0.0)
                )

            return p_minus, q_minus, p_plus, q_plus, q_prime, n_prime, s_prime

        # sample momentum and compute hamiltonian
        p_0 = self._prec_L @ self._proposal_dist.get_sample()
        hamiltonian_0 = self._get_hamiltonian(p_0, self._state)

        # sample u for slice sampling
        u = self._rng.random() * np.exp(- hamiltonian_0)

        # init the position and momemtum variables for this iteration
        q_minus = self._state
        q_plus = self._state
        p_minus = p_plus = p_0

        # init the stopping criterion, the tree depth, and the number of nodes in C (q_0 is in C)
        s = 1
        j = 0
        n = 1

        # build the tree by doubling its size until stopping criterion is met
        while s == 1:
            v = self._rng.choice([-1, 1])
            if v == -1:
                trajectory.append({"type": "left"})
                p_minus, q_minus, _, _, q_prime, n_prime, s_prime = build_tree(p_minus, q_minus, u, v, j)
            else:
                trajectory.append({"type": "right"})
                _, _, p_plus, q_plus, q_prime, n_prime, s_prime = build_tree(p_plus, q_plus, u, v, j)

            if s_prime == 1:
                if self._rng.binomial(1, min(1., n_prime / n)):
                    self._state = q_prime

            n = n + n_prime
            s = (
                    s_prime *
                    int(np.dot((q_plus - q_minus), p_minus) > 0.0) *
                    int(np.dot((q_plus - q_minus), p_plus) > 0.0)
            )
            j += 1

        # update chain
        self.chain[self._iter, :] = self._state
        self._iter += 1

        if self._plot:
            self._plot_trajectory(trajectory)

@register_sampler('DualAveragingNUTS')
class DualAveragingNUTS(NaiveNUTS):
    """
    Dual averaging NUTS sampler class for sampling from a target distribution using the efficient version of the
    algorithm with dual averaging of the step size epsilon.
    """
    def __init__(
            self,
            target: Distribution,
            epsilon_bar_0: float = 1.,
            gamma: float = 0.05,
            t_0: int = 10,
            kappa: float = 0.75,
            M_adapt: int = 1000,
            target_acceptance_rate: float = 0.65,
            **kwargs
    ):
        """
        Initialize the HamiltonianMonteCarlo class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        prec (np.ndarray, optional): Preconditioner matrix for the momentum distribution. Default is None.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        plot (bool, optional): Whether to plot the sampling process. Default is False.
        plot_limits (Tuple[float, float], optional): Limits for the plot. Default is None.
        """
        super().__init__(target=target, **kwargs)

        self._gamma = gamma
        self._t_0 = t_0
        self._kappa = kappa
        self._epsilon_bar = epsilon_bar_0
        self._H_bar = 0.
        self._M_adapt = M_adapt
        self._delta = target_acceptance_rate

        epsilon_0 = self._find_reasonable_epsilon()
        self._mu = np.log(10 * epsilon_0)
        self._epsilon = epsilon_0
        print(f"epsilon 0: {self._epsilon}")

        self._acceptance_rate = 0.0

    def _find_reasonable_epsilon(self) -> float:
        """
        Find a reasonable epsilon for the leapfrog step size.

        Returns:
        float: A reasonable epsilon value.
        """
        epsilon = 1.
        p = self._prec_L @ self._proposal_dist.get_sample()
        p_prime, q_prime = self._leap_frog_step(p, self._state, 1, epsilon)
        a = 2 * (np.exp( - self._get_hamiltonian(p_prime, q_prime) + self._get_hamiltonian(p, self._state)) > 0.5) - 1.

        while np.exp(- self._get_hamiltonian(p_prime, q_prime) + self._get_hamiltonian(p, self._state))**a > 2**(-a):
            epsilon = epsilon * 2.**a
            p_prime, q_prime = self._leap_frog_step(p, self._state, 1, epsilon)
        return epsilon

    @override
    def _step(self):
        trajectory = []

        def build_tree(
                p: np.ndarray,
                q: np.ndarray,
                u: np.ndarray,
                v: int,
                j: int,
                epsilon: float,
                p_0: np.ndarray,
                q_0: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, int]:
            """Build a tree of leap frog steps.

            Args:
                p (np.ndarray): Momentum.
                q (np.ndarray): Position.
                u (np.ndarray): Random variable for slice sampling.
                v (int): Direction of the leapfrog step.
                j (int): Current depth of the tree.
                epsilon (float): Step size.
                p_0 (np.ndarray): Initial momentum.
                q_0 (np.ndarray): Initial position.

            Returns:
                p_minus (np.ndarray): Momentum of leftmost leaf.
                q_minus (np.ndarray): Position of leftmost leaf.
                p_plus (np.ndarray): Momentum of rightmost leaf.
                q_plus (np.ndarray): Position of rightmost leaf.
                q_prime (np.ndarray): Position of the candidate leaf.
                n_prime (int): Number of candidate leaves in C'.
                s_prime (int): Whether conditions for continuation are satisfied (0 or 1).
                alpha (float): Acceptance probability.
                n_alpha (int): Number of accepted samples.
            """

            # base case: take a single leap-frog step
            if j == 0:
                self._n_evals[self._iter] += 1
                p_p, q_p = self._leap_frog_step(p, q, v, epsilon)
                ham_1 = self._get_hamiltonian(p_p, q_p)
                ham_0 = self._get_hamiltonian(p_0, q_0)
                n_p = int(u < np.exp(- ham_1))
                s_p = int(np.log(u) < self._delta_max - ham_1)
                alpha = min(1., np.exp(- ham_1 + ham_0))

                if self._plot:
                    if n_p:
                        trajectory.append({"type": "accept", "from": q, "to": q_p})
                    else:
                        trajectory.append({"type": "reject", "from": q, "to": q_p})


                return p_p, q_p, p_p, q_p, q_p, n_p, s_p, alpha, 1

            # build left and right subtrees recursively
            # (each subtree has a left and right subtree, if not a node)
            else:
                p_minus, q_minus, p_plus, q_plus, q_p, n_p, s_p, a_p, n_a_p = build_tree(
                    p, q, u, v, j - 1, epsilon, p_0, q_0
                )

                if v == -1:
                    p_minus, q_minus, _, _, q_p_p, n_p_p, s_p_p, a_p_p, n_a_p_p = build_tree(
                        p_minus, q_minus, u, v, j - 1, epsilon, p_0, q_0
                    )
                else:
                    _, _, p_plus, q_plus, q_p_p, n_p_p, s_p_p, a_p_p, n_a_p_p = build_tree(
                        p_plus, q_plus, u, v, j - 1, epsilon, p_0, q_0
                    )

                # update all quantities
                n_sum = n_p + n_p_p
                if n_sum > 0:
                    if self._rng.binomial(1, n_p_p / (n_sum)):
                        q_prime = q_p_p

                n_prime = n_sum
                a_p = a_p + a_p_p
                n_a_p = n_a_p + n_a_p_p

                s_prime = (
                        s_p
                        * s_p_p
                        * int(np.dot((q_plus - q_minus), p_minus) > 0.0)
                        * int(np.dot((q_plus - q_minus), p_plus) > 0.0)
                )

            return p_minus, q_minus, p_plus, q_plus, q_p, n_p, s_p, a_p, n_a_p

        # sample momentum and compute hamiltonian
        p_0 = self._prec_L @ self._proposal_dist.get_sample()
        hamiltonian_0 = self._get_hamiltonian(p_0, self._state)

        # sample u for slice sampling
        u = self._rng.random() * np.exp(- hamiltonian_0)

        # init the position and momemtum variables for this iteration
        q_minus = self._state
        q_plus = self._state
        p_minus = p_plus = p_0

        # init the stopping criterion, the tree depth, and the number of nodes in C (q_0 is in C)
        s = 1
        j = 0
        n = 1

        # build the tree by doubling its size until stopping criterion is met
        while s == 1:
            v = self._rng.choice([-1, 1])
            if v == -1:
                trajectory.append({"type": "left"})
                p_minus, q_minus, _, _, q_prime, n_prime, s_prime, a, n_a = build_tree(
                    p_minus, q_minus, u, v, j, self._epsilon, p_0, self._state
                )
            else:
                trajectory.append({"type": "right"})
                _, _, p_plus, q_plus, q_prime, n_prime, s_prime, a, n_a = build_tree(
                    p_plus, q_plus, u, v, j, self._epsilon, p_0, self._state
                )

            if s_prime == 1:
                if self._rng.binomial(1, min(1., n_prime / n)):
                    self._state = q_prime

            n = n + n_prime
            s = (
                    s_prime *
                    int(np.dot((q_plus - q_minus), p_minus) > 0.0) *
                    int(np.dot((q_plus - q_minus), p_plus) > 0.0)
            )
            j += 1

        self._acceptance_rate += a / n_a / self._n_samples

        # update H_bar, epsilon, and epsilon_bar
        if self._iter <= self._M_adapt:
            self._H_bar = (
                    (1 - 1 / (self._iter + self._t_0)) * self._H_bar
                    + 1 / (self._iter + self._t_0) * (self._delta - a / n_a)
            )

            log_eps = self._mu - np.sqrt(self._iter) / self._gamma * self._H_bar
            term_1 = self._iter ** (-self._kappa) * log_eps
            term_2 = (1 - self._iter**(-self._kappa)) * np.log(self._epsilon_bar)

            self._epsilon = np.exp(log_eps)
            self._epsilon_bar = np.exp(term_1 + term_2)
        else:
            self._epsilon = self._epsilon_bar

        # save latest state
        self.chain[self._iter, :] = self._state
        self._iter += 1

        if self._plot:
            self._plot_trajectory(trajectory)

    @override
    def run(self):
        super().run()
        logger.info(f"Acceptance rate: {self._acceptance_rate}")


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
    Sampler = RandomWalkMetropolisSampler(posterior, n_samples=n_samples, sigma=np.sqrt(1.5), rng=rng,
                                          prec=cov)
    Sampler.run()
    samples = Sampler.chain
    n_vis = 500

    x = np.linspace(plot_limits[0][0], plot_limits[0][1], 100)
    y = np.linspace(plot_limits[1][0], plot_limits[1][1], 70)
    gx, gy = np.meshgrid(x, y)
    gz = np.zeros_like(gx)
    grad_z = np.zeros((gx.shape[0], gx.shape[1], 2))

    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            gz[i, j] = np.exp(posterior.log_density(np.array([gx[i, j], gy[i, j]])))
            grad_z[i, j, :] = posterior.grad_log_density(np.array([gx[i, j], gy[i, j]]))

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
