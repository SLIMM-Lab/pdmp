import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from typing import Tuple
from tqdm import tqdm

from src.distributions import Distribution, MultivariateNormal
from src.distributions import get_sample
from src import logger

import seaborn as sns


class StepSampler:

    def __init__(self,
                 target: Distribution,
                 n_samples: int = 10000,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 prec: np.ndarray = None,
                 cov_factor: float = 1.):
        """
        Initialize the StepSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        prec (np.ndarray, optional): Preconditioner matrix. Default is None.
        """
        self.target_ = target
        self.dim_ = self.target_.get_dim()
        self.n_samples_ = n_samples
        self.state_ = np.zeros(self.dim_, dtype=np.float64)
        self.chain_ = np.zeros((self.n_samples_, self.dim_), dtype=np.float64)
        self.iter_ = 1
        self.n_accept_ = 1
        self.n_accept_last_ = 1
        self.rescale_interval_ = 100
        self.cov_factor_ = cov_factor

        if rng is None and seed is None:
            self.rng_ = np.random.default_rng(0)
        elif rng is None:
            self.rng_ = np.random.default_rng(seed)
        else:
            self.rng_ = rng

        self.proposal_dist_ = MultivariateNormal(np.zeros(self.dim_),
                                                 np.eye(self.dim_),
                                                 rng=self.rng_)

        if prec is None:
            self.prec_ = np.eye(self.dim_, dtype=np.float64)
        elif (prec.shape[0] == prec.shape[1] == self.dim_) and (len(prec.shape) == 2):
            self.prec_ = prec
        elif len(prec.shape) == 1 and len(prec) == self.dim_:
            self.prec_ = np.diagonal(prec)
        else:
            raise Exception("No valid preconditioner specified.")

        self.prec_L_ = np.linalg.cholesky(self.prec_)

    def reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        self.iter_ = 1
        self.n_accept_ = 1
        self.n_accept_last_ = 1

        # check if self.target_ has a method get_prior_sample, and if so, use it
        if x_0 is not None:
            self.state_ = x_0
        elif hasattr(self.target_, 'get_prior_sample'):
            self.state_ = self.target_.get_prior_sample()
        else:
            self.state_ = get_sample(self.target_.get_dim(), rng=self.rng_)
        self.chain_ = np.zeros((self.n_samples_, self.dim_))
        self.chain_[0, :] = self.state_

    def get_sample_covariance(self) -> np.ndarray:
        """
        Get the sample covariance of the chain.

        Returns:
        np.ndarray: The sample covariance matrix.
        """
        if self.dim_ == 1:
            return np.array([[np.var(self.chain_)]])
        else:
            return np.cov(self.chain_, rowvar=True)

    def set_preconditioner(self, cov):
        """
        Set the preconditioner matrix.

        Parameters:
        cov (np.ndarray): The covariance matrix to set as the preconditioner.
        """
        self.prec_ = cov
        self.prec_L_ = np.linalg.cholesky(self.prec_)

    def step(self):
        """
        Perform a single step.
        """
        raise NotImplementedError("The step method must be implemented in a subclass.")

    def run(self):
        """
        Run the Metropolis-Hastings sampler.
        """
        raise NotImplementedError("The step method must be implemented in a subclass.")

    def write_data(self, folder: str):
        """
        Write the chain data to a file.

        Parameters:
        filename (str): The name of the folder to write the data to.
        """

        data = {
            'acceptance_rate': self.n_accept_ / self.n_samples_,
        }

        with open(os.path.join(folder, 'other.pkl'), 'wb') as f:
            pickle.dump(data, f)

        np.savetxt(os.path.join(folder, 'samples.dat'), self.chain_)


class MetropolisHastingsSampler(StepSampler):
    """
    A class to perform Metropolis-Hastings sampling.
    """

    def __init__(self,
                 target: Distribution,
                 sigma: float = 0.5,
                 n_samples: int = 10000,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 prec: np.ndarray = None,
                 cov_factor: float = 1.,
                 x_0: np.ndarray = None):
        """
        Initialize the MetropolisHastingsSampler class.

        Parameters:
        target (Distribution): The target distribution to sample from.
        sigma (float, optional): The standard deviation of the proposal distribution. Default is 0.5.
        n_samples (int, optional): The number of samples to generate. Default is 10000.
        rng (np.random.Generator, optional): Random number generator. Default is None.
        seed (int, optional): Seed for the random number generator. Default is None.
        prec (np.ndarray, optional): Preconditioner matrix. Default is None.
        """
        super().__init__(target=target, n_samples=n_samples, rng=rng, seed=seed, prec=prec, cov_factor=cov_factor)
        self.sigma_ = sigma
        self.log_density_old_ = 0.
        self.reset(x_0)
        self.proposals_ = np.zeros_like(self.chain_)
        self.proposals_[0, :] = self.state_
        self.accepted_ = np.zeros(self.n_samples_, dtype=bool)
        self.accepted_[0] = True

    def reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        super().reset(x_0)
        self.log_density_old_ = self.target_.log_density(self.state_)

    def step(self):
        """
        Perform a single Metropolis-Hastings step.
        """
        proposal = self.state_ + self.sigma_ * self.prec_L_ @ self.proposal_dist_.get_sample()
        self.proposals_[self.iter_, :] = proposal
        log_density_new = self.target_.log_density(proposal)
        # log_density_current = self.target_.log_density(self.state_)

        if (log_density_new - self.log_density_old_) > np.log(self.rng_.uniform()):
            self.state_ = proposal
            self.log_density_old_ = log_density_new
            self.n_accept_ += 1
            self.n_accept_last_ += 1
            self.accepted_[self.iter_] = True

        self.chain_[self.iter_, :] = self.state_
        self.iter_ += 1

    def run(self):
        """
        Run the Metropolis-Hastings sampler.
        """

        with tqdm(total=self.n_samples_) as pbar:
            for i in range(1, self.n_samples_):

                if i == 1000:
                    self.set_preconditioner(self.cov_factor_ * 2.38**2/self.dim_ * self.get_sample_covariance())

                if i % self.rescale_interval_ == 0:
                    logger.info(f"Iteration: {i}")
                    logger.info(f"Acceptance rate: {self.n_accept_last_ / self.rescale_interval_}")
                    if self.n_accept_last_ / self.rescale_interval_ < 0.2:
                        self.set_preconditioner(0.9 * self.prec_)
                        logger.info("Decrease")
                    elif self.n_accept_last_ / self.rescale_interval_ > 0.25:
                        self.set_preconditioner(1.1 * self.prec_)
                        logger.info("Increase")
                    self.n_accept_last_ = 0
                self.step()
                pbar.update()
        logger.info(f"Total acceptance rate: {self.n_accept_ / self.n_samples_}")


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
                 x_0: np.ndarray = None):
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
        self.grad_log_density_ = np.zeros(self.dim_, dtype=np.float64)
        self.adjusted_ = adjusted
        self.reset(x_0)

    def reset(self, x_0: np.ndarray = None):
        """
        Reset the sampler state.
        """
        super().reset(x_0)
        self.log_density_ = self.target_.log_density(self.state_)
        self.grad_log_density_ = self.target_.grad_log_density(self.state_)

    def log_proposal_density(self, y: np.ndarray, x: np.ndarray, grad_x: np.ndarray) -> float:
        """
        Calculate the log proposal density.

        Parameters:
        y (np.ndarray): The proposed state.
        x (np.ndarray): The current state.
        grad_x (np.ndarray): The gradient of the log density at the current state.

        Returns:
        float: The log proposal density.
        """
        diff = y - x - 0.5 * self.sigma_ ** 2 * self.prec_ @ grad_x
        return - 0.5 * np.linalg.norm(np.linalg.solve(self.sigma_ * self.prec_L_, diff))**2

    def step(self):
        """
        Perform a single Langevin dynamics step.
        """
        self.randn_ = self.proposal_dist_.get_sample()
        # self.randn_ = np.array([0.8037, -1.715])
        prop = (self.state_ + self.sigma_ * self.prec_L_ @ self.randn_
                + 0.5 * self.sigma_ ** 2 * self.prec_ @ self.grad_log_density_)
        log_density_prop = self.target_.log_density(prop)
        grad_log_density_prop = self.target_.grad_log_density(prop)
        log_numerator = log_density_prop + self.log_proposal_density(self.state_, prop, grad_log_density_prop)
        log_denominator = self.log_density_ + self.log_proposal_density(prop, self.state_, self.grad_log_density_)

        if not self.adjusted_ or ((log_numerator - log_denominator) > np.log(self.rng_.uniform())):
            self.state_ = prop
            self.chain_[self.iter_, :] = prop
            self.log_density_ = log_density_prop
            self.grad_log_density_ = grad_log_density_prop
            self.n_accept_ += 1
            self.n_accept_last_ += 1
        else:
            self.chain_[self.iter_, :] = self.state_

        self.iter_ += 1

    def run(self):
        """
        Run the Langevin dynamics sampler.
        """
        with tqdm(total=self.n_samples_) as pbar:
            for i in range(1, self.n_samples_):
                if i == 1000:
                    self.set_preconditioner(self.cov_factor_ * np.power(self.dim_, -1./3.) * self.get_sample_covariance())

                if i % self.rescale_interval_ == 0:
                    logger.info(f"Iteration: {i}")
                    logger.info(f"Acceptance rate: {self.n_accept_last_ / self.rescale_interval_}")
                    if self.n_accept_last_ / self.rescale_interval_ < 0.5:
                        self.set_preconditioner(0.9 * self.prec_)
                        logger.info("Decrease")
                    elif self.n_accept_last_ / self.rescale_interval_ > 0.6:
                        self.set_preconditioner(1.1 * self.prec_)
                        logger.info("Increase")
                    self.n_accept_last_ = 0
                self.step()
                pbar.update()
        logger.info(f"Acceptance rate: {self.n_accept_ / self.n_samples_}")


class HamiltonianMonteCarlo(StepSampler):
    """
    Hamiltonian Monte Carlo (HMC) sampler class for sampling from a target distribution using HMC.
    """

    def __init__(self,
                 target: Distribution,
                 step_scale: float = 0.1,
                 leap_frog_steps: int = 20,
                 n_samples: int = 10000,
                 prec: np.ndarray = None,
                 rng: np.random.Generator = None,
                 seed: int = None,
                 plot: bool = False,
                 x_0: np.ndarray = None,
                 plot_limits: Tuple[float, float] = None):
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

        self.step_scale_ = step_scale
        self.leap_frog_steps_ = leap_frog_steps
        self.n_accepted_ = 1
        self.p_old_ = 0.0

        self.prec_inv_ = np.linalg.inv(self.prec_)
        self.prec_det_ = np.linalg.det(self.prec_)

        super().reset(x_0)

        if plot and (self.dim_ == 1 or self.dim_ == 2):
            self.plot_ = True
            self.init_plot(plot_limits)
        else:
            self.plot_ = False

    def leap_frog_step_(self, p0: np.ndarray, q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single leapfrog step.

        Parameters:
        p0 (np.ndarray): Initial momentum.
        q0 (np.ndarray): Initial position.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Updated momentum and position.
        """
        p = p0 + self.step_scale_ * 0.5 * self.target_.grad_log_density(q0)
        q = q0 + self.step_scale_ * self.prec_inv_ @ p
        p = p + self.step_scale_ * 0.5 * self.target_.grad_log_density(q)
        return p, q

    def get_hamiltonian(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Hamiltonian.

        Parameters:
        p (np.ndarray): Momentum.
        q (np.ndarray): Position.

        Returns:
        float: The Hamiltonian value.
        """
        potential = - self.target_.log_density(q)
        kinetic = 0.5 * p @ self.prec_inv_ @ p
        kinetic = kinetic + 0.5 * np.log((2. * np.pi) ** self.dim_ * self.prec_det_)

        return potential + kinetic

    def step_(self):
        """
        Perform a single HMC step.
        """
        p_hist = np.empty((self.leap_frog_steps_ + 1, self.dim_))
        q_hist = np.empty((self.leap_frog_steps_ + 1, self.dim_))

        p_hist[0, :] = p_0 = self.prec_L_ @ self.proposal_dist_.get_sample()
        q_hist[0, :] = q_0 = self.state_

        hamiltonian_0 = self.get_hamiltonian(p_0, q_0)

        for i in range(1, self.leap_frog_steps_ + 1):
            p_hist[i, :], q_hist[i, :] = self.leap_frog_step_(p_hist[i - 1, :], q_hist[i - 1, :])

        hamiltonian = self.get_hamiltonian(p_hist[-1, :], q_hist[-1, :])

        accept = (- hamiltonian + hamiltonian_0) > np.log(self.rng_.uniform())
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
                # self.ax_.arrow(q_0[0], self.p_old_, 0, p_0[0] - self.p_old_, alpha=0.5,
                #                color='k', width=aw, length_includes_head=True, head_width=7*aw)
                # self.ax_.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4.)
                self.ax_.plot(*q_hist.transpose(), *p_hist.transpose(), marker=".", markersize=4., color=color)

                if accept:
                    self.p_old_ = p_hist[-1, 0]

        self.chain_[self.iter_, :] = self.state_
        self.iter_ += 1

    def run(self):
        """
        Run the HMC sampler.
        """
        with tqdm(total=self.n_samples_) as pbar:
            for i in range(1, self.n_samples_):
                self.step_()
                pbar.update()
                if self.plot_:
                    logger.info(f"Iteration: {i}")

                    # self.fig_.fig.show()
                elif i % 1000 == 0:
                    logger.info(f"Iteration: {i}")
        logger.info(f"Acceptance rate: {self.n_accepted_ / self.n_samples_}")

        if self.plot_:
            self.fig_.fig.show()
            self.fig_.savefig(f"plots/hmc_2d_l2.p_x")

    def init_plot(self, plot_limits: Tuple[float, float] = None):
        """
        Initialize the plot for the Hamiltonian Monte Carlo sampler.

        This method sets up the plotting environment, including the figure and axes,
        and configures the plot limits and labels based on the dimensionality of the
        target distribution.

        Note:
            This method is only applicable if the dimensionality of the target distribution
            is 1 or 2 and the `plot` parameter is set to True during initialization.
        """

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
                    self.gz_[i, j] = np.exp(self.target_.log_density(np.array([self.gx_[i, j],
                                                                               self.gy_[i, j]])))
                else:
                    self.gz_[i, j] = np.exp(self.target_.log_density(np.array([self.gx_[i, j]])) +
                                            self.proposal_dist_.log_density(np.array([self.gy_[i, j]])))

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
