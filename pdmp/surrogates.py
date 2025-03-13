import os.path
import sys
import warnings
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

import numpy as np

from typing import override, Optional
from scipy.stats import qmc
from typing import cast
from tqdm import tqdm
from linear_operator.utils.warnings import NumericalWarning

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, ZeroMean, ConstantMeanGrad
from gpytorch.kernels import RBFKernel, ScaleKernel, RBFKernelGrad
from gpytorch.distributions.multivariate_normal import MultivariateNormal as gpyMultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal

from pdmp.distributions import Distribution, MultivariateNormal, Posterior, find_mean, find_curvature
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp import logger

dtype = torch.float64


class SurrogateModel(object):
    """
    Base class for surrogate models.
    """
    def __init__(self):
        """
        Initialize the surrogate model.
        """

    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the surrogate model at a point.

        Parameters:
        x (np.ndarray): The point at which the surrogate model is to be evaluated.

        Returns:
        float: The value of the surrogate model at the point x.
        """
        raise NotImplementedError

    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Compute the gradient of the surrogate model at a point.

        Parameters:
        x (np.ndarray): The point at which the gradient is to be computed.
        idx (int, optional): The index of the component of the grad to be computed. Default is None.

        Returns:
        float: The gradient of the surrogate model at the point x.
        """
        raise NotImplementedError

    def add_data(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:
        """
        Add data to the surrogate model.

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        dy_dx (np.ndarray): The gradient of the output data.
        """
        raise NotImplementedError

    def train(self, *args, **kwargs) -> None:
        """
        Train the surrogate model.
        """
        raise NotImplementedError


class LaplaceSurrogate(SurrogateModel):
    """
    Laplace approximation to the target distribution.
    """
    def __init__(
            self,
            target: Distribution,
            rng: np.random.Generator,
            mean: np.ndarray = None,
            cov: np.ndarray = None,
            x_0: np.ndarray = None,
            **kwargs
    ):
        """
        Initialize the Laplace approximation. If mean and or cov are not provided, they are computed using the target.

        Parameters:
        target (Distribution): The target distribution.
        mean (np.ndarray): The mean of the Laplace approximation.
        cov (np.ndarray): The covariance matrix of the Laplace approximation.
        x_0 (np.ndarray): The initial point for the Laplace approximation.
        """
        super().__init__()

        if isinstance(target, Posterior):
            target = cast(Posterior, target)

        if mean is None or cov is None:
            assert target is not None, "Target distribution must be provided if mean and cov are not provided."

            if mean is None:
                self.mean = find_mean(target, x_0)
            else:
                self.mean = mean
            if cov is None:
                self.cov = find_curvature(target, mean)
            else:
                self.cov = cov
        else:
            self.mean = mean
            self.cov = cov

        self.gaussian = MultivariateNormal(self.mean, self.cov, rng=rng)
        self.delta = self.gaussian.log_density(self.mean) - target.log_density(self.mean)

    @classmethod
    def from_dict(
            cls,
            config: dict,
            target: Distribution,
            rng: np.random.Generator
    ):
        """
        Create a Laplace approximation from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        target (Distribution): The target distribution.
        rng (np.random.Generator): The random number generator.

        Returns:
        LaplaceSurrogate: The Laplace approximation.
        """
        mean = np.array(config['mean']) if 'mean' in config else None
        cov = np.array(config['cov']) if 'cov' in config else None
        return cls(
            mean=mean,
            cov=cov,
            target=target,
            rng=rng
        )

    def eval(self, x: np.ndarray, delta: bool=False, **kwargs) -> np.ndarray:
        """
        Evaluate the Laplace approximation at a point.

        Parameters:
        x (np.ndarray): The point at which the Laplace approximation is to be evaluated.

        Returns:
        float: The value of the Laplace approximation at the point x.
        """
        return self.gaussian.log_density(x) - delta * self.delta

    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Compute the gradient of the Laplace approximation at a point.

        Parameters:
        x (np.ndarray): The point at which the gradient is to be computed.
        idx (int, optional): The index of the component of the gradient to be computed. Default is None.

        Returns:
        np.ndarray: The gradient of the Laplace approximation at the point x.
        """
        if idx is None:
            return self.gaussian.grad_log_density(x)
        else:
            return self.gaussian.grad_log_density(x)[idx]

    def add_data(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:
        """
        Add data to the Laplace approximation. Nothing to do here!

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data
        """
        pass

    def train(self, *args, **kwargs) -> None:
        """
        Update the Laplace approximation. Nothing to do here!

        Parameters:
        args: Additional arguments.
        kwargs: Additional keyword arguments.

        Returns:
        None
        """
        pass

    def get_samples(self, n: int) -> np.ndarray:
        """
        Get samples from the Laplace approximation.

        Parameters:
        n (int): The number of samples to generate.

        Returns:
        np.ndarray: An array of samples from the Laplace approximation.
        """
        return self.gaussian.get_sample(n)


class NeuralNetwork(SurrogateModel):
    """
    Neural network surrogate model based on PyTorch.
    """

    def __init__(
            self,
            target: Distribution,
            hidden_layers: list,
            rng: np.random.Generator,
            n_samples: int = 100,
            epochs:  int = 5000,
            batch_size: int = 20,
            weight_decay: float = 0.0,
            val_split: float = 0.3,
            patience: int = 100,
            print_every: int = 10,
            lr: float = 1e-3,
            lr_scheduler: str = None,
            lr_scheduler_params: dict = None,
            train_on_init: bool = True,
            update_model: list = None,
            **kwargs
    ):
        """
        Initialize the neural network surrogate model.

        Parameters:
        layer_sizes (list): List containing the number of neurons per layer.
        lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        if update_model is None:
            update_model = []
        hidden_layers = [target.get_dim()] + hidden_layers + [1]
        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if i < len(hidden_layers) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

        self.laplace = LaplaceSurrogate.from_dict(kwargs, target=target, rng=rng)

        # init all data
        self.x_data = None
        self.y_data = None
        self.x_data_new = []
        self.y_data_new = []
        self.n_data_buffer = 0
        if update_model is None:
            update_model = []
        self.update_model = copy.deepcopy(update_model)

        self.training_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'val_split': val_split,
            'weight_decay': weight_decay,
            'patience': patience,
            'print_every': print_every,
            'lr': lr,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_params': lr_scheduler_params
        }

        # add initial training data to update model array
        if len(update_model) > 0:
            for i in range(0, len(update_model)):
                self.update_model[i] += n_samples

        if train_on_init:
            samples = self.laplace.get_samples(n_samples)
            self.x_data = torch.tensor(samples, dtype=dtype)
            self.y_data = torch.zeros(n_samples)

            for i in range(n_samples):
                self.y_data[i] = torch.tensor(
                    target.log_density(samples[i]) - self.laplace.eval(samples[i], delta=True),
                    dtype=dtype
                )

            self.train(**self.training_params)

    @classmethod
    def from_dict(
            cls,
            config: dict,
            target: Distribution,
            rng: np.random.Generator,
            train_on_init: bool = True
    ):
        """
        Create a neural network surrogate model from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        rng (np.random.Generator, optional): The random number generator. Default is None.

        Returns:
        NeuralNetwork: The neural network surrogate model.
        """

        return cls(
            target=target,
            rng=rng,
            train_on_init=train_on_init,
            **config
        )

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=dtype)
        with torch.no_grad():
            return self.model(x_tensor).numpy() + self.laplace.eval(x, delta=True)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=dtype, requires_grad=True)
        y_tensor = self.model(x_tensor)
        gradients = grad(
            outputs=y_tensor,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_tensor),
            create_graph=True
        )[0]

        if idx is None:
            return gradients.detach().numpy() + self.laplace.grad(x)
        else:
            return gradients[idx].detach().numpy() + self.laplace.grad(x, idx=idx)

    @override
    def add_data(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:

        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        n = x.shape[0]

        self.x_data_new.append(x)
        self.y_data_new.append(y)

        self.n_data_buffer += n

        if len(self.update_model) > 0:
            if self.n_data_buffer + self.x_data.shape[0] >= self.update_model[0]:
                self.update_model.pop(0)

                x_new = np.concat(self.x_data_new)
                y_new = np.concat(self.y_data_new)

                for i in range(len(y_new)):
                    y_new[i] -= self.laplace.eval(x_new[i], delta=True)

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self.x_data = torch.vstack((self.x_data, x_new))
                self.y_data = torch.hstack((self.y_data, y_new))

                self.train(**self.training_params)

                self.x_data_new = []
                self.y_data_new = []
                self.n_data_buffer = 0

    def save_model(self, path: str = 'neural_network.th') -> None:
        """
        Save the neural network model to a file.

        Parameters:
        path (str): The path to the file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = 'neural_network.th') -> None:
        """
        Load the neural network model from a file.

        Parameters:
        path (str): The path to the file.
        """
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def train(
            self,
            *args,
            epochs: int,
            batch_size: int,
            print_every: int,
            patience: int,
            lr: float,
            val_split: float,
            weight_decay: float,
            lr_scheduler: str,
            lr_scheduler_params: dict,
            **kwargs
    ) -> None:
        """
        Train the neural network surrogate model using stored data.

        Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        print_every (int): Print training information every print_every epochs.
        patience (int): Number of epochs to wait for improvement in validation loss before stopping early.
        lr (float): Learning rate for the optimizer.
        lr_scheduler_step (int): Number of epochs after which to reduce the learning rate.
        lr_scheduler_gamma (float): Factor by which to reduce the learning rate.
        """

        train_losses = []
        val_losses = []
        patience_counter = 0

        num_val = int(val_split * len(self.x_data))
        x_train, x_val = self.x_data[:-num_val], self.x_data[-num_val:]
        y_train, y_val = self.y_data[:-num_val], self.y_data[-num_val:]

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if lr_scheduler == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_params)
        else:
            scheduler = None

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_state = None

        last_lr = lr

        logger.warning("Training neural network surrogate model ...")

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=epochs, file=sys.stdout, dynamic_ncols=True, leave=True, disable=disable_tqdm) as pbar:
            for epoch in range(epochs):

                self.model.train()

                if (epoch % print_every) == 0:
                    pbar.clear()
                    logger.debug(f"Epoch {epoch}/{epochs}")
                    pbar.refresh()
                    pbar.update(print_every)

                for x_batch, y_batch in train_loader:

                    optimizer.zero_grad()
                    y_pred = self.model(x_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    train_losses.append(loss.detach().numpy())
                    loss.backward()
                    optimizer.step()


                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(x_val).squeeze(), y_val).item()
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            pbar.clear()
                            logger.warning(f"Early stopping at epoch {epoch}.")
                            logger.warning(f"with best validation loss {best_val_loss:.6f}")
                            break

                if isinstance(scheduler, optim.lr_scheduler.StepLR):
                    scheduler.step()

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                if last_lr > scheduler.get_last_lr()[-1]:
                    pbar.clear()
                    last_lr = scheduler.get_last_lr()[-1]
                    logger.warning(f"At Iter {epoch} new learning rate: {last_lr}")
                    pbar.refresh()

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.save_model()

        # plot training and validation loss
        fig, ax = get_2d_despined_figure(
            figsize=(5, 3.5),
            equal_axes=False,
            axes_label=('Epoch', 'MSE'),
            keep_ticks=True
        )

        ax.semilogy(np.linspace(0, epoch, len(train_losses)), train_losses, label='Train')
        ax.semilogy(np.linspace(0, epoch, len(val_losses)), val_losses, label='Validation')
        ax.set_ylim(1e-5, 4e1)
        ax.legend()

        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures/training_validation_loss_{self.x_data.shape[0]}.pdf')


class ExactGPModel(ExactGP):
    """
    Exact Gaussian process model based on GPyTorch.
    """
    def __init__(
            self,
            train_x: torch.tensor,
            train_y: torch.tensor,
            likelihood: _GaussianLikelihoodBase,
            ard_num_dims: int
    ):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))

    def forward(self, x: torch.tensor) -> gpyMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpyMultivariateNormal(mean_x, covar_x)


class GaussianProcessBase(SurrogateModel):
    """
    Base class for Gaussian process surrogate models.
    """
    def __init__(
            self,
            rng: np.random.Generator,
            target: Distribution,
            lbfgs_steps: int = 5,
            n_restarts: int = 50,
            n_samples: int = 100,
            print_every: int = 1,
            lr: float = 0.5,
            tolerance_grad: float = 1e-7,
            tolerance_change: float = 1e-9,
            update_model: list = None,
            retrain_threshold: int = 1000,
            eval_strategy: str = 'mean',
            **kwargs
    ):
        """
        Initialize the Gaussian process surrogate model.

        Parameters:
        target (Distribution): The target distribution.
        rng (np.random.Generator): The random number generator.
        train_iters (int): Number of training iterations.
        n_restarts (int): Number of restarts for hyperparameter optimization.
        lr (float): Learning rate for the optimizer.
        lr_scheduler (str): Learning rate scheduler.
        lr_scheduler_params (dict): Learning rate scheduler parameters.
        print_every (int): Print training information every print_every iterations.
        update_model (list): List of number of samples after which to update the model.
        retrain_when_update (bool): Retrain the model when new data is added.
        """
        super().__init__()

        # get laplace approximation
        self.laplace = LaplaceSurrogate.from_dict(kwargs, target=target, rng=rng)
        self.rng = rng

        # init all data
        self.x_data: torch.tensor = None
        self.y_data: torch.tensor = None
        self.x_data_new = []
        self.y_data_new = []
        self.n_data_buffer = 0
        if update_model is None:
            update_model = []
        self.update_model = copy.deepcopy(update_model)
        for i in range(len(update_model)):
            self.update_model[i] += n_samples

        self.retrain_threshold = retrain_threshold

        # default is add_data_on because it changes to add_data_off when update_model is empty
        self.add_data_ = self.add_data_on

        self.training_params = {
            'lbfgs_steps': lbfgs_steps,
            'n_restarts': n_restarts,
            'print_every': print_every,
            'lr': lr,
            'tolerance_grad': tolerance_grad,
            'tolerance_change': tolerance_change
        }

        # these will be overwritten in derived classes
        self.model: Optional[gpytorch.models.ExactGP] = None
        self.likelihood: Optional[gpytorch.likelihoods._GaussianLikelihoodBase] = None

        if eval_strategy == 'mean':
            self.eval_ = self.eval_mean
            self.grad_ = self.grad_mean
        elif eval_strategy == 'mean_plus_std':
            self.eval_ = self.eval_mean_plus_std
            self.grad_ = self.grad_mean_plus_std
        else:
            raise ValueError(f"Evaluation strategy {eval_strategy} not implemented.\n" +
                             "Choose from: 'mean', 'mean_plus_std'")

    @classmethod
    def from_dict(
            cls,
            config: dict,
            target: Distribution,
            rng: np.random.Generator = None
    ):
        """
        Create a Gaussian process surrogate model from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        target (Distribution): The target distribution.
        rng (np.random.Generator, optional): The random number generator. Default is None.

        Returns:
        GaussianProcess: The Gaussian process surrogate model.
        """
        return cls(
            target=target,
            rng=rng,
            **config
        )

    @override
    def add_data(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:
        """
        Add data to the Gaussian process surrogate model. Nothing to do here!

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data
        """
        return self.add_data_(x, y, dy_dx)

    def add_data_on(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:
        """
        Add data to the Gaussian process surrogate model. Nothing to do here!

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data
        dy_dx (np.ndarray): The gradient of the output data with respect to x
        """
        raise NotImplementedError

    def add_data_off(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:
        """
        Add data to the Gaussian process surrogate model. Nothing to do here!

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data
        dy_dx (np.ndarray): The gradient of the output data with respect to x
        """
        pass

    @override
    def train(
            self,
            *args,
            lbfgs_steps: int,
            n_restarts: int,
            print_every: int,
            lr: float,
            tolerance_grad: float,
            tolerance_change: float,
            **kwargs
    ) -> None:
        """
        Update the Gaussian process surrogate model.
        """

        logger.warning(f"Training {self.__class__.__name__} surrogate model ...")

        # set train data and put model in training mode
        self.model.set_train_data(self.x_data, self.y_data, strict=False)
        self.model.train()
        self.likelihood.train()

        # define loss
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ
        warnings.simplefilter("ignore", category=NumericalWarning)

        # get hyperparameter initialisations
        n_params = int(sum([np.prod(param.shape) for name, param in self.model.named_parameters()]))
        sampler = qmc.LatinHypercube(n_params, scramble=True, rng=self.rng)
        sample = sampler.random(n_restarts)
        scaled_sample = qmc.scale(sample, -10, 20)
        initial_params = torch.tensor(scaled_sample, dtype=dtype)

        # init best model and loss
        best_model = None
        best_loss = float('inf')

        # plot training and validation loss
        fig, ax = get_2d_despined_figure(
            figsize=(5, 3.5),
            equal_axes=False,
            axes_label=('Iter', 'MLL'),
            keep_ticks=True
        )
        losses_min = 1e10
        losses_max = -1e10
        best_iter = 0
        train_losses_all = []

        with tqdm(
            total=n_restarts,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            disable=disable_tqdm
        ) as pbar:

            for restart in range(n_restarts):

                train_losses = []

                # init optimizer and scheduler
                optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=lr,
                    max_iter=lbfgs_steps,
                    tolerance_grad=tolerance_grad,
                    tolerance_change=tolerance_change,
                    line_search_fn='strong_wolfe'
                )

                def closure():
                    optimizer.zero_grad()
                    output = self.model(self.x_data)
                    loss = -mll(output, self.y_data)
                    loss.backward()
                    return loss

                # set hyperparameters and initialise model
                hyper_params = {}
                counter = 0
                for name, param in self.model.named_parameters():
                    shape = param.shape
                    length = int(np.prod(shape))
                    hyper_params[name] = initial_params[restart, counter:counter + length].reshape(shape)
                    counter += length

                self.model.initialize(**hyper_params)

                # print hyperparameter values
                pbar.clear()
                logger.info(f"Restart {restart}/{n_restarts} with \n" + self.log_state_dict())
                pbar.refresh()

                # try block needed to catch non-PSD errors. in that case, interation is just skipped
                try:
                    for i in range(lbfgs_steps):

                        loss = optimizer.step(closure)
                        train_losses.append(loss.item())

                        # print current hyperparams and loss
                        if (i % print_every) == 0:
                            pbar.clear()
                            logger.debug(
                                f"   Iter {i}/{lbfgs_steps}"
                                + f" Loss: {loss.item():.3f},"
                                + self.log_state_dict(end_of_line=', ')
                            )
                            pbar.refresh()

                    # check if current model is best
                    if loss.item() < best_loss:
                        logger.info(f'New best loss: {loss.item()} with params:')
                        logger.info(self.log_state_dict())
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_loss = loss.item()
                        best_iter = restart

                # catch exception if non-PSD error occurs
                except gpytorch.linear_operator.utils.errors.NotPSDError as e:
                    pbar.clear()
                    logger.info(f"Non-PSD error: {e}")
                    pbar.refresh()

                except gpytorch.linear_operator.utils.errors.NanError as e:
                    pbar.clear()
                    logger.info(f"NaN error: {e}")
                    pbar.refresh()

                train_losses_np = np.array(train_losses)
                if len(train_losses) > 0:
                    losses_min = np.min((np.min(train_losses_np), losses_min))
                train_losses_all.append(train_losses_np)
                pbar.update()

        # load best model print parameters
        self.model.load_state_dict(best_model)
        logger.warning(f"Best model with loss {best_loss} and params:\n" + self.log_state_dict())
        self.save_model()

        for i in range(n_restarts):
            if i == best_iter:
                ax.plot(train_losses_all[i], color='C0', alpha=1, lw=1.5)
            else:
                ax.plot(train_losses_all[i], color='C0', alpha=0.3, lw=1.)

        ax.set_ylim(losses_min - 0.2*np.abs(losses_min), losses_min + 10.)

        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures/mll_{self.x_data.shape[0]}.pdf')

        # set model into evaluation mode
        self.model.eval()
        self.likelihood.eval()

    def log_state_dict(self, end_of_line: str = '\n') -> str:
        """
        Log the state dictionary of the model.

        Parameters
        end_of_line : str
            The end of line character. Default is '\n'.

        Returns
            str: The state dictionary of the model as a string
        """

        named_params = ""

        for name, param in self.model.named_parameters():
            named_params += f"  {name}: {param.data}" + end_of_line

        return named_params

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.eval_(x, **kwargs)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        return self.grad_(x, idx=idx)

    def eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the mean of the Gaussian process model.

        Parameters:
        x (np.ndarray): The input data.
        kwargs: Additional keyword arguments

        Returns:
        np.ndarray: The mean of the Gaussian process model.
        """
        raise NotImplementedError

    def eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the mean of the Gaussian process model and add the standard deviation.

        Parameters:
        x (np.ndarray): The input data.
        kwargs: Additional keyword arguments

        Returns:
        np.ndarray: The mean of the Gaussian process model plus the standard deviation.
        """
        raise NotImplementedError

    def grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Compute the gradient of the mean of the Gaussian process model.

        Parameters:
        x (np.ndarray): The input data.
        idx (int): The index of the component of the gradient to be computed. Default is None.

        Returns:
        np.ndarray: The gradient of the mean of the Gaussian process model.
        """
        raise NotImplementedError

    def grad_mean_plus_std(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Compute the gradient of the mean of the Gaussian process model plus the standard deviation.

        Parameters:
        x (np.ndarray): The input data.
        idx (int): The index of the component of the gradient to be computed. Default is None.

        Returns:
        np.ndarray: The gradient of the mean of the Gaussian process model plus the standard deviation.
        """
        raise NotImplementedError

    def save_model(self, path: str = 'model_params') -> None:
        """
        Save the neural network model to a file.

        Parameters:
        path (str): The path to the file.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), os.path.join(path, 'model_params.th'))
        np.savetxt(os.path.join(path, 'x_data.dat'), self.x_data.numpy())
        np.savetxt(os.path.join(path, 'y_data.dat'), self.y_data.numpy())

    def load_model(self, path: str = 'model_params') -> None:
        """
        Load the neural network model from a file.

        Parameters:
        path (str): The path to the file.
        """

        model_params = torch.load(os.path.join(path, 'model_params.th'))
        self.x_data = torch.tensor(np.loadtxt(os.path.join(path, 'x_data.dat')), dtype=dtype)
        self.y_data = torch.tensor(np.loadtxt(os.path.join(path, 'y_data.dat')), dtype=dtype)

        self.model.load_state_dict(model_params)
        self.model.set_train_data(self.x_data, self.y_data, strict=False)
        self.model.eval()
        self.likelihood.eval()

class GaussianProcess(GaussianProcessBase):
    """
    Gaussian process surrogate model based on GPyTorch.
    """
    def __init__(
            self,
            rng: np.random.Generator,
            target: Distribution,
            train_on_init: bool = True,
            n_samples: int = 100,
            **kwargs
    ):
        """
        Initialize the Gaussian process surrogate model.
        """
        super().__init__(rng=rng, target=target, n_samples=n_samples, **kwargs)

        # define likelihood, get model, and set optimizer
        self.likelihood = GaussianLikelihood()
        self.model = ExactGPModel(None, None, self.likelihood, target.get_dim())

        # train model unless specified otherwise
        if train_on_init:
            samples = self.laplace.get_samples(n_samples)
            self.x_data = torch.tensor(samples, dtype=dtype)
            self.y_data = torch.zeros(n_samples)

            for i in range(n_samples):
                self.y_data[i] = torch.tensor(
                    target.log_density(samples[i]) - self.laplace.eval(samples[i], delta=True),
                    dtype=dtype
                )

            self.train(**self.training_params)

        logger.info(f'{self.__class__.__name__} surrogate model initialized.')

    @override
    def add_data_on(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:

        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        n = x.shape[0]

        self.x_data_new.append(x)
        self.y_data_new.append(y)

        self.n_data_buffer += n

        if len(self.update_model) > 0:
            if self.n_data_buffer + self.x_data.shape[0] >= self.update_model[0]:
                self.update_model.pop(0)

                x_new = np.concat(self.x_data_new)
                y_new = np.concat(self.y_data_new)

                for i in range(len(y_new)):
                    y_new[i] -= self.laplace.eval(x_new[i], delta=True)

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self.x_data = torch.vstack((self.x_data, x_new))
                self.y_data = torch.hstack((self.y_data, y_new))

                if len(self.x_data) < self.retrain_threshold:
                    logger.info(f'Retraining model with {len(self.x_data)} data points.')
                    self.train(**self.training_params)
                else:
                    logger.info(f'Updating model with {len(self.x_data)} data points.')
                    self.model.set_train_data(self.x_data, self.y_data, strict=False)

                self.x_data_new = []
                self.y_data_new = []
                self.n_data_buffer = 0
        else:
            self.add_data_ = self.add_data_off

    @override
    def eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=False
        )

        with (
            torch.no_grad(),
            gpytorch.settings.skip_posterior_variances(True)
        ):
            return self.model(x_tensor).mean.squeeze().numpy() + self.laplace.eval(x, delta=True)

    @override
    def eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=False
        )

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(True)
        ):
            y = self.model(x_tensor)

        gp = y.mean.squeeze().numpy() + y.variance.sqrt().squeeze().numpy()

        return gp + self.laplace.eval(x, delta=True)

    @override
    def grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=True
        )

        with gpytorch.settings.skip_posterior_variances(True), gpytorch.settings.fast_computations(
                False, False, False):
            y_tensor = self.model(x_tensor).mean
            gradients = grad(
                outputs=y_tensor,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(y_tensor),
                create_graph=True
            )[0].squeeze().detach().numpy()

        if idx is None:
            return gradients + self.laplace.grad(x)
        else:
            return gradients[idx] + self.laplace.grad(x, idx=idx)

    @override
    def grad_mean_plus_std(self, x: np.ndarray, idx: int = None):
        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=True
        )
        with (
            gpytorch.settings.skip_posterior_variances(False),
            gpytorch.settings.fast_computations(False, False, False),
            gpytorch.settings.fast_pred_var(True)
        ):
            model_output = self.model(x_tensor)
            y_tensor = model_output.mean + model_output.variance.sqrt()
            gradients = grad(
                outputs=y_tensor,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(y_tensor),
                create_graph=True
            )[0].squeeze().detach().numpy()

        if idx is None:
            return gradients + self.laplace.grad(x)
        else:
            return gradients[idx] + self.laplace.grad(x, idx=idx)


class DerivativeGPModel(ExactGP):
    """
    Gaussian process model based on GPyTorch that also observes gradients.
    """
    def __init__(
            self,
            train_x: torch.tensor,
            train_y: torch.tensor,
            likelihood: _GaussianLikelihoodBase,
            ard_num_dims: int
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMeanGrad()
        self.covar_module = ScaleKernel(RBFKernelGrad(ard_num_dims=ard_num_dims))

    def forward(self, x: torch.tensor) -> MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class DerivativeGaussianProcess(GaussianProcessBase):
    """
    Gaussian process surrogate model based on GPyTorch that also observes gradients.
    """

    def __init__(
            self,
            rng: np.random.Generator,
            target: Distribution,
            train_on_init: bool = True,
            n_samples: int = 100,
            **kwargs
    ):
        """
        Initialize the Derivative Gaussian process surrogate model.
        """
        super().__init__(rng=rng, target=target, n_samples=n_samples, **kwargs)

        # define likelihood, get model, and set optimizer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=target.get_dim() + 1)
        self.model = DerivativeGPModel(None, None, self.likelihood, target.get_dim())

        # train model unless specified otherwise
        if train_on_init:
            samples = self.laplace.get_samples(n_samples)
            self.x_data = torch.tensor(samples, dtype=dtype)
            self.y_data = torch.zeros(n_samples, target.get_dim() + 1, dtype=dtype)

            for i in range(n_samples):
                self.y_data[i, 0] = torch.tensor(
                    target.log_density(samples[i]) - self.laplace.eval(samples[i], delta=True),
                    dtype=dtype
                )
                self.y_data[i, 1:] = torch.tensor(
                    target.grad_log_density(samples[i]) - self.laplace.grad(samples[i]),
                    dtype=dtype
                )

            self.train(**self.training_params)

        logger.info(f'{self.__class__.__name__} surrogate model initialized.')

    @override
    def add_data_on(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None) -> None:

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        dy_dx = np.atleast_2d(dy_dx)
        y = np.hstack((y, dy_dx))

        n = x.shape[0]

        self.x_data_new.append(x)
        self.y_data_new.append(y)

        self.n_data_buffer += n

        if len(self.update_model) > 0:
            if self.n_data_buffer + self.x_data.shape[0] >= self.update_model[0]:
                self.update_model.pop(0)

                x_new = np.concat(self.x_data_new)
                y_new = np.concat(self.y_data_new)

                for i in range(len(y_new)):
                    y_new[i, 0]  -= self.laplace.eval(x_new[i], delta=True)
                    y_new[i, 1:] -= self.laplace.grad(x_new[i])

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self.x_data = torch.vstack((self.x_data, x_new))
                self.y_data = torch.vstack((self.y_data, y_new))

                if len(self.x_data) < self.retrain_threshold:
                    self.train(**self.training_params)
                else:
                    self.model.set_train_data(self.x_data, self.y_data, strict=False)

                self.x_data_new = []
                self.y_data_new = []
                self.n_data_buffer = 0

        else:
            self.add_data_ = self.add_data_off

    @override
    def eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=False
        )

        with (
            torch.no_grad(),
            gpytorch.settings.skip_posterior_variances(True),
            gpytorch.settings.fast_computations(False, False, False),
        ):
            return self.model(x_tensor).mean[:, 0].squeeze().detach().numpy() + self.laplace.eval(x, delta=True)

    @override
    def eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=False
        )

        with (
            torch.no_grad(),
            gpytorch.settings.fast_computations(False, False, False),
            gpytorch.settings.fast_pred_var(True)
        ):
            y = self.model(x_tensor)

        gp = y.mean[:, 0].squeeze().detach().numpy() + y.variance[:, 0].sqrt().squeeze().detach().numpy()
        return gp + self.laplace.eval(x, delta=True)

    @override
    def grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=True
        )

        with (
            torch.no_grad(),
            gpytorch.settings.skip_posterior_variances(True),
            gpytorch.settings.fast_computations(False, False, False)
        ):
            gradient = self.model(x_tensor).mean[:, 1:].squeeze().detach().numpy()

        if idx is None:
            return gradient + self.laplace.grad(x)
        else:
            return gradient[idx] + self.laplace.grad(x, idx=idx)

    @override
    def grad_mean_plus_std(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(
            np.atleast_2d(x),
            dtype=dtype,
            requires_grad=True
        )

        with (
            torch.no_grad(),
            gpytorch.settings.fast_computations(False, False, False),
            gpytorch.settings.fast_pred_var(True)
        ):
            y = self.model(x_tensor)
            gradient = y.mean[:, 1:].squeeze().detach().numpy() + y.variance[:, 1:].sqrt().squeeze().detach().numpy()

        if idx is None:
            return gradient + self.laplace.grad(x)
        else:
            return gradient[idx] + self.laplace.grad(x, idx=idx)
