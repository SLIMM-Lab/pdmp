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
    """Base class for surrogate models."""

    gaussian: Optional[MultivariateNormal] = None
    """Laplace approximation of the target distribution as base model."""

    def __init__(self, *args, **kwargs):
        """Initialize the surrogate model.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.
        """

    @classmethod
    def from_dict(cls, target: Distribution, rng: np.random.Generator,
                  **kwargs):
        """Create a surrogate model derived class from a dictionary.

        Args:
            target: The target distribution.
            rng: The random number generator.
            kwargs: Additional keyword arguments.

        Returns:
            SurrogateModel: The surrogate model.
        """
        return cls(target=target, rng=rng, **kwargs)

    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the surrogate model at a point.

        Args:
            x: The point at which the surrogate model is to be evaluated.

        Returns:
            np.ndarray: The value of the surrogate model at the point x.
        """
        raise NotImplementedError

    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """Compute the gradient of the surrogate model at a point.

        Args:
            x: The point at which the gradient is to be computed.
            idx: The index of the component of the grad to be computed. Default is None.

        Returns:
            np.ndarray: The gradient of the surrogate model at the point x.
        """
        raise NotImplementedError

    def add_data(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dy_dx: np.ndarray = None) -> None:
        """Add data to the surrogate model.

        Args:
            x: The input data.
            y: The output data.
            dy_dx: The gradient of the output data. Default is None.
        """
        pass

    def train(self, *args, **kwargs):
        """Train the surrogate model.

        Args:
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        pass


class ConstantSurrogate(SurrogateModel):
    """Constant surrogate model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.array(0.0)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            return np.zeros_like(x)
        else:
            return np.array(0.0)


class RandomConstantSurrogate(SurrogateModel):
    """Random constant surrogate model."""

    def __init__(self,
                 rng: np.random.Generator,
                 *args,
                 var: float = 1.0,
                 **kwargs):
        """Initialize the random constant surrogate model.

        Args:
            rng: The random number generator.
            args: Additional positional arguments.
            var: The variance of the random constant surrogate model.
            kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._var = var
        self._rng = rng

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.array(0.0)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        if idx is None:
            return self._rng.uniform(-0.5 * self._var,
                                     0.5 * self._var,
                                     size=len(x))
        else:
            return np.array(self._rng.uniform(-0.5 * self._var,
                                              0.5 * self._var))


class LaplaceSurrogate(SurrogateModel):
    """Laplace approximation to the target distribution."""

    def __init__(self,
                 target: Distribution,
                 rng: np.random.Generator,
                 *args,
                 mean: np.ndarray = None,
                 cov: np.ndarray = None,
                 x_0: np.ndarray = None,
                 **kwargs):
        """Initialize the Laplace approximation. If mean and or cov are not provided, they are computed using the target.

        Args:
            target: The target distribution.
            rng: The random number generator.
            args: Additional keyword arguments.
            mean: The mean of the Laplace approximation.
            cov: The covariance matrix of the Laplace approximation.
            x_0: The initial point for the Laplace approximation.
            kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        if isinstance(target, Posterior):
            target = cast(Posterior, target)

        if mean is None or cov is None:
            assert target is not None, "Target distribution must be provided if mean and cov are not provided."

            if mean is None:
                self._mean = find_mean(target, x_0)
                mean = self._mean
            else:
                self._mean = mean
            if cov is None:
                self._cov = find_curvature(target, mean)
            else:
                self._cov = cov
        else:
            self._mean = mean
            self._cov = cov

        self.gaussian = MultivariateNormal(self._mean, self._cov, rng=rng)
        self._delta = self.gaussian.log_density(
            self._mean) - target.log_density(self._mean)

    @override
    def eval(self, x: np.ndarray, delta: bool = False, **kwargs) -> np.ndarray:
        """Evaluate the Laplace approximation at a point.

        Args:
            x: The point at which the Laplace approximation is to be evaluated.
            delta: If True, return the value of the Laplace approximation at the point x minus the delta term. Default is False.
            kwargs: Additional keyword arguments.

        Returns:
            float: The value of the Laplace approximation at the point x.
        """
        return self.gaussian.log_density(x) - delta * self._delta

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        if idx is None:
            return self.gaussian.grad_log_density(x)
        else:
            return self.gaussian.grad_log_density(x)[idx]

    def get_samples(self, n: int) -> np.ndarray:
        """Get samples from the Laplace approximation.

        Args:
            n: The number of samples to generate.

        Returns:
            np.ndarray: An array of samples from the Laplace approximation.
        """
        return self.gaussian.get_sample(n)


class NeuralNetwork(SurrogateModel):
    """Neural network surrogate model based on PyTorch."""

    def __init__(self,
                 target: Distribution,
                 hidden_layers: list,
                 rng: np.random.Generator,
                 *args,
                 n_samples: int = 100,
                 epochs: int = 5000,
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
                 **kwargs):
        """Initialize the neural network surrogate model.

        Args:
            target: The target distribution.
            hidden_layers: The number of hidden layers in the neural network.
            rng: The random number generator.
            args: Additional positional arguments.
            n_samples: The number of samples to generate.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            weight_decay: Weight decay for the optimizer.
            val_split: Fraction of data to use for validation.
            patience: Number of epochs to wait for improvement in validation loss before stopping early.
            print_every: Print training information every print_every epochs.
            lr: Learning rate for the optimizer.
            lr_scheduler: Learning rate scheduler.
            lr_scheduler_params: Learning rate scheduler parameters.
            train_on_init: Whether to train the model on initialization.
            update_model: List of number of samples after which to update the model.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if update_model is None:
            update_model = []
        hidden_layers = [target.dim] + hidden_layers + [1]
        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if i < len(hidden_layers) - 2:
                layers.append(nn.Tanh())
        self._model = nn.Sequential(*layers)

        self._laplace = LaplaceSurrogate.from_dict(target=target,
                                                   rng=rng,
                                                   **kwargs)

        # init all data
        self._x_data = None
        self._y_data = None
        self._x_data_new = []
        self._y_data_new = []
        self._n_data_buffer = 0
        if update_model is None:
            update_model = []
        self._update_model = copy.deepcopy(update_model)

        self._training_params = {
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
                self._update_model[i] += n_samples

        if train_on_init:
            samples = self._laplace.get_samples(n_samples)
            self._x_data = torch.tensor(samples, dtype=dtype)
            self._y_data = torch.zeros(n_samples)

            for i in range(n_samples):
                self._y_data[i] = torch.tensor(
                    target.log_density(samples[i]) -
                    self._laplace.eval(samples[i], delta=True),
                    dtype=dtype)

            self.train(**self._training_params)

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=dtype)
        with torch.no_grad():
            return self._model(x_tensor).numpy() + self._laplace.eval(
                x, delta=True)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=dtype, requires_grad=True)
        y_tensor = self._model(x_tensor)
        gradients = grad(outputs=y_tensor,
                         inputs=x_tensor,
                         grad_outputs=torch.ones_like(y_tensor),
                         create_graph=True)[0]

        if idx is None:
            return gradients.detach().numpy() + self._laplace.grad(x)
        else:
            return gradients[idx].detach().numpy() + self._laplace.grad(x,
                                                                        idx=idx)

    @override
    def add_data(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dy_dx: np.ndarray = None) -> None:

        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        n = x.shape[0]

        self._x_data_new.append(x)
        self._y_data_new.append(y)

        self._n_data_buffer += n

        if len(self._update_model) > 0:
            if self._n_data_buffer + self._x_data.shape[
                    0] >= self._update_model[0]:
                self._update_model.pop(0)

                x_new = np.concat(self._x_data_new)
                y_new = np.concat(self._y_data_new)

                for i in range(len(y_new)):
                    y_new[i] -= self._laplace.eval(x_new[i], delta=True)

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self._x_data = torch.vstack((self._x_data, x_new))
                self._y_data = torch.hstack((self._y_data, y_new))

                self.train(**self._training_params)

                self._x_data_new = []
                self._y_data_new = []
                self._n_data_buffer = 0

    def save_model(self, path: str = 'neural_network.th') -> None:
        """Save the neural network model to a file.

        Args:
            path: The path to the file. Default is 'neural_network.th'.
        """
        torch.save(self._model.state_dict(), path)

    def load_model(self, path: str = 'neural_network.th') -> None:
        """Load the neural network model from a file.

        Args:
            path: The path to the file. Default is 'neural_network.th'.
        """
        self._model.load_state_dict(torch.load(path, weights_only=True))

    def train(self, *args, epochs: int, batch_size: int, print_every: int,
              patience: int, lr: float, val_split: float, weight_decay: float,
              lr_scheduler: str, lr_scheduler_params: dict, **kwargs):
        """Train the neural network surrogate model using stored data.

        Args:
            args: Additional positional arguments.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            print_every: Print training information every print_every epochs.
            patience: Number of epochs to wait for improvement in validation loss before stopping early.
            lr: Learning rate for the optimizer.
            val_split: Fraction of data used for validation.
            weight_decay: Weight decay for the optimizer.
            lr_scheduler: Type of learning rate scheduler. Admissible values are 'StepLR' and 'ReduceLROnPlateau'.
            lr_scheduler_params: Parameters for the learning rate scheduler.
            kwargs: Additional keyword arguments.
        """

        train_losses = []
        val_losses = []
        patience_counter = 0

        num_val = int(val_split * len(self._x_data))
        x_train, x_val = self._x_data[:-num_val], self._x_data[-num_val:]
        y_train, y_val = self._y_data[:-num_val], self._y_data[-num_val:]

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        optimizer = optim.Adam(self._model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)

        if lr_scheduler == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  **lr_scheduler_params)
        elif lr_scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **lr_scheduler_params)
        else:
            scheduler = None

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_state = None

        last_lr = lr

        logger.warning("Training neural network surrogate model ...")

        # disable tqdm if running on a cluster
        disable_tqdm = 'PBS_ENVIRONMENT' in os.environ or 'SLURM_JOB_ID' in os.environ

        with tqdm(total=epochs,
                  file=sys.stdout,
                  dynamic_ncols=True,
                  leave=True,
                  disable=disable_tqdm) as pbar:
            for epoch in range(epochs):

                self._model.train()

                if (epoch % print_every) == 0:
                    pbar.clear()
                    logger.debug(f"Epoch {epoch}/{epochs}")
                    pbar.refresh()
                    pbar.update(print_every)

                for x_batch, y_batch in train_loader:

                    optimizer.zero_grad()
                    y_pred = self._model(x_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    train_losses.append(loss.detach().numpy())
                    loss.backward()
                    optimizer.step()

                self._model.eval()
                with torch.no_grad():
                    val_loss = criterion(self._model(x_val).squeeze(),
                                         y_val).item()
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(
                            self._model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            pbar.clear()
                            logger.warning(f"Early stopping at epoch {epoch}.")
                            logger.warning(
                                f"with best validation loss {best_val_loss:.6f}"
                            )
                            break

                if isinstance(scheduler, optim.lr_scheduler.StepLR):
                    scheduler.step()

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                if last_lr > scheduler.get_last_lr()[-1]:
                    pbar.clear()
                    last_lr = scheduler.get_last_lr()[-1]
                    logger.warning(
                        f"At Iter {epoch} new learning rate: {last_lr}")
                    pbar.refresh()

        if best_model_state:
            self._model.load_state_dict(best_model_state)
            self.save_model()

        # plot training and validation loss
        fig, ax = get_2d_despined_figure(figsize=(5, 3.5),
                                         equal_axes=False,
                                         axes_label=('Epoch', 'MSE'),
                                         keep_ticks=True)

        ax.semilogy(np.linspace(0, epoch, len(train_losses)),
                    train_losses,
                    label='Train')
        ax.semilogy(np.linspace(0, epoch, len(val_losses)),
                    val_losses,
                    label='Validation')
        ax.set_ylim(1e-5, 4e1)
        ax.legend()

        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(
            f'figures/training_validation_loss_{self._x_data.shape[0]}.pdf')


class ExactGPModel(ExactGP):
    """Exact Gaussian process model based on GPyTorch."""

    def __init__(self, train_x: torch.tensor, train_y: torch.tensor,
                 likelihood: _GaussianLikelihoodBase, ard_num_dims: int):
        """Initialize the exact Gaussian process model.

        Args:
            train_x: The training input data.
            train_y: The training output data.
            likelihood: The likelihood function.
            ard_num_dims: The number of dimensions for the ARD kernel.
        """

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self._mean_module = ConstantMean()
        self._covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))

    def forward(self, x: torch.tensor) -> gpyMultivariateNormal:
        mean_x = self._mean_module(x)
        covar_x = self._covar_module(x)
        return gpyMultivariateNormal(mean_x, covar_x)


class GaussianProcessBase(SurrogateModel):
    """Base class for Gaussian process surrogate models."""

    def __init__(self,
                 target: Distribution,
                 rng: np.random.Generator,
                 *args,
                 n_samples: int = 100,
                 lbfgs_steps: int = 5,
                 n_restarts: int = 50,
                 lr: float = 0.5,
                 tolerance_grad: float = 1e-7,
                 tolerance_change: float = 1e-9,
                 update_model: list = None,
                 retrain_threshold: int = 1000,
                 eval_strategy: str = 'mean',
                 print_every: int = 1,
                 **kwargs):
        """Initialize the Gaussian process surrogate model.

        Args:
            target: The target distribution.
            rng: The random number generator.
            args: Additional positional arguments.
            n_samples: The number of training data to use.
            lbfgs_steps: Number of lbfgs iterations during hyperparameter fitting.
            n_restarts: Number of restarts for hyperparameter optimization.
            lr: Learning rate for the optimizer.
            update_model: List of number of samples after which to update the model.
            retrain_threshold: Number of samples after which to retrain the model.
            eval_strategy: Evaluation strategy. Admissible values are 'mean' and 'mean_plus_std'.
            print_every: Print training information every print_every iterations.
            kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        # get laplace approximation
        self._laplace = LaplaceSurrogate.from_dict(target=target,
                                                   rng=rng,
                                                   **kwargs)
        self._rng = rng

        # init all data
        self._x_data: torch.tensor = None
        self._y_data: torch.tensor = None
        self._x_data_new = []
        self._y_data_new = []
        self._n_data_buffer = 0
        if update_model is None:
            update_model = []
        self._update_model = copy.deepcopy(update_model)
        for i in range(len(update_model)):
            self._update_model[i] += n_samples

        self._retrain_threshold = retrain_threshold

        # default is add_data_on because it changes to add_data_off when update_model is empty
        self._add_data = self._add_data_on

        self._training_params = {
            'lbfgs_steps': lbfgs_steps,
            'n_restarts': n_restarts,
            'print_every': print_every,
            'lr': lr,
            'tolerance_grad': tolerance_grad,
            'tolerance_change': tolerance_change
        }

        # these will be overwritten in derived classes
        self._model: Optional[gpytorch.models.ExactGP] = None
        self._likelihood: Optional[
            gpytorch.likelihoods._GaussianLikelihoodBase] = None

        if eval_strategy == 'mean':
            self._eval = self._eval_mean
            self._grad = self._grad_mean
        elif eval_strategy == 'mean_plus_std':
            self._eval = self._eval_mean_plus_std
            self._grad = self._grad_mean_plus_std
        else:
            raise ValueError(
                f"Evaluation strategy {eval_strategy} not implemented.\n" +
                "Choose from: 'mean', 'mean_plus_std'")

    @override
    def add_data(self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray = None):
        """Wrapper for the _add_data_on and _add_data_off methods.

        Depending on the training stage, _add_data either points to _add_data_on or _add_data_off.

        Args:
            x: The input data.
            y: The output data.
            dy_dx: The gradient of the output data. Default is None.
        """
        return self._add_data(x, y, dy_dx)

    def _add_data_on(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     dy_dx: np.ndarray = None):
        """Add data to the Gaussian process surrogate model.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The output data
            dy_dx (np.ndarray): The gradient of the output data with respect to x
        """
        raise NotImplementedError

    def _add_data_off(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      dy_dx: np.ndarray = None):
        """Add data to the Gaussian process surrogate model. Nothing to do here!

        Args:
            x: The input data.
            y: The output data
            dy_dx: The gradient of the output data with respect to x
        """
        pass

    @override
    def train(self, *args, lbfgs_steps: int, n_restarts: int, print_every: int,
              lr: float, tolerance_grad: float, tolerance_change: float,
              **kwargs):
        """Update the Gaussian process surrogate model.

        Args:
            args: Additional positional arguments.
            lbfgs_steps: Number of lbfgs iterations during hyperparameter fitting.
            n_restarts: Number of restarts for hyperparameter optimization.
            print_every: Print training information every print_every iterations.
            lr: Learning rate for the optimizer.
            tolerance_grad: Tolerance for the gradient.
            tolerance_change: Tolerance for the change in loss.
            kwargs: Additional keyword arguments.
        """

        logger.info(f"Training {self.__class__.__name__} surrogate model ...")

        # set train data and put model in training mode
        self._model.set_train_data(self._x_data, self._y_data, strict=False)
        self._model.train()
        self._likelihood.train()

        # define loss
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        # disable tqdm if running on a cluster
        disable_tqdm = ('PBS_ENVIRONMENT' in os.environ or
                        'SLURM_JOB_ID' in os.environ or
                        'GIO_LAUNCHED_DESKTOP_FILE' in os.environ)
        warnings.simplefilter("ignore", category=NumericalWarning)

        # get hyperparameter initialisations
        n_params = int(
            sum([
                np.prod(param.shape)
                for name, param in self._model.named_parameters()
            ]))
        sampler = qmc.LatinHypercube(n_params, scramble=True, rng=self._rng)
        sample = sampler.random(n_restarts)
        scaled_sample = qmc.scale(sample, -10, 20)
        initial_params = torch.tensor(scaled_sample, dtype=dtype)

        # init best model and loss
        best_model = None
        best_loss = float('inf')

        # plot training and validation loss
        fig, ax = get_2d_despined_figure(figsize=(5, 3.5),
                                         equal_axes=False,
                                         axes_label=('Iter', 'MLL'),
                                         keep_ticks=True)
        losses_min = 1e10
        losses_max = -1e10
        best_iter = 0
        train_losses_all = []

        with tqdm(total=n_restarts,
                  file=sys.stdout,
                  dynamic_ncols=True,
                  leave=True,
                  disable=disable_tqdm) as pbar:

            for restart in range(n_restarts):

                train_losses = []

                # init optimizer and scheduler
                optimizer = torch.optim.LBFGS(self._model.parameters(),
                                              lr=lr,
                                              max_iter=lbfgs_steps,
                                              tolerance_grad=tolerance_grad,
                                              tolerance_change=tolerance_change,
                                              line_search_fn='strong_wolfe')

                def closure():
                    optimizer.zero_grad()
                    output = self._model(self._x_data)
                    loss = -mll(output, self._y_data)
                    loss.backward()
                    return loss

                # set hyperparameters and initialise model
                hyper_params = {}
                counter = 0
                for name, param in self._model.named_parameters():
                    shape = param.shape
                    length = int(np.prod(shape))
                    hyper_params[name] = initial_params[restart,
                                                        counter:counter +
                                                        length].reshape(shape)
                    counter += length

                self._model.initialize(**hyper_params)

                # print hyperparameter values
                pbar.clear()
                logger.info(f"Restart {restart}/{n_restarts} with \n" +
                            self._log_state_dict())
                pbar.refresh()

                # try block needed to catch non-PSD errors. in that case, interation is just skipped
                try:
                    for i in range(lbfgs_steps):

                        loss = optimizer.step(closure)
                        train_losses.append(loss.item())

                        # print current hyperparams and loss
                        if (i % print_every) == 0:
                            pbar.clear()
                            logger.debug(f"   Iter {i}/{lbfgs_steps}" +
                                         f" Loss: {loss.item():.3f}," +
                                         self._log_state_dict(end_of_line=', '))
                            pbar.refresh()

                    # check if current model is best
                    if loss.item() < best_loss:
                        logger.info(
                            f'New best loss: {loss.item()} with params:')
                        logger.info(self._log_state_dict())
                        best_model = copy.deepcopy(self._model.state_dict())
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

                except RuntimeError as e:
                    pbar.clear()
                    logger.info(f"Runtime error: {e}")
                    pbar.refresh()

                train_losses_np = np.array(train_losses)
                if len(train_losses) > 0:
                    # check if train_losses_np contains nan
                    if not np.any(np.isnan(train_losses_np)):
                        losses_min = np.min(
                            (np.min(train_losses_np), losses_min))
                train_losses_all.append(train_losses_np)
                pbar.update()

        # load best model print parameters
        self._model.load_state_dict(best_model)
        logger.warning(f"Best model with loss {best_loss} and params:\n" +
                       self._log_state_dict())
        self.save_model()

        for i in range(n_restarts):
            if i == best_iter:
                ax.plot(train_losses_all[i], color='C0', alpha=1, lw=1.5)
            else:
                ax.plot(train_losses_all[i], color='C0', alpha=0.3, lw=1.)

        ax.set_ylim(losses_min - 0.2 * np.abs(losses_min), losses_min + 10.)

        if not os.path.exists('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures/mll_{self._x_data.shape[0]}.pdf')

        # set model into evaluation mode
        self._model.eval()
        self._likelihood.eval()

    def _log_state_dict(self, end_of_line: str = '\n') -> str:
        """Log the state dictionary of the model.

        Args:
            end_of_line :The end of line character. Default is '\n'.

        Returns
            str: The state dictionary of the model as a string
        """

        named_params = ""

        for name, param in self._model.named_parameters():
            named_params += f"  {name}: {param.data}" + end_of_line

        return named_params

    @override
    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._eval(x, **kwargs)

    @override
    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        return self._grad(x, idx=idx)

    def _eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the mean of the Gaussian process model.

        Args:
            x: The input data.
            kwargs: Additional keyword arguments

        Returns:
            np.ndarray: The mean of the Gaussian process model.
        """
        raise NotImplementedError

    def _eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the mean of the Gaussian process model and add the standard deviation.

        Args:
            x: The input data.
            kwargs: Additional keyword arguments

        Returns:
            np.ndarray: The mean of the Gaussian process model plus the standard deviation.
        """
        raise NotImplementedError

    def _grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """Compute the gradient of the mean of the Gaussian process model.

        Args:
            x: The input data.
            idx: The index of the component of the gradient to be computed. Default is None.

        Returns:
            np.ndarray: The gradient of the mean of the Gaussian process model.
        """
        raise NotImplementedError

    def _grad_mean_plus_std(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """Compute the gradient of the mean of the Gaussian process model plus the standard deviation.

        Args:
            x: The input data.
            idx: The index of the component of the gradient to be computed. Default is None.

        Returns:
            np.ndarray: The gradient of the mean of the Gaussian process model plus the standard deviation.
        """
        raise NotImplementedError

    def save_model(self, path: str = 'model_params'):
        """Save the neural network model to a file.

        Args:
            path: The path to the file. Default is 'model_params'.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self._model.state_dict(),
                   os.path.join(path, 'model_params.th'))
        np.savetxt(os.path.join(path, 'x_data.dat'), self._x_data.numpy())
        np.savetxt(os.path.join(path, 'y_data.dat'), self._y_data.numpy())

    def load_model(self, path: str = 'model_params'):
        """Load the neural network model from a file.

        Args:
            path: The path to the file. Default is 'model_params'.
        """

        model_params = torch.load(os.path.join(path, 'model_params.th'))
        self._x_data = torch.tensor(np.loadtxt(os.path.join(path,
                                                            'x_data.dat')),
                                    dtype=dtype)
        self._y_data = torch.tensor(np.loadtxt(os.path.join(path,
                                                            'y_data.dat')),
                                    dtype=dtype)

        self._model.load_state_dict(model_params)
        self._model.set_train_data(self._x_data, self._y_data, strict=False)
        self._model.eval()
        self._likelihood.eval()


class GaussianProcess(GaussianProcessBase):
    """Gaussian process surrogate model based on GPyTorch."""

    def __init__(self,
                 target: Distribution,
                 rng: np.random.Generator,
                 *args,
                 train_on_init: bool = True,
                 n_samples: int = 100,
                 **kwargs):
        """Initialize the Gaussian process surrogate model.

        Args:
            target: The target distribution.
            rng: The random number generator.
            args: Additional positional arguments.
            train_on_init: Whether to train the model on initialization.
            n_samples: The number of training data to use.
            kwargs: Additional keyword arguments.
        """
        super().__init__(target, rng, *args, n_samples=n_samples, **kwargs)

        # define likelihood, get model, and set optimizer
        self._likelihood = GaussianLikelihood()
        self._model = ExactGPModel(None, None, self._likelihood,
                                   target.dim)

        # train model unless specified otherwise
        if train_on_init:
            samples = self._laplace.get_samples(n_samples)
            self._x_data = torch.tensor(samples, dtype=dtype)
            self._y_data = torch.zeros(n_samples)

            for i in range(n_samples):
                self._y_data[i] = torch.tensor(
                    target.log_density(samples[i]) -
                    self._laplace.eval(samples[i], delta=True),
                    dtype=dtype)

            self.train(**self._training_params)

        logger.info(f'{self.__class__.__name__} surrogate model initialized.')

    @override
    def _add_data_on(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     dy_dx: np.ndarray = None):

        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        n = x.shape[0]

        self._x_data_new.append(x)
        self._y_data_new.append(y)

        self._n_data_buffer += n

        if len(self._update_model) > 0:
            if self._n_data_buffer + self._x_data.shape[
                    0] >= self._update_model[0]:
                self._update_model.pop(0)

                x_new = np.concat(self._x_data_new)
                y_new = np.concat(self._y_data_new)

                for i in range(len(y_new)):
                    y_new[i] -= self._laplace.eval(x_new[i], delta=True)

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self._x_data = torch.vstack((self._x_data, x_new))
                self._y_data = torch.hstack((self._y_data, y_new))

                if len(self._x_data) < self._retrain_threshold:
                    logger.info(
                        f'Retraining model with {len(self._x_data)} data points.'
                    )
                    self.train(**self._training_params)
                else:
                    logger.info(
                        f'Updating model with {len(self._x_data)} data points.')
                    self._model.set_train_data(self._x_data,
                                               self._y_data,
                                               strict=False)

                self._x_data_new = []
                self._y_data_new = []
                self._n_data_buffer = 0
        else:
            self._add_data = self._add_data_off

    @override
    def _eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=False)

        with (torch.no_grad(),
              gpytorch.settings.skip_posterior_variances(True)):
            return self._model(x_tensor).mean.squeeze().numpy(
            ) + self._laplace.eval(x, delta=True)

    @override
    def _eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=False)

        with (torch.no_grad(), gpytorch.settings.fast_pred_var(True)):
            y = self._model(x_tensor)

        gp = y.mean.squeeze().numpy() + y.variance.sqrt().squeeze().numpy()

        return gp + self._laplace.eval(x, delta=True)

    @override
    def _grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=True)

        with gpytorch.settings.skip_posterior_variances(
                True), gpytorch.settings.fast_computations(False, False, False):
            y_tensor = self._model(x_tensor).mean
            gradients = grad(outputs=y_tensor,
                             inputs=x_tensor,
                             grad_outputs=torch.ones_like(y_tensor),
                             create_graph=True)[0].squeeze().detach().numpy()

        if idx is None:
            return gradients + self._laplace.grad(x)
        else:
            return gradients[idx] + self._laplace.grad(x, idx=idx)

    @override
    def _grad_mean_plus_std(self, x: np.ndarray, idx: int = None):
        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=True)
        with (gpytorch.settings.skip_posterior_variances(False),
              gpytorch.settings.fast_computations(False, False, False),
              gpytorch.settings.fast_pred_var(True)):
            model_output = self._model(x_tensor)
            y_tensor = model_output.mean + model_output.variance.sqrt()
            gradients = grad(outputs=y_tensor,
                             inputs=x_tensor,
                             grad_outputs=torch.ones_like(y_tensor),
                             create_graph=True)[0].squeeze().detach().numpy()

        if idx is None:
            return gradients + self._laplace.grad(x)
        else:
            return gradients[idx] + self._laplace.grad(x, idx=idx)


class DerivativeGPModel(ExactGP):
    """Gaussian process model based on GPyTorch that also observes gradients."""

    def __init__(self, train_x: torch.tensor, train_y: torch.tensor,
                 likelihood: _GaussianLikelihoodBase, ard_num_dims: int):
        """Initialize the Gaussian process model.

        Args:
            train_x: The training input data.
            train_y: The training output data.
            likelihood: The likelihood function.
            ard_num_dims: The number of dimensions for the ARD kernel.
        """

        super().__init__(train_x, train_y, likelihood)
        self._mean_module = ConstantMeanGrad()
        self._covar_module = ScaleKernel(
            RBFKernelGrad(ard_num_dims=ard_num_dims))

    @override
    def forward(self, x: torch.tensor) -> MultitaskMultivariateNormal:
        mean_x = self._mean_module(x)
        covar_x = self._covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class DerivativeGaussianProcess(GaussianProcessBase):
    """Gaussian process surrogate model based on GPyTorch that also observes gradients."""

    def __init__(self,
                 target: Distribution,
                 rng: np.random.Generator,
                 *args,
                 train_on_init: bool = True,
                 n_samples: int = 100,
                 **kwargs):
        """Initialize the Derivative Gaussian process surrogate model.

        Args:
            target: The target distribution.
            rng: The random number generator.
            args: Additional positional arguments.
            train_on_init: Whether to train the model on initialization.
            n_samples: The number of training data to use.
            kwargs: Additional keyword arguments.
        """
        super().__init__(target, rng, *args, n_samples=n_samples, **kwargs)

        # define likelihood, get model, and set optimizer
        self._likelihood = MultitaskGaussianLikelihood(
            num_tasks=target.dim + 1)
        self._model = DerivativeGPModel(None, None, self._likelihood,
                                        target.dim)

        # train model unless specified otherwise
        if train_on_init:
            samples = self._laplace.get_samples(n_samples)
            self._x_data = torch.tensor(samples, dtype=dtype)
            self._y_data = torch.zeros(n_samples,
                                       target.dim + 1,
                                       dtype=dtype)

            for i in range(n_samples):
                self._y_data[i, 0] = torch.tensor(
                    target.log_density(samples[i]) -
                    self._laplace.eval(samples[i], delta=True),
                    dtype=dtype)
                self._y_data[i, 1:] = torch.tensor(
                    target.grad_log_density(samples[i]) -
                    self._laplace.grad(samples[i]),
                    dtype=dtype)

            self.train(**self._training_params)

        logger.info(f'{self.__class__.__name__} surrogate model initialized.')

    @override
    def _add_data_on(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     dy_dx: np.ndarray = None) -> None:

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        dy_dx = np.atleast_2d(dy_dx)
        y = np.hstack((y, dy_dx))

        n = x.shape[0]

        self._x_data_new.append(x)
        self._y_data_new.append(y)

        self._n_data_buffer += n

        if len(self._update_model) > 0:
            if self._n_data_buffer + self._x_data.shape[
                    0] >= self._update_model[0]:
                self._update_model.pop(0)

                x_new = np.concat(self._x_data_new)
                y_new = np.concat(self._y_data_new)

                for i in range(len(y_new)):
                    y_new[i, 0] -= self._laplace.eval(x_new[i], delta=True)
                    y_new[i, 1:] -= self._laplace.grad(x_new[i])

                x_new = torch.tensor(x_new, dtype=dtype)
                y_new = torch.tensor(y_new, dtype=dtype)

                self._x_data = torch.vstack((self._x_data, x_new))
                self._y_data = torch.vstack((self._y_data, y_new))

                if len(self._x_data) < self._retrain_threshold:
                    self.train(**self._training_params)
                else:
                    self._model.set_train_data(self._x_data,
                                               self._y_data,
                                               strict=False)

                self._x_data_new = []
                self._y_data_new = []
                self._n_data_buffer = 0

        else:
            self._add_data = self._add_data_off

    @override
    def _eval_mean(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=False)

        with (
                torch.no_grad(),
                gpytorch.settings.skip_posterior_variances(True),
                gpytorch.settings.fast_computations(False, False, False),
        ):
            return self._model(x_tensor).mean[:, 0].squeeze().detach().numpy(
            ) + self._laplace.eval(x, delta=True)

    @override
    def _eval_mean_plus_std(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=False)

        with (torch.no_grad(),
              gpytorch.settings.fast_computations(False, False, False),
              gpytorch.settings.fast_pred_var(True)):
            y = self._model(x_tensor)

        gp = y.mean[:, 0].squeeze().detach().numpy() + y.variance[:, 0].sqrt(
        ).squeeze().detach().numpy()
        return gp + self._laplace.eval(x, delta=True)

    @override
    def _grad_mean(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=True)

        with (torch.no_grad(), gpytorch.settings.skip_posterior_variances(True),
              gpytorch.settings.fast_computations(False, False, False)):
            gradient = self._model(
                x_tensor).mean[:, 1:].squeeze().detach().numpy()

        if idx is None:
            return gradient + self._laplace.grad(x)
        else:
            return gradient[idx] + self._laplace.grad(x, idx=idx)

    @override
    def _grad_mean_plus_std(self, x: np.ndarray, idx: int = None) -> np.ndarray:

        x_tensor = torch.tensor(np.atleast_2d(x),
                                dtype=dtype,
                                requires_grad=True)

        with (torch.no_grad(),
              gpytorch.settings.fast_computations(False, False, False),
              gpytorch.settings.fast_pred_var(True)):
            y = self._model(x_tensor)
            gradient = y.mean[:, 1:].squeeze().detach().numpy(
            ) + y.variance[:, 1:].sqrt().squeeze().detach().numpy()

        if idx is None:
            return gradient + self._laplace.grad(x)
        else:
            return gradient[idx] + self._laplace.grad(x, idx=idx)


SURROGATE_REGISTRY = {
    'Laplace': LaplaceSurrogate,
    'Constant': ConstantSurrogate,
    'RandomConstant': RandomConstantSurrogate,
    'NeuralNetwork': NeuralNetwork,
    'GaussianProcess': GaussianProcess,
    'DerivativeGaussianProcess': DerivativeGaussianProcess
}
