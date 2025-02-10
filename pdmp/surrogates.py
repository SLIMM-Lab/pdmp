import os.path
import sys
import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

import numpy as np
import matplotlib.pyplot as plt

from typing import cast
from tqdm import tqdm

from pdmp.distributions import Distribution, MultivariateNormal, Posterior, find_mean, find_curvature
from pdmp.plotting_utils import get_2d_despined_figure
from pdmp import logger


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

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data to the surrogate model.

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        """
        raise NotImplementedError

    def update(self) -> None:
        """
        Update the surrogate model.
        """
        raise NotImplementedError


class LaplaceSurrogate(SurrogateModel):
    """
    Laplace approximation to the target distribution.
    """
    def __init__(
            self,
            target: Distribution = None,
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

        self.gaussian = MultivariateNormal(self.mean, self.cov)
        self.delta = self.gaussian.log_density(self.mean) - target.log_density(self.mean)

    @classmethod
    def from_dict(
            cls,
            config: dict,
            target = None,
            rng: np.random.Generator = None
    ):
        """
        Create a Laplace approximation from a dictionary.

        Parameters:
        config (dict): The configuration dictionary.
        rng (np.random.Generator, optional): The random number generator. Default is None.

        Returns:
        LaplaceSurrogate: The Laplace approximation.
        """
        mean = np.array(config['mean']) if 'mean' in config else None
        cov = np.array(config['cov']) if 'cov' in config else None
        return cls(
            mean=mean,
            cov=cov,
            target=target
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

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data to the Laplace approximation. Nothing to do here!
        :param x: The input data.
        :param y: The output data.
        :return: None
        """
        pass

    def update(self) -> None:
        """
        Update the Laplace approximation. Nothing to do here!
        :return: None
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
            lr: float = 1e-3,
            n_samples: int = 100,
            epochs: int = 5000,
            batch_size: int = 20,
            val_split: float = 0.3,
            patience: int = 100,
            print_every: int = 10,
            lr_scheduler_step: bool = None,
            lr_scheduler_gamma: bool = None,
            train_on_init: bool = True,
            rng: np.random.Generator = None,
            **kwargs
    ):
        """
        Initialize the neural network surrogate model.

        Parameters:
        layer_sizes (list): List containing the number of neurons per layer.
        lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        hidden_layers = [target.get_dim()] + hidden_layers + [1]
        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if i < len(hidden_layers) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if lr_scheduler_gamma is not None and lr_scheduler_step is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
        else:
            self.scheduler = None
        self.criterion = nn.MSELoss()
        self.losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

        self.laplace = LaplaceSurrogate.from_dict(kwargs, target=target, rng=rng)

        if train_on_init:
            samples = self.laplace.get_samples(n_samples)
            self.x_data = torch.tensor(samples, dtype=torch.float32)
            self.y_data = torch.zeros(n_samples)

            num_val = int(val_split * len(self.x_data))
            self.x_train, self.x_val = self.x_data[:-num_val], self.x_data[-num_val:]
            self.y_train, self.y_val = self.y_data[:-num_val], self.y_data[-num_val:]

            for i in range(n_samples):
                self.y_data[i] = torch.tensor(
                    target.log_density(samples[i]) - self.laplace.eval(samples[i], delta=True),
                    dtype=torch.float32
                )

            self.update(epochs=epochs, batch_size=batch_size, patience=patience, print_every=print_every)

    @classmethod
    def from_dict(
            cls,
            config: dict,
            target: Distribution,
            rng: np.random.Generator = None,
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

    def eval(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the neural network surrogate model at a point.

        Parameters:
        x (np.ndarray): The point at which the neural network surrogate model is to be evaluated.

        Returns:
        float: The value of the neural network surrogate model at the point x.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x_tensor).numpy() + self.laplace.eval(x, delta=True)

    def grad(self, x: np.ndarray, idx: int = None) -> np.ndarray:
        """
        Compute the gradient of the neural network surrogate model at a point.

        Parameters:
        x (np.ndarray): The point at which the gradient is to be computed.
        idx (int, optional): The index of the component of the gradient to be computed. Default is None.

        Returns:
        np.ndarray: The gradient of the neural network surrogate model at the point x.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
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

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data to the neural network surrogate model.

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        """
        self.x_data.append(torch.tensor(x, dtype=torch.float32))
        self.y_data.append(torch.tensor(y, dtype=torch.float32))

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

    def update(
            self,
            epochs: int = 5000,
            batch_size: int = 20,
            print_every: int = 10,
            patience: int = 10,
            lr_scheduler_step: int = None,
            lr_scheduler_gamma: float = None
    ) -> None:
        """
        Train the neural network surrogate model using stored data.

        Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        print_every (int): Print training information every print_every epochs.
        patience (int): Number of epochs to wait for improvement in validation loss before stopping early.
        lr_scheduler_step (int): Number of epochs after which to reduce the learning rate.
        lr_scheduler_gamma (float): Factor by which to reduce the learning rate.
        """

        train_losses = []
        val_losses = []

        train_dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

                    self.optimizer.zero_grad()
                    y_pred = self.model(x_batch).squeeze()
                    loss = self.criterion(y_pred, y_batch)
                    train_losses.append(loss.detach().numpy())
                    loss.backward()
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.model.eval()
                with torch.no_grad():
                    val_loss = self.criterion(self.model(self.x_val).squeeze(), self.y_val).item()
                    val_losses.append(val_loss)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_state = self.model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            pbar.clear()
                            logger.warning(f"Early stopping at epoch {epoch} with validation loss {val_loss:.6f}")
                            break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
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
        fig.savefig(f'figures/training_validation_loss.pdf')
        # plt.show()


        # # Clear stored data
        # self.x_data.clear()
        # self.y_data.clear()

    # def update(self, target_grad_fn) -> None:
    #     """
    #     Update the neural network surrogate model using added data.
    #
    #     Parameters:
    #     target_grad_fn (callable): A function that computes the gradient of the true negative log-pdf.
    #     """
    #     if not self.x_data:
    #         return
    #
    #     x_batch = torch.stack(self.x_data)
    #     y_batch = torch.stack(self.y_data)
    #
    #     self.optimizer.zero_grad()
    #
    #     # Compute loss on function values
    #     y_pred = self.model(x_batch).squeeze()
    #     loss = self.criterion(y_pred, y_batch)
    #
    #     # Compute gradient penalty
    #     x_batch.requires_grad = True
    #     y_pred = self.model(x_batch)
    #     grad_pred = grad(outputs=y_pred, inputs=x_batch, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    #     grad_target = target_grad_fn(x_batch.detach().numpy())
    #     grad_penalty = torch.mean(
    #         torch.clamp(torch.abs(grad_pred) - torch.abs(torch.tensor(grad_target, dtype=torch.float32)), min=0) ** 2)
    #
    #     total_loss = loss + grad_penalty
    #     total_loss.backward()
    #     self.optimizer.step()
    #
    #     # Clear stored data
    #     self.x_data.clear()
    #     self.y_data.clear()
