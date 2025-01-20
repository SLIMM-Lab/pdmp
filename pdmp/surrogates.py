import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

import numpy as np

from typing import cast
from scipy.optimize import minimize

from pdmp.distributions import Distribution, MultivariateNormal, Posterior
from pdmp import logger


class SurrogateModel(object):
    """
    Base class for surrogate models.
    """
    def __init__(self):
        """
        Initialize the surrogate model.
        """

    def eval(self, x: np.ndarray) -> np.ndarray:
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
    def __init__(self,
                 target: Distribution = None,
                 mean: np.ndarray = None,
                 cov: np.ndarray = None,
                 x_0: np.ndarray = None):
        """
        Initialize the Laplace approximation.

        Parameters:
        target (Distribution): The target distribution.
        x_0 (np.ndarray): The initial point for the Laplace approximation.
        approximation (dict, optional): The approximation to be used. Default is None.
        """
        super().__init__()

        if isinstance(target, Posterior):
            target = cast(Posterior, target)

        if mean is None or cov is None:
            assert target is not None, "Target distribution must be provided if mean and cov are not provided."
            n_log_post = lambda x: - target.log_density(x)
            n_grad_log_post = lambda x: - target.grad_log_density(x)

            if mean is None:
                if x_0 is None:
                    logger.warning("No initial point provided ... attempting to get sample from target.")
                    success = False

                    if hasattr(target, 'get_sample'):
                        try:
                            x_0 = target.get_sample()
                            success = True
                        except NotImplementedError as e:
                            logger.warning("  Method get_sample not implemented for target.")

                    if hasattr(target, 'get_mean'):
                        try:
                            x_0 = target.get_mean()
                            success = True
                        except NotImplementedError as e:
                            logger.warning("  Method get_mean not implemented for target.")

                    if not success and hasattr(target, 'prior_') and hasattr(target.prior_, 'get_sample'):
                        try:
                            x_0 = target.prior_.get_sample()
                            success = True
                        except NotImplementedError as e:
                            logger.warning("  Method get_sample not implemented for prior.")

                    if not success and hasattr(target, 'prior_') and hasattr(target.prior_, 'get_mean'):
                        try:
                            x_0 = target.prior_.get_mean()
                        except NotImplementedError as e:
                            logger.warning("  Method get_mean not implemented for prior.")

                    if not success:
                        x_0 = np.zeros(target.get_dim())

                self.mean = minimize(n_log_post, x_0, jac=n_grad_log_post, method='BFGS').x
            else:
                self.mean = mean
            if cov is None:
                self.cov = - np.linalg.inv(target.hessian_log_density(self.mean))
            else:
                self.cov = cov
        else:
            self.mean = mean
            self.cov = cov

        self.gaussian = MultivariateNormal(self.mean, self.cov)

    @classmethod
    def from_dict(cls, config: dict, target = None, rng: np.random.Generator = None):
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

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Laplace approximation at a point.

        Parameters:
        x (np.ndarray): The point at which the Laplace approximation is to be evaluated.

        Returns:
        float: The value of the Laplace approximation at the point x.
        """
        return self.gaussian.log_density(x)

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

    def __init__(self,
                 target: Distribution,
                 layer_sizes: list,
                 lr: float = 1e-3,
                 x_0: np.ndarray = None):
        """
        Initialize the neural network surrogate model.

        Parameters:
        layer_sizes (list): List containing the number of neurons per layer.
        lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        layer_sizes = [target.get_dim()] + layer_sizes + [1]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        laplace = LaplaceSurrogate(target, x_0=x_0)
        n_samples = 100
        samples = laplace.get_samples(n_samples)

        self.x_data = torch.tensor(samples, dtype=torch.float32)
        self.y_data = torch.zeros(n_samples)

        for i in range(n_samples):
            self.y_data[i] = torch.tensor(target.log_density(samples[i]), dtype=torch.float32)

        self.update()

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the neural network surrogate model at a point.

        Parameters:
        x (np.ndarray): The point at which the neural network surrogate model is to be evaluated.

        Returns:
        float: The value of the neural network surrogate model at the point x.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x_tensor).numpy()

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
            return gradients.detach().numpy()
        else:
            return gradients[idx].detach().numpy()

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data to the neural network surrogate model.

        Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        """
        self.x_data.append(torch.tensor(x, dtype=torch.float32))
        self.y_data.append(torch.tensor(y, dtype=torch.float32))

    def update(self, epochs: int = 5000, batch_size: int = 20) -> None:
        """
        Train the neural network surrogate model using stored data.

        Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        """

        dataset = torch.utils.data.TensorDataset(self.x_data, self.y_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch).squeeze()
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

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
