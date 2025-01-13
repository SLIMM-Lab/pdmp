import numpy as np

from scipy.optimize import minimize

from pdmp.distributions import Distribution, MultivariateNormal
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

        if mean is None or cov is None:
            assert target is not None, "Target distribution must be provided if mean and cov are not provided."
            n_log_post = lambda x: - target.log_density(x)
            n_grad_log_post = lambda x: - target.grad_log_density(x)

            if mean is None:
                self.mean = minimize(n_log_post, x_0, jac=n_grad_log_post, method='BFGS').x
            else:
                self.mean = mean
            if cov is None:
                self.cov = np.linalg.inv(target.hessian_log_density(self.mean))
            else:
                self.cov = cov
        else:
            self.mean = mean
            self.cov = cov

        self.gaussian = MultivariateNormal(self.mean, self.cov)

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
