from typing import Callable, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from pdmp import logger

SMALL = 1e-10
LARGE = 1e10


class Basis:
    def __init__(self, n: int):
        """
        Initialize the Basis class.

        Parameters:
        n (int): The number of basis functions.
        """
        self.n_ = n
        self.support_ = np.zeros((n, 2))

    def __call__(self, x: np.ndarray, i: int = None) -> np.ndarray:
        """
        Evaluate the basis functions at given points.

        Parameters:
        x (np.ndarray): The points at which to evaluate the basis functions.
        i (int, optional): The index of the specific basis function to evaluate. If None, evaluate all basis functions.
                           Default is None.

        Returns:
        np.ndarray: The evaluated basis functions.
        """
        pass

    def get_n(self) -> int:
        """
        Get the number of basis functions.

        Returns:
        int: The number of basis functions.
        """
        return self.n_

    def get_support(self, i: int = None) -> np.ndarray:
        """
        Get the support of the basis functions.

        Parameters:
        i (int, optional): The index of the specific basis function to get the support for. If None, get the support for
                           all basis functions. Default is None.

        Returns:
        np.ndarray: The support of the basis functions.
        """
        pass

class PiecewiseConstantBasis(Basis):
    def __init__(self, n: int, interval: Tuple[float, float]):
        """
        Initialize the PiecewiseConstantBasis class.

        Parameters:
        n (int): The number of basis functions.
        interval (Tuple[float, float]): The interval over which the basis functions are defined.
        """
        super().__init__(n)
        self.interval_ = interval
        self.basis_ = []

        for i in range(self.n_):
            self.basis_.append(
                lambda x, i=i: np.piecewise(
                    x,
                    [x < (i / self.n_),
                    x == (i / self.n_),
                    ((i / self.n_) < x) & (x < ((i + 1) / self.n_)),
                    x == ((i + 1) / self.n_),
                    x > ((i + 1) / self.n_)], [0, 0.5, 1, 0.5, 0]
                )
            )

        self.support_[:, 0] = np.linspace(interval[0], interval[1], n + 1)[:-1]
        self.support_[:, 1] = np.linspace(interval[0], interval[1], n + 1)[1:]

    def __call__(self, x: np.ndarray, i: int = None) -> np.ndarray:
        """
        Evaluate the piecewise constant basis functions at given points.

        Parameters:
        x (np.ndarray): The points at which to evaluate the basis functions.
        i (int, optional): The index of the specific basis function to evaluate. If None, evaluate all basis functions.
                           Default is None.

        Returns:
        np.ndarray: The evaluated basis functions.
        """
        if i is None:
            return np.array([basis(x) for basis in self.basis_]).T
        else:
            return self.basis_[i](x)

    def get_support(self, i: int = None) -> np.ndarray:
        """
        Get the support of the piecewise constant basis functions.

        Parameters:
        i (int, optional): The index of the specific basis function to get the support for. If None, get the support for
                           all basis functions. Default is None.

        Returns:
        np.ndarray: The support of the basis functions.
        """
        if i is None:
            return self.support_
        else:
            return self.support_[i]

def squared_exponential_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0, l: float = 1.0, **kwargs) -> float:
    """
    Compute the squared exponential kernel between two points.

    Parameters:
    x (np.ndarray): The first point.
    y (np.ndarray): The second point.
    sigma (float, optional): The standard deviation parameter of the kernel. Default is 1.0.
    l (float, optional): The length scale parameter of the kernel. Default is 1.0.
    **kwargs: Additional keyword arguments.

    Returns:
    float: The computed kernel value.
    """
    return sigma ** 2 * np.exp(-((x - y) ** 2) / (2 * l ** 2))

def compute_coefficients(kernel: Callable[[np.ndarray, np.ndarray, float, float], float],
                         basis: Basis,
                         interval: Tuple[float, float],
                         weights: Callable[[np.ndarray], float] = None,
                         kernel_params: Dict = None) -> np.ndarray:
    """
    Compute the coefficients for the given basis functions using the specified kernel.

    Parameters:
    kernel (Callable[[np.ndarray, np.ndarray, float, float], float]): The kernel function to use.
    basis (Basis): The basis functions.
    interval (Tuple[float, float]): The interval over which to compute the coefficients.
    weights (Callable[[np.ndarray], float], optional): A function to compute weights. Default is None.
    kernel_params (Dict[str, float], optional): Parameters for the kernel function. Default is None.

    Returns:
    np.ndarray: The computed coefficients.
    """

    n = basis.get_n()

    if weights is None:
        weights = lambda x: 1.

    def integrand(x: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
        return kernel(x, y, **kernel_params) * basis(x, p) * basis(y, q) * weights(x) * weights(y)

    coefficients = np.zeros((n, n))
    coefficients_norm = np.zeros((n, n))

    logger.debug("Computing coefficients for basis functions")
    for i in range(n):
        for j in range(i, n):
            logger.debug(f"   {i} and {j}")
            int_x = (np.max([interval[0], basis.get_support(i)[0]]),
                     np.min([interval[1], basis.get_support(i)[1]]))
            int_y = (np.max([interval[0], basis.get_support(j)[0]]),
                     np.min([interval[1], basis.get_support(j)[1]]))
            coefficients[i, j], tol = integrate.nquad(integrand, [int_x, int_y], args=(i, j))
            logger.debug(f"     {coefficients[i, j]:.4}")

    # normalize the coefficients
    for i in range(n):
        for j in range(i, n):
            coefficients_norm[i, j] = coefficients_norm[j, i] = coefficients[i, j] / np.sqrt(
                coefficients[i, i] * coefficients[j, j])

    logger.debug("Normalized coefficientes:\n  ", coefficients_norm)

    return coefficients_norm


def monte_carlo_2d(func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   x_lim: Tuple[float, float],
                   y_lim: Tuple[float, float],
                   num_samples: int = 100000) -> float:
    """
    Perform Monte Carlo integration over a 2D domain. Mostly for testing purposes.

    Parameters:
    func (Callable[[np.ndarray, np.ndarray], np.ndarray]): The function to integrate. It should take two numpy arrays (x and y coordinates) and return an array of function values.
    x_lim (Tuple[float, float]): The limits for the x-axis as a tuple (min, max).
    y_lim (Tuple[float, float]): The limits for the y-axis as a tuple (min, max).
    num_samples (int, optional): The number of random samples to generate. Default is 100000.

    Returns:
    float: The estimated integral of the function over the specified domain.
    """
    x_samples = np.random.uniform(x_lim[0], x_lim[1], num_samples)
    y_samples = np.random.uniform(y_lim[0], y_lim[1], num_samples)
    sample_values = func(x_samples, y_samples)
    area = (x_lim[1] - x_lim[0]) * (y_lim[1] - y_lim[0])
    return np.mean(sample_values) * area


if __name__ == '__main__':

    kernel_params = {'sigma': 1., 'l': 0.5}

    n_b = 5
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_b, interval)
    coefficients = compute_coefficients(squared_exponential_kernel, basis, interval, kernel_params=kernel_params)

    # Test the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_vals = np.linspace(0, 1, 100)
    for i in range(n_b):
        ax.plot(x_vals, basis(x_vals, i), label=f'basis_{i}')
    ax.legend()
    plt.show()

    # plot the variance function
    Phi = basis(x_vals).T
    cov = Phi.T @ coefficients @ Phi

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_vals, np.diag(cov))
    plt.show()

    # plot full covariance matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c = ax.imshow(cov)
    fig.colorbar(c)
    plt.show()
