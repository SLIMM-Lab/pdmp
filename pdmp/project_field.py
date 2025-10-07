from typing import Callable, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special

from pdmp import logger

SMALL = 1e-10
LARGE = 1e10


class Basis:
    """Base class for basis functions."""

    def __init__(self, n: int):
        """Initialize the Basis class.

        Args:
            n: The number of basis functions.
        """
        self.n_ = n
        self.support_ = np.zeros((n, 2))
        self.norms_ = np.zeros((n, n))

    def __call__(self, x: np.ndarray, i: int = None) -> np.ndarray:
        """Evaluate the basis functions at given points.

        Args:
            x: The points at which to evaluate the basis functions.
            i: The index of the specific basis function to evaluate. If None, evaluate all basis functions.
               Default is None.

        Returns:
        np.ndarray: The evaluated basis functions.
        """
        pass

    def get_n(self) -> int:
        """Get the number of basis functions.

        Returns:
            int: The number of basis functions.
        """
        return self.n_

    def get_support(self, i: int = None) -> np.ndarray:
        """Get the support of the basis functions.

        Args:
        i: The index of the specific basis function to get the support for. If None, get the support for all basis functions. Default is None.

        Returns:
        np.ndarray: The support of the basis functions.
        """
        pass

    def get_norms(self) -> np.ndarray:
        """Get the norms of the basis functions, i.e. ||φᵢ|| = sqrt(∫ φᵢ(x)² dx).

        Returns:
            np.ndarray: The norms of the basis functions.
        """
        pass


class PiecewiseConstantBasis(Basis):

    def __init__(self, n: int, interval: Tuple[float, float]):
        """Initialize the PiecewiseConstantBasis class.

        Args:
            n: The number of basis functions.
            interval: The interval over which the basis functions are defined.
        """
        super().__init__(n)
        self.interval_ = interval
        self.basis_ = []

        for i in range(self.n_):
            self.basis_.append(
                lambda x, i=i: np.piecewise(x, [
                    x < (i / self.n_), x == (i / self.n_), ((i / self.n_) < x) &
                    (x < ((i + 1) / self.n_)), x == ((i + 1) / self.n_), x > (
                        (i + 1) / self.n_)
                ], [0, 0.5, 1, 0.5, 0]))

        self.support_[:, 0] = np.linspace(interval[0], interval[1], n + 1)[:-1]
        self.support_[:, 1] = np.linspace(interval[0], interval[1], n + 1)[1:]

        self.norms_ = np.diag(np.sqrt((interval[1] - interval[0]) / n) * np.ones(n))

    def __call__(self, x: np.ndarray, i: int = None) -> np.ndarray:
        """Evaluate the piecewise constant basis functions at given points.

        Args:
            x: The points at which to evaluate the basis functions.
            i: The index of the specific basis function to evaluate. If None, evaluate all basis functions. Default is None.

        Returns:
            np.ndarray: The evaluated basis functions.
        """
        if i is None:
            return np.array([basis(x) for basis in self.basis_]).T
        else:
            return self.basis_[i](x)

    def get_support(self, i: int = None) -> np.ndarray:
        """Get the support of the piecewise constant basis functions.

        Args:
            i: The index of the specific basis function to get the support for. If None, get the support for all basis functions. Default is None.

        Returns:
            np.ndarray: The support of the basis functions.
        """
        if i is None:
            return self.support_
        else:
            return self.support_[i]

    def get_norms(self) -> np.ndarray:
        """Get the norms of the piecewise constant basis functions.

        Returns:
            np.ndarray: The norms of the basis functions.
        """
        return self.norms_


class NormPiecewiseConstantBasis(PiecewiseConstantBasis):
    def __init__(self, n: int, interval: Tuple[float, float]):
        """Initialize the Normalized Piecewise Constant Basis class.

        Args:
            n: The number of basis functions.
            interval: The interval over which the basis functions are defined.
        """
        super().__init__(n, interval)
        self.norms_ = (interval[1] - interval[0]) / np.sqrt(n) * np.eye(n)

    def __call__(self, x: np.ndarray, i: int = None) -> np.ndarray:
        """Evaluate the normalized piecewise constant basis functions at given points.

        Args:
            x: The points at which to evaluate the basis functions.
            i: The index of the specific basis function to evaluate. If None, evaluate all basis functions. Default is None.

        Returns:
            np.ndarray: The evaluated basis functions.
        """
        if i is None:
            return np.array([basis(x) for basis in self.basis_]).T / np.diag(self.norms_)
        else:
            return self.basis_[i](x) / self.norms_[i,i]

    def get_norms(self) -> np.ndarray:
        """Get the norms of the normalized piecewise constant basis functions.

        Returns:
            np.ndarray: The norms of the basis functions.
        """
        return self.norms_


def squared_exponential_kernel(x: np.ndarray,
                               y: np.ndarray,
                               sigma: float = 1.0,
                               l: float = 1.0,
                               **kwargs) -> float:
    """Compute the squared exponential kernel between two points.

    Args:
        x: The first point.
        y: The second point.
        sigma: The standard deviation parameter of the kernel. Default is 1.0.
        l: The length scale parameter of the kernel. Default is 1.0.
        kwargs: Additional keyword arguments.

    Returns:
        float: The computed kernel value.
    """
    return sigma**2 * np.exp(-((x - y)**2) / (2 * l**2))


def compute_coefficients(kernel: Callable[
    [np.ndarray, np.ndarray, float, float], float],
                         basis: Basis,
                         interval: Tuple[float, float],
                         weights: Callable[[np.ndarray], float] = None,
                         kernel_params: Dict = None) -> np.ndarray:
    """Compute the coefficients for the given basis functions using the specified kernel.

    Args:
        kernel: The kernel function to use.
        basis: The basis functions.
        interval: The interval over which to compute the coefficients.
        weights: A function to compute weights. Default is None.
        kernel_params: Parameters for the kernel function. Default is None.

    Returns:
        np.ndarray: The computed coefficients.
    """

    n = basis.get_n()

    if weights is None:
        weights = lambda x: 1.

    def integrand(x: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
        return kernel(x, y, **kernel_params) * basis(x, p) * basis(
            y, q) * weights(x) * weights(y)

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
            coefficients[i, j], tol = integrate.nquad(integrand, [int_x, int_y],
                                                      args=(i, j))
            logger.debug(f"     {coefficients[i, j]:.4}")

    # normalize the coefficients
    for i in range(n):
        for j in range(i, n):
            coefficients_norm[i, j] = coefficients_norm[j, i] = coefficients[
                i, j] / np.sqrt(coefficients[i, i] * coefficients[j, j])

    logger.debug(f"Normalized coefficientes:\n {coefficients_norm}")

    return coefficients_norm

def compute_coefficients_norm(kernel: Callable[
    [np.ndarray, np.ndarray, float, float], float],
                         basis: Basis,
                         interval: Tuple[float, float],
                         weights: Callable[[np.ndarray], float] = None,
                         kernel_params: Dict = None) -> np.ndarray:
    """Compute the coefficients for the given basis functions using the specified kernel.

    Args:
        kernel: The kernel function to use.
        basis: The basis functions.
        interval: The interval over which to compute the coefficients.
        weights: A function to compute weights. Default is None.
        kernel_params: Parameters for the kernel function. Default is None.

    Returns:
        np.ndarray: The computed coefficients.
    """

    n = basis.get_n()

    if weights is None:
        weights = lambda x: 1.

    def integrand(x: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
        return kernel(x, y, **kernel_params) * basis(x, p) * basis(
            y, q) * weights(x) * weights(y)


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
            coefficients[i, j], tol = integrate.nquad(integrand, [int_x, int_y],
                                                      args=(i, j))
            logger.debug(f"     {coefficients[i, j]:.4}")

    # normalize the coefficients
    for i in range(n):
        for j in range(i, n):
            coefficients_norm[i, j] = coefficients_norm[j, i] = coefficients[i, j]


    logger.debug(f"Normalized coefficientes:\n {coefficients_norm}")

    return coefficients_norm

def compute_cell_average_covariance(basis: PiecewiseConstantBasis,
                                   sigma: float = 1.0,
                                   ell: float = 1.0,
                                   return_correlation: bool = True) -> np.ndarray:

    """Compute the covariance matrix for cell-averaged basis functions with a squared exponential kernel.
    Args:
        basis: The piecewise constant basis functions.
        sigma: The standard deviation parameter of the kernel. Default is 1.0.
        ell: The length scale parameter of the kernel. Default is 1.0.
        return_correlation: If True, return the correlation matrix instead of the covariance matrix. Default is True.
    Returns:
        np.ndarray: The computed covariance or correlation matrix.
    """
    n = basis.get_n()
    cov = np.zeros((n, n))
    support = basis.get_support()
    for i in range(n):
        for j in range(i, n):
            a, b = support[i]
            c, d = support[j]
            term1 = special.erf((b - c) / (np.sqrt(2) * ell))
            term2 = special.erf((b - d) / (np.sqrt(2) * ell))
            term3 = special.erf((a - c) / (np.sqrt(2) * ell))
            term4 = special.erf((a - d) / (np.sqrt(2) * ell))
            cov[i, j] = cov[j, i] = (sigma**2 * ell * np.sqrt(np.pi / 2) / ((b - a) * (d - c))) * (term1 - term2 - term3 + term4)
    if return_correlation:
        stddev = np.sqrt(np.diag(cov))
        corr = cov / np.outer(stddev, stddev)
        return corr
    else:
        return cov

def monte_carlo_2d(func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   x_lim: Tuple[float, float],
                   y_lim: Tuple[float, float],
                   num_samples: int = 100000) -> float:
    """Perform Monte Carlo integration over a 2D domain. Mostly for testing purposes.

    Args:
        func: The function to integrate. It should take two numpy arrays (x and y coordinates) and return an array of function values.
        x_lim: The limits for the x-axis as a tuple (min, max).
        y_lim: The limits for the y-axis as a tuple (min, max).
        num_samples: The number of random samples to generate. Default is 100000.

    Returns:
        float: The estimated integral of the function over the specified domain.
    """
    x_samples = np.random.uniform(x_lim[0], x_lim[1], num_samples)
    y_samples = np.random.uniform(y_lim[0], y_lim[1], num_samples)
    sample_values = func(x_samples, y_samples)
    area = (x_lim[1] - x_lim[0]) * (y_lim[1] - y_lim[0])
    return np.mean(sample_values) * area


def get_gaussian_random_field_projection_from_dict(
        config: Dict[str, Any], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Load the projection configuration.

    Args:
        config: The projection configuration.
        kwargs: Additional keyword arguments.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The projection function.
    """

    kernel_params = config['kernel_params']
    interval = config.get('interval', (0, 1))
    d = config['dim']
    mean = np.ones(d) * config['mean']
    cov = compute_coefficients(squared_exponential_kernel,
                               PiecewiseConstantBasis(d, interval),
                               interval,
                               kernel_params=kernel_params)

    return mean, cov

def get_gaussian_random_field_projection_norm_from_dict(
        config: Dict[str, Any], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Load the projection configuration.

    Args:
        config: The projection configuration.
        kwargs: Additional keyword arguments.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The projection function.
    """

    kernel_params = config['kernel_params']
    interval = config.get('interval', (0, 1))
    d = config['dim']
    mean = np.ones(d) * config['mean']
    cov = compute_coefficients_norm(squared_exponential_kernel,
                               NormPiecewiseConstantBasis(d, interval),
                               interval,
                               kernel_params=kernel_params)

    return mean, cov


if __name__ == '__main__':

    kernel_params = {'sigma': 1., 'l': 0.5}

    n_b = 2
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_b, interval)
    basis_2 = NormPiecewiseConstantBasis(n_b, interval)

    print("Computing coefficients numerically...")
    coefficients_numerical = compute_coefficients(squared_exponential_kernel,
                                                basis,
                                                interval,
                                                kernel_params=kernel_params)

    print("Computing coefficients analytically...")
    # coefficients_new = compute_coefficients_analytical(basis, interval, **kernel_params)
    coefficients_new = compute_coefficients_norm(squared_exponential_kernel,
                                                  basis_2,
                                                  interval,
                                                  kernel_params=kernel_params)

    print("\nNumerical coefficients:")
    print(coefficients_numerical)
    print("\nAnalytical coefficients:")
    print(coefficients_new)
    print("\nDifference (Numerical - Analytical):")
    print(coefficients_numerical - coefficients_new)
    print(f"\nMax absolute difference: {np.max(np.abs(coefficients_numerical - coefficients_new)):.8f}")

    # Test the basis functions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot basis functions
    x_vals = np.linspace(0, 1, 100)
    for i in range(n_b):
        axes[0, 0].plot(x_vals, basis(x_vals, i), label=f'basis_{i}')
    axes[0, 0].legend()
    axes[0, 0].set_title("Piecewise Constant Basis Functions")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("φᵢ(x)")

    # Plot variance function for both methods
    Phi = basis(x_vals).T
    Phi_2 = basis_2(x_vals).T
    cov_numerical = Phi.T @ coefficients_numerical @ Phi
    cov_analytical = Phi_2.T @ coefficients_new @ Phi_2

    axes[0, 1].plot(x_vals, np.diag(cov_numerical), label='Numerical', linestyle='--')
    axes[0, 1].plot(x_vals, np.diag(cov_analytical), label='Analytical', linestyle='-')
    axes[0, 1].legend()
    axes[0, 1].set_title("Variance Function Comparison")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Var(f(x))")

    # Plot coefficient matrices
    im1 = axes[1, 0].imshow(cov_analytical, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1, 0].set_title("Numerical Coefficients")
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(coefficients_numerical, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1, 1].set_title("Analytical Coefficients")
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # # Plot difference matrix
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # diff = coefficients_numerical - coefficients_new
    # im = ax.imshow(diff, cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    # ax.set_title(f"Difference Matrix (Max: {np.max(np.abs(diff)):.2e})")
    # plt.colorbar(im, ax=ax)
    # plt.show()
