import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

def central_moment_from_skeleton(t: np.ndarray, x: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute the central moment of a piecewise linear curve defined by its skeleton.

    Parameters:
    t (np.ndarray): 1D array of time points.
    x (np.ndarray): 2D array of positions corresponding to the time points.
    v (np.ndarray): 2D array of velocities corresponding to the segments between time points.
    degree (int): The degree of the moment to compute.

    Returns:
    np.ndarray: The computed central moment of the specified degree.
    """

    # Compute the mean of the curve
    if degree != 1:
        mean = central_moment_from_skeleton(t, x, v, 1)
    else:
        mean = np.zeros(x.shape[1])

    n_events, d = x.shape

    n_segments = n_events - 1
    total_integral = np.zeros_like(mean)

    for i in range(n_segments):
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]
        v0 = v[i]

        def integrand(k):
            return (t1 - t0)**(k + 1) / (k + 1)

        a = x0 - mean
        for k in range(degree + 1):
            # binomial_coeff = binom(degree, k)
            # integral = np.zeros_like(mean)
            total_integral += binom(degree, k) * a ** (degree - k) * v0 ** k * integrand(k)

    return total_integral / t[-1]

def grad_fd(f: callable, x: np.ndarray, h: float = 1e-5, n: int = None) -> np.ndarray:
    """
    Compute the gradient of a function using finite differences.

    Parameters:
    f (callable): The function for which the gradient is to be computed. It should take a numpy array as input and return a scalar.
    x (np.ndarray): The point at which the gradient is to be computed. It should be a 1D numpy array.
    h (float, optional): The step size for the finite difference approximation. Default is 1e-5.
    n (int, optional): The number of outputs of the function. Default is None.

    Returns:
    np.ndarray: The gradient of the function at the point x. It will be a 1D numpy array of the same length as x.
    """
    if n is None:
        y = f(x)
        if isinstance(y, float):
            n = 1
        else:
            n = len(y)

    m = len(x)
    grad = np.zeros((n, m))
    for i in range(m):
        grad[:,i] = (f(x + h * np.eye(m)[i]) - f(x - h * np.eye(m)[i])) / (2 * h)
    return grad

def hessian_fd(f: callable, x: np.ndarray, h: float = 1e-5, n: int = None) -> np.ndarray:
    """
    Compute the Hessian matrix of a function using finite differences.

    Parameters:
    f (callable): The function for which the Hessian is to be computed. It should take a numpy array as input and return a scalar.
    x (np.ndarray): The point at which the Hessian is to be computed. It should be a 1D numpy array.
    h (float, optional): The step size for the finite difference approximation. Default is 1e-5.
    n (int, optional): The number of outputs of the function. Default is None.

    Returns:
    np.ndarray: The Hessian matrix of the function at the point x. It will be a 2D numpy array of shape (n, n) where n is the length of x.
    """
    if n is None:
        y = f(x)
        if isinstance(y, float):
            n = 1
        else:
            n = len(y)

    m = len(x)
    hess = np.zeros((n, m, m))
    for i in range(m):
        for j in range(m):
            hess[:, i, j] = (f(x + h * np.eye(m)[i] + h * np.eye(m)[j]) -
                             f(x - h * np.eye(m)[i] + h * np.eye(m)[j]) -
                             f(x + h * np.eye(m)[i] - h * np.eye(m)[j]) +
                             f(x - h * np.eye(m)[i] - h * np.eye(m)[j])) / (4 * h**2)
    return hess


if __name__ == '__main__':

    # ---------------------------- test pd curve moments ----------------------------
    # Define times, positions and velocities
    t = np.array([0, 1, 2, 5])
    x = np.array([[0, 0], [-1, -1], [-2, 0], [1, 3]])
    v = np.array([[-1, -1], [-1, 1], [1, 1]])

    # Define the power n for the statistical moment
    n = 1

    # Compute the integral
    mean = central_moment_from_skeleton(t, x, v, n)
    variance = central_moment_from_skeleton(t, x, v, 2)
    print("Mean along the piecewise linear curve:", mean)
    print("Std along the piecewise linear curve:", np.sqrt(variance))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(*x.T, 'o-')
    ax.scatter(*mean, c='r')
    ax.axis('equal')
    plt.show()
