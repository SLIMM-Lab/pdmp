import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


# Define the squared exponential kernel function
def squared_exponential_kernel(x, y, sigma=1., l=1.):
    return sigma ** 2 * np.exp(-((x - y) ** 2) / (2 * l ** 2))


def compute_coefficients(n, kernel, basis, interval, weights=None):
    """
    Compute coefficients for the random field projection.
    :param n:
    :param m:
    :param kernel:
    :param funcs:
    :return:
    """

    if weights is None:
        weights = lambda x: 1.

    def integrand(x, y, n, m):
        return kernel(x, y) * basis[n](x) * basis[m](y) * weights(x) * weights(y)

    coefficients = np.zeros((n, n))
    print(f"Computing coefficients for basis functions")
    for i in range(n):
        for j in range(i, n):
            print(f"   {i} and {j}")
            coefficients[i, j] = integrate.dblquad(integrand,
                                                   interval[0], interval[1], interval[0], interval[1],
                                                   args=(i, j))[0]

    for i in range(n):
        for j in range(i, n):
            # normalize the coefficients
            coefficients[i, j] = coefficients[j, i] = coefficients[i, j] / np.sqrt(
                coefficients[i, i] * coefficients[j, j])

    return coefficients


if __name__ == '__main__':

    n_b = 5

    # Define the basis functions
    basis = []

    for i in range(n_b):
        basis.append(lambda x, i=i: np.piecewise(x,
                                                 [x < (i / n_b),
                                                  x == (i / n_b),
                                                  ((i / n_b) < x) & (x < ((i + 1) / n_b)),
                                                  x == ((i + 1) / n_b),
                                                  x > ((i + 1) / n_b)], [0, 0.5, 1, 0.5, 0]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Test the basis functions
    x_vals = np.linspace(0, 1, 10)
    for i in range(n_b):
        ax.plot(x_vals, basis[i](x_vals), label=f'basis_{i}')
    ax.legend()
    plt.show()

    interval = [0, 1]

    coefficients = compute_coefficients(n_b, squared_exponential_kernel, basis, interval)

    Phi = np.array([basis[i](x_vals) for i in range(n_b)])

    cov = Phi.T @ coefficients @ Phi

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_vals, np.diag(cov))

    plt.show()

    # plot full covariance matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c = ax.imshow(cov)
    fig.colorbar(c)
    plt.show()

    print(f"Coefficients:\n{coefficients}")



    print("Done!")
