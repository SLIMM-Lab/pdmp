import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pyvbmc import VBMC


def log_joint(x: np.ndarray, a: float = 1.15, b: float = 0.5, rho: float = 0.9) -> np.ndarray:
    x = np.atleast_2d(x)
    mean = np.zeros(2)
    cov = np.array([[1, rho], [rho, 1]])

    u1 = x[:, 0] / a
    u2 = x[:, 1] - b * (u1 ** 2 + a ** 2)

    return multivariate_normal(mean, cov).logpdf(np.vstack((u1, u2)).T)

    # return - 1 / (2 * 1 - rho**2) * (u1**2 + a**2)

    # return ((- 1. /( 2. * (1. - rho**2)))
    #         * ( (x[:,0]/a)**2 + a**2 * ( x[:,1] - b * (x[:,0]/a)**2 - b * a**2 )**2
    #            - 2*rho*x[:,0] * (x[:,1] - b * (x[:,0]/a)**2 - b*a**2)))


if __name__ == '__main__':
    x, y = np.meshgrid(np.linspace(-3, 3, 100),
                       np.linspace(-2, 7, 100))
    x_flat = np.vstack((x.flatten(), y.flatten())).T

    # p = np.exp(log_joint(x_flat))

    LB = np.full((1, 2), -np.inf)
    UB = np.full((1, 2), np.inf)

    PLB = np.full((1,2), -3)
    PUB = np.full((1,2), 3)

    x0 = np.array([[1. ,1.]])

    vbmc = VBMC(log_joint, x0, LB, UB, PLB, PUB)
    vp, results = vbmc.optimize()

    print(results)

    # fig, ax = plt.subplots()
    # ax.contourf(x, y, p.reshape(100, 100))
    # # ax.scatter(x,y)
    # fig.show()

    n_samples = int(3e5)
    Xs, _ = vp.sample(n_samples)

    # Easily compute statistics such as moments, credible intervals, etc.
    post_mean = np.mean(Xs, axis=0)  # Posterior mean
    post_cov = np.cov(Xs.T)  # Posterior covariance matrix
    print("The approximate posterior mean is:", post_mean)
    print("The approximate posterior covariance matrix is:\n", post_cov)

    vp.plot()

    print("done")