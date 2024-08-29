import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def normal_pdf(x, mean, cov):
    return multivariate_normal.p_x(x, mean, cov)


# Gradient of the log PDF
def grad_log_pdf(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    return -inv_cov @ (x - mean)


# Intensity function for the non-homogeneous Poisson process (example)
def rates(i, t, x, v, gamma, params):
    mu = params['mean']
    sigma_inv = params['inv_cov']
    rate = v[i] * sigma_inv[i] @ (x - mu + t * v)
    rate = max(0, rate) + gamma
    return rate


def bounds(i, t, x, gamma, params):
    mu = params['mean']
    sigma_inv = params['inv_cov']
    a = np.abs(inv_cov[i]) @ np.abs((x - mean)) + gamma
    b = np.sum(np.abs(inv_cov[i]))
    return a + b * t


# Function to simulate switching times using Cinlar's method
def simulate_switching_times(x, v, gamma, params):
    t = - np.log(np.random.uniform(0, 1, x.shape[0]))
    # t = np.array([0.59857602, 3.59142201])
    # t = np.array([1.25593076, 0.92322315])

    mean = params['mean']
    inv_cov = params['inv_cov']

    a = np.abs(inv_cov) @ np.abs((x - mean)) + gamma
    b = np.sum(np.abs(inv_cov), axis=1)

    tau = np.divide((np.sqrt(a ** 2 + 2 * b * t) - a), b, where=b != 0)

    return tau

# def cinlars_method(f, x, dt=0.01):
def cinlars_method(f, x, v, gamma, params, dt=0.001):

    tau = - np.log(np.random.uniform(0, 1, x.shape[0]))
    # tau = np.array([1.25593076, 0.92322315])
    t = np.zeros_like(tau)

    for i in range(len(tau)):

        # print(f"Compting switching time for i = {i}")

        int = 0.
        int_new = 0.

        while (int < tau[i]):
            # print(f"int: {int:.5f}, tau: {t[i]:.5f}")
            t[i] += dt
            # int = int_new
            int += np.trapz(np.array([f(i, t[i], x, v, gamma, params),
                                      f(i, t[i] + dt, x, v, gamma, params)]), dx=dt)

        # t_new = t[i] + (tau - int) / f(t[i])
        # int_new = int + np.trapz(f(np.linspace(t[i], t_new, 2)), dx=t_new - t[i])
        # t[i] = t_new

    return t


# ZigZag Sampler
def zigzag_sampler(start, velocity, gamma, rate, bound, n_events, params):
    samples = [start]
    current_pos = start
    times = np.zeros(n_events)
    positions = np.zeros((n_events, start.shape[0]))
    velocities = np.zeros((n_events, start.shape[0]))
    positions[0] = start
    velocities[0] = velocity

    for i in range(0, n_events - 1):

        if i % 50 == 0:
            print(f"Sampling event {i}")

        # Sample the time to the next event
        taus = simulate_switching_times(positions[i], velocities[i], gamma, params)
        # print(taus)

        # taus = cinlars_method(rates, positions[i], velocities[i], gamma, params, dt=0.01)
        # print(taus_c)

        # j = np.random.multinomial(1, taus/np.sum(taus)).argmin()
        # j = np.argmin(taus)
        # T = taus[j]

        # j = np.argmin(taus)
        # T = taus[j]
        # times[i+1] = times[i] + T
        # positions[i+1] = positions[i] + T * velocities[i]
        # velocities[i+1] = velocities[i]
        # velocities[i + 1, j] *= -1

        # j = np.random.multinomial(1, taus/np.sum(taus)).argmin()
        j = np.argmin(taus)
        T = taus[j]
        m = rates(j, T, positions[i], velocities[i], gamma, params)
        M = bound(j, T, positions[i], gamma, params)
        p = m / M
        u = np.random.uniform(0, 1)
        print(u)
        velocities[i+1] = velocities[i]
        # print(f"m: {m}, M: {M}")
        # print(f"p: {p}, u: {u}")
        # if np.random.uniform(0, 1) < m / M:
        # if np.random.uniform(0, 1) < p:
        if u < p:
            velocities[i+1, j] *= -1

        # Update position
        positions[i+1] = positions[i] + T * velocities[i]

    return times, positions, velocities


# Visualization
def plot_path(samples, mean, cov):
    plt.figure(figsize=(8, 6))
    plt.plot(samples[:, 0], samples[:, 1], linestyle='-', color='blue', lw=1., alpha=0.5, zorder=5)
    # plt.plot(samples[:, 0], samples[:, 1], marker='o', markersize=1, linestyle='-', color='blue', lw=0.5)
    plt.title("ZigZag Sampler Path in 2D Normal Distribution")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)

    # Plot the contour of the 2D Normal distribution
    x, y = np.mgrid[mean[0]-1:mean[0]+1:.01, mean[1]-1:mean[1]+1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    plt.contourf(x, y, rv.p_x(pos), cmap='Oranges')
    plt.colorbar()
    plt.savefig("./ZigZag_Sampler_Path")
    plt.show()


if __name__ == '__main__':
    # Simulation Parameters
    # mean = np.array([3, 3])
    # cov = np.array([[1, 0], [0, 1]])
    mean = np.array([3.26736155, 4.09819007])
    inv_cov = np.array([[14.68933863, 3.91809474], [3.91809474, 4.32853531]])
    cov = np.linalg.inv(inv_cov)
    # inv_cov = np.linalg.inv(cov)
    params = {'mean': mean, 'inv_cov': inv_cov}

    start = np.array([4, 4])
    velocity = np.array([1, -1])
    n_samples = 20
    gamma = 0.01

    # Simulate the ZigZag Sampler
    times, positions, velocities = zigzag_sampler(start, velocity, gamma, rates, bounds, n_samples, params)

    # print(f"Times:\n{times}\n\n")
    # print(f"Positions:\n{positions}\n\n")
    # print(f"Velocities:\n{velocities}\n\n")

    print("Define the 2D Normal Distribution")
    plot_path(positions, mean, cov)
    print("done")
