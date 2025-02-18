import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

def central_moment_from_skeleton(
        t: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        degree: int,
        mean: np.ndarray = None
) -> np.ndarray:
    """
    Compute the central moment of a piecewise linear curve defined by its skeleton.

    Parameters:
    t (np.ndarray): 1D array of time points.
    x (np.ndarray): 2D array of positions corresponding to the time points.
    v (np.ndarray): 2D array of velocities corresponding to the segments between time points.
    degree (int): The degree of the moment to compute.
    mean (np.ndarray, optional): The mean of the curve. If None, it will be computed from the skeleton.

    Returns:
    np.ndarray: The computed central moment of the specified degree.
    """

    # Compute the mean of the curve
    if degree != 1 and mean is None:
        mean = central_moment_from_skeleton(t, x, v, 1)

    if degree == 1:
        mean = np.zeros_like(x[0])

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

    if t[-1] < 1e-10:
        return x[0]
    return total_integral / t[-1]


def running_mean(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the running mean of the process at each recorded time.
    The mean at time t[i] is computed by reusing the integral (or "mean contribution")
    up to time t[i-1] and adding the contribution of the new interval [t[i-1], t[i]].

    Parameters:
        t (np.ndarray): 1D array of time points.
        x (np.ndarray): 2D array of positions.
        v (np.ndarray): 2D array of velocities (one per segment).

    Returns:
        np.ndarray: Array of running means, one for each time in t.
    """
    n_events, d = x.shape
    running_means = np.zeros((n_events, d))

    # At time t[0] we define the mean to be x[0] (assuming t[0] == 0).
    running_means[0] = x[0]

    # We'll accumulate the time integral of x(t) in total_integral.
    # That is, after i steps, total_integral = ∫₀^(t[i]) x(s) ds.
    total_integral = np.zeros(d)

    # Loop over segments
    for i in range(1, n_events):
        dt = t[i] - t[i - 1]
        # For the current segment, build a "local" skeleton where time starts at 0.
        t_seg = np.array([0, dt])
        x_seg = np.array([x[i - 1], x[i]])
        # v is defined per segment so we have one velocity per segment.
        v_seg = np.array([v[i - 1]])

        # Compute the mean of x on the segment [t[i-1], t[i]].
        # (Since degree==1, the function computes:
        #   (x[i-1]*dt + 0.5*v[i-1]*dt^2) / dt = x[i-1] + 0.5*v[i-1]*dt )
        seg_mean = central_moment_from_skeleton(t_seg, x_seg, v_seg, 1)

        # The contribution of this segment to the total time-integral is seg_mean * dt.
        total_integral += seg_mean * dt

        # The running mean up to time t[i] is the total integral divided by t[i].
        running_means[i] = total_integral / t[i]

    return running_means


def running_mean_at_queries(t: np.ndarray, x: np.ndarray, v: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """
    Compute the running mean of the process at arbitrary query times using the skeleton (t, x, v).

    The running mean at time T is given by
        m(T) = (∫₀ᵀ x(s) ds) / T.

    This function first computes the cumulative integral at the skeleton times and then,
    for each query time T (which may lie between skeleton times), it adds the contribution
    from the incomplete (partial) segment.

    Parameters:
        t (np.ndarray): 1D array of skeleton time points (assumed sorted, with t[0]=0).
        x (np.ndarray): 2D array of positions corresponding to t.
        v (np.ndarray): 2D array of velocities (one per segment between skeleton nodes).
        t_query (np.ndarray): 1D array of query times at which to compute the running mean.

    Returns:
        np.ndarray: Array of running means at each query time.
    """
    n_events, d = x.shape
    cum_int = np.zeros((n_events, d))  # cumulative integral I(t) at skeleton times

    # At t[0], we define I(0)=0.
    for i in range(1, n_events):
        dt = t[i] - t[i - 1]

        # Define a local skeleton for the segment [t[i-1], t[i]] with time reset to 0.
        t_seg = np.array([0, dt])
        x_seg = np.array([x[i - 1], x[i]])
        v_seg = np.array([v[i - 1]])

        # For degree 1, the function returns:
        #   seg_mean = (x[i-1]*dt + 0.5*v[i-1]*dt^2) / dt = x[i-1] + 0.5*v[i-1]*dt.
        seg_mean = central_moment_from_skeleton(t_seg, x_seg, v_seg, 1)

        # The contribution to the integral over the segment is seg_mean*dt.
        cum_int[i] = cum_int[i - 1] + seg_mean * dt

    results = np.zeros((len(t_query), d))

    for idx, T in enumerate(t_query):

        # For T <= t[0] (e.g. T==0), we return the initial position.
        if T <= t[0]:
            results[idx] = x[0]

        # If T is beyond the last skeleton time, use the full cumulative integral.
        elif T >= t[-1]:
            results[idx] = cum_int[-1] / t[-1]
        else:
            # Find the last skeleton time that is <= T.
            j = np.searchsorted(t, T, side='right') - 1

            # If T exactly equals a skeleton time, use the precomputed cumulative integral.
            if T == t[j]:
                results[idx] = cum_int[j] / T
            else:
                # Compute the partial contribution from the segment [t[j], T].
                dt_partial = T - t[j]
                t_seg_partial = np.array([0, dt_partial])

                # For a partial segment, the position is linear:
                # at time 0: x[j] and at time dt_partial: x[j] + v[j]*dt_partial.
                x_seg_partial = np.array([x[j], x[j] + v[j] * dt_partial])
                v_seg_partial = np.array([v[j]])
                seg_mean_partial = central_moment_from_skeleton(t_seg_partial, x_seg_partial, v_seg_partial, 1)
                partial_int = seg_mean_partial * dt_partial

                # Total integral up to T is the sum of the cumulative integral up to t[j] plus the partial contribution.
                total_int = cum_int[j] + partial_int
                results[idx] = total_int / T

    return results


def running_variance(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the running variance of the process at each recorded time.

    For a process with raw moments
        I1(t) = ∫₀ᵗ x(s) ds   and   I2(t) = ∫₀ᵗ x(s)² ds,
    the running mean and variance at time t are given by
        m(t) = I1(t) / t,   var(t) = I2(t) / t - m(t)².

    This function accumulates I1 and I2 incrementally by computing, for each segment [t[i-1], t[i]],
    the contribution of that segment using the existing central_moment_from_skeleton function with the mean forced to zero.

    Parameters:
        t (np.ndarray): 1D array of time points.
        x (np.ndarray): 2D array of positions.
        v (np.ndarray): 2D array of velocities (one per segment).

    Returns:
        np.ndarray: Array of running variances, one for each time in t.
    """
    n_events, d = x.shape
    running_variances = np.zeros((n_events, d))

    # I1_total and I2_total will store the accumulated raw moment integrals.
    I1_total = np.zeros(d)
    I2_total = np.zeros(d)

    # At time t[0] we have only one point, so variance is zero.
    running_variances[0] = 0.0

    for i in range(1, n_events):
        dt = t[i] - t[i - 1]

        # Define a local skeleton for the segment [t[i-1], t[i]] with time reset to 0.
        t_seg = np.array([0, dt])
        x_seg = np.array([x[i - 1], x[i]])
        v_seg = np.array([v[i - 1]])

        # Compute the raw (non-central) moment contributions.
        # For degree==1, the function automatically uses mean=0.
        seg_I1 = central_moment_from_skeleton(t_seg, x_seg, v_seg, 1, mean=np.zeros_like(x_seg[0])) * dt
        seg_I2 = central_moment_from_skeleton(t_seg, x_seg, v_seg, 2, mean=np.zeros_like(x_seg[0])) * dt

        I1_total += seg_I1
        I2_total += seg_I2

        running_mean = I1_total / t[i]
        running_variances[i] = I2_total / t[i] - running_mean ** 2

    return running_variances


def running_variance_at_queries(t: np.ndarray, x: np.ndarray, v: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """
    Compute the running variance of the process at arbitrary query times using the skeleton (t, x, v).

    For a process with raw moments
        I1(T) = ∫₀ᵀ x(s) ds   and   I2(T) = ∫₀ᵀ x(s)² ds,
    the running variance is given by
        Var(T) = I2(T)/T - (I1(T)/T)².

    This function first computes the cumulative integrals I1 and I2 at the skeleton times.
    Then, for each query time T (which may lie between skeleton times), it adds the partial
    contribution from the current segment.

    Parameters:
        t (np.ndarray): 1D array of skeleton time points (assumed sorted, with t[0]=0).
        x (np.ndarray): 2D array of positions corresponding to t.
        v (np.ndarray): 2D array of velocities (one per segment between skeleton nodes).
        t_query (np.ndarray): 1D array of query times at which to compute the running variance.

    Returns:
        np.ndarray: Array of running variances at each query time.
    """
    n_events, d = x.shape

    # cum_I1 and cum_I2 store the cumulative raw integrals at the skeleton times.
    cum_I1 = np.zeros((n_events, d))
    cum_I2 = np.zeros((n_events, d))

    # Compute cumulative integrals over the skeleton.
    # At t[0] we have I1(0)=0 and I2(0)=0.
    for i in range(1, n_events):
        dt = t[i] - t[i - 1]

        # Build a local skeleton for the segment [t[i-1], t[i]].
        t_seg = np.array([0, dt])
        x_seg = np.array([x[i - 1], x[i]])
        v_seg = np.array([v[i - 1]])

        # Force mean=0 to compute raw moments.
        seg_I1 = central_moment_from_skeleton(t_seg, x_seg, v_seg, 1, mean=np.zeros_like(x_seg[0])) * dt
        seg_I2 = central_moment_from_skeleton(t_seg, x_seg, v_seg, 2, mean=np.zeros_like(x_seg[0])) * dt

        cum_I1[i] = cum_I1[i - 1] + seg_I1
        cum_I2[i] = cum_I2[i - 1] + seg_I2

    results = np.zeros((len(t_query), d))

    for idx, T in enumerate(t_query):

        # For T <= t[0] (e.g. T==0), we define the variance as zero.
        if T <= t[0]:
            results[idx] = 0.0

        # If T is beyond the last skeleton time, use the full cumulative integrals.
        elif T >= t[-1]:
            mean_T = cum_I1[-1] / t[-1]
            results[idx] = cum_I2[-1] / t[-1] - mean_T ** 2
        else:
            # Find the last skeleton time that is <= T.
            j = np.searchsorted(t, T, side='right') - 1

            if T == t[j]:
                mean_T = cum_I1[j] / t[j]
                results[idx] = cum_I2[j] / t[j] - mean_T ** 2
            else:
                dt_partial = T - t[j]
                t_seg_partial = np.array([0, dt_partial])

                # For the partial segment, positions are linear: from x[j] to x[j] + v[j]*dt_partial.
                x_seg_partial = np.array([x[j], x[j] + v[j] * dt_partial])
                v_seg_partial = np.array([v[j]])
                seg_I1_partial = central_moment_from_skeleton(
                    t_seg_partial, x_seg_partial, v_seg_partial, 1, mean=np.zeros_like(x_seg_partial[0])
                ) * dt_partial
                seg_I2_partial = central_moment_from_skeleton(
                    t_seg_partial, x_seg_partial, v_seg_partial, 2, mean=np.zeros_like(x_seg_partial[0])
                ) * dt_partial

                I1_total = cum_I1[j] + seg_I1_partial
                I2_total = cum_I2[j] + seg_I2_partial
                mean_T = I1_total / T
                results[idx] = I2_total / T - mean_T ** 2

    return results


def running_sample_mean(samples: np.ndarray) -> np.ndarray:
    """
    Compute the running sample mean of a set of samples at each recorded time.

    Parameters:
        samples (np.ndarray): 2D array of samples where each row is a sample.

    Returns:
        np.ndarray: Array of running sample means, one for each time point.
    """
    n_samples, d = samples.shape
    running_means = np.zeros((n_samples, d))

    running_means[0] = samples[0]
    running_means[1] = (samples[0] + samples[1]) / 2

    for i in range(2, n_samples):
        running_means[i] = (i - 1) / i * (running_means[i - 1] + samples[i] / (i - 1))

    return running_means


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
