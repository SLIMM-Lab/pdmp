import arviz
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom


def central_moment_from_skeleton(t: np.ndarray,
                                 x: np.ndarray,
                                 v: np.ndarray,
                                 degree: int,
                                 mean: np.ndarray = None) -> np.ndarray:
    """Compute the central moment of a piecewise linear curve defined by its skeleton.

    Args:
        t: 1D array of time points.
        x: 2D array of positions corresponding to the time points.
        v: 2D array of velocities corresponding to the segments between time points.
        degree: The degree of the moment to compute.
        mean: The mean of the curve. If None, it will be computed from the skeleton.

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
            total_integral += binom(degree,
                                    k) * a**(degree - k) * v0**k * integrand(k)

    if t[-1] < 1e-10:
        return x[0]
    return total_integral / t[-1]


def running_mean(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute the running mean of the process at each recorded time.

    The mean at time t[i] is computed by reusing the integral (or "mean contribution")
    up to time t[i-1] and adding the contribution of the new interval [t[i-1], t[i]].

    Args:
        t: 1D array of time points.
        x: 2D array of positions.
        v: 2D array of velocities (one per segment).

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


def running_mean_at_queries(t: np.ndarray, x: np.ndarray, v: np.ndarray,
                            t_query: np.ndarray) -> np.ndarray:
    """Compute the running mean of the process at arbitrary query times using the skeleton (t, x, v).

    The running mean at time T is given by
        m(T) = (∫₀ᵀ x(s) ds) / T.

    This function first computes the cumulative integral at the skeleton times and then,
    for each query time T (which may lie between skeleton times), it adds the contribution
    from the incomplete (partial) segment.

    Args:
        t: 1D array of skeleton time points (assumed sorted, with t[0]=0).
        x: 2D array of positions corresponding to t.
        v: 2D array of velocities (one per segment between skeleton nodes).
        t_query: 1D array of query times at which to compute the running mean.

    Returns:
        np.ndarray: Array of running means at each query time.
    """
    n_events, d = x.shape
    cum_int = np.zeros(
        (n_events, d))  # cumulative integral I(t) at skeleton times

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
                seg_mean_partial = central_moment_from_skeleton(
                    t_seg_partial, x_seg_partial, v_seg_partial, 1)
                partial_int = seg_mean_partial * dt_partial

                # Total integral up to T is the sum of the cumulative integral up to t[j] plus the partial contribution.
                total_int = cum_int[j] + partial_int
                results[idx] = total_int / T

    return results


def running_variance(t: np.ndarray, x: np.ndarray,
                     v: np.ndarray) -> np.ndarray:
    """Compute the running variance of the process at each recorded time.

    For a process with raw moments
        I1(t) = ∫₀ᵗ x(s) ds   and   I2(t) = ∫₀ᵗ x(s)² ds,
    the running mean and variance at time t are given by
        m(t) = I1(t) / t,   var(t) = I2(t) / t - m(t)².

    This function accumulates I1 and I2 incrementally by computing, for each segment [t[i-1], t[i]],
    the contribution of that segment using the existing central_moment_from_skeleton function with the mean forced to zero.

    Args:
        t: 1D array of time points.
        x: 2D array of positions.
        v: 2D array of velocities (one per segment).

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
        seg_I1 = central_moment_from_skeleton(
            t_seg, x_seg, v_seg, 1, mean=np.zeros_like(x_seg[0])) * dt
        seg_I2 = central_moment_from_skeleton(
            t_seg, x_seg, v_seg, 2, mean=np.zeros_like(x_seg[0])) * dt

        I1_total += seg_I1
        I2_total += seg_I2

        running_mean = I1_total / t[i]
        running_variances[i] = I2_total / t[i] - running_mean**2

    return running_variances


def running_variance_at_queries(t: np.ndarray, x: np.ndarray, v: np.ndarray,
                                t_query: np.ndarray) -> np.ndarray:
    """Compute the running variance of the process at arbitrary query times using the skeleton (t, x, v).

    For a process with raw moments
        I1(T) = ∫₀ᵀ x(s) ds   and   I2(T) = ∫₀ᵀ x(s)² ds,
    the running variance is given by
        Var(T) = I2(T)/T - (I1(T)/T)².

    This function first computes the cumulative integrals I1 and I2 at the skeleton times.
    Then, for each query time T (which may lie between skeleton times), it adds the partial
    contribution from the current segment.

    Args:
        t: 1D array of skeleton time points (assumed sorted, with t[0]=0).
        x: 2D array of positions corresponding to t.
        v: 2D array of velocities (one per segment between skeleton nodes).
        t_query: 1D array of query times at which to compute the running variance.

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
        seg_I1 = central_moment_from_skeleton(
            t_seg, x_seg, v_seg, 1, mean=np.zeros_like(x_seg[0])) * dt
        seg_I2 = central_moment_from_skeleton(
            t_seg, x_seg, v_seg, 2, mean=np.zeros_like(x_seg[0])) * dt

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
            results[idx] = cum_I2[-1] / t[-1] - mean_T**2
        else:
            # Find the last skeleton time that is <= T.
            j = np.searchsorted(t, T, side='right') - 1

            if T == t[j]:
                mean_T = cum_I1[j] / t[j]
                results[idx] = cum_I2[j] / t[j] - mean_T**2
            else:
                dt_partial = T - t[j]
                t_seg_partial = np.array([0, dt_partial])

                # For the partial segment, positions are linear: from x[j] to x[j] + v[j]*dt_partial.
                x_seg_partial = np.array([x[j], x[j] + v[j] * dt_partial])
                v_seg_partial = np.array([v[j]])
                seg_I1_partial = central_moment_from_skeleton(
                    t_seg_partial,
                    x_seg_partial,
                    v_seg_partial,
                    1,
                    mean=np.zeros_like(x_seg_partial[0])) * dt_partial
                seg_I2_partial = central_moment_from_skeleton(
                    t_seg_partial,
                    x_seg_partial,
                    v_seg_partial,
                    2,
                    mean=np.zeros_like(x_seg_partial[0])) * dt_partial

                I1_total = cum_I1[j] + seg_I1_partial
                I2_total = cum_I2[j] + seg_I2_partial
                mean_T = I1_total / T
                results[idx] = I2_total / T - mean_T**2

    return results


def running_sample_mean(samples: np.ndarray) -> np.ndarray:
    """Compute the running sample mean of a set of samples at each recorded time.

    Args:
        samples: 2D array of samples where each row is a sample.

    Returns:
        np.ndarray: Array of running sample means, one for each time point.
    """
    n_samples, d = samples.shape
    running_means = np.zeros((n_samples, d))

    running_means[0] = samples[0]
    running_means[1] = (samples[0] + samples[1]) / 2

    for i in range(2, n_samples):
        running_means[i] = (i - 1) / i * (running_means[i - 1] + samples[i] /
                                          (i - 1))

    return running_means


def running_sample_variance(samples: np.ndarray) -> np.ndarray:
    """Compute the running sample variance of a set of samples at each recorded time.

    Args:
        samples: 2D array of samples where each row is a sample.

    Returns:
        np.ndarray: Array of running sample variances, one for each time point.
    """
    n_samples, d = samples.shape
    running_means = np.zeros((n_samples, d))
    running_vars = np.zeros((n_samples, d))

    running_means[0] = samples[0]
    running_vars[0] = 0.0

    for i in range(1, n_samples):
        delta = samples[i] - running_means[i - 1]
        running_means[i] = running_means[i - 1] + delta / (i + 1)
        running_vars[i] = ((i - 1) * running_vars[i - 1] + delta *
                           (samples[i] - running_means[i])) / i

    return running_vars


def compute_ess_zigzag(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    num_batches: int = 1000,
    avg: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the effective sample size (ESS) of a zigzag process.

    The ESS is computed using the batch means method, which estimates the variance
    of the sample mean by dividing the total time by the number of batches. The ESS
    is then the total time divided by the autocorrelation of the process, which in
    turn is given by the ratio of the asymptotic variance to the variance of the process.

    Mathematically, the ESS for coordinate h (observable) is computed as follows:

    1. Estimate the mean under the target distribution π:
        π̂(h) = (1/τ) ∫₀^τ h(Ξ(s)) ds

    2. Estimate the variance under the target distribution π:
        Var_π(h) = (1/τ) ∫₀^τ h(Ξ(s))² ds - [π̂(h)]²

    3. Estimate the asymptotic variance (σ²_h) via batch means:
        σ²_h ≈ (τ/B-1) ∑_{i=1}^B [ (1/(τ/B)) ∫_{(i-1)τ/B}^{iτ/B} h(Ξ(s)) ds - π̂(h) ]²

    4. Compute ESS using:
        ESS = τ * Var_π(h) / σ²_h

    Args:
        t: 1D array of time points.
        x: 2D array of positions.
        v: 2D array of velocities.
        num_batches: Number of batches to use for the batch means method.
        avg: If True, return the average ESS across all coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - The ESS for each coordinate or the average ESS if avg is True.
            - The autocorrelation of the processes or their avg if avg is True.
            - The mean of the process.
            - The variance of the process.
    """
    total_time = t[-1] - t[0]
    batch_edges = np.linspace(t[0], t[-1], num_batches + 1)
    num_coords = x.shape[1]

    batch_means = np.zeros((num_batches, num_coords))

    # Compute batch means for each coordinate
    for b in range(num_batches):
        batch_start, batch_end = batch_edges[b], batch_edges[b + 1]
        batch_length = batch_end - batch_start

        integral = np.zeros(num_coords)

        for i in range(len(t) - 1):
            seg_start, seg_end = t[i], t[i + 1]
            seg_x, seg_v = x[i], v[i]

            # Check for overlap
            if seg_end <= batch_start or seg_start >= batch_end:
                continue

            # Overlap interval
            interval_start = np.max((seg_start, batch_start))
            interval_end = np.min((seg_end, batch_end))
            dt = interval_end - interval_start
            offset = interval_start - seg_start

            integral += seg_x * dt + 0.5 * seg_v * (2 * offset * dt + dt**2)

        batch_means[b] = integral / batch_length

    # Integrate entire trajectory to estimate mean and variance under target distribution
    dt = np.diff(t)[:, np.newaxis]  # shape: (N, 1)
    xi, vi = x[:-1], v[:-1]  # shape: (N, d)

    total_integral_h = np.sum(xi * dt + 0.5 * vi * dt**2, axis=0)
    total_integral_h2 = np.sum(xi**2 * dt + xi * vi * dt**2 +
                               (vi**2 / 3) * dt**3,
                               axis=0)

    mean_pi = total_integral_h / total_time
    mean_h2_pi = total_integral_h2 / total_time

    var_pi = mean_h2_pi - mean_pi**2

    # Asymptotic variance from batch means (for each coordinate)
    asymp_variance = (total_time / num_batches) * np.var(
        batch_means, axis=0, ddof=1)

    autocorrelation = asymp_variance / var_pi
    ess = total_time / autocorrelation

    if avg:
        return np.array(np.mean(ess)), np.array(
            np.mean(autocorrelation)), mean_pi, var_pi
    else:
        return ess, autocorrelation, mean_pi, var_pi


def sample_equidistant_along_path(positions: np.ndarray,
                                  velocities: np.ndarray,
                                  times: np.ndarray,
                                  t_k: float = None,
                                  N: int = 1000) -> np.ndarray:
    """N samples uniformly spaced along the path corresponting to interval [0,t_k].

    Args:
        positions: 2D array of positions at the skeleton times.
        velocities: 2D array of velocities (one per segment).
        times: 1D array of skeleton time points.
        t_k: The end time for sampling. If None, uses the last time in `times`.
        N: Number of samples to draw. Default is 1000.

    Returns:
        x: 2D array of sampled positions, shape (N, d), where d is the dimension of the positions.
    """
    if t_k is None:
        t_k = times[-1]

    u = np.linspace(0, t_k, N)  # N uniform times
    x = np.empty((N, positions.shape[1]))
    seg_idx = np.searchsorted(times, u, side="right") - 1
    dt = u - np.asarray(times)[seg_idx]
    x[:] = np.asarray(
        positions)[seg_idx] + dt[:, None] * np.asarray(velocities)[seg_idx]
    return x  # shape (N,d)


def compute_ess_zigzag_from_samples(
        t: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        n_samples: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the effective sample size (ESS) of a zigzag process from samples.

    Args:
        x: 2D array of positions (samples).
        t: 1D array of time points corresponding to the samples.
        v: 2D array of velocities (one per segment).
        n_samples: Number of samples to consider. If None, all samples are used.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The ESS for each coordinate or the average ESS if n_samples is specified.
            - The autocorrelation of the processes or their avg if n_samples is specified.
    """

    if n_samples is None:
        n_samples = int(x.shape[0] * 10)

    samples = sample_equidistant_along_path(x, v, t, t[-1], n_samples)

    dataset = arviz.from_dict(posterior={"param": samples[None, :, :]})
    ess = arviz.ess(dataset).to_array().values.flatten()
    autocorr = len(samples) / ess

    return ess, autocorr


def grad_fd(f: callable,
            x: np.ndarray,
            h: float = 1e-5,
            n: int = None) -> np.ndarray:
    """Compute the gradient of a function using finite differences.

    Args:
        f: The function for which the gradient is to be computed. It should take a numpy array as input and return a scalar.
        x: The point at which the gradient is to be computed. It should be a 1D numpy array.
        h: The step size for the finite difference approximation. Default is 1e-5.
        n: The number of outputs of the function. Default is None.

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
        grad[:,
             i] = (f(x + h * np.eye(m)[i]) - f(x - h * np.eye(m)[i])) / (2 * h)
    return grad


def hessian_fd(f: callable,
               x: np.ndarray,
               h: float = 1e-5,
               n: int = None) -> np.ndarray:
    """Compute the Hessian matrix of a function using finite differences.

    Args:
        f: The function for which the Hessian is to be computed. It should take a numpy array as input and return a scalar.
        x: The point at which the Hessian is to be computed. It should be a 1D numpy array.
        h: The step size for the finite difference approximation. Default is 1e-5.
        n: The number of outputs of the function. Default is None.

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
            hess[:, i,
                 j] = (f(x + h * np.eye(m)[i] + h * np.eye(m)[j]) -
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
