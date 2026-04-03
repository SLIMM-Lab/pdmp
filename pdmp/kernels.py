"""Kernel functions for Gaussian processes.

This module provides shared kernel (covariance) functions used across the
codebase.  Two parameterization conventions coexist:

- **Inverse squared length-scale** (``rho``): used by :func:`rbf_kernel_matrix`
  and the Kennedy–O'Hagan likelihood.  The kernel entry is
  ``C_ij = exp(-sum_k rho_k (x_ik - x_jk)^2)``.

- **Amplitude + length scale** (``sigma``, ``l``): used by
  :func:`squared_exponential_kernel` and the random-field projection code.
  The kernel value is ``sigma^2 exp(-(x - y)^2 / (2 l^2))``.

Conversion (1-D, unit amplitude): ``rho = 1 / (2 l^2)``.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vectorised kernel matrix (rho parameterization)
# ---------------------------------------------------------------------------

def rbf_kernel_matrix(x_locs: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ARD squared-exponential (RBF) kernel matrix.

    C_ij = exp(-sum_k rho_k (x_ik - x_jk)^2)

    Args:
        x_locs: (m, d_x) sensor locations.
        rho: (d_x,) inverse squared length-scale per dimension.

    Returns:
        (m, m) positive-definite kernel matrix.
    """
    x_locs = np.atleast_2d(x_locs)
    rho = np.atleast_1d(rho).astype(float)
    # Weighted squared distances: sum_k rho_k (x_ik - x_jk)^2
    # Compute per-dimension squared differences scaled by rho
    diff = x_locs[:, np.newaxis, :] - x_locs[np.newaxis, :, :]  # (m, m, d_x)
    sq_dist = np.sum(rho[np.newaxis, np.newaxis, :] * diff**2, axis=2)  # (m, m)
    return np.exp(-sq_dist)


def _squared_diff_per_dim(x_locs: np.ndarray) -> np.ndarray:
    """Per-dimension squared difference matrices.

    Args:
        x_locs: (m, d_x) sensor locations.

    Returns:
        (d_x, m, m) array where [k, i, j] = (x_ik - x_jk)^2.
    """
    x_locs = np.atleast_2d(x_locs)
    diff = x_locs[:, np.newaxis, :] - x_locs[np.newaxis, :, :]  # (m, m, d_x)
    return np.transpose(diff**2, (2, 0, 1))  # (d_x, m, m)


def rbf_kernel_matrix_drho(x_locs: np.ndarray, rho: np.ndarray,
                           k: int) -> np.ndarray:
    """Derivative of the RBF kernel matrix w.r.t. rho_k.

    dC/d(rho_k) = -D_k * C   (element-wise)

    where D_k[i,j] = (x_ik - x_jk)^2.

    Args:
        x_locs: (m, d_x) sensor locations.
        rho: (d_x,) inverse squared length-scales.
        k: dimension index.

    Returns:
        (m, m) derivative matrix.
    """
    C = rbf_kernel_matrix(x_locs, rho)
    D_k = _squared_diff_per_dim(x_locs)[k]
    return -D_k * C


# ---------------------------------------------------------------------------
# Scalar kernel (sigma / l parameterization)
# ---------------------------------------------------------------------------

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
