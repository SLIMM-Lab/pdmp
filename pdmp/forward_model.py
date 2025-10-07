import numpy as np
import matplotlib.pyplot as plt

from typing import override

from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp import logger


class Model:
    """Base class for the forward model"""

    def __init__(self):
        pass

    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the forward model

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: model evaluation
        """
        raise NotImplementedError

    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the gradient (jacobian) of the forward model outputs with respect to the parameters

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: gradient of model evaluation
        """
        raise NotImplementedError

    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the hessian of the forward model outputs with respect to the parameters

        Args:
            params: parameter values
            kwargs: additional keyword arguments

        Returns:
            np.ndarray: hessian of model evaluation
        """
        raise NotImplementedError

    def get_dim_in(self) -> int:
        """Get dimension of the model outputs

        Returns:
            int: dimension of the model
        """
        raise NotImplementedError

    def get_dim_out(self) -> int:
        """Get dimension of the model outputs

        Returns:
            int: dimension of the model outputs
        """
        raise NotImplementedError

    def get_n_settings(self) -> int:
        """Get number of settings.

        Returns:
            int: Number of settings. Default is 1 if not overridden by derived classes.
        """
        return 1

    @classmethod
    def from_dict(cls, config: dict):
        """Create a model from a dictionary configuration.

        Args:
            config: configuration dictionary

        Returns:
            Model: model
        """
        raise NotImplementedError


class PiecewiseConstantModel(Model):
    """    Forward model for deformation of a 1d bar with piecewise constant Young's modulus."""

    def __init__(self, F: np.ndarray, n_params: int, x_obs: np.ndarray):
        """Initialize the model

        Args:
            F: collection of prescribed forces for each setting
            n_params: number of parameters
            x_obs: observed x values
        """
        super().__init__()
        self.F_vals = F  # Actual values of F
        self.n_settings_ = self.F_vals.shape[0]  # Number of settings
        # assert self.n_settings_ == len(x_obs), "Number of settings does not match number of observations"
        self.x_obs_ = x_obs  # Observations

        self.n_params = n_params  # Number of parameters

    def eval_E(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate Young's modulus E(x) without sympy.

        For x in ((k-1)/n, k/n] we have E(x) = params[k-1]. For x=0 we assign E(0)=params[0].

        Args:
            params: array of shape (n_params,)
            x: 1D array of points in [0,1]
        Returns:
            1D array of E values with shape (len(x),)
        """
        params = np.asarray(params, dtype=float)
        x = np.asarray(x, dtype=float)
        n = self.n_params
        if params.shape[0] != n:
            raise ValueError(f"params must have length {n}, got {params.shape[0]}")
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("All x must be within [0, 1]")
        # Interval index I = ceil(x * n); clip to [1, n]
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        return params[I - 1]

    def eval(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic evaluation of u

        Formula for x in ((I-1)/n, I/n], with I = ceil(x*n), n = number of parameters:
            u(x) = F * [ (1/n) * sum_{k=1}^{I-1} 1/p_k + (x - (I-1)/n)/p_I ].
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        n = self.n_params
        if params.shape[0] != n:
            raise ValueError("params must be a vector of length n_params for analytic evaluation")
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p = 1.0 / params
        cumsum_inv = np.concatenate(([0.0], np.cumsum(inv_p)))  # length n+1
        base_sum = (1.0 / n) * cumsum_inv[I - 1]
        final_term = (x - (I - 1) / n) / params[I - 1]
        return self.F_vals[idx] * (base_sum + final_term)

    def eval_grad(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic gradient w.r.t. parameters.

        For x in interval I (1-based):
            d/d p_k u(x) = -F/n * 1/p_k^2            for k < I
                           -F * (x - (I-1)/n)/p_I^2  for k = I
                           0                         for k > I.
        Returns array shape (len(x), n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        n = self.n_params
        if params.shape[0] != n:
            raise ValueError("params must be a vector of length n_params for analytic gradient")
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p2 = 1.0 / (params**2)
        base_prefix = -(1.0 / n) * inv_p2
        G = np.zeros((m, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            stop = Ii - 1
            if stop > 0:
                G[row, :stop] = base_prefix[:stop]
            G[row, Ii - 1] = -(x[row] - (Ii - 1) / n) / (params[Ii - 1]**2)
        return Fval * G

    def eval_hessian(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic Hessian.

        Non-zero only on diagonal. For I = ceil(x*n):
            d2/d p_k^2 u(x) = F * 2/n * 1/p_k^3                        for k < I
                               F * 2*(x - (I-1)/n)/p_I^3              for k = I
            Mixed partials are zero.
        Returns array shape (len(x), n_params, n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        n = self.n_params
        if params.shape[0] != n:
            raise ValueError("params must be a vector of length n_params for analytic Hessian")
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p3 = 1.0 / (params**3)
        H = np.zeros((m, n, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            if Ii - 1 > 0:
                # k < I contributes 2/n * 1/p_k^3
                H[row, :Ii - 1, :Ii - 1] = np.diag(2.0 / n * inv_p3[:Ii - 1])
            # k = I
            H[row, Ii - 1, Ii - 1] = 2.0 * (x[row] - (Ii - 1) / n) * inv_p3[Ii - 1]
        return Fval * H

    @override
    def get_dim_in(self) -> int:
        return self.n_params  # Return number of parameters

    def get_dim_out(self) -> int:
        return len(self.x_obs_)

    @override
    def get_n_settings(self):
        return self.n_settings_  # Return number of settings

    @classmethod
    def from_dict(cls, config: dict):
        """Create a PiecewiseConstantModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary

        Returns:
            PiecewiseConstantModel: piecewise constant model
        """
        F = np.array(config['F'])
        n_params = config['dim']

        if 'x_obs' in config:
            x_obs = np.array(config['x_obs'])
        elif 'n_obs_loc' in config:
            x_obs = np.linspace(0, 1, config['n_obs_loc'] + 1)[1:]
        else:
            raise ValueError("Observation locations not provided")

        return cls(F, n_params, x_obs)


def get_piececwise_constant_model(
    n_params: int,
    n_obs_loc: int,
    n_obs: int = 1,
    sigma_obs: float = 0.025,
    mean: float = 4.,
    rng: np.random.Generator = np.random.default_rng(0),
    kernel_params: dict[str, float] = None
) -> tuple[PiecewiseConstantModel, np.ndarray, np.ndarray, np.ndarray]:
    """Get a piecewise constant model with noisy observations

    Args:
        n_params: number of parameters
        n_obs_loc: number of observation locations
        n_obs: number of observations
        sigma_obs: observation noise
        mean: mean for the prior
        rng: random number generator
        kernel_params: kernel parameters

    Returns:
        tuple[PiecewiseConstantModel, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            - model
            - observations
            - ground truth
            - prior_cov
    """

    if kernel_params is None:
        kernel_params = {'sigma': 1., 'l': 0.3}

    # get the prior field
    interval = (0, 1)
    basis = PiecewiseConstantBasis(n_params, interval)
    prior_cov = compute_coefficients(squared_exponential_kernel,
                                     basis,
                                     interval,
                                     kernel_params=kernel_params)

    # define observation locations
    x_obs = np.linspace(0, 1, n_obs_loc + 1)[1:]

    # set up the forward model
    F = [1.]
    F = np.array([item for item in F for i in range(n_obs)])
    model = PiecewiseConstantModel(F, n_params, x_obs)

    # generate ground truth from a multi-variate normal distribution
    params_gt = rng.multivariate_normal(mean * np.ones(n_params), prior_cov)
    logger.info(f"Ground truth: {params_gt}")

    # generate n_obs noisy observations of u_gt
    u_obs = np.zeros((len(F), len(x_obs)))

    for i in range(len(F)):
        u_gt = model.eval(params_gt, x_obs, idx=i)
        u_obs[i] = u_gt + rng.normal(0, sigma_obs, (1, u_gt.shape[0]))

    return model, u_obs, params_gt, prior_cov


class LinearModel(Model):
    """Forward model for a linear system (mainly for testing purposes)"""

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """Initialize the model

        Args:
            A: matrix A
            b: vector b
        """
        super().__init__()
        assert A.shape[0] == b.shape[0], "Dimensions do not match"
        self.dim_in_ = A.shape[1]
        self.dim_out_ = A.shape[0]
        self.A_ = A
        self.b_ = b

    @classmethod
    def from_dict(cls, config: dict):
        """Create a LinearModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary

        Returns:
            LinearModel: linear model
        """
        A = np.array(config['A'])
        b = np.array(config['b'])
        return cls(A, b)

    @override
    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        return self.A_ @ params + self.b_

    @override
    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        return self.A_

    @override
    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        s = self.A_.shape
        return np.zeros((s[0], s[1], s[1]))

    @override
    def get_dim_in(self) -> int:
        return self.dim_in_

    @override
    def get_dim_out(self) -> int:
        return self.dim_out_


def get_model(config: dict):
    """Get a model from a configuration dictionary

    Args:
        config: configuration dictionary

    Returns:
        Model: model
    """
    if config['name'] == 'PiecewiseConstant':
        return PiecewiseConstantModel.from_dict(config)
    elif config['name'] == 'Linear':
        return LinearModel.from_dict(config)
    else:
        raise ValueError(f"Model {config['name']} not recognized.")


if __name__ == '__main__':

    # Use non-interactive backend if display may not be available
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:  # pragma: no cover
        pass

    # Example piecewise constant model
    F = np.array([1., 2.])
    x_obs = np.array([0.1, 0.2])
    n_params = 2
    model = PiecewiseConstantModel(F, n_params, x_obs)
    print(
        model.eval_E(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3, 0.6, 1.1])))

    # x = np.linspace(0, 1, 100)
    x = np.array([0.1, 0.2, 0.3, 0.6, 1.0])
    # params = np.array([0.1, 0.2])
    params_all = np.linspace(1, 5, 100)
    params_all = np.vstack((params_all, np.ones_like(params_all))).T
    # grads = model.eval_grad(np.array(x), np.array([0.1, 0.2]))

    for j in range(len(F)):
        grads = np.zeros((params_all.shape[0], len(x)))
        u = np.zeros((params_all.shape[0], len(x)))
        for i, params in enumerate(params_all):
            grads[i] = model.eval_grad(params, x, j)[:, 1]
            u[i] = model.eval(params, x, j)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # ax.plot(params_all[:, 0], grads[:, 0], label='dE/dp_0')
        # ax.plot(params_all[:, 0], grads[:, 4], label='dE/dp_0')
        for i in range(len(x)):
            ax.plot(params_all[:, 0], u[:, i], label=f'u(x={x[i]})')
        ax.legend()
        plt.show()

    print('Done!')

    # Example linear model
    rng = np.random.default_rng(0)

    n = 2
    m = 3
    A = rng.random((m, n))
    b = rng.random(m)
    model = LinearModel(A, b)
    # model = get_linear_model()
    # n = model.get_dim()
    # m = n

    x_true = rng.random(n)
    y_true = model.eval(x_true)
    print(f"True output: {y_true}")

    # # create a noisy observation
    # noise = 0.1 * np.random.randn(m)
    # y_obs = y_true + noise
    # print(f"Noisy observation: {y_obs}")

    # evaluate the model on a grid and plot contours
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = model.eval(np.array([x[i], y[j]]))[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.contourf(X, Y, Z, 100)
    plt.show()
