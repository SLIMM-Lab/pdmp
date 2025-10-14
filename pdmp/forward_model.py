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

    def __init__(self, F: np.ndarray, n_params: int, x_obs: np.ndarray, field=None):
        """Initialize the model

        Args:
            F: collection of prescribed forces for each setting
            n_params: number of parameters (ignored if field provided)
            x_obs: observed x values
            field: Optional GaussianRandomField providing coefficient dimension
        """
        super().__init__()
        self.F_vals = F  # Actual values of F
        self.n_settings_ = self.F_vals.shape[0]  # Number of settings
        self.x_obs_ = x_obs  # Observations
        # store field if provided
        self.field = field
        if self.field is not None:
            self.n_params = int(self.field.dim)
        else:
            self.n_params = n_params  # Number of parameters
        if self.n_params is None:
            raise ValueError("Either n_params or field must define parameter dimension.")

    def _midpoints(self) -> np.ndarray:
        """Return midpoints of the n_params equal sub-intervals of [0,1]."""
        n = self.n_params
        return (np.arange(n) + 0.5) / n

    def _J_params_eff(self) -> np.ndarray:
        """Jacobian of effective piecewise constants with respect to model parameters.

        If a GaussianRandomField is attached, params_eff = Φ(mid) @ params, hence J = Φ(mid).
        Otherwise J is the identity.
        Shape: (n_eff, n_params) where n_eff == n_params == self.n_params.
        """
        n = self.n_params
        if self.field is None:
            return np.eye(n)
        mid = self._midpoints()
        # design_matrix returns shape (len(mid), dim) == (n, n)
        return self.field.design_matrix(mid)

    def _effective_params(self, params: np.ndarray) -> np.ndarray:
        """Return the piecewise constant parameter vector used by the analytic formula.

        If a GaussianRandomField is attached, evaluate it at cell midpoints to obtain
        effective piecewise constant values (these coincide with coefficients
        for a PiecewiseConstantBasis). Otherwise, return params directly.
        """
        if self.field is None:
            return np.asarray(params, dtype=float)
        n = self.n_params
        midpoints = self._midpoints()
        return self.field.evaluate(params, midpoints)

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
        params_eff = self._effective_params(params)
        return params_eff[I - 1]

    def eval(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic evaluation of u

        Formula for x in ((I-1)/n, I/n], with I = ceil(x*n), n = number of parameters:
            u(x) = F * [ (1/n) * sum_{k=1}^{I-1} 1/p_k + (x - (I-1)/n)/p_I ].
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError("effective params must be a vector of length n_params for analytic evaluation")
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p = 1.0 / params_eff
        cumsum_inv = np.concatenate(([0.0], np.cumsum(inv_p)))  # length n+1
        base_sum = (1.0 / n) * cumsum_inv[I - 1]
        final_term = (x - (I - 1) / n) / params_eff[I - 1]
        return self.F_vals[idx] * (base_sum + final_term)

    def eval_grad(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic gradient w.r.t. model parameters (coefficients).

        Internally, the analytic formulas are expressed w.r.t. the effective piecewise-constant
        values p = params_eff. When a GaussianRandomField is attached, p = Φ(mid) @ a (linear in
        the coefficients a). Using the chain rule: du/da = (du/dp) @ Φ(mid).

        Returns array shape (len(x), n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError("effective params must be a vector of length n_params for analytic gradient")
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p2 = 1.0 / (params_eff**2)
        base_prefix = -(1.0 / n) * inv_p2
        # Gradient w.r.t effective params p
        G_eff = np.zeros((m, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            stop = Ii - 1
            if stop > 0:
                G_eff[row, :stop] = base_prefix[:stop]
            G_eff[row, Ii - 1] = -(x[row] - (Ii - 1) / n) / (params_eff[Ii - 1]**2)
        G_eff *= Fval
        # Chain rule to parameters: J = ∂p/∂a = Φ(mid)
        J = self._J_params_eff()
        return G_eff @ J

    def eval_hessian(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0) -> np.ndarray:
        """Analytic Hessian w.r.t. model parameters (coefficients).

        With p = Φ(mid) @ a linear in a, the Hessian transforms as:
            H_a = J^T H_p J, where J = ∂p/∂a = Φ(mid).
        Returns array shape (len(x), n_params, n_params).
        """
        if x is None:
            x = self.x_obs_
        x = np.asarray(x, dtype=float)
        params = np.asarray(params, dtype=float)
        params_eff = self._effective_params(params)
        n = self.n_params
        if params_eff.shape[0] != n:
            raise ValueError("effective params must be a vector of length n_params for analytic Hessian")
        m = x.size
        I = np.ceil(x * n).astype(int)
        np.clip(I, 1, n, out=I)
        inv_p3 = 1.0 / (params_eff**3)
        # Hessian w.r.t effective params p
        H_eff = np.zeros((m, n, n), dtype=float)
        Fval = self.F_vals[idx]
        for row, Ii in enumerate(I):
            if Ii - 1 > 0:
                H_eff[row, :Ii - 1, :Ii - 1] = np.diag(2.0 / n * inv_p3[:Ii - 1])
            H_eff[row, Ii - 1, Ii - 1] = 2.0 * (x[row] - (Ii - 1) / n) * inv_p3[Ii - 1]
        H_eff *= Fval
        # Transform to parameters via J^T H J
        J = self._J_params_eff()
        # For each output row, compute J^T H_eff[row] J
        H = np.zeros((m, n, n), dtype=float)
        JT = J.T
        for row in range(m):
            H[row] = JT @ H_eff[row] @ J
        return H

    @override
    def get_dim_in(self) -> int:
        return self.n_params  # Return number of parameters

    def get_dim_out(self) -> int:
        return len(self.x_obs_)

    @override
    def get_n_settings(self):
        return self.n_settings_  # Return number of settings

    @classmethod
    def from_dict(cls, config: dict, field=None):
        """Create a PiecewiseConstantModel from a dictionary configuration.

        Args:
            config (dict): configuration dictionary
            field (GaussianRandomField, optional): field providing coefficient dimension

        Returns:
            PiecewiseConstantModel: piecewise constant model
        """
        F = np.array(config['F'])
        if field is not None:
            n_params = field.dim
        else:
            n_params = config['dim']

        if 'x_obs' in config:
            x_obs = np.array(config['x_obs'])
        elif 'n_obs_loc' in config:
            x_obs = np.linspace(0, 1, config['n_obs_loc'] + 1)[1:]
        else:
            raise ValueError("Observation locations not provided")

        return cls(F, n_params, x_obs, field=field)


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


def get_model(config: dict, field=None):
    """Get a model from a configuration dictionary

    Args:
        config: configuration dictionary
        field: Optional GaussianRandomField to inject into model (if supported)

    Returns:
        Model: model
    """
    if config['name'] == 'PiecewiseConstant':
        return PiecewiseConstantModel.from_dict(config, field=field)
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
