import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from pdmp.project_field import compute_coefficients, squared_exponential_kernel, PiecewiseConstantBasis
from pdmp import logger


class Model:
    """
    Base class for the forward model
    """

    def __init__(self):
        pass

    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the forward model
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: model evaluation
        """
        raise NotImplementedError

    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the gradient (jacobian) of the forward model outputs with respect to the parameters
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: gradient of model evaluation
        """
        raise NotImplementedError

    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the hessian of the forward model outputs with respect to the parameters
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: hessian of model evaluation
        """
        raise NotImplementedError

    def get_dim_in(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model
        """
        raise NotImplementedError

    def get_dim_out(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model outputs
        """
        raise NotImplementedError

    def get_n_settings(self) -> int:
        """
        Get number of settings.

        Returns:
            int: Number of settings. Default is 1 if not overridden by derived classes.
        """
        return 1

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create a model from a dictionary configuration.
        Args:
            config (dict): configuration dictionary
        Returns:
            Model: model
        """
        raise NotImplementedError


class PiecewiseConstantModel(Model):
    """
    Forward model for deformation of a 1d bar with piecewise constant Young's modulus
    """

    def __init__(self, F: np.ndarray, n_params: int, x_obs: np.ndarray):
        """
        Initialize the model
        Args:
            F (np.ndarray): collection of prescribed forces for each setting
            n_params (int): number of parameters
            x_obs (np.ndarray): observed x values
        """
        super().__init__()
        self.F = sy.symbols('F')  # Symbolic representation of F
        self.F_vals = F  # Actual values of F
        self.n_settings_ = self.F_vals.shape[0]  # Number of settings
        # assert self.n_settings_ == len(x_obs), "Number of settings does not match number of observations"
        self.x_obs_ = x_obs  # Observations

        self.n_params = n_params  # Number of parameters
        self.params = sy.symbols([f'p_{str(i)}' for i in range(n_params)
                                 ])  # Symbolic parameters
        self.params_val = np.zeros(n_params)  # Initial parameter values
        self.x = sy.symbols('x')  # Symbolic variable x
        self.u = sy.symbols('u', cls=sy.Function)  # Symbolic function u
        offset = 1.  # Offset for Young's modulus

        # Define Young's modulus as a piecewise function
        self.E = offset + (self.params[0] - offset) * sy.Heaviside(self.x)
        for i in range(n_params - 1):
            self.E += (self.params[i + 1] -
                       self.params[i]) * sy.Heaviside(self.x -
                                                      (i + 1) / self.n_params)
        self.E = self.E.rewrite(sy.Piecewise)
        logger.debug("Young's modulus:")
        logger.debug(f"  {self.E}")

        u_i = []  # List to store piecewise functions for u
        conditions = []  # List to store conditions for piecewise functions

        # Define piecewise functions for u
        for i in range(self.n_params):
            u_i.append(sy.S((self.x - (i / self.n_params)) / self.params[i]))
            conditions.append(sy.S(self.x < (i + 1) / self.n_params))
            for j in range(i):
                u_i[i] += (1 / self.n_params) / self.params[j]

        conditions[-1] = sy.S('1')  # Last condition is always true
        self.u = self.F * sy.Piecewise(*zip(
            u_i, conditions))  # Define u as a piecewise function
        self.u_np = sy.lambdify((self.x, self.F, *self.params), self.u,
                                'numpy')  # Convert u to a numpy function

        # Compute gradient of u with respect to parameters
        gradient = [sy.diff(self.u, param) for param in self.params]
        logger.debug("Gradient:")
        [logger.debug("  " + grad.__str__()) for grad in gradient]
        self.gradient = [
            sy.lambdify((self.x, self.F, *self.params), grad, 'numpy')
            for grad in gradient
        ]

        # Compute Hessian of u with respect to parameters
        hessian = [
            [sy.diff(grad, param) for param in self.params] for grad in gradient
        ]
        logger.debug("Hessian:")
        [[logger.debug("  " + hess.__str__())
          for hess in hessian_row]
         for hessian_row in hessian]
        self.hessian = [[
            sy.lambdify((self.x, self.F, *self.params), hess, 'numpy')
            for hess in hessian_row
        ]
                        for hessian_row in hessian]

    def eval_E(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Young's modulus at given x values
        Args:
            params (np.ndarray): parameter values
            x (np.ndarray): x values
        Returns:
            np.ndarray: Young's modulus evaluated at x
        """
        E = self.E.subs([*zip(self.params, params)
                        ])  # Substitute parameter values into E
        return np.array([E.subs(self.x, x_i) for x_i in x
                        ])  # Evaluate E at each x value

    def eval(self,
             params: np.ndarray,
             x: np.ndarray = None,
             idx: int = None) -> np.ndarray:
        """
        Evaluate displacements u at given x locations for setting idx
        Args:
            params (np.ndarray): parameter values
            x (np.ndarray): x values
            idx (int): setting index
        Returns:
            np.ndarray: u evaluated at x
        """
        if x is None:
            x = self.x_obs_  # Use observed x values if none provided

        if idx is None:
            idx = 0  # Default index is 0

        if len(params) == self.n_params:
            return self.u_np(x, self.F_vals[idx],
                             *params)  # Evaluate u with given parameters
        else:
            if len(params.shape) == 1:
                params = params[:, None]  # Reshape parameters if necessary
            assert params.shape[
                1] == self.n_params, "Array dimensions do not match"
            return self.u_np(x, self.F_vals[idx],
                             *[param[:, None] for param in params.T])

    def eval_grad(self, params: np.ndarray, x: np.ndarray = None, idx: int = 0):
        """
        Evaluate gradient of u with respect to parameters
        Args:
            params (np.ndarray): parameter values
            x (np.ndarray): x values
            idx (int): setting index
        Returns:
            np.ndarray: gradient of u with respect to parameters
        """
        if x is None:
            x = self.x_obs_  # Use observed x values if none provided

        if idx is None:
            idx = 0  # Default index is 0
        return np.array(
            [[grad(x_i, self.F_vals[idx], *params)
              for grad in self.gradient]
             for x_i in x])  # Evaluate gradient

    def eval_hessian(self,
                     params: np.ndarray,
                     x: np.ndarray = None,
                     idx: int = 0) -> np.ndarray:
        """
        Evaluate Hessian of u with respect to parameters
        Args:
            params (np.ndarray): parameter values
            x (np.ndarray): x values
            idx (int): setting index
        Returns:
            np.ndarray: Hessian of u with respect to parameters
        """
        if x is None:
            x = self.x_obs_  # Use observed x values if none provided

        if idx is None:
            idx = 0  # Default index is 0

        hessian = np.zeros(
            (len(x), self.n_params, self.n_params))  # Initialize Hessian array
        for i, x_i in enumerate(x):
            for j, hessian_row in enumerate(self.hessian):
                for k, hess in enumerate(hessian_row):
                    hessian[i, j, k] = hess(x_i, self.F_vals[idx],
                                            *params)  # Evaluate Hessian
        return hessian

    def get_dim_in(self) -> int:
        """
        Get dimension of the model
        Returns:
            int: dimension of the model
        """
        return self.n_params  # Return number of parameters

    def get_dim_out(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model outputs
        """
        return len(self.x_obs_)

    def get_n_settings(self):
        """
        Get number of settings
        Returns:
            int: number of settings
        """
        return self.n_settings_  # Return number of settings

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create a PiecewiseConstantModel from a dictionary configuration.
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
    """
    Get a piecewise constant model with noisy observations
    Args:
        n_params (int): number of parameters
        n_obs_loc (int): number of observation locations
        n_obs (int): number of observations
        sigma_obs (float): observation noise
        mean (float): mean for the prior
        rng (np.random.Generator): random number generator
        kernel_params (dict[str, float]): kernel parameters
    Returns:
        tuple[PiecewiseConstantModel, np.ndarray, np.ndarray, np.ndarray]:
            model, observations, ground truth, prior_cov
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
    """
    Forward model for a linear system (mainly for testing purposes)
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Initialize the model
        Args:
            A (np.ndarray): matrix A
            b (np.ndarray): vector b
        """
        super().__init__()
        assert A.shape[0] == b.shape[0], "Dimensions do not match"
        self.dim_in_ = A.shape[1]
        self.dim_out_ = A.shape[0]
        self.A_ = A
        self.b_ = b

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create a LinearModel from a dictionary configuration.
        Args:
            config (dict): configuration dictionary
        Returns:
            LinearModel: linear model
        """
        A = np.array(config['A'])
        b = np.array(config['b'])
        return cls(A, b)

    def eval(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the forward model
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: model evaluation
        """
        return self.A_ @ params + self.b_

    def eval_grad(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the gradient (jacobian) of the forward model outputs with respect to the parameters
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: gradient of model evaluation
        """
        return self.A_

    def eval_hessian(self, params: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluate the hessian of the forward model outputs with respect to the parameters
        Args:
            params (np.ndarray): parameter values
            **kwargs: additional arguments
        Returns:
            np.ndarray: hessian of model evaluation
        """
        s = self.A_.shape
        return np.zeros((s[0], s[1], s[1]))

    def get_dim_in(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model
        """
        return self.dim_in_

    def get_dim_out(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model outputs
        """
        return self.dim_out_


def get_model(config: dict):
    """
    Get a model from a configuration dictionary
    Args:
        config (dict): configuration dictionary
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
