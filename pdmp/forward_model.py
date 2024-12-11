import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from pdmp.utils import grad_fd, hessian_fd


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

    def get_dim(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model
        """
        raise NotImplementedError

    def get_n_settings(self) -> int:
        """
        Get number of settings.

        Returns:
            int: Number of settings. Default is 1 if not overridden by derived classes.
        """
        return 1

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
        self.params = sy.symbols([f'p_{str(i)}' for i in range(n_params)])  # Symbolic parameters
        self.params_val = np.zeros(n_params)  # Initial parameter values
        self.x = sy.symbols('x')  # Symbolic variable x
        self.u = sy.symbols('u', cls=sy.Function)  # Symbolic function u
        offset = 1.  # Offset for Young's modulus

        # Define Young's modulus as a piecewise function
        self.E = offset + (self.params[0] - offset) * sy.Heaviside(self.x)
        for i in range(n_params - 1):
            self.E += (self.params[i + 1] - self.params[i]) * sy.Heaviside(self.x - (i + 1) / self.n_params)
        self.E = self.E.rewrite(sy.Piecewise)
        print(f"Young's modulus:\n{self.E}\n")

        u_i = []  # List to store piecewise functions for u
        conditions = []  # List to store conditions for piecewise functions

        # Define piecewise functions for u
        for i in range(self.n_params):
            u_i.append(sy.S((self.x - (i/self.n_params)) / self.params[i]))
            conditions.append(sy.S(self.x < (i + 1) / self.n_params))
            for j in range(i):
                u_i[i] += (1 / self.n_params) / self.params[j]

        conditions[-1] = sy.S('1')  # Last condition is always true
        self.u = self.F * sy.Piecewise(*zip(u_i, conditions))  # Define u as a piecewise function
        self.u_np = sy.lambdify((self.x, self.F, *self.params), self.u, 'numpy')  # Convert u to a numpy function

        # Compute gradient of u with respect to parameters
        gradient = [sy.diff(self.u, param) for param in self.params]
        print("Gradient:\n")
        [print(grad) for grad in gradient]
        print("\n")
        self.gradient = [sy.lambdify((self.x, self.F, *self.params), grad, 'numpy') for grad in gradient]

        # Compute Hessian of u with respect to parameters
        hessian = [[sy.diff(grad, param) for param in self.params] for grad in gradient]
        print("Hessian:\n")
        [[print(hess) for hess in hessian_row] for hessian_row in hessian]
        print("\n")
        self.hessian = [[sy.lambdify((self.x, self.F, *self.params), hess, 'numpy') for hess in hessian_row] for hessian_row in hessian]

    def eval_E(self, params: np.ndarray,x: np.ndarray) -> np.ndarray:
        """
        Evaluate Young's modulus at given x values
        Args:
            params (np.ndarray): parameter values
            x (np.ndarray): x values
        Returns:
            np.ndarray: Young's modulus evaluated at x
        """
        E = self.E.subs([*zip(self.params, params)])  # Substitute parameter values into E
        return np.array([E.subs(self.x, x_i) for x_i in x])  # Evaluate E at each x value

    def eval(self, params: np.ndarray, x: np.ndarray=None, idx: int=None) -> np.ndarray:
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
            return self.u_np(x, self.F_vals[idx], *params)  # Evaluate u with given parameters
        else:
            if len(params.shape) == 1:
                params = params[:, None]  # Reshape parameters if necessary
            assert params.shape[1] == self.n_params, "Array dimensions do not match"
            return self.u_np(x, self.F_vals[idx], *[param[:, None] for param in params.T])

    def eval_grad(self, params: np.ndarray, x: np.ndarray=None, idx: int=0):
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
        return np.array([[grad(x_i, self.F_vals[idx], *params) for grad in self.gradient] for x_i in x])  # Evaluate gradient

    def eval_hessian(self, params: np.ndarray, x: np.ndarray=None, idx: int=0) -> np.ndarray:
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

        hessian = np.zeros((len(x), self.n_params, self.n_params))  # Initialize Hessian array
        for i, x_i in enumerate(x):
            for j, hessian_row in enumerate(self.hessian):
                for k, hess in enumerate(hessian_row):
                    hessian[i, j, k] = hess(x_i, self.F_vals[idx], *params)  # Evaluate Hessian
        return hessian

    def get_dim(self) -> int:
        """
        Get dimension of the model
        Returns:
            int: dimension of the model
        """
        return self.n_params  # Return number of parameters

    def get_n_settings(self):
        """
        Get number of settings
        Returns:
            int: number of settings
        """
        return self.n_settings_  # Return number of settings


class LinearModel:
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
        self.dim_ = A.shape[0]
        self.A_ = A
        self.b_ = b

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

    def get_dim(self) -> int:
        """
        Get dimension of the model outputs
        Returns:
            int: dimension of the model
        """
        return self.dim_


if __name__ == '__main__':

    # Example piecewise constant model
    F = np.array([1., 2.])
    x_obs = np.array([0.1, 0.2])
    n_params = 2
    model = PiecewiseConstantModel(F, n_params, x_obs)
    print(model.eval_E(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3, 0.6, 1.1])))

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
