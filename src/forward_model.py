import numpy as np
import matplotlib.pyplot as plt
import sympy as sy


class ForwardModel:
    def __init__(self, F, n_params):
        self.F = sy.symbols('F')
        self.F_vals = np.array(F)
        self.n_settings_ = self.F_vals.shape[0]
        self.n_params = n_params
        self.params = sy.symbols([f'p_{str(i)}' for i in range(n_params)])
        self.params_val = np.zeros(n_params)
        self.x = sy.symbols('x')
        self.u = sy.symbols('u', cls=sy.Function)
        offset = 1.

        self.E = offset + (self.params[0] - offset) * sy.Heaviside(self.x)
        for i in range(n_params - 1):
            self.E += (self.params[i + 1] - self.params[i]) * sy.Heaviside(self.x - (i + 1) / self.n_params)
        self.E = self.E.rewrite(sy.Piecewise)
        print(f"Young's modulus:\n{self.E}\n")

        u_i = []
        conditions = []

        for i in range(self.n_params):
            u_i.append(sy.S((self.x - (i/self.n_params)) / self.params[i]))
            conditions.append(sy.S(self.x < (i + 1) / self.n_params))
            for j in range(i):
                u_i[i] += (1 / self.n_params) / self.params[j]

        conditions[-1] = sy.S('1')
        self.u = self.F * sy.Piecewise(*zip(u_i, conditions))
        self.u_np = sy.lambdify((self.x, self.F, *self.params), self.u, 'numpy')

        gradient = [sy.diff(self.u, param) for param in self.params]
        print("Gradient:\n")
        [print(grad) for grad in gradient]
        print("\n")
        self.gradient = [sy.lambdify((self.x, self.F, *self.params), grad, 'numpy') for grad in gradient]

        hessian = [[sy.diff(grad, param) for param in self.params] for grad in gradient]
        print("Hessian:\n")
        [[print(hess) for hess in hessian_row] for hessian_row in hessian]
        print("\n")
        self.hessian = [[sy.lambdify((self.x, self.F, *self.params), hess, 'numpy') for hess in hessian_row] for hessian_row in hessian]

    def eval_E(self, x, params):
        E = self.E.subs([*zip(self.params, params)])
        return np.array([E.subs(self.x, x_i) for x_i in x])

    def eval(self, x, params, idx=None):
        if idx is None:
            idx = 0

        if len(params) == self.n_params:
            return self.u_np(x, self.F_vals[idx], *params)
        else:
            if len(params.shape) == 1:
                params=params[:, None]
            assert params.shape[1] == self.n_params, "Array dimensions do not match"
            return self.u_np(x, self.F_vals[idx], *[param[:, None] for param in params.T])

    def eval_grad(self, x, params, idx=0):
        if idx is None:
            idx = 0
        return np.array([[grad(x_i, self.F_vals[idx], *params) for grad in self.gradient] for x_i in x])

    def eval_hessian(self, x, params, idx=0):
        if idx is None:
            idx = 0

        hessian = np.zeros((len(x), self.n_params, self.n_params))
        for i, x_i in enumerate(x):
            for j, hessian_row in enumerate(self.hessian):
                for k, hess in enumerate(hessian_row):
                    hessian[i, j, k] = hess(x_i, self.F_vals[idx], *params)
        return hessian

    def get_dim(self):
        return self.n_params

    def get_n_settings(self):
        return self.n_settings_


if __name__ == '__main__':

    # Example usage
    F = [1., 2.]
    n_params = 2
    model = ForwardModel(F, n_params)
    print(model.eval_E(np.array([0.1, 0.2, 0.3, 0.6, 1.1]), np.array([0.1, 0.2])))


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
            grads[i] = model.eval_grad(x, params, j)[:,1]
            u[i] = model.eval(x, params, j)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # ax.plot(params_all[:, 0], grads[:, 0], label='dE/dp_0')
        # ax.plot(params_all[:, 0], grads[:, 4], label='dE/dp_0')
        for i in range(len(x)):
            ax.plot(params_all[:, 0], u[:, i], label=f'u(x={x[i]})')
        ax.legend()
        plt.show()

    print('Done!')