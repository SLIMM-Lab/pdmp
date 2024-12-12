import numpy as np

from pdmp.forward_model import LinearModel, PiecewiseConstantModel
from pdmp.utils import grad_fd, hessian_fd

SMALL = -1e-6
LARGE = 1e6


def test_linear_model():

    rng = np.random.default_rng(0)

    n, m = rng.integers(2, 10, size=2)
    A = rng.random((n, m))
    b = rng.random(n)
    model = LinearModel(A, b)

    # test gradient
    x = rng.random(m)
    grad = model.eval_grad(x)
    grad_finite_diff = grad_fd(model.eval, x, n=n)
    assert np.allclose(grad, grad_finite_diff, atol=1e-5)

    # test hessian
    hess = model.eval_hessian(x)
    hess_finite_diff = hessian_fd(model.eval, x, n=n)
    assert np.allclose(hess, hess_finite_diff, atol=1e-5)

def test_piecewise_constant_model():

    rng = np.random.default_rng(0)

    n, n_obs, n_loc = 3, 2, 5
    F = rng.random(n_obs)
    x_obs = np.linspace(0, 1, n_loc + 1)[1:]
    model = PiecewiseConstantModel(F, n, x_obs)

    # test gradient
    params = rng.random(n)
    grad = model.eval_grad(params)
    grad_finite_diff = grad_fd(model.eval, params, n=n_loc)
    assert np.allclose(grad, grad_finite_diff, atol=1e-5)

    # test hessian
    hess = model.eval_hessian(params)
    hess_finite_diff = hessian_fd(model.eval, params, n=n_loc, h=1e-5)
    assert np.allclose(hess, hess_finite_diff, atol=1e-5)
