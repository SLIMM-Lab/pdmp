import os
import numpy as np
import matplotlib.pyplot as plt
from pdmp.forward_model import JaxFemModel

from pdmp.distributions import GaussianLikelihood
from pdmp.plotting_utils import get_2d_despined_figure

save_fig = True
path = './figures'

# make dir if not exists
if save_fig and not os.path.exists(path):
    os.makedirs(path)

if __name__ == "__main__":
    model = JaxFemModel(d_x=1.0, d_y=1.0, d_z=2.5, n_params=1, h=0.125, traction=[0, 15e-4, 0])

    theta = np.array([1.5])
    print(f"theta = {theta}")

    # Forward eval
    y = model.eval(theta, save=True)
    print(f"eval(theta) -> y shape: {y.shape}, y = {y}")

    # Linearize + VJP
    y_lin, vjp_fun = model.linearize(theta)
    print(f"linearize(theta) -> y shape: {y_lin.shape}")

    v = np.ones_like(y_lin)
    g_vjp = vjp_fun(v)
    print(f"vjp(theta, v) shape: {g_vjp.shape}, value: {g_vjp}")

    # Direct eval_vjp
    g_vjp2 = model.eval_vjp(theta, idx=0, v=v)
    print(f"eval_vjp(theta, v) shape: {g_vjp2.shape}, value: {g_vjp2}")

    # Full Jacobian
    J = model.eval_grad(theta)
    print(f"eval_grad(theta) -> J shape: {J.shape}, value: {J}")

    # Likelihood test
    sigma = 0.01
    # Generate observations from a different parameter value to get non-zero gradients
    theta_true = np.array([1.6])
    y_obs = np.array([model.eval(theta_true)])
    likelihood = GaussianLikelihood(model, y_obs, sigma)

    # Test evaluation at theta (different from theta_true)
    log_pdf = likelihood.log_density(theta)

    # Manual calculation: -0.5 * sum((y - y_obs)**2 / sigma**2) - const
    y_at_theta = model.eval(theta)  # Re-evaluate at theta for manual check
    diff = y_at_theta - y_obs
    expected_log_pdf = -0.5 * np.sum((diff / sigma) ** 2) - 0.5 * len(y_at_theta) * np.log(2 * np.pi * sigma**2)

    print(f"theta_true = {theta_true}, y_obs = {y_obs}")
    print(f"Likelihood log_pdf: {log_pdf}")
    print(f"Expected log_pdf:   {expected_log_pdf}")
    assert np.allclose(log_pdf, expected_log_pdf), "Likelihood evaluation mismatch!"
    print("GaussianLikelihood test passed.")

    # Test likelihood gradient
    print("\n--- Testing Likelihood Gradient ---")
    grad_log_pdf = likelihood.grad_log_density(theta)
    print(f"grad_log_density(theta) shape: {grad_log_pdf.shape}, value: {grad_log_pdf}")

    # Validate gradient with finite differences
    epsilon = 1e-5
    grad_fd = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus = theta.copy()
        theta_minus[i] -= epsilon

        log_pdf_plus = likelihood.log_density(theta_plus)
        log_pdf_minus = likelihood.log_density(theta_minus)

        grad_fd[i] = (log_pdf_plus - log_pdf_minus) / (2 * epsilon)

    print(f"Finite difference gradient: {grad_fd}")
    print(f"Relative error: {np.abs(grad_log_pdf - grad_fd) / (np.abs(grad_fd) + 1e-10)}")

    assert np.allclose(grad_log_pdf, grad_fd, rtol=1e-4, atol=1e-6), \
        f"Likelihood gradient mismatch! Analytical: {grad_log_pdf}, FD: {grad_fd}"
    print("Likelihood gradient test passed!")

    # plot log-likelihood for some interval and gradient at one point
    thetas = np.linspace(0.5, 4, 50)
    log_pdfs = np.array([np.exp(likelihood.log_density(np.array([th]))) for th in thetas])
    fig, ax = get_2d_despined_figure(figsize=(5, 3), equal_axes=False, keep_ticks=True)
    ax.plot(thetas, log_pdfs)

    p_theta = np.exp(likelihood.log_density(theta))

    theta_minus = theta - 0.1
    theta_plus = theta + 0.1
    g_minus = p_theta + (theta_minus - theta) * p_theta * grad_log_pdf
    g_plus = p_theta + (theta_plus - theta) *  p_theta * grad_log_pdf
    ax.plot([theta_minus, theta_plus], [g_minus, g_plus],
            color='C1')

    ax.set_xlabel('Global stiffness (GPa)')
    ax.set_ylabel('Likelihood')
    if save_fig:
        fig.savefig(f'{path}/likelihood_plot.pdf')
    plt.show()


    thetas = np.linspace(0.5, 2.5, 10)
    # thetas = np.linspace(0.5, 2.5, 2) * 1e3
    y_th = np.zeros((thetas.shape[0], y.shape[0]))

    for i, th in enumerate(thetas):
        y_th[i] = model.eval(th)

    min = theta - 0.4
    max = theta + 0.4

    J_min = y + J[:, 0] * (min - theta)
    J_max = y + J[:, 0] * (max - theta)

    Js = np.vstack((J_min, J_max))

    from pdmp.plotting_utils import get_2d_despined_figure

    fig, ax = get_2d_despined_figure(figsize=(5, 3), equal_axes=False, keep_ticks=True)
    labels = [r'$u_x$', r'$u_y$', r'$u_z$']

    for i in range(y_th.shape[1]):
        ax.plot(thetas, y_th[:, i], c=f'C{i}', label=labels[i])
        ax.plot(np.array([min, max]), Js[:, i], c=f'C{i}', linestyle='--')

    ax.set_xlabel('Global stiffness (GPa)')
    ax.set_ylabel('Observed displacement')
    ax.legend()

    if save_fig:
        fig.savefig(f'{path}/sensor_gradients.pdf')

    plt.show()
