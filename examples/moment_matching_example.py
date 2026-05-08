"""Example: Moment matching for Bayesian inference and forward UQ.

This script demonstrates the two moment-matching drivers added to the repo:

  run_laplace.py         — Laplace approximation of a posterior.
                           Finds the MAP and approximates the posterior as a
                           Gaussian N(MAP, -inv(Hessian)).

  forward_uq_moment.py   — Pushes a Gaussian posterior's moments through a
                           forward model using two methods:
                             * Unscented transform (UT): deterministic
                               sigma points, exact to 3rd order for Gaussian
                               inputs, handles non-smooth outputs gracefully.
                             * First-order linearization: Gauss propagation
                               Sigma_y = J Sigma_x J^T via a central
                               finite-difference Jacobian.

Both methods are compared against Monte Carlo references so the approximation
quality is visible. No FEM or YAML files are required; the example is entirely
self-contained.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless; remove / change for interactive use
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pdmp.distributions import MultivariateNormal, find_mean, find_curvature

# ── helpers re-implemented inline so the example is self-contained ────────────

def _ut_sigma_points(mu, cov, alpha=1.0, beta=2.0, kappa=0.0):
    d = mu.shape[0]
    lam = alpha**2 * (d + kappa) - d
    c   = np.sqrt(d + lam)
    L   = np.linalg.cholesky(cov)
    pts = np.empty((2 * d + 1, d))
    pts[0] = mu
    for i in range(d):
        pts[1 + i]     = mu + c * L[:, i]
        pts[1 + d + i] = mu - c * L[:, i]
    wm      = np.full(2 * d + 1, 1.0 / (2.0 * (d + lam)))
    wc      = wm.copy()
    wm[0]   = lam / (d + lam)
    wc[0]   = lam / (d + lam) + (1.0 - alpha**2 + beta)
    return pts, wm, wc


def _fd_jacobian(f, x, h=1e-3):
    d  = x.size
    f0 = f(x)
    J  = np.empty((f0.size, d))
    for i in range(d):
        e      = np.zeros(d); e[i] = h
        J[:, i] = (f(x + e) - f(x - e)) / (2.0 * h)
    return J, f0


def _confidence_ellipse(ax, mean, cov, n_std=2.0, **kw):
    """Draw a covariance ellipse on *ax* for a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2.0 * n_std * np.sqrt(vals)
    ell   = Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)
    ax.add_patch(ell)


def _mrw_samples(log_density, dim, n_samples, step=0.3, seed=0):
    """Metropolis random-walk sampler on an arbitrary log-density."""
    rng     = np.random.default_rng(seed)
    chain   = np.empty((n_samples, dim))
    x       = np.zeros(dim)
    lp_x    = log_density(x)
    accepted = 0
    for i in range(n_samples):
        x_prop  = x + step * rng.standard_normal(dim)
        lp_prop = log_density(x_prop)
        if np.log(rng.random()) < lp_prop - lp_x:
            x, lp_x = x_prop, lp_prop
            accepted += 1
        chain[i] = x
    rate = accepted / n_samples
    return chain, rate


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Laplace approximation of a non-Gaussian posterior
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 62)
print("Example 1: Laplace approximation of a non-Gaussian posterior")
print("=" * 62)

# Target: log p(x) = -0.5*(x0^2) - 0.5*(x1^4)
# The x1-marginal is heavier-tailed than Gaussian (quartic, not quadratic).
# MAP is at the origin; the Laplace approximation underestimates the x1 spread.

class _QuarticPosterior:
    """log p(x) = -0.5*(x[0]^2 + x[1]^4); heavier-tailed in dim 1."""
    dim = 2
    rng = np.random.default_rng(0)

    def log_density(self, x):
        return float(-0.5 * (x[0]**2 + x[1]**4))

    def grad_log_density(self, x):
        return np.array([-x[0], -2.0 * x[1]**3])

    def hessian_log_density(self, x):
        return np.array([[-1.0, 0.0], [0.0, -6.0 * x[1]**2]])


target = _QuarticPosterior()

# ── Laplace approximation ──────────────────────────────────────────────────
map_point = find_mean(target, x_0=np.array([0.1, 0.1]))
laplace_cov = find_curvature(target, mean=map_point)
laplace_gaussian = MultivariateNormal(map_point, laplace_cov,
                                      rng=np.random.default_rng(1))
laplace_samples = laplace_gaussian.get_sample(5000)

print(f"\nMAP point            : {map_point}")
print(f"Laplace mean         : {map_point}")
print(f"Laplace std (dim 0)  : {np.sqrt(laplace_cov[0,0]):.4f}  (true: 1.0000)")
print(f"Laplace std (dim 1)  : {np.sqrt(laplace_cov[1,1]):.4f}"
      f"  (true MC std will be larger — heavy tail)")

# ── MC reference via Metropolis random walk ────────────────────────────────
print("\nRunning Metropolis random walk (5 000 samples)…")
chain, rate = _mrw_samples(target.log_density, dim=2, n_samples=5000,
                           step=0.4, seed=42)
burnin = 500
mc_samples = chain[burnin:]
print(f"  acceptance rate : {rate:.3f}")
print(f"  MC mean         : {mc_samples.mean(axis=0)}")
print(f"  MC std (dim 0)  : {mc_samples.std(axis=0)[0]:.4f}")
print(f"  MC std (dim 1)  : {mc_samples.std(axis=0)[1]:.4f}")

diff_std = mc_samples.std(axis=0) - np.sqrt(np.diag(laplace_cov))
print(f"\n  Laplace underestimates std by: {diff_std} "
      f"(positive = Laplace too narrow)")

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Forward UQ — UT vs linearization vs MC
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Example 2: Forward UQ — UT vs linearization vs MC")
print("=" * 62)

# Posterior moments (mimicking ITZ Laplace output in natural [theta] space).
mu_lat  = np.array([0.5, 1.2, -0.3])
Sig_lat = np.diag([0.3, 0.2, 0.5])

def _transform(theta):
    """Latent → physical: ITZ-like sigmoid/exp/sigmoid chain."""
    rho   = 1.0 / (1.0 + np.exp(-theta[0]))          # ∈ (0, 1)
    l     = np.exp(theta[1])                           # ∈ (0, ∞)  [µm]
    E_inf = 130.0 / (1.0 + np.exp(-theta[2]))         # ∈ (0, 130) [GPa]
    return rho, l, E_inf

def _forward(theta):
    """Analytic mock of ITZ-like output quantities.

    avg_stress  — linear in E_matrix  → smooth, UT ≈ lin
    max_vm      — quadratic in E_matrix → mild nonlinearity, UT > lin
    max_comp    — max(rho, l/100, E_inf/130) → piecewise, lin unreliable
    """
    rho, l, E_inf = _transform(theta)
    E_mat      = E_inf * (1.0 - (1.0 - rho) * np.exp(-10.0 / l))
    avg_stress = E_mat * 0.01
    max_vm     = E_mat**2 * 0.001
    max_comp   = max(rho, l / 100.0, E_inf / 130.0)
    return np.array([avg_stress, max_vm, max_comp])

output_names = ['avg_stress', 'max_vm', 'max(rho,l/100,E/130)']

# ── MC reference ──────────────────────────────────────────────────────────
rng_mc  = np.random.default_rng(0)
lat_mc  = rng_mc.multivariate_normal(mu_lat, Sig_lat, size=10_000)
out_mc  = np.array([_forward(s) for s in lat_mc])
mu_mc   = out_mc.mean(axis=0)
std_mc  = out_mc.std(axis=0)

# ── Unscented transform ───────────────────────────────────────────────────
pts, wm, wc = _ut_sigma_points(mu_lat, Sig_lat, alpha=1.0, beta=2.0, kappa=0.0)
outs_ut     = np.array([_forward(p) for p in pts])
mu_ut       = (wm[:, None] * outs_ut).sum(axis=0)
diffs       = outs_ut - mu_ut
cov_ut      = (wc[:, None, None] * diffs[:, :, None] * diffs[:, None, :]).sum(axis=0)
std_ut      = np.sqrt(np.diag(cov_ut))

# ── Linearization (central FD Jacobian) ──────────────────────────────────
J, f0   = _fd_jacobian(_forward, mu_lat, h=1e-3)
mu_lin  = f0
cov_lin = J @ Sig_lat @ J.T
std_lin = np.sqrt(np.diag(cov_lin))

print(f"\n{'Quantity':<24}  {'MC mean':>8}  {'UT mean':>8}  {'Lin mean':>8}")
print("-" * 56)
for i, name in enumerate(output_names):
    print(f"  {name:<22}  {mu_mc[i]:>8.4f}  {mu_ut[i]:>8.4f}  {mu_lin[i]:>8.4f}")

print(f"\n{'Quantity':<24}  {'MC std':>8}  {'UT std':>8}  {'Lin std':>8}")
print("-" * 56)
for i, name in enumerate(output_names):
    print(f"  {name:<22}  {std_mc[i]:>8.4f}  {std_ut[i]:>8.4f}  {std_lin[i]:>8.4f}")

# Highlight mean divergence — the primary failure mode of linearisation is a
# biased mean (not std) when the forward model is strongly nonlinear.
print("\nMean divergence (|μ_UT − μ_lin| / σ_MC):")
for i, name in enumerate(output_names):
    rel_diff = abs(mu_ut[i] - mu_lin[i]) / (std_mc[i] + 1e-12)
    note = "<<< LIN UNRELIABLE" if rel_diff > 0.05 else "OK"
    print(f"  {name:<22}  |UT-lin|/MC_std = {rel_diff:.3f}  {note}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("Creating figures…")
print("=" * 62)

fig_dir = os.path.join(os.path.dirname(__file__), "figures", "moment_matching")
os.makedirs(fig_dir, exist_ok=True)

# ── Figure 1: Example 1 — scatter + marginals ──────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig1.suptitle("Example 1 — Laplace vs. MC for a non-Gaussian posterior",
              fontsize=11)

# 2-D scatter
ax = axes[0]
ax.scatter(mc_samples[:, 0], mc_samples[:, 1],
           s=3, alpha=0.2, color='steelblue', label='MC samples')
ax.scatter(laplace_samples[:, 0], laplace_samples[:, 1],
           s=3, alpha=0.1, color='tomato', label='Laplace samples')
_confidence_ellipse(ax, map_point, laplace_cov, n_std=2,
                    edgecolor='tomato', facecolor='none', lw=2,
                    label='Laplace 2σ')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.legend(fontsize=8, markerscale=3)
ax.set_title("Joint distribution")

# Dim-0 marginal
for ax_i, dim, xlbl in zip(axes[1:], [0, 1],
                            [r'$\theta_0$ (Gaussian — Laplace exact)',
                             r'$\theta_1$ (quartic — Laplace too narrow)']):
    ax = ax_i
    grid = np.linspace(mc_samples[:, dim].min(), mc_samples[:, dim].max(), 200)
    from scipy.stats import norm
    lap_pdf = norm.pdf(grid, loc=map_point[dim],
                       scale=np.sqrt(laplace_cov[dim, dim]))
    ax.hist(mc_samples[:, dim], bins=50, density=True,
            alpha=0.4, color='steelblue', label='MC')
    ax.plot(grid, lap_pdf, color='tomato', lw=2, label='Laplace N')
    ax.set_xlabel(xlbl)
    ax.set_ylabel('Density' if dim == 0 else '')
    ax.legend(fontsize=8)

fig1.tight_layout()
path1 = os.path.join(fig_dir, 'example1_laplace_vs_mc.pdf')
fig1.savefig(path1, bbox_inches='tight')
print(f"  Saved {path1}")
plt.close(fig1)

# ── Figure 2: Example 2 — bar chart + UT correlation heatmap ───────────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Example 2 — Forward UQ: UT vs linearization vs MC",
              fontsize=11)

# Bar chart: mean ± std per method
ax = axes[0]
x_pos = np.arange(len(output_names))
width = 0.25
methods    = ['MC',      'UT',    'Lin']
means_all  = [mu_mc,     mu_ut,   mu_lin]
stds_all   = [std_mc,    std_ut,  std_lin]
colors     = ['steelblue','seagreen','tomato']

for k, (method, means, stds, color) in enumerate(
        zip(methods, means_all, stds_all, colors)):
    ax.bar(x_pos + (k - 1) * width, means, width,
           yerr=stds, capsize=4, label=method, color=color, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(output_names, fontsize=9)
ax.set_ylabel('Output value (± 1 std)')
ax.set_title('Mean ± std per method')
ax.legend(fontsize=9)
ax.axhline(0, color='k', lw=0.5, ls='--')

# UT covariance heatmap (correlation)
ax = axes[1]
std_ut_safe = np.where(std_ut > 0, std_ut, 1.0)
corr_ut     = cov_ut / (std_ut_safe[:, None] * std_ut_safe[None, :])
im = ax.imshow(corr_ut, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(range(len(output_names)))
ax.set_yticks(range(len(output_names)))
ax.set_xticklabels(output_names, rotation=25, ha='right', fontsize=8)
ax.set_yticklabels(output_names, fontsize=8)
plt.colorbar(im, ax=ax, label='Correlation')
for i in range(len(output_names)):
    for j in range(len(output_names)):
        ax.text(j, i, f'{corr_ut[i, j]:.2f}', ha='center', va='center',
                fontsize=8, color='k')
ax.set_title('UT output correlation matrix')

fig2.tight_layout()
path2 = os.path.join(fig_dir, 'example2_ut_vs_lin.pdf')
fig2.savefig(path2, bbox_inches='tight')
print(f"  Saved {path2}")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("Examples completed successfully.")
print("=" * 62)
