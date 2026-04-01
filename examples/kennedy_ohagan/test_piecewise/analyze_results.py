#!/usr/bin/env python3
"""Analyze and plot posterior from K&O piecewise-constant calibration.

Run from this directory after completing run.sh.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.dat")

THETA_NAMES = [r"$E_1$", r"$E_2$"]
PSI_NAMES = [r"$\log\sigma^2_\delta$", r"$\log\sigma^2_\varepsilon$",
             r"$\log\rho$"]
PARAM_NAMES = THETA_NAMES + PSI_NAMES


def load_samples():
    path = os.path.join(RESULTS_DIR, "samples.dat")
    samples = np.loadtxt(path)
    print(f"Loaded {samples.shape[0]} samples, dim={samples.shape[1]}")
    return samples


def main():
    samples = load_samples()
    n_params = samples.shape[1]
    n_theta = 2
    burn_in = samples.shape[0] // 4

    gt = np.loadtxt(GT_PATH) if os.path.exists(GT_PATH) else None
    if gt is not None:
        print(f"\nGround truth (log-space for psi):")
        for i, name in enumerate(PARAM_NAMES[:n_params]):
            print(f"  {name} = {gt[i]:.4f}")

    print(f"\nPosterior summaries (after {burn_in} burn-in):")
    post = samples[burn_in:]
    for i in range(n_params):
        lo, med, hi = np.percentile(post[:, i], [5, 50, 95])
        print(f"  {PARAM_NAMES[i]:30s}: median={med:.4f}  90%-CI=[{lo:.4f}, {hi:.4f}]")

    # ── Trace plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params),
                             sharex=True)
    if n_params == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(samples[:, i], lw=0.4, alpha=0.7)
        ax.axvline(burn_in, color='r', linestyle='--', lw=0.8,
                   label='burn-in')
        ax.set_ylabel(PARAM_NAMES[i])
        if gt is not None:
            ax.axhline(gt[i], color='k', linestyle=':', lw=1.2,
                       label='truth')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Trace plots", y=1.01)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "traces.pdf"), bbox_inches='tight')
    print("\nSaved: results/traces.pdf")

    # ── Marginal posteriors for theta ─────────────────────────────────────
    fig, axes = plt.subplots(1, n_theta, figsize=(4 * n_theta, 4))
    if n_theta == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(post[:, i], bins=60, density=True, alpha=0.7,
                color='steelblue')
        ax.set_xlabel(THETA_NAMES[i])
        ax.set_ylabel("density")
        if gt is not None:
            ax.axvline(gt[i], color='k', linestyle='--', lw=1.5,
                       label=f"truth={gt[i]:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle(r"$\theta$ posterior marginals")
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "theta_marginals.pdf"),
                bbox_inches='tight')
    print("Saved: results/theta_marginals.pdf")

    # ── Marginal posteriors for psi ───────────────────────────────────────
    n_psi = n_params - n_theta
    if n_psi > 0:
        fig, axes = plt.subplots(1, n_psi, figsize=(4 * n_psi, 4))
        if n_psi == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            i = n_theta + j
            ax.hist(post[:, i], bins=60, density=True, alpha=0.7,
                    color='darkorange')
            ax.set_xlabel(PSI_NAMES[j])
            ax.set_ylabel("density")
            if gt is not None:
                ax.axvline(gt[i], color='k', linestyle='--', lw=1.5,
                           label=f"truth={gt[i]:.2f}")
            ax.legend(fontsize=8)
        fig.suptitle(r"$\psi$ (hyper-parameter) posterior marginals")
        fig.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "psi_marginals.pdf"),
                    bbox_inches='tight')
        print("Saved: results/psi_marginals.pdf")


if __name__ == "__main__":
    main()
