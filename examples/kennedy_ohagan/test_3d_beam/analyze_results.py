#!/usr/bin/env python3
"""Analyze and plot posteriors from K&O 3D beam calibration.

Handles RWM with K&O discrepancy (rwm/) and RWM without discrepancy (rwm_no_ko/).

The RWM sampler operates in LATENT space:
    xi[0] = pre-Sigmoid input for rho   (physical rho = sigmoid(xi[0]))
    xi[1] = log(l)                      (physical l   = exp(xi[1]))
    xi[2:5] (K&O only)                  = [log_s2d, log_s2e, log_rho_ko]  (Identity)
Samples are back-transformed to physical space for theta so they can be compared
with ground_truth.dat, which stores physical [rho, l, log_s2d, log_s2e, log_rho_ko].

Run from the test_3d_beam/ directory after running the samplers.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RWM_DIR = os.path.join(HERE, "rwm")
RWM_NO_KO_DIR = os.path.join(HERE, "rwm_no_ko")
GT_PATH = os.path.join(HERE, "ground_truth.dat")

PARAM_NAMES = [r"$\theta_\rho$", r"$\theta_l$",
               r"$\log\sigma^2_\delta$", r"$\log\sigma^2_\varepsilon$",
               r"$\log\rho_\mathrm{KO}$"]
THETA_NAMES = PARAM_NAMES[:2]
PSI_NAMES = PARAM_NAMES[2:]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_samples(directory: str, label: str):
    """Load RWM samples (kept in latent/theta space)."""
    path = os.path.join(directory, "samples.dat")
    if not os.path.exists(path):
        print(f"{label} samples not found: {path}")
        return None
    samples = np.loadtxt(path)
    print(f"{label}: loaded {samples.shape[0]} samples, dim={samples.shape[1]}")
    return samples


def _print_acceptance_rate(directory: str, label: str) -> None:
    """Print RWM acceptance rate from accepted.dat (0/1 per iteration)."""
    path = os.path.join(directory, "accepted.dat")
    if not os.path.exists(path):
        print(f"{label}: accepted.dat not found")
        return
    accepted = np.loadtxt(path)
    rate = float(np.mean(accepted))
    print(f"{label}: acceptance rate = {rate:.3f}  "
          f"({int(accepted.sum())}/{len(accepted)})")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_summary(name, samples, param_names, burn_in_frac=0.25):
    if samples is None:
        return None
    n_burn = int(len(samples) * burn_in_frac)
    post = samples[n_burn:]
    n_params = samples.shape[1]
    print(f"\n{name} posterior (burn-in={n_burn}):")
    for i in range(n_params):
        lo, med, hi = np.percentile(post[:, i], [5, 50, 95])
        print(f"  {param_names[i]:35s}: median={med:.3f}  90%-CI=[{lo:.3f}, {hi:.3f}]")
    return post


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_traces(rwm_post, rwm_no_ko_post, gt):
    reference = next(s for s in [rwm_post, rwm_no_ko_post] if s is not None)
    n_params = reference.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=False)
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if rwm_post is not None:
            ax.plot(rwm_post[:, i], lw=0.3, alpha=0.6, color="steelblue",
                    label="RWM (K&O)")
        if rwm_no_ko_post is not None and i < rwm_no_ko_post.shape[1]:
            ax.plot(rwm_no_ko_post[:, i], lw=0.3, alpha=0.6, color="forestgreen",
                    label="RWM (no K&O)")
        ax.set_ylabel(PARAM_NAMES[i])
        if gt is not None:
            ax.axhline(gt[i], color="k", lw=1.2, ls="--", label="truth")
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("post-burn-in iteration")
    fig.suptitle("Trace plots (post burn-in, theta in physical space)", y=1.01)
    fig.tight_layout()
    return fig


def _plot_marginals(rwm_post, rwm_no_ko_post, gt, param_indices, title):
    n = len(param_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, param_indices):
        if rwm_post is not None:
            ax.hist(rwm_post[:, i], bins=60, density=True, alpha=0.55,
                    color="steelblue", label="RWM (K&O)")
        if rwm_no_ko_post is not None and i < rwm_no_ko_post.shape[1]:
            ax.hist(rwm_no_ko_post[:, i], bins=60, density=True, alpha=0.55,
                    color="forestgreen", label="RWM (no K&O)")
        ax.set_xlabel(PARAM_NAMES[i])
        ax.set_ylabel("density")
        if gt is not None:
            ax.axvline(gt[i], color="k", lw=1.5, ls="--",
                       label=f"truth={gt[i]:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ko", action="store_true",
                        help="Skip loading and plotting the no-K&O RWM results")
    args = parser.parse_args()

    gt = np.loadtxt(GT_PATH) if os.path.exists(GT_PATH) else None
    if gt is not None:
        gt_latent = gt.copy()
        gt_latent[0] = np.log(gt[0] / (1.0 - gt[0]))  # logit(rho)
        gt_latent[1] = np.log(gt[1])                    # log(l)
        # gt_latent[2:] unchanged (psi already in log space)
        print("Ground truth (theta in latent space, psi in log space):")
        for i, name in enumerate(PARAM_NAMES):
            print(f"  {name:35s} = {gt_latent[i]:.4f}  (physical: {gt[i]:.4f})")
    else:
        gt_latent = None

    print("\nAcceptance rates:")
    _print_acceptance_rate(RWM_DIR, "RWM (K&O)")
    if not args.no_ko:
        _print_acceptance_rate(RWM_NO_KO_DIR, "RWM (no K&O)")

    rwm_samples = _load_samples(RWM_DIR, "RWM (K&O)")
    rwm_no_ko_samples = (None if args.no_ko
                         else _load_samples(RWM_NO_KO_DIR, "RWM (no K&O)"))

    burn_in_frac = 0.25
    rwm_post = _print_summary("RWM (K&O)", rwm_samples, PARAM_NAMES, burn_in_frac)
    rwm_no_ko_post = _print_summary("RWM (no K&O)", rwm_no_ko_samples,
                                    THETA_NAMES, burn_in_frac)

    out_dir = HERE
    os.makedirs(out_dir, exist_ok=True)

    if rwm_post is not None or rwm_no_ko_post is not None:
        fig = _plot_traces(rwm_post, rwm_no_ko_post, gt_latent)
        p = os.path.join(out_dir, "traces.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"\nSaved: {p}")
        plt.close(fig)

        fig = _plot_marginals(rwm_post, rwm_no_ko_post, gt_latent, [0, 1],
                              r"$\theta$ posterior marginals")
        p = os.path.join(out_dir, "theta_marginals.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
        plt.close(fig)

    if rwm_post is not None:
        fig = _plot_marginals(rwm_post, None, gt_latent, [2, 3, 4],
                              r"$\psi$ (hyper-parameter) posterior marginals")
        p = os.path.join(out_dir, "psi_marginals.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
