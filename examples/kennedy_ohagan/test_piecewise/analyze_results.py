#!/usr/bin/env python3
"""Analyze and plot posteriors from K&O piecewise-constant calibration.

Handles RWM (rwm/), BPS (bps/), and a standard likelihood RWM (rwm_no_ko/).
BPS positions live in whitened ξ-space and are mapped back to the latent
[theta, psi] space via the affine transformation stored in bps/config.yaml.

Run from the test_piecewise/ directory after running both samplers.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RWM_DIR = os.path.join(HERE, "rwm")
BPS_DIR = os.path.join(HERE, "bps")
RWM_NO_KO_DIR = os.path.join(HERE, "rwm_no_ko")
GT_PATH = os.path.join(HERE, "ground_truth.dat")

PARAM_NAMES = [r"$E_1$", r"$E_2$",
               r"$\log\sigma^2_\delta$", r"$\log\sigma^2_\varepsilon$",
               r"$\log\rho$"]
THETA_NAMES = PARAM_NAMES[:2]
PSI_NAMES = PARAM_NAMES[2:]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_rwm_samples():
    """Load RWM samples (already in latent space)."""
    path = os.path.join(RWM_DIR, "samples.dat")
    if not os.path.exists(path):
        print(f"RWM samples not found: {path}")
        return None
    samples = np.loadtxt(path)
    print(f"RWM: loaded {samples.shape[0]} samples, dim={samples.shape[1]}")
    return samples


def _load_bps_samples():
    """Load BPS positions from whitened space and transform back to latent space."""
    pos_path = os.path.join(BPS_DIR, "positions.dat")
    cfg_path = os.path.join(BPS_DIR, "config.yaml")
    if not os.path.exists(pos_path):
        print(f"BPS positions not found: {pos_path}")
        return None

    positions = np.loadtxt(pos_path)
    print(f"BPS: loaded {positions.shape[0]} positions, dim={positions.shape[1]}")

    # Back-transform ξ → [theta, psi] via the affine map stored in the config.
    # Change into bps/ so that relative paths in config.yaml (e.g. ../observations.dat)
    # resolve correctly, then restore the original directory.
    from pdmp.loader import get_config, get_target
    import numpy as _np
    rng = _np.random.default_rng(0)
    config = get_config(cfg_path)
    _orig_dir = os.getcwd()
    os.chdir(BPS_DIR)
    try:
        target = get_target(config["problem"], rng=rng)
    finally:
        os.chdir(_orig_dir)
    # target is a TransformedDistribution; _transformation is AffineTransformation
    transform = target._transformation
    samples = np.array([transform.transform(xi) for xi in positions])
    print(f"BPS: transformed to latent space, shape={samples.shape}")
    return samples


def _load_rwm_no_ko_samples():
    """Load RWM samples from the standard (no K&O discrepancy) run."""
    path = os.path.join(RWM_NO_KO_DIR, "samples.dat")
    if not os.path.exists(path):
        print(f"RWM (no K&O) samples not found: {path}")
        return None
    samples = np.loadtxt(path)
    print(f"RWM (no K&O): loaded {samples.shape[0]} samples, dim={samples.shape[1]}")
    return samples


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

def _plot_traces(rwm_post, bps_post, rwm_no_ko_post, gt):
    # Use the highest-dimensional available sample set to determine n_params
    reference = next(s for s in [rwm_post, bps_post] if s is not None)
    n_params = reference.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=False)
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if rwm_post is not None:
            ax.plot(rwm_post[:, i], lw=0.3, alpha=0.6, color="steelblue",
                    label="RWM (K&O)")
        if bps_post is not None:
            ax.plot(bps_post[:, i], lw=0.3, alpha=0.6, color="darkorange",
                    label="BPS (K&O)")
        if rwm_no_ko_post is not None and i < rwm_no_ko_post.shape[1]:
            ax.plot(rwm_no_ko_post[:, i], lw=0.3, alpha=0.6, color="forestgreen",
                    label="RWM (no K&O)")
        ax.set_ylabel(PARAM_NAMES[i])
        if gt is not None:
            ax.axhline(gt[i], color="k", lw=1.2, ls="--", label="truth")
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("post-burn-in iteration")
    fig.suptitle("Trace plots (post burn-in)", y=1.01)
    fig.tight_layout()
    return fig


def _plot_marginals(rwm_post, bps_post, rwm_no_ko_post, gt, param_indices, title):
    n = len(param_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, param_indices):
        if rwm_post is not None:
            ax.hist(rwm_post[:, i], bins=60, density=True, alpha=0.55,
                    color="steelblue", label="RWM (K&O)")
        if bps_post is not None:
            ax.hist(bps_post[:, i], bins=60, density=True, alpha=0.55,
                    color="darkorange", label="BPS (K&O)")
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
    parser.add_argument("--no-bps", action="store_true",
                        help="Skip loading and plotting BPS results")
    parser.add_argument("--no-ko", action="store_true",
                        help="Skip loading and plotting the no-K&O RWM results")
    args = parser.parse_args()

    gt = np.loadtxt(GT_PATH) if os.path.exists(GT_PATH) else None
    if gt is not None:
        print("Ground truth (log-space for psi):")
        for i, name in enumerate(PARAM_NAMES):
            print(f"  {name:35s} = {gt[i]:.4f}")

    rwm_samples = _load_rwm_samples()
    bps_samples = None if args.no_bps else _load_bps_samples()
    rwm_no_ko_samples = None if args.no_ko else _load_rwm_no_ko_samples()

    burn_in_frac = 0.25
    rwm_post = _print_summary("RWM (K&O)", rwm_samples, PARAM_NAMES, burn_in_frac)
    bps_post = _print_summary("BPS (K&O)", bps_samples, PARAM_NAMES, burn_in_frac)
    rwm_no_ko_post = _print_summary("RWM (no K&O)", rwm_no_ko_samples,
                                    THETA_NAMES, burn_in_frac)

    out_dir = HERE
    os.makedirs(out_dir, exist_ok=True)

    ko_available = rwm_post is not None or bps_post is not None
    if ko_available:
        fig = _plot_traces(rwm_post, bps_post, rwm_no_ko_post, gt)
        p = os.path.join(out_dir, "traces.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"\nSaved: {p}")
        plt.close(fig)

    if ko_available or rwm_no_ko_post is not None:
        fig = _plot_marginals(rwm_post, bps_post, rwm_no_ko_post, gt, [0, 1],
                              r"$\theta$ posterior marginals")
        p = os.path.join(out_dir, "theta_marginals.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
        plt.close(fig)

    if ko_available:
        fig = _plot_marginals(rwm_post, bps_post, None, gt, [2, 3, 4],
                              r"$\psi$ (hyper-parameter) posterior marginals")
        p = os.path.join(out_dir, "psi_marginals.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
