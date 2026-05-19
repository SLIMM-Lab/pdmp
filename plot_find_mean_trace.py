#!/usr/bin/env python3
"""Trace plots for find_mean BFGS optimization diagnostics.

Usage:
    python plot_find_mean_trace.py [trace_file] [-o output.png]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot find_mean optimization trace from CSV")
    parser.add_argument("trace_file",
                        nargs="?",
                        default="find_mean_trace.csv",
                        help="Path to trace CSV (default: find_mean_trace.csv)")
    parser.add_argument("--output",
                        "-o",
                        default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.trace_file)
    except FileNotFoundError:
        print(f"Error: '{args.trace_file}' not found", file=sys.stderr)
        sys.exit(1)

    param_cols = [c for c in df.columns if c.startswith("x_")]
    n_params = len(param_cols)
    iters = df["iteration"]

    fig, (ax_obj, ax_grad, ax_params) = plt.subplots(3, 1, figsize=(10, 11))

    ax_obj.plot(iters, df["neg_log_post"], color="C0")
    ax_obj.set_ylabel(r"$-\log p(x)$")
    ax_obj.set_title("Negative log-posterior")
    ax_obj.grid(True, alpha=0.3)
    ax_obj.set_xlabel("Iteration")

    ax_grad.plot(iters, df["grad_norm"], color="C1")
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel(r"$\|\nabla \log p(x)\|$")
    ax_grad.set_title("Gradient norm")
    ax_grad.grid(True, which="both", alpha=0.3)
    ax_grad.set_xlabel("Iteration")

    many = n_params > 10
    alpha = 0.25 if many else 0.7
    for i, col in enumerate(param_cols):
        label = col if not many else None
        ax_params.plot(iters, df[col], alpha=alpha, color=f"C{i % 10}", label=label)
    ax_params.set_ylabel("Parameter value")
    ax_params.set_title(f"Parameter traces  ({n_params} parameters)")
    ax_params.grid(True, alpha=0.3)
    ax_params.set_xlabel("Iteration")
    if not many:
        ax_params.legend(fontsize="small", ncol=min(n_params, 4))

    fig.suptitle(f"find_mean diagnostics — {args.trace_file}", fontsize=11)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()