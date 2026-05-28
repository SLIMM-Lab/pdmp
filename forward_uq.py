#!/usr/bin/env python
"""Forward Uncertainty Quantification driver script.

Draws samples from a distribution (or loads them from a file), evaluates a
forward model on each sample, and stores the outputs together with the column
names and LaTeX labels needed to plot them. Plotting is handled separately by
``plot_forward_uq.py``.

Usage:
    # Sample from a distribution defined in the config
    python forward_uq.py path/to/config.yaml

    # Then plot the results
    python plot_forward_uq.py <output_dir>
"""

import argparse
import json
import os

import numpy as np

from pdmp.loader import get_config
from pdmp.forward_model import get_model
from pdmp.distributions import get_prior
from pdmp.random_field import get_field, get_jax_field
from pdmp.logger_setup import suppress_external_loggers


def flatten_output(out):
    """Flatten a model output to a 1-D numpy array.

    Handles both plain ndarray/scalar outputs and dict outputs (e.g. RVEModel).
    Returns (flat_array, legend) where legend is a list of column labels.
    """
    if isinstance(out, dict):
        parts, legend = [], []
        for key, val in out.items():
            arr = np.atleast_1d(np.array(val, dtype=float)).ravel()
            parts.append(arr)
            if arr.size == 1:
                legend.append(key)
            else:
                legend.extend([f'{key}[{i}]' for i in range(arr.size)])
        return np.concatenate(parts), legend
    else:
        arr = np.atleast_1d(np.array(out, dtype=float)).ravel()
        return arr, [f'out[{i}]' for i in range(arr.size)]


def main():
    parser = argparse.ArgumentParser(description='Forward UQ driver')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    config = get_config(args.config)

    # Optional field (needed by some models)
    field = None
    model_cfg = config['model']
    if isinstance(model_cfg, dict) and 'field' in model_cfg:
        field_cfg = model_cfg['field']
        field_name = field_cfg.get('name', '')
        if field_name.startswith('Jax'):
            field = get_jax_field(field_cfg)
        else:
            field = get_field(field_cfg)

    # Get samples: from file or by drawing from distribution
    uq_cfg = config.get('forward_uq', {})

    if 'samples_file' in uq_cfg:
        samples = np.loadtxt(uq_cfg['samples_file'])
        if samples.ndim == 1:
            samples = samples[:, None]
        print(
            f'Loaded {samples.shape[0]} samples from {uq_cfg["samples_file"]}')
    else:
        seed = int(uq_cfg.get('seed', 42))
        n_samples = int(uq_cfg.get('n_samples', 1000))
        rng = np.random.default_rng(seed)
        distribution = get_prior(config['distribution'], rng=rng, field=field)
        samples = distribution.get_sample(n_samples)

    model = get_model(model_cfg, field=field)
    suppress_external_loggers()

    # Evaluate model on each sample, flattening dict outputs
    column_names = None
    rows = []
    for i, s in enumerate(samples):
        raw = model.eval(s)
        flat, names = flatten_output(raw)
        if column_names is None:
            column_names = names
        rows.append(flat)
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f'  Evaluated {i + 1}/{len(samples)}', flush=True)

    outputs = np.array(rows)

    # Map raw column names to model-specific LaTeX labels (fall back to name).
    label_map = getattr(model, 'LATEX_LABELS', {})
    labels = [label_map.get(name, name) for name in column_names]

    # Save results
    out_dir = config.get('output', {}).get('dir', './results_forward_uq')
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir, 'samples.dat'), samples)
    np.savetxt(os.path.join(out_dir, 'outputs.dat'),
               outputs,
               header='\t'.join(column_names))

    # Metadata so plot_forward_uq.py can label columns without the model.
    metadata = {
        'n_samples': int(samples.shape[0]),
        'input_dim': int(samples.shape[1]),
        'output_dim': int(outputs.shape[1]),
        'columns': [{
            'name': name,
            'label': label
        } for name, label in zip(column_names, labels)],
    }
    with open(os.path.join(out_dir, 'outputs_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Geometry for spatial plots of *_cell location columns (RVE models).
    if hasattr(model, 'export_geometry'):
        geom = model.export_geometry()
        np.savetxt(os.path.join(out_dir, 'cell_centroids.dat'),
                   geom['cell_centroids'])
        with open(os.path.join(out_dir, 'geometry.json'), 'w') as f:
            json.dump({
                'L': geom['L'],
                'fibers': geom['fibers']
            }, f, indent=2)

    mean_out = np.mean(outputs, axis=0)
    std_out = np.std(outputs, axis=0)
    print(f'\nSaved {len(samples)} samples to {out_dir}/')
    print(f'Input dim:  {samples.shape[1]}')
    print(f'Output dim: {outputs.shape[1]}')
    for i, name in enumerate(column_names):
        print(f'  {name}: mean = {mean_out[i]:.4f}, std = {std_out[i]:.4f}')
    print(f'\nTo plot: python plot_forward_uq.py {out_dir}')


if __name__ == '__main__':
    main()
