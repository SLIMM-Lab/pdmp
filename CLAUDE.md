# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pdmp` is a Bayesian inference framework for solving inverse problems in computational mechanics. It implements Piecewise Deterministic Markov Process (PDMP) samplers (ZigZag, Bouncy Particle) alongside traditional MCMC, paired with JAX-based finite element forward models and surrogate modeling.

## Environment Setup

```bash
mamba env create -f environment.yml
conda activate pdmp-jax

# jax-fem must be installed separately (pinned to commit c3fbcb3)
git clone https://github.com/deepmodeling/jax-fem.git /tmp/jax-fem
cd /tmp/jax-fem && git checkout c3fbcb3
pip install /tmp/jax-fem
```

## Commands

```bash
# Run all tests with coverage
pytest --cov=pdmp

# Run a single test file
pytest tests/test_distributions.py -v

# Run a single test
pytest tests/test_distributions.py::test_name -v

# Profile with snakeviz
python -m cProfile -o out.prof script.py && snakeviz out.prof
```

## Architecture

### Data flow

```
YAML config → loader.get_config()
                 ├─ get_target()   → Distribution (prior × likelihood = Posterior)
                 ├─ get_surrogate() → SurrogateModel (optional, accelerates likelihood)
                 └─ get_sampler()  → Sampler
                                        └─ .run() → samples
```

### Registry pattern

All samplers, surrogate models, and forward models self-register via decorators:

```python
@register_sampler('ZigZag')
class ZigZagSampler(Sampler): ...
```

`loader.py` is the single entry point that wires everything together via `SAMPLER_REGISTRY`, `SURROGATE_REGISTRY`, and `MODEL_REGISTRY`. Adding a new sampler means implementing `Sampler.run()`, `Sampler.write_data()`, `Sampler.from_dict()`, and applying `@register_sampler('Name')`.

### Key modules

| Module | Role |
|---|---|
| `sampler.py` | Base `Sampler` class + `SAMPLER_REGISTRY` |
| `mcmc.py` | `StepSampler` — adaptive-step MCMC |
| `zigzag.py` | `ZigZagSampler` — continuous PDMP sampler |
| `bouncy_particle.py` | `BouncyParticleSampler` — velocity-reflecting PDMP |
| `distributions.py` | `Distribution`, `MultivariateNormal`, `Posterior`, `TransformedDistribution`; includes numerically robust Cholesky with jitter |
| `forward_model.py` | `PiecewiseConstantModel`, `JaxFemModel`; JAX autodiff compatible |
| `random_field.py` | Spatial field types: `JaxConstantField`, `JaxExponentialRecoveryField`, `JaxGaussianRandomField` |
| `project_field.py` | Basis function implementations for field discretization |
| `surrogates.py` | `LaplaceSurrogate`, `NeuralNetwork`, `GaussianProcess`, `DerivativeGaussianProcess`, `ConstantSurrogate` |
| `loader.py` | Factory functions (`get_target`, `get_sampler`, `get_surrogate`, `get_config`) + YAML ↔ NumPy conversion |

### YAML configuration structure

```yaml
name: BayesianInverse
problem:
  name: BayesianInverse
  prior: { ... }
  likelihood: { ... }
  model:
    name: JaxFemModel
    field:
      name: JaxExponentialRecoveryField  # prefix "Jax" triggers get_jax_field()
sampler:
  name: ZigZag   # must match a key in SAMPLER_REGISTRY
  n_max: 100000
surrogate:
  name: GaussianProcess
output:
  dir: ./results
```

`loader.get_config()` loads YAML and converts numeric lists to NumPy arrays automatically (except `hidden_layers` and `update_model` keys).

### JAX vs NumPy fields

`loader.get_target()` checks the field `name` prefix: names starting with `Jax` use `get_jax_field()` (JAX arrays, autodiff compatible); others use `get_field()` (NumPy). Forward models that need gradients (e.g., `JaxFemModel`) require JAX fields.

### FEM integration

`JaxFemModel` wraps jax-fem PDE solvers. Mesh files live in `msh/`. Sensor placement and boundary face naming conventions are documented in `docs/sensor_configuration.md` and `docs/exponential_recovery_field.md`.

**jax-fem local installation:** The jax-fem library is installed from a local clone at `/home/leon/Nextcloud/Documents/projects/gradient_samplers/jax-fem`. Source files and example applications live there — read them directly when debugging or understanding jax-fem internals:

- Core source: `/home/leon/Nextcloud/Documents/projects/gradient_samplers/jax-fem/jax-fem/jax_fem/`
- Example applications: `/home/leon/Nextcloud/Documents/projects/gradient_samplers/jax-fem/jax-fem/applications/`