import numpy as np

from pdmp.random_field import get_field, GaussianRandomField
from pdmp.project_field import PiecewiseConstantBasis, NormPiecewiseConstantBasis


def test_default_basis_piecewise_constant():
    cfg = {
        'name': 'GaussianRandomField',
        'dim': 4,
        'mean': 0.0,
        'kernel_params': {'sigma': 1.0, 'l': 0.2},
    }
    field = get_field(cfg)
    assert isinstance(field, GaussianRandomField)
    assert isinstance(field.basis, PiecewiseConstantBasis)
    assert field.dim == 4


def test_norm_piecewise_constant_basis():
    cfg = {
        'name': 'GaussianRandomField',
        'dim': 5,
        'mean': 1.0,
        'basis': 'NormPiecewiseConstant',
        'kernel_params': {'sigma': 1.0, 'l': 0.5},
    }
    field = get_field(cfg)
    assert isinstance(field, GaussianRandomField)
    assert isinstance(field.basis, NormPiecewiseConstantBasis)
    # ensure covariance is positive definite (Cholesky will work through distribution usage)
    eigvals = np.linalg.eigvalsh(field.cov)
    assert np.all(eigvals > -1e-8)


def test_invalid_basis_name():
    cfg = {
        'name': 'GaussianRandomField',
        'dim': 3,
        'mean': 0.0,
        'basis': 'UnknownBasis',
        'kernel_params': {'sigma': 1.0, 'l': 0.3},
    }
    try:
        get_field(cfg)
    except ValueError as e:
        assert 'Unsupported basis' in str(e)
    else:
        raise AssertionError('Expected ValueError for invalid basis name')

