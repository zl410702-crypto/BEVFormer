import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / 'tools/debug_static_sca_numerics.py'
SPEC = importlib.util.spec_from_file_location('debug_static_sca', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_arrays_reports_exact_values():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = MODULE.compare_arrays(values, values.copy())
    assert result['exact'] is True
    assert result['allclose'] is True
    assert result['max_abs'] == 0.0
    assert result['first_mismatching_index'] is None


def test_compare_arrays_reports_first_bitwise_divergence_without_relaxing():
    reference = np.array([[1.0, 2.0]], dtype=np.float32)
    static = reference.copy()
    static[0, 1] = np.nextafter(static[0, 1], np.float32(3.0))
    result = MODULE.compare_arrays(reference, static)
    assert result['exact'] is False
    assert result['first_mismatching_index'] == [0, 1]
    assert result['reference_value'] == 2.0
    assert result['static_value'] > 2.0
    assert result['allclose'] is True


def test_strict_allclose_constants_are_not_relaxed():
    assert MODULE.RTOL == 1e-5
    assert MODULE.ATOL == 1e-8
