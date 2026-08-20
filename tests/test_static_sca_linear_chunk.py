import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / 'tools/search_static_sca_linear_chunk.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('search_static_sca_chunk', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('num_query,chunk_size', [(7, 3), (6, 3), (2, 8)])
def test_fixed_chunk_linear_matches_unchunked_cpu(num_query, chunk_size):
    torch.manual_seed(5)
    linear = torch.nn.Linear(4, 3)
    values = torch.randn(2, num_query, 4)
    expected = linear(values)
    actual = MODULE.fixed_chunk_linear(values, linear, chunk_size)
    assert actual.shape == expected.shape
    # This unit test checks mathematical/padding semantics. Bitwise equality
    # across GEMM shapes is exactly what the CUDA search script investigates.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_fixed_chunk_linear_rejects_invalid_contract():
    linear = torch.nn.Linear(4, 3)
    with pytest.raises(ValueError):
        MODULE.fixed_chunk_linear(torch.randn(2, 4), linear, 3)
    with pytest.raises(ValueError):
        MODULE.fixed_chunk_linear(torch.randn(2, 3, 4), linear, 0)
