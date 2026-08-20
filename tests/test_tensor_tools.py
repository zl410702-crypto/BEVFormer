import importlib.util
from pathlib import Path

import numpy as np
import torch


TOOLS = Path(__file__).parents[1] / 'tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tensor_dumper_supported_types(tmp_path):
    module = load('tensor_dump')
    dumper = module.TensorDumper(tmp_path)
    for name, value in (
            ('float32', torch.tensor([1.0], dtype=torch.float32)),
            ('float16', torch.tensor([2.0], dtype=torch.float16)),
            ('integer', torch.tensor([3], dtype=torch.int64)),
            ('boolean', torch.tensor([True], dtype=torch.bool))):
        record = dumper.dump(name, value)
        assert np.load(record['path'], allow_pickle=False).dtype == value.numpy().dtype
    assert len(dumper.records) == 4


def test_compare_and_shape_mismatch():
    module = load('tensor_compare')
    a = np.asarray([1.0, 2.0], dtype=np.float32)
    metrics = module.compare_arrays(a, a.copy())
    assert metrics['MAE'] == 0.0
    assert metrics['MaxAE'] == 0.0
    assert np.isclose(metrics['Cosine Similarity'], 1.0)
    assert module.compare_arrays(a, a.reshape(1, 2)) is None
