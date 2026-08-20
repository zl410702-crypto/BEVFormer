#!/usr/bin/env python
"""Consistent NumPy tensor dumping and metadata collection."""

import json
from pathlib import Path

import numpy as np
import torch


SUPPORTED_KINDS = frozenset('biuf')


def to_numpy(value):
    """Detach a tensor and return a CPU NumPy array without changing values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    if value.dtype.kind not in SUPPORTED_KINDS:
        raise TypeError('Unsupported tensor dtype: {}'.format(value.dtype))
    return value


def tensor_statistics(value):
    array = to_numpy(value)
    finite = array[np.isfinite(array)] if array.dtype.kind == 'f' else array
    stats = {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'min': None,
        'max': None,
        'mean': None,
        'std': None,
        'nan_count': int(np.isnan(array).sum()) if array.dtype.kind == 'f' else 0,
        'inf_count': int(np.isinf(array).sum()) if array.dtype.kind == 'f' else 0,
    }
    if finite.size:
        numeric = finite.astype(np.float64, copy=False)
        stats.update(min=float(numeric.min()), max=float(numeric.max()),
                     mean=float(numeric.mean()), std=float(numeric.std()))
    return stats


class TensorDumper:
    """Write .npy tensors under one root and retain JSON-safe statistics."""

    def __init__(self, root):
        self.root = Path(root)
        self.records = []

    def dump(self, name, value, relative_path=None):
        array = to_numpy(value)
        relative = Path(relative_path or name)
        if relative.suffix != '.npy':
            relative = relative.with_suffix('.npy')
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(destination), array, allow_pickle=False)
        record = {'name': name, **tensor_statistics(array),
                  'path': str(destination)}
        self.records.append(record)
        return record

    def write_manifest(self, path=None):
        destination = Path(path) if path else self.root / 'tensor_manifest.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w') as stream:
            json.dump(self.records, stream, indent=2)
            stream.write('\n')
        return destination

