#!/usr/bin/env python
"""Compare two .npy tensor dumps."""

import argparse

import numpy as np

from tensor_dump import tensor_statistics


def compare_arrays(a, b):
    if a.shape != b.shape:
        return None
    af = a.astype(np.float64, copy=False).reshape(-1)
    bf = b.astype(np.float64, copy=False).reshape(-1)
    diff = af - bf
    denominator = np.linalg.norm(af) * np.linalg.norm(bf)
    cosine = float(np.dot(af, bf) / denominator) if denominator else (
        1.0 if np.array_equal(af, bf) else 0.0)
    return {
        'MAE': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        'MaxAE': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'RMSE': float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        'Cosine Similarity': cosine,
    }


def print_stats(label, array):
    stats = tensor_statistics(array)
    print('\n{}'.format(label))
    for key in ('shape', 'dtype', 'min', 'max', 'mean', 'std',
                'nan_count', 'inf_count'):
        print('{}: {}'.format(key, stats[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a')
    parser.add_argument('b')
    args = parser.parse_args()
    a = np.load(args.a, allow_pickle=False)
    b = np.load(args.b, allow_pickle=False)
    print('=' * 48)
    print('Tensor Compare')
    print('=' * 48)
    print_stats('A', a)
    print_stats('B', b)
    metrics = compare_arrays(a, b)
    print('\nDifference')
    if metrics is None:
        print('SHAPE MISMATCH')
        return
    for name, value in metrics.items():
        print('{}: {}'.format(name, value))


if __name__ == '__main__':
    main()
