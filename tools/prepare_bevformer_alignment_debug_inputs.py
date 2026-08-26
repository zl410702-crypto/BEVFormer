#!/usr/bin/env python
"""Prepare exact Frame-1 host affine-grid and linear-index debug inputs."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--runtime-frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def make_linear_index(grid, channels, height, width):
    """Use the exact deployment rewrite operations and return Gather indices."""
    if tuple(grid.shape) != (1, height, width, 2):
        raise ValueError('unexpected grid shape {}'.format(tuple(grid.shape)))
    x = ((grid[..., 0] + 1.0) * float(width) - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * float(height) - 1.0) * 0.5
    x_rounded = torch.round(x)
    y_rounded = torch.round(y)
    x_index = x_rounded.clamp(0.0, float(width - 1)).to(torch.long)
    y_index = y_rounded.clamp(0.0, float(height - 1)).to(torch.long)
    linear = (y_index * width + x_index).reshape(1, 1, -1)
    linear = linear.expand(1, channels, height * width).contiguous()
    return x, y, x_rounded, y_rounded, linear


def main():
    args = parse_args()
    fixture = torch.load(str(args.fixture), map_location='cpu')
    grid = fixture['grid'].detach().to(dtype=torch.float32).contiguous()
    prev_bev = fixture['prev_bev']
    if prev_bev.dtype != torch.float32 or prev_bev.dim() != 4:
        raise ValueError('unexpected prev_bev contract')
    _, channels, height, width = prev_bev.shape
    x, y, x_rounded, y_rounded, linear = make_linear_index(
        grid, channels, height, width)

    if args.output.exists():
        raise RuntimeError('refusing to overwrite {}'.format(args.output))
    shutil.copytree(str(args.runtime_frame), str(args.output))
    grid.reshape(1, height * width, 2).numpy().tofile(
        str(args.output / 'golden_affine_grid.bin'))
    # TensorRT 8.x exposes integer execution bindings as INT32. The debug ONNX
    # explicitly casts this input back to the original Gather INT64 contract.
    linear.to(dtype=torch.int32).numpy().tofile(
        str(args.output / 'golden_linear_index.bin'))

    reference_dir = args.output / 'alignment_reference'
    reference_dir.mkdir()
    tensors = {
        'affine_grid': grid.reshape(1, height * width, 2),
        'x_pixel': x,
        'y_pixel': y,
        'x_rounded': x_rounded,
        'y_rounded': y_rounded,
        'linear_index': linear,
        'gathered': torch.gather(prev_bev.reshape(1, channels, -1), 2, linear),
        'rotated_nchw': fixture['reference_output'],
    }
    for name, tensor in tensors.items():
        np.save(str(reference_dir / (name + '.npy')),
                tensor.detach().cpu().numpy(), allow_pickle=False)
    metadata = {
        'affine_grid_shape': list(grid.reshape(1, -1, 2).shape),
        'affine_grid_dtype': str(grid.dtype),
        'linear_index_shape': list(linear.shape),
        'linear_index_dtype': str(linear.dtype),
        'linear_index_runtime_dtype': 'torch.int32',
        'linear_index_onnx_gather_dtype': 'torch.int64',
        'source_fixture': str(args.fixture),
        'source_grid_is_official_pytorch_capture': True,
    }
    with (args.output / 'alignment_debug_metadata.json').open('w') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
