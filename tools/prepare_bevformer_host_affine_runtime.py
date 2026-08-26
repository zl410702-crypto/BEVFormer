#!/usr/bin/env python
"""Prepare the production affine_grid runtime binding from Frame motion state."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from deployment.host_affine_contract import prepare_host_affine_grid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference-grid', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise RuntimeError('refusing to overwrite {}'.format(args.output))
    previous = np.fromfile(str(args.runtime_frame / 'prev_bev.bin'), np.float32)
    if previous.size != 2500 * 256:
        raise ValueError('prev_bev byte contract changed')
    previous = torch.from_numpy(previous.reshape(2500, 1, 256)).to(args.device)
    history = previous[:, 0].reshape(50, 50, 256).permute(2, 0, 1).contiguous()
    can_bus = np.fromfile(str(args.runtime_frame / 'can_bus.bin'), np.float32)
    if can_bus.shape != (18,):
        raise ValueError('can_bus contract changed')
    angle = torch.tensor([can_bus[-1]], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        grid = prepare_host_affine_grid(history, angle, [100, 100])
    grid_cpu = grid.detach().cpu().contiguous().numpy()

    metrics = None
    if args.reference_grid:
        reference = np.load(str(args.reference_grid), allow_pickle=False)
        if reference.shape != grid_cpu.shape or reference.dtype != np.float32:
            raise ValueError('reference affine-grid contract changed')
        difference = np.abs(reference.astype(np.float64) -
                            grid_cpu.astype(np.float64))
        metrics = {
            'max_abs_error': float(difference.max()),
            'mean_abs_error': float(difference.mean()),
            'mismatch_count': int(np.count_nonzero(reference != grid_cpu)),
        }
        if metrics['max_abs_error'] != 0.0:
            raise RuntimeError('host affine grid is not bitwise Golden: {}'.format(
                metrics))

    shutil.copytree(str(args.runtime_frame), str(args.output))
    grid_cpu.tofile(str(args.output / 'affine_grid.bin'))
    metadata = {
        'name': 'affine_grid',
        'shape': [1, 2500, 2],
        'dtype': 'float32',
        'layout': 'B,HW,XY',
        'contiguous': True,
        'angle_degrees': float(can_bus[-1]),
        'device_used_for_preparation': args.device,
        'golden_validation': metrics,
    }
    (args.output / 'affine_grid_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
