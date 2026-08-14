#!/usr/bin/env python
"""Validate the deployment NEAREST sampler using a captured temporal fixture."""

import argparse
import importlib.util
from pathlib import Path

import torch

from deployment.grid_sample_rewrite import (
    GridSampleNearestRewriteWrapper, bev_grid_sample_nearest)


DEFAULT_SECOND_TOKEN = '3950bd41f74548429c0f7700ff3d8269'
FORBIDDEN_NODE_TERMS = ('gridsample', 'grid_sampler')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=str(
        Path('golden_samples') / DEFAULT_SECOND_TOKEN /
        'temporal_grid_sample_reference.pt'))
    parser.add_argument('--onnx-output', default=
                        'artifacts/grid_sample_nearest_rewrite_opset13.onnx')
    parser.add_argument('--atol', type=float, default=1e-5)
    return parser.parse_args()


def error_metrics(reference, rewrite):
    absolute = (reference - rewrite).abs()
    denominator = reference.abs().clamp_min(1e-12)
    relative = absolute / denominator
    return {
        'shape_equal': reference.shape == rewrite.shape,
        'dtype_equal': reference.dtype == rewrite.dtype,
        'max_abs_error': float(absolute.max()) if absolute.numel() else 0.0,
        'mean_abs_error': float(absolute.mean()) if absolute.numel() else 0.0,
        'max_rel_error': float(relative.max()) if relative.numel() else 0.0,
    }


def validate_fixture(path, atol):
    fixture = torch.load(str(path), map_location='cpu')
    required = {'prev_bev', 'grid', 'reference_output', 'sample_token',
                'timestamp'}
    missing = required - set(fixture)
    if missing:
        raise ValueError('Fixture is missing: {}'.format(sorted(missing)))
    rewrite = bev_grid_sample_nearest(fixture['prev_bev'], fixture['grid'])
    metrics = error_metrics(fixture['reference_output'], rewrite)
    passed = (metrics['shape_equal'] and metrics['dtype_equal'] and
              metrics['max_abs_error'] <= atol)
    return fixture, metrics, passed


def export_and_check(fixture, output_path):
    if importlib.util.find_spec('onnx') is None:
        return {'status': 'NOT RUN', 'remaining': 'NOT CHECKED', 'ops': []}
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = GridSampleNearestRewriteWrapper().eval()
    torch.onnx.export(wrapper, (fixture['prev_bev'], fixture['grid']),
                      str(output_path), opset_version=13,
                      input_names=['input', 'grid'], output_names=['output'])
    model = onnx.load(str(output_path))
    ops = sorted(set(node.op_type for node in model.graph.node))
    remaining = any(any(term in node.op_type.lower() for term in
                        FORBIDDEN_NODE_TERMS) for node in model.graph.node)
    return {'status': 'FAIL' if remaining else 'PASS',
            'remaining': 'YES' if remaining else 'NO', 'ops': ops}


def main():
    args = parse_args()
    fixture_path = Path(args.fixture)
    if not fixture_path.is_file():
        raise FileNotFoundError(
            'Real temporal fixture not found: {}. Run '
            'tools/capture_temporal_rotation.py on a CUDA-enabled host.'
            .format(fixture_path))
    fixture, metrics, passed = validate_fixture(fixture_path, args.atol)
    print('Reference shape: {}'.format(tuple(fixture['reference_output'].shape)))
    print('Rewrite shape: {}'.format(tuple(
        bev_grid_sample_nearest(fixture['prev_bev'], fixture['grid']).shape)))
    print('Max abs error: {}'.format(metrics['max_abs_error']))
    print('Mean abs error: {}'.format(metrics['mean_abs_error']))
    print('Max relative error: {}'.format(metrics['max_rel_error']))
    print('Numerical validation: {}'.format('PASS' if passed else 'FAIL'))
    export = export_and_check(fixture, args.onnx_output)
    print('ONNX opset13 export: {}'.format(export['status']))
    print('GridSample remaining in graph: {}'.format(export['remaining']))
    if export['ops']:
        print('ONNX op_type set: {}'.format(', '.join(export['ops'])))
    if not passed or export['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
