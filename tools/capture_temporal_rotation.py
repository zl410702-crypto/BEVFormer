#!/usr/bin/env python
"""Capture the real second-frame torchvision NEAREST grid-sampler fixture."""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional_tensor as functional_tensor
from mmcv.parallel import collate

from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               DEFAULT_TOKEN, find_sample_index,
                               prepare_golden_sample, run_model_inference)
from validate_grid_sample_rewrite import error_metrics
from deployment.grid_sample_rewrite import bev_grid_sample_nearest


DEFAULT_SECOND_TOKEN = '3950bd41f74548429c0f7700ff3d8269'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-token', default=DEFAULT_TOKEN)
    parser.add_argument('--second-token', default=DEFAULT_SECOND_TOKEN)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-root', default='golden_samples')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


class GridSampleCapture:
    def __init__(self):
        self.fixture = None
        self.operator_names = []

    def __call__(self, original, input_tensor, grid, **kwargs):
        if kwargs != {'mode': 'nearest', 'padding_mode': 'zeros',
                      'align_corners': False}:
            raise ValueError('Unexpected torchvision grid_sample args: {}'.format(
                kwargs))
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities,
                                    record_shapes=True) as profiler:
            output = original(input_tensor, grid, **kwargs)
        self.operator_names = sorted(set(
            str(getattr(event, 'key', getattr(event, 'name', 'unknown')))
            for event in profiler.key_averages()))
        self.fixture = {
            'prev_bev': input_tensor.detach().cpu(),
            'grid': grid.detach().cpu(),
            'reference_output': output.detach().cpu(),
            'mode': kwargs['mode'],
            'padding_mode': kwargs['padding_mode'],
            'align_corners': kwargs['align_corners'],
        }
        return output


def main():
    args = parse_args()
    context = prepare_golden_sample(
        args.first_token, args.config, args.checkpoint, args.seed)
    dataset = context['dataset']
    first_index = context['index']
    second_index = find_sample_index(dataset.data_infos, args.second_token)
    first_info = dataset.data_infos[first_index]
    second_info = dataset.data_infos[second_index]
    if first_info['scene_token'] != second_info['scene_token']:
        raise ValueError('First and second tokens are not in the same scene')
    if second_info['timestamp'] <= first_info['timestamp']:
        raise ValueError('Second token must be temporally after first token')

    run_model_inference(context['model'], context['batch'])
    if context['model'].module.prev_frame_info['prev_bev'] is None:
        raise RuntimeError('First inference did not produce prev_bev')

    second_data = dataset[second_index]
    second_batch = collate([second_data], samples_per_gpu=1)
    capture = GridSampleCapture()
    original = functional_tensor.grid_sample

    def captured(input_tensor, grid, **kwargs):
        return capture(original, input_tensor, grid, **kwargs)

    functional_tensor.grid_sample = captured
    try:
        run_model_inference(context['model'], second_batch)
    finally:
        functional_tensor.grid_sample = original
    if capture.fixture is None:
        raise RuntimeError('Second-frame inference did not execute grid_sample')

    capture.fixture.update({
        'sample_token': args.second_token,
        'timestamp': second_info['timestamp'],
        'first_sample_token': args.first_token,
        'scene_token': second_info['scene_token'],
        'operator_names': capture.operator_names,
    })
    rewrite = bev_grid_sample_nearest(
        capture.fixture['prev_bev'], capture.fixture['grid'])
    metrics = error_metrics(capture.fixture['reference_output'], rewrite)
    capture.fixture['rewrite_metrics'] = metrics
    output_dir = Path(args.output_root) / args.second_token
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / 'temporal_grid_sample_reference.pt'
    torch.save(capture.fixture, str(fixture_path))
    metadata_path = output_dir / 'temporal_grid_sample_reference.json'
    metadata = {
        'sample_token': args.second_token,
        'timestamp': second_info['timestamp'],
        'scene_token': second_info['scene_token'],
        'prev_bev_shape': list(capture.fixture['prev_bev'].shape),
        'grid_shape': list(capture.fixture['grid'].shape),
        'output_shape': list(capture.fixture['reference_output'].shape),
        'dtype': str(capture.fixture['prev_bev'].dtype),
        'mode': capture.fixture['mode'],
        'padding_mode': capture.fixture['padding_mode'],
        'align_corners': capture.fixture['align_corners'],
        'operator_names': capture.operator_names,
        'rewrite_metrics': metrics,
    }
    with metadata_path.open('w') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print('Captured fixture: {}'.format(fixture_path))
    print('Runtime operators: {}'.format(', '.join(capture.operator_names)))
    print('Max abs error: {}'.format(metrics['max_abs_error']))
    print('Mean abs error: {}'.format(metrics['mean_abs_error']))


if __name__ == '__main__':
    main()
