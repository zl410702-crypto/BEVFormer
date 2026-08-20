#!/usr/bin/env python
"""Run consecutive nuScenes samples through the official stateful test path."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from mmcv.parallel import collate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from golden_sequence_recorder import GoldenSequenceRecorder
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_TOKEN,
                               prepare_golden_sample, result_arrays,
                               run_model_inference, validate_pipeline_item)
from tensor_compare import compare_arrays
from tensor_dump import TensorDumper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--num-frames', type=int, default=3)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-root', default='golden_tensors')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def consecutive_indices(data_infos, token, count):
    if count < 1:
        raise ValueError('--num-frames must be positive')
    by_token = {info['token']: index for index, info in enumerate(data_infos)}
    if token not in by_token:
        raise ValueError('Sample token not found: {}'.format(token))
    indices = []
    current = token
    scene = data_infos[by_token[token]]['scene_token']
    previous_timestamp = None
    for frame in range(count):
        if not current or current not in by_token:
            raise ValueError('NuScenes next chain ended before frame {}'.format(frame))
        index = by_token[current]
        info = data_infos[index]
        if info['scene_token'] != scene:
            raise ValueError('NuScenes next chain crossed scene boundary')
        if previous_timestamp is not None and info['timestamp'] <= previous_timestamp:
            raise ValueError('Timestamps are not strictly increasing')
        if indices and info.get('prev') != data_infos[indices[-1]]['token']:
            raise ValueError('Broken reciprocal prev/next chain at {}'.format(current))
        indices.append(index)
        previous_timestamp = info['timestamp']
        current = info.get('next')
    return indices


def json_values(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [float(item) for item in value]


def print_sequence(infos):
    print('Golden Sequence')
    print('=' * 48)
    print('Scene Token: {}'.format(infos[0]['scene_token']))
    for frame, info in enumerate(infos):
        print('\nframe_{:03d}'.format(frame))
        print('Frame Index : {}'.format(info.get('frame_idx', frame)))
        print('token       : {}'.format(info['token']))
        print('timestamp   : {}'.format(info['timestamp']))
        print('prev / next : {} / {}'.format(info.get('prev'), info.get('next')))


def main():
    args = parse_args()
    context = prepare_golden_sample(args.token, args.config, args.checkpoint,
                                    args.seed)
    dataset = context['dataset']
    indices = consecutive_indices(dataset.data_infos, args.token, args.num_frames)
    infos = [dataset.data_infos[index] for index in indices]
    print_sequence(infos)

    sequence_id = '{}_{}'.format(infos[0]['scene_token'], infos[0]['token'][:8])
    output_dir = Path(args.output_root) / sequence_id
    dumper = TensorDumper(output_dir)
    model = context['model']
    recorder = GoldenSequenceRecorder(model.module, dumper).install()
    sequence_frames = []
    previous_absolute = None
    try:
        for frame, (index, info) in enumerate(zip(indices, infos)):
            recorder.frame = 'frame_{:03d}'.format(frame)
            data = dataset[index]
            _, meta = validate_pipeline_item(data, info, info['token'])
            absolute_can_bus = copy.deepcopy(meta['can_bus'])
            had_prev_bev = model.module.prev_frame_info['prev_bev'] is not None
            result = run_model_inference(
                model, collate([data], samples_per_gpu=1))
            boxes, scores, labels = result_arrays(result)
            dumper.dump('detection.decoded_boxes', boxes,
                        recorder.path('detection/decoded_boxes.npy'))
            dumper.dump('detection.decoded_scores', scores,
                        recorder.path('detection/decoded_scores.npy'))
            dumper.dump('detection.decoded_labels', labels,
                        recorder.path('detection/decoded_labels.npy'))
            delta_translation = ([0.0, 0.0, 0.0] if previous_absolute is None
                                 else (absolute_can_bus[:3] - previous_absolute[:3]).tolist())
            delta_yaw = (0.0 if previous_absolute is None else
                         float(absolute_can_bus[-1] - previous_absolute[-1]))
            sequence_frames.append({
                'frame': frame,
                'frame_index': int(info.get('frame_idx', frame)),
                'sample_token': info['token'],
                'timestamp': int(info['timestamp']),
                'prev_sample': info.get('prev', ''),
                'next_sample': info.get('next', ''),
                'can_bus_absolute': json_values(absolute_can_bus),
                'ego2global_translation': json_values(info['ego2global_translation']),
                'ego2global_rotation': json_values(info['ego2global_rotation']),
                'translation_delta': delta_translation,
                'yaw_delta_degrees': delta_yaw,
                'prev_bev_present_before_forward': had_prev_bev,
                'detection_count': int(len(scores)),
            })
            previous_absolute = absolute_can_bus
    finally:
        recorder.close()

    payload = {
        'scene_token': infos[0]['scene_token'],
        'sequence_id': sequence_id,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'frames': sequence_frames,
        'tensors': dumper.records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'sequence.json').open('w') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    dumper.write_manifest()

    print('\n' + '=' * 48)
    print('Temporal Sequence Summary')
    print('=' * 48)
    for item in sequence_frames:
        final_record = next(record for record in dumper.records
                            if record['name'] == 'bev.final_bev' and
                            'frame_{:03d}/'.format(item['frame']) in record['path'])
        print('\nFrame {} token={} timestamp={} scene={}'.format(
            item['frame'], item['sample_token'], item['timestamp'],
            infos[0]['scene_token']))
        print('prev_bev={} translation_delta={} rotation_delta={} final_bev={}'.format(
            item['prev_bev_present_before_forward'], item['translation_delta'],
            item['yaw_delta_degrees'], final_record['shape']))
    for frame in range(1, len(sequence_frames)):
        previous = np.load(str(output_dir / 'frame_{:03d}/bev/final_bev.npy'.format(frame - 1)))
        stored = np.load(str(output_dir / 'frame_{:03d}/temporal/prev_bev_stored.npy'.format(frame)))
        metrics = compare_arrays(previous, stored)
        print('\nFrame{} final_bev -> Frame{} prev_bev_stored: {}'.format(
            frame - 1, frame, metrics))

    print('\nFrame | Module | Tensor | Shape | Dtype | Min | Max | Mean | Path')
    for record in dumper.records:
        parts = Path(record['path']).parts
        marker = next((part for part in parts if part.startswith('frame_')), '-')
        module = Path(record['path']).parent.name
        print('{} | {} | {} | {} | {} | {} | {} | {} | {}'.format(
            marker, module, record['name'], record['shape'], record['dtype'],
            record['min'], record['max'], record['mean'], record['path']))
    print('\nWrote golden sequence to {}'.format(output_dir))


if __name__ == '__main__':
    main()
