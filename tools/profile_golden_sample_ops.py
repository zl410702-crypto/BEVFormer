#!/usr/bin/env python
"""Profile operators executed by the real BEVFormer Golden Sample forward."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               DEFAULT_TOKEN, prepare_golden_sample,
                               result_arrays, run_model_inference)


TIME_FIELDS = ('self_cpu_time_total', 'cpu_time_total',
               'self_cuda_time_total', 'cuda_time_total')
RISK_TERMS = ('mmcv', 'deform', 'grid_sampler', 'grid_sample', 'scatter',
              'gather', 'index', 'masked', 'sampling', 'attention')


class ModelInferenceFailure(RuntimeError):
    """Mark an exception raised by the model rather than the profiler."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-root', default='golden_samples')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--row-limit', type=int, default=200)
    parser.add_argument('--with-stack', action='store_true')
    parser.add_argument('--event-limit', type=int, default=100000,
                        help='Maximum detailed events to save (0 means unlimited)')
    return parser.parse_args()


def optional_number(event, name):
    value = getattr(event, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_name(event):
    return str(getattr(event, 'key', getattr(event, 'name', 'unknown')))


def json_value(value):
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def aggregate_operators(key_averages):
    operators = []
    for event in key_averages:
        item = {
            'name': event_name(event),
            'calls': int(getattr(event, 'count', 0)),
        }
        for field in TIME_FIELDS:
            item[field] = optional_number(event, field)
        operators.append(item)
    return sorted(operators, key=lambda item: item['name'])


def detailed_events(profiler, limit):
    getter = getattr(profiler, 'events', None)
    events = getter() if callable(getter) else getattr(
        profiler, 'function_events', [])
    details = []
    for event in events:
        name = event_name(event)
        if not (name.startswith(('aten::', 'mmcv')) or
                any(term in name.lower() for term in RISK_TERMS)):
            continue
        shapes = getattr(event, 'input_shapes', None)
        details.append({
            'name': name,
            'input_shapes': json_value(shapes) if shapes is not None else [],
            'cpu_time': optional_number(event, 'cpu_time_total'),
            'cuda_time': optional_number(event, 'cuda_time_total'),
        })
        if limit > 0 and len(details) >= limit:
            break
    return details


def potential_risk_operators(operators):
    risks = []
    for operator in operators:
        name = operator['name']
        lowered = name.lower()
        namespace = name.split('::', 1)[0] if '::' in name else None
        is_custom_namespace = namespace not in (None, 'aten', 'cuda')
        if is_custom_namespace or any(term in lowered for term in RISK_TERMS):
            risks.append(name)
    return sorted(set(risks))


def write_reports(output_dir, token, operators, events):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'operator_profile.json'
    csv_path = output_dir / 'operator_profile.csv'
    events_path = output_dir / 'operator_events.json'
    payload = {
        'sample_token': token,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'operators': operators,
    }
    with json_path.open('w') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('name', 'calls') + TIME_FIELDS)
        writer.writeheader()
        writer.writerows(operators)
    with events_path.open('w') as stream:
        json.dump({'sample_token': token, 'events': events}, stream, indent=2)
        stream.write('\n')
    return json_path, csv_path, events_path


def check_golden_reference(reference_path, boxes, scores, labels):
    result = {'pass': True, 'boxes_3d_max_abs_error': None,
              'scores_3d_max_abs_error': None}
    with np.load(str(reference_path)) as reference:
        for name, actual in (('boxes_3d', boxes), ('scores_3d', scores),
                             ('labels_3d', labels)):
            expected = reference[name]
            shape_matches = actual.shape == expected.shape
            result[name + '_shape_matches'] = shape_matches
            if not shape_matches:
                result['pass'] = False
                continue
            if name == 'labels_3d':
                labels_match = np.array_equal(actual, expected)
                result['labels_3d_exact_match'] = labels_match
                result['pass'] = result['pass'] and labels_match
            else:
                error = (float(np.max(np.abs(actual - expected)))
                         if actual.size else 0.0)
                result[name + '_max_abs_error'] = error
                result['pass'] = result['pass'] and np.array_equal(
                    actual, expected)
    return result


def profile_forward(model, batch, with_stack):
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    inference_started = False
    inference_completed = False
    inference_error = None
    try:
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    with_stack=with_stack) as profiler:
            inference_started = True
            try:
                result = run_model_inference(model, batch)
                inference_completed = True
            except Exception as error:
                inference_error = error
                raise
    except Exception as error:
        if inference_error is not None:
            raise ModelInferenceFailure(
                'Model inference failure: {}'.format(inference_error)
            ) from inference_error
        phase = ('after inference completed' if inference_completed else
                 'before inference started' if not inference_started else
                 'while profiler was active')
        raise RuntimeError(
            'CUDA profiler activity unavailable or profiler failure {}: {}'
            .format(phase, error)) from error
    cuda_times = [max(value for value in (
        optional_number(event, 'cuda_time_total'),
        optional_number(event, 'self_cuda_time_total'), 0.0)
                      if value is not None)
                  for event in profiler.key_averages()]
    if not any(value is not None and value > 0 for value in cuda_times):
        raise RuntimeError(
            'CUDA profiler activity unavailable: inference completed, but no '
            'CUDA timing events were captured')
    return result, profiler


def main():
    args = parse_args()
    context = prepare_golden_sample(
        args.token, args.config, args.checkpoint, args.seed)
    result, profiler = profile_forward(
        context['model'], context['batch'], args.with_stack)
    key_averages = profiler.key_averages()
    try:
        table = key_averages.table(sort_by='self_cuda_time_total',
                                   row_limit=args.row_limit)
    except (KeyError, AttributeError, RuntimeError):
        table = key_averages.table(sort_by='cuda_time_total',
                                   row_limit=args.row_limit)
    print(table)

    operators = aggregate_operators(key_averages)
    print('\nUnique operators (alphabetical)')
    for operator in operators:
        print('{:<60} {}'.format(operator['name'], operator['calls']))

    events = detailed_events(profiler, args.event_limit)
    output_dir = Path(args.output_root) / args.token
    paths = write_reports(output_dir, args.token, operators, events)

    boxes, scores, labels = result_arrays(result)
    reference_path = output_dir / 'detections.npz'
    if not reference_path.is_file():
        raise FileNotFoundError('Golden reference not found: {}'.format(
            reference_path))
    check = check_golden_reference(reference_path, boxes, scores, labels)
    print('\nGolden comparison: boxes shape={}, scores shape={}, labels shape={}'
          .format(check['boxes_3d_shape_matches'],
                  check['scores_3d_shape_matches'],
                  check['labels_3d_shape_matches']))
    print('boxes_3d max abs error: {}'.format(
        check['boxes_3d_max_abs_error']))
    print('scores_3d max abs error: {}'.format(
        check['scores_3d_max_abs_error']))
    print('labels_3d exact match: {}'.format(
        check.get('labels_3d_exact_match', False)))
    print('Golden reference check: {}'.format('PASS' if check['pass'] else 'FAIL'))

    risks = potential_risk_operators(operators)
    print('\nPotential deployment-risk operators')
    for name in risks:
        print(name)
    print('\nUnique operator count: {}'.format(len(operators)))
    print('Reports: {}'.format(', '.join(str(path) for path in paths)))
    if not check['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
