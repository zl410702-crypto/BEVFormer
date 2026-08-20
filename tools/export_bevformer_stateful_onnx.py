#!/usr/bin/env python
"""Export and audit a stateful single-frame BEVFormer tiny ONNX model."""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import onnx
import torch

from deployment.bevformer_onnx_wrapper import (
    BEVFormerONNXWrapper, CUSTOM_DOMAIN, CUSTOM_OP_TYPE,
    deployment_export_overrides)
from deployment.bevformer_stateful_onnx_wrapper import (
    BEVFormerStatefulONNXWrapper)
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               pipeline_image_tensor, prepare_golden_sample,
                               validate_pipeline_item)
from tensor_compare import compare_arrays


DEFAULT_FRAME1_TOKEN = '3950bd41f74548429c0f7700ff3d8269'
DEFAULT_SEQUENCE = Path(
    'golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3')
DEFAULT_OUTPUT = Path('artifacts/bevformer_tiny_stateful_opset13.onnx')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_FRAME1_TOKEN)
    parser.add_argument('--sequence-root', type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--allow-validation-failure', action='store_true',
        help=('Export a structural debug candidate after Golden validation '
              'fails; disabled by default and not deployment-ready.'))
    return parser.parse_args()


def metrics(expected, actual):
    result = compare_arrays(expected, actual)
    if result is None:
        raise ValueError('Shape mismatch: {} != {}'.format(
            expected.shape, actual.shape))
    return result


def prepare_runtime_meta(meta, current_absolute, previous_absolute):
    runtime = copy.deepcopy(meta)
    # Match BEVFormer.forward_test: subtract in the metadata's float64 domain,
    # then cast when creating the model input tensor.
    can_bus = np.asarray(current_absolute, dtype=np.float64).copy()
    previous = np.asarray(previous_absolute, dtype=np.float64)
    can_bus[:3] -= previous[:3]
    can_bus[-1] -= previous[-1]
    runtime['can_bus'] = can_bus
    return runtime


def first_frame_runtime_meta(meta, absolute):
    runtime = copy.deepcopy(meta)
    can_bus = np.asarray(absolute, dtype=np.float64).copy()
    can_bus[:3] = 0.0
    can_bus[-1] = 0.0
    runtime['can_bus'] = can_bus
    return runtime


def compute_shift(can_bus, model):
    """Host implementation of the exact reference shift equations."""
    head = model.pts_bbox_head
    transformer = head.transformer
    delta_x = np.asarray([can_bus[0]], dtype=np.float64)
    delta_y = np.asarray([can_bus[1]], dtype=np.float64)
    ego_angle = np.asarray([can_bus[-2] / np.pi * 180.0], dtype=np.float64)
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180.0
    bev_angle = ego_angle - translation_angle
    grid_y = head.real_h / head.bev_h
    grid_x = head.real_w / head.bev_w
    shift_y = translation_length * np.cos(
        bev_angle / 180.0 * np.pi) / grid_y / head.bev_h
    shift_x = translation_length * np.sin(
        bev_angle / 180.0 * np.pi) / grid_x / head.bev_w
    shift = np.stack((shift_x * transformer.use_shift,
                      shift_y * transformer.use_shift), axis=-1)
    return torch.from_numpy(shift.astype(np.float32))


def tensor_inputs(meta, model):
    can_bus = torch.from_numpy(
        np.asarray(meta['can_bus'], dtype=np.float32)).unsqueeze(0).cuda()
    shift = compute_shift(meta['can_bus'], model).cuda()
    lidar2img = torch.from_numpy(
        np.asarray(meta['lidar2img'], dtype=np.float32)).unsqueeze(0).cuda()
    return can_bus, shift, lidar2img


def golden_outputs(root, frame):
    base = root / frame
    return tuple(np.load(str(path), allow_pickle=False) for path in (
        base / 'detection/cls_scores.npy',
        base / 'detection/bbox_preds.npy',
        base / 'bev/final_bev.npy'))


def compare_triplet(expected, actual):
    names = ('all_cls_scores', 'all_bbox_preds', 'bev_embed')
    report = {}
    for name, reference, output in zip(names, expected, actual):
        report[name] = metrics(reference, output.detach().cpu().numpy())
        report[name]['shape'] = list(reference.shape)
    return report


def assert_close_report(report, atol=1e-5, allow_failure=False):
    failed = {name: values for name, values in report.items()
              if values['MaxAE'] > atol}
    if failed:
        if not allow_failure:
            raise RuntimeError('Wrapper validation failed: {}'.format(failed))
        print('WARNING: exporting structural candidate after validation '
              'failure: {}'.format(failed))


def value_description(value):
    tensor = value.type.tensor_type
    dims = []
    for dimension in tensor.shape.dim:
        if dimension.HasField('dim_value'):
            dims.append(dimension.dim_value)
        elif dimension.HasField('dim_param'):
            dims.append(dimension.dim_param)
        else:
            dims.append('?')
    return {'name': value.name, 'shape': dims,
            'dtype': onnx.TensorProto.DataType.Name(tensor.elem_type)}


def descendants(graph, source):
    consumers = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
    seen_values = {source}
    seen_nodes = set()
    queue = deque([source])
    while queue:
        value = queue.popleft()
        for index in consumers.get(value, []):
            if index in seen_nodes:
                continue
            seen_nodes.add(index)
            for output in graph.node[index].output:
                if output not in seen_values:
                    seen_values.add(output)
                    queue.append(output)
    return seen_values, seen_nodes, consumers.get(source, [])


def audit_onnx(path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    counts = Counter((node.domain or 'ai.onnx', node.op_type)
                     for node in model.graph.node)
    custom_nodes = [node for node in model.graph.node
                    if node.domain == CUSTOM_DOMAIN and
                    node.op_type == CUSTOM_OP_TYPE]
    if len(custom_nodes) != 12:
        raise RuntimeError('Expected 12 custom nodes, found {}'.format(
            len(custom_nodes)))
    invalid_custom = [node.name for node in custom_nodes
                      if len(node.input) != 5 or
                      not any(attribute.name == 'im2col_step' and
                              attribute.i == 64 for attribute in node.attribute)]
    if invalid_custom:
        raise RuntimeError('Invalid custom node schema: {}'.format(
            invalid_custom))
    expected_inputs = {'images', 'prev_bev', 'can_bus', 'shift',
                       'lidar2img', 'use_prev_bev'}
    inputs = [value_description(value) for value in model.graph.input]
    outputs = [value_description(value) for value in model.graph.output]
    if {item['name'] for item in inputs} != expected_inputs:
        raise RuntimeError('Unexpected graph inputs: {}'.format(inputs))
    dependencies = {}
    output_names = {value.name for value in model.graph.output}
    for name in expected_inputs:
        values, nodes, direct = descendants(model.graph, name)
        dependencies[name] = {
            'direct_consumer_indices': direct,
            'node_count': len(nodes),
            'reaches_graph_output': bool(values & output_names),
            'custom_node_count_downstream': sum(
                1 for index in nodes
                if model.graph.node[index].domain == CUSTOM_DOMAIN),
        }
        if not dependencies[name]['reaches_graph_output']:
            raise RuntimeError('Input is not connected to output: {}'.format(
                name))
    alignment_ops = {name: counts[('ai.onnx', name)] for name in (
        'Sin', 'Cos', 'MatMul', 'Round', 'Clip', 'GatherElements', 'Gather',
        'Reshape', 'Transpose') if counts[('ai.onnx', name)]}
    if not alignment_ops.get('Sin') or not alignment_ops.get('Cos'):
        raise RuntimeError('Dynamic rotation trigonometry is absent')
    shape_inference = {'status': 'PASS'}
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        shape_inference['value_info_count'] = len(inferred.graph.value_info)
    except Exception as error:
        shape_inference = {'status': 'PARTIAL/FAIL',
                           'reason': '{}: {}'.format(type(error).__name__, error)}
    return {
        'file': str(path), 'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'ir_version': model.ir_version,
        'producer': '{} {}'.format(model.producer_name, model.producer_version),
        'opsets': [{'domain': item.domain or 'ai.onnx',
                    'version': item.version} for item in model.opset_import],
        'node_count': len(model.graph.node), 'inputs': inputs,
        'outputs': outputs,
        'custom_ops': {'domain': CUSTOM_DOMAIN, 'op_type': CUSTOM_OP_TYPE,
                       'count': len(custom_nodes), 'input_count': 5,
                       'im2col_step': 64},
        'input_dependencies': dependencies,
        'alignment_standard_ops': alignment_ops,
        'grid_sample_node_count': sum(
            count for (domain, op), count in counts.items()
            if 'gridsample' in op.lower() or 'grid_sampler' in op.lower()),
        'shape_inference': shape_inference,
        'onnx_checker': 'PASS',
    }


def print_report(title, report):
    print('\n{}'.format(title))
    print('=' * 64)
    for name in ('bev_embed', 'all_cls_scores', 'all_bbox_preds'):
        values = report[name]
        print('{} shape={} MAE={} MaxAE={} RMSE={} Cosine={}'.format(
            name, values['shape'], values['MAE'], values['MaxAE'],
            values['RMSE'], values['Cosine Similarity']))


def main():
    args = parse_args()
    sequence = json.loads((args.sequence_root / 'sequence.json').read_text())
    frames = sequence['frames']
    if len(frames) < 2 or frames[1]['sample_token'] != args.token:
        raise ValueError('Sequence Frame 1 does not match --token')
    context = prepare_golden_sample(
        args.token, args.config, args.checkpoint, args.seed)
    model = context['model'].module
    frame1_meta = prepare_runtime_meta(
        context['metas'], frames[1]['can_bus_absolute'],
        frames[0]['can_bus_absolute'])
    frame1_images = context['image_tensor'].unsqueeze(0).cuda()
    frame1_prev = torch.from_numpy(np.load(
        str(args.sequence_root / 'frame_000/bev/final_bev.npy'),
        allow_pickle=False)).cuda()
    can_bus1, shift1, lidar2img1 = tensor_inputs(frame1_meta, model)
    gate1 = torch.ones(1, dtype=torch.float32, device='cuda')
    wrapper1 = BEVFormerStatefulONNXWrapper(model, frame1_meta).cuda().eval()
    with torch.no_grad():
        outputs1 = wrapper1(frame1_images, frame1_prev, can_bus1, shift1,
                            lidar2img1, gate1)
    frame1_report = compare_triplet(
        golden_outputs(args.sequence_root, 'frame_001'), outputs1)
    print_report('PyTorch Stateful Wrapper vs Golden Frame 1', frame1_report)
    assert_close_report(
        frame1_report, allow_failure=args.allow_validation_failure)

    zero_prev = torch.zeros_like(frame1_prev)
    with torch.no_grad():
        zero_outputs = wrapper1(frame1_images, zero_prev, can_bus1, shift1,
                                lidar2img1, gate1)
    sensitivity = compare_triplet(
        tuple(output.detach().cpu().numpy() for output in outputs1),
        zero_outputs)
    if all(values['MaxAE'] == 0.0 for values in sensitivity.values()):
        raise RuntimeError('prev_bev sensitivity check produced no difference')
    print_report('Frame 1 real prev_bev vs zero prev_bev sensitivity', sensitivity)

    frame0_info = context['dataset'].data_infos[context['index'] - 1]
    if frame0_info['token'] != frames[0]['sample_token']:
        raise ValueError('Frame 0 is not immediately before Frame 1')
    frame0_data = context['dataset'][context['index'] - 1]
    _, frame0_pipeline_meta = validate_pipeline_item(
        frame0_data, frame0_info, frame0_info['token'])
    frame0_meta = first_frame_runtime_meta(
        frame0_pipeline_meta, frames[0]['can_bus_absolute'])
    frame0_images = pipeline_image_tensor(frame0_data).unsqueeze(0).cuda()
    can_bus0, shift0, lidar2img0 = tensor_inputs(frame0_meta, model)
    gate0 = torch.zeros(1, dtype=torch.float32, device='cuda')
    official0 = BEVFormerONNXWrapper(
        model, frame0_meta, use_prev_bev=False).cuda().eval()
    wrapper0 = BEVFormerStatefulONNXWrapper(model, frame0_meta).cuda().eval()
    with torch.no_grad():
        official0_outputs = official0(frame0_images)
        stateful0_outputs = wrapper0(
            frame0_images, torch.zeros_like(frame1_prev), can_bus0, shift0,
            lidar2img0, gate0)
    frame0_report = compare_triplet(
        tuple(output.detach().cpu().numpy() for output in official0_outputs),
        stateful0_outputs)
    print_report('Frame 0 gate=0 vs official prev_bev=None', frame0_report)
    assert_close_report(
        frame0_report, allow_failure=args.allow_validation_failure)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), deployment_export_overrides():
        torch.onnx.export(
            wrapper1,
            (frame1_images, frame1_prev, can_bus1, shift1, lidar2img1, gate1),
            str(args.output), opset_version=13,
            input_names=['images', 'prev_bev', 'can_bus', 'shift',
                         'lidar2img', 'use_prev_bev'],
            output_names=['all_cls_scores', 'all_bbox_preds', 'bev_embed'],
            do_constant_folding=False, verbose=False)
    audit = audit_onnx(args.output)
    audit['pytorch_validation'] = {
        'frame1_wrapper_vs_golden': frame1_report,
        'frame1_real_vs_zero_prev_sensitivity': sensitivity,
        'frame0_gate0_vs_none': frame0_report,
    }
    audit['runtime_semantics'] = {
        'prev_bev': {'shape': [2500, 1, 256], 'layout': 'Nbev,B,C'},
        'bev_embed': {'shape': [2500, 1, 256], 'layout': 'Nbev,B,C'},
        'can_bus': ('18 float32 values after BEVFormer.forward_test delta '
                    'processing; index -1 rotation delta is degrees'),
        'shift': ('[shift_x, shift_y] computed by the reference host formula'),
        'lidar2img': 'per-frame camera projection matrices',
        'use_prev_bev': 'float32 scalar gate: exactly 0 or 1',
    }
    audit_path = args.output.with_name(args.output.stem + '_audit.json')
    with audit_path.open('w') as stream:
        json.dump(audit, stream, indent=2)
        stream.write('\n')
    print('\nStateful ONNX Audit')
    print('=' * 64)
    print(json.dumps(audit, indent=2))
    print('Audit JSON: {}'.format(audit_path))


if __name__ == '__main__':
    main()
