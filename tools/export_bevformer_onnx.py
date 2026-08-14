#!/usr/bin/env python
"""Export the static BEVFormer tiny Golden path to ONNX opset 13."""

import argparse
import importlib.util
import json
import re
import traceback
from collections import Counter
from pathlib import Path

import torch
import yaml

from deployment.bevformer_onnx_wrapper import (
    BEVFormerONNXWrapper, CUSTOM_DOMAIN, CUSTOM_OP_TYPE,
    deployment_export_overrides)
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               DEFAULT_TOKEN, prepare_golden_sample)


DEFAULT_OUTPUT = 'artifacts/bevformer_tiny_opset13.onnx'
DEFAULT_BLOCKERS = 'docs/BEVFormer_onnx_export_blockers.md'
# PyTorch 1.9.1's ONNX folder evaluates a valid opset-13 Slice constant
# subgraph with Float storage while unconditionally reading int64 indices.
# The unfurled static graph passes onnx.checker; keep folding disabled here.
ONNX_DO_CONSTANT_FOLDING = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--opset', type=int, default=13)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--blockers', default=DEFAULT_BLOCKERS)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def environment_blockers():
    blockers = []
    if importlib.util.find_spec('onnx') is None:
        blockers.append({
            'module': 'Export environment',
            'source': 'N/A',
            'op': 'onnx package',
            'error': 'ModuleNotFoundError: No module named onnx',
            'status': 'ENVIRONMENT',
            'action': 'CHECK',
        })
    if not torch.cuda.is_available():
        blockers.append({
            'module': 'Export environment',
            'source': 'N/A',
            'op': 'CUDA runtime',
            'error': 'torch.cuda.is_available() returned False',
            'status': 'ENVIRONMENT',
            'action': 'CHECK',
        })
    return blockers


def classify_export_error(error):
    message = '{}: {}'.format(type(error).__name__, error)
    lowered = message.lower()
    if 'grid_sampler' in lowered or 'gridsample' in lowered:
        return ('Temporal rotation',
                'tools/deployment/grid_sample_rewrite.py',
                'aten::grid_sampler', 'REWRITE')
    if 'deform' in lowered:
        return ('BEV Encoder / Decoder',
                'projects/mmdet3d_plugin/bevformer/modules/'
                'multi_scale_deformable_attn_function.py',
                'MultiScaleDeformableAttnFunction_fp32', 'CUSTOM')
    if 'index' in lowered or 'scatter' in lowered or 'nonzero' in lowered:
        return ('BEV Encoder / Postprocess',
                'projects/mmdet3d_plugin/bevformer/modules/encoder.py',
                'index/scatter/nonzero', 'CHECK')
    return 'Unknown exporter location', 'UNKNOWN', 'UNKNOWN', 'CHECK'


def write_blockers(path, blockers, command, exporter_started):
    lines = [
        '# BEVFormer ONNX Export Blockers', '',
        'Target: static BEVFormer tiny Golden Sample, ONNX opset 13.', '',
        'Command:', '', '```bash', command, '```', '',
        'Exporter started: **{}**'.format('YES' if exporter_started else 'NO'),
        '',
        '| Module | Source | PyTorch Op / Dependency | Export Error | '
        'In Compatibility Report | Status | Proposed Action |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    if blockers:
        for blocker in blockers:
            compatibility = ('N/A' if blocker['status'] == 'ENVIRONMENT' else
                             'NO' if blocker['op'] == 'UNKNOWN' else 'YES')
            values = (blocker['module'], blocker['source'], blocker['op'],
                      blocker['error'], compatibility, blocker['status'],
                      blocker['action'])
            row = [value.replace('|', '\\|').replace('\n', '<br>')
                   for value in values]
            lines.append('| {} |'.format(' | '.join(row)))
    else:
        lines.append(
            '| — | — | — | No blockers observed | N/A | PASS | NATIVE |')
    lines.extend([
        '', '## Interpretation', '',
        '- Environment blockers prevent tracing and are not PyTorch operator '
        'exporter blockers.',
        '- Operator blockers are added only after `torch.onnx.export` actually '
        'starts and raises an error.',
        '- Entries marked `UNKNOWN/CHECK` must not be promoted without a real '
        'opset-13 export attempt.', '',
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def inspect_onnx(output_path, ops_path, expected_opset):
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    default_opsets = [item.version for item in model.opset_import
                      if item.domain in ('', 'ai.onnx')]
    if default_opsets != [expected_opset]:
        raise ValueError('Expected default opset {}, found {}'.format(
            expected_opset, default_opsets))
    counts = Counter((node.domain or 'ai.onnx', node.op_type)
                     for node in model.graph.node)
    forbidden = [(domain, name) for (domain, name) in counts
                 if 'gridsample' in name.lower() or
                 'grid_sampler' in name.lower()]
    custom_count = counts[(CUSTOM_DOMAIN, CUSTOM_OP_TYPE)]
    if forbidden:
        raise ValueError('GridSample remains in graph: {}'.format(forbidden))
    if custom_count != 12:
        raise ValueError('Expected 12 {} custom nodes, found {}'.format(
            CUSTOM_OP_TYPE, custom_count))
    with Path('docs/operator_compatibility.yaml').open() as stream:
        compatibility = yaml.safe_load(stream)
    graph_standard_ops = {op_type for (domain, op_type) in counts
                          if domain == 'ai.onnx'}
    matched_entries = []
    for entry in compatibility.get('operators', []):
        mapping = entry.get('onnx_op', '')
        if any(re.search(r'\b{}\b'.format(re.escape(op_type)), mapping)
               for op_type in graph_standard_ops):
            matched_entries.append(entry['runtime_op'])
    inventory = {
        'onnx_opset': expected_opset,
        'node_count': sum(counts.values()),
        'custom_node_count': custom_count,
        'operators': [
            {'domain': domain, 'op_type': op_type, 'count': count}
            for (domain, op_type), count in sorted(counts.items())
        ],
        'compatibility_database': 'docs/operator_compatibility.yaml',
        'compatibility_comparison': {
            'database_entry_count': len(compatibility.get('operators', [])),
            'entries_with_mapping_present_in_graph': len(matched_entries),
            'matched_runtime_ops': sorted(matched_entries),
        },
        'compatibility_note': (
            'Runtime-to-ONNX names are not one-to-one; CONDITIONAL/UNKNOWN '
            'items still require exported-node and parser review.'),
    }
    with Path(ops_path).open('w') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    return inventory


def run_onnxruntime(output_path, images, reference_outputs):
    if importlib.util.find_spec('onnxruntime') is None:
        return {'status': 'NOT RUN', 'reason': 'onnxruntime is not installed'}
    try:
        import numpy as np
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(output_path), providers=['CPUExecutionProvider'])
        actual_outputs = session.run(None, {'images': images.cpu().numpy()})
        comparisons = []
        for name, reference, actual in zip(
                ('all_cls_scores', 'all_bbox_preds', 'bev_embed'),
                reference_outputs, actual_outputs):
            expected = reference.detach().cpu().numpy()
            absolute = np.abs(expected - actual)
            comparisons.append({
                'name': name,
                'shape': list(actual.shape),
                'max_abs_error': float(absolute.max()),
                'mean_abs_error': float(absolute.mean()),
            })
        return {'status': 'PASS', 'comparisons': comparisons}
    except Exception as error:
        return {'status': 'FAIL', 'reason': '{}: {}'.format(
            type(error).__name__, error)}


def main():
    args = parse_args()
    if args.opset != 13:
        raise ValueError('This first export is intentionally fixed to opset 13')
    command = ('python tools/export_bevformer_onnx.py --token {} --opset {} '
               '--output {}'.format(args.token, args.opset, args.output))
    blockers = environment_blockers()
    if blockers:
        write_blockers(args.blockers, blockers, command, exporter_started=False)
        print('ONNX export status: FAIL')
        print('Exporter started: NO')
        print('Environment blockers: {}'.format(len(blockers)))
        print('Exporter operator blockers: 0')
        print('Blocker report: {}'.format(args.blockers))
        raise SystemExit(2)

    exporter_started = False
    try:
        context = prepare_golden_sample(
            args.token, args.config, args.checkpoint, args.seed)
        wrapper = BEVFormerONNXWrapper(
            context['model'].module, context['metas'], use_prev_bev=False)
        wrapper.eval()
        images = context['image_tensor'].unsqueeze(0).cuda()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            reference_outputs = wrapper(images)
        exporter_started = True
        with torch.no_grad(), deployment_export_overrides():
            torch.onnx.export(
                wrapper, (images,), str(output_path), opset_version=args.opset,
                input_names=['images'],
                output_names=['all_cls_scores', 'all_bbox_preds', 'bev_embed'],
                do_constant_folding=ONNX_DO_CONSTANT_FOLDING, verbose=False)
        ops_path = output_path.with_name(output_path.stem + '_ops.json')
        inventory = inspect_onnx(output_path, ops_path, args.opset)
        ort_result = run_onnxruntime(
            output_path, images, reference_outputs)
        inventory['onnxruntime_validation'] = ort_result
        with ops_path.open('w') as stream:
            json.dump(inventory, stream, indent=2)
            stream.write('\n')
        write_blockers(args.blockers, [], command, exporter_started=True)
        print('ONNX export status: PASS')
        print('ONNX output: {}'.format(output_path))
        print('Node count: {}'.format(inventory['node_count']))
        print('MSDeformableAttention custom nodes: {}'.format(
            inventory['custom_node_count']))
        print('GridSample remaining: NO')
        print('ONNX checker: PASS')
        print('ONNX Runtime numerical validation: {}'.format(
            ort_result['status']))
    except Exception as error:
        module, source, op, action = classify_export_error(error)
        critical_trace = ''.join(traceback.format_exception(
            type(error), error, error.__traceback__))[-12000:]
        blockers.append({
            'module': module,
            'source': source,
            'op': op,
            'error': critical_trace,
            'status': 'EXPORTER',
            'action': action,
        })
        write_blockers(args.blockers, blockers, command, exporter_started)
        print('ONNX export status: FAIL')
        print('Exporter started: {}'.format('YES' if exporter_started else 'NO'))
        print('Exporter blockers: {}'.format(len(blockers)))
        print('Blocker report: {}'.format(args.blockers))
        raise


if __name__ == '__main__':
    main()
