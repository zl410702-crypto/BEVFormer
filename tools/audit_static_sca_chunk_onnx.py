#!/usr/bin/env python
"""Audit that Stateful ONNX preserves the static-SCA Linear chunk contract."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


LINEARS = ('sampling_offsets', 'attention_weights')


def constant_hits(graph, target):
    hits = []
    for index, node in enumerate(graph.node):
        if node.op_type != 'Constant':
            continue
        for attribute in node.attribute:
            if attribute.name != 'value':
                continue
            value = numpy_helper.to_array(attribute.t)
            if np.any(value == target):
                hits.append({'index': index, 'name': node.name,
                             'shape': list(value.shape)})
    return hits


def audit(path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    producers = {output: (index, node)
                 for index, node in enumerate(model.graph.node)
                 for output in node.output}
    linears = []
    for layer in range(3):
        for linear_name in LINEARS:
            weight = ('model.pts_bbox_head.transformer.encoder.layers.{}.'
                      'attentions.1.deformable_attention.{}.weight').format(
                          layer, linear_name)
            matmuls = []
            for index, node in enumerate(model.graph.node):
                if node.op_type != 'MatMul' or len(node.input) < 2:
                    continue
                weight_producer = producers.get(node.input[1])
                if (weight_producer is not None and
                        weight in weight_producer[1].input):
                    input_producer = producers.get(node.input[0])
                    matmuls.append({
                        'index': index,
                        'name': node.name,
                        'input_op': (input_producer[1].op_type
                                     if input_producer else 'initializer'),
                    })
            linears.append({
                'encoder_layer': layer,
                'linear': linear_name,
                'matmul_count': len(matmuls),
                'matmuls': matmuls,
                'fixed_four_chunk_pattern': (
                    len(matmuls) == 4 and
                    Counter(item['input_op'] for item in matmuls) ==
                    Counter({'Slice': 3, 'Pad': 1})),
            })
    counts = Counter(node.op_type for node in model.graph.node)
    hits_640 = constant_hits(model.graph, 640)
    hits_604 = constant_hits(model.graph, 604)
    preserved = (all(item['fixed_four_chunk_pattern'] for item in linears)
                 and counts['NonZero'] == 0 and not hits_604 and hits_640)
    return {
        'file': str(path),
        'onnx_checker': 'PASS',
        'fixed_chunk': 640,
        'static_query_contract': [6, 2500, 256],
        'linear_contracts': linears,
        'operator_counts': {name: counts[name] for name in (
            'NonZero', 'MatMul', 'Slice', 'Pad', 'Concat', 'Shape', 'Range',
            'ScatterND')},
        'constant_640_count': len(hits_640),
        'constant_604_count': len(hits_604),
        'visibility_dynamic_shape_reintroduced': counts['NonZero'] != 0,
        'single_nq2500_matmul_detected': any(
            item['matmul_count'] == 1 for item in linears),
        'onnx_execution_contract_preserved': bool(preserved),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.model)
    output = args.output or args.model.with_name(
        args.model.stem + '_chunk_audit.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['onnx_execution_contract_preserved']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
