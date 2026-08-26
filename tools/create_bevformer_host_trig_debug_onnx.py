#!/usr/bin/env python
"""Create a debug ONNX that replaces only TRT alignment trig outputs."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from create_bevformer_fp32_debug_onnx import BOUNDARIES


TRIG_REPLACEMENTS = {
    '1024': ('rotation_cos', 'Cos_322', 4),
    '1025': ('rotation_sin', 'Sin_323', 4),
}

ARITHMETIC_BOUNDARIES = {
    '1059': [1, 2, 3],       # theta
    '1352': [1, 2500, 3],    # flattened base grid
    '1353': [1, 3, 2],       # transposed theta
    '1355': [1, 3, 2],       # transposed theta / [W/2, H/2]
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arithmetic-debug', action='store_true')
    return parser.parse_args()


def replace_alignment_trig(model):
    """Replace only the audited trig consumers; fail on graph drift."""
    nodes = {node.name: node for node in model.graph.node}
    report = {}
    for old_value, (input_name, producer_name, expected_consumers) in \
            TRIG_REPLACEMENTS.items():
        producer = nodes.get(producer_name)
        if producer is None or list(producer.output) != [old_value]:
            raise RuntimeError('{} does not exclusively produce {}'.format(
                producer_name, old_value))
        replaced = 0
        for node in model.graph.node:
            if node is producer:
                continue
            for index, value in enumerate(node.input):
                if value == old_value:
                    node.input[index] = input_name
                    replaced += 1
        if replaced != expected_consumers:
            raise RuntimeError('{} consumer count {} != {}'.format(
                old_value, replaced, expected_consumers))
        model.graph.input.append(helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, []))
        report[input_name] = replaced
    return report


def main():
    args = parse_args()
    model = onnx.load(str(args.input))
    produced = {name for node in model.graph.node for name in node.output}
    report = replace_alignment_trig(model)

    existing_outputs = {item.name for item in model.graph.output}
    for debug_name, (tensor_name, shape) in BOUNDARIES.items():
        if tensor_name not in produced:
            raise RuntimeError('{} source {} is absent'.format(
                debug_name, tensor_name))
        if tensor_name not in existing_outputs:
            model.graph.output.append(helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, shape))
    if args.arithmetic_debug:
        for tensor_name, shape in ARITHMETIC_BOUNDARIES.items():
            if tensor_name not in produced:
                raise RuntimeError('arithmetic source {} is absent'.format(
                    tensor_name))
            if tensor_name not in existing_outputs:
                model.graph.output.append(helper.make_tensor_value_info(
                    tensor_name, TensorProto.FLOAT, shape))
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    for input_name, count in report.items():
        print('{} replacement consumers: {}'.format(input_name, count))
    print('Original Cos/Sin nodes retained only as debug outputs.')


if __name__ == '__main__':
    main()
