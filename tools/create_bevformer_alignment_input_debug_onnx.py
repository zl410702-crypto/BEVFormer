#!/usr/bin/env python
"""Create fail-closed host-grid or host-index alignment debug ONNX graphs."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from create_bevformer_fp32_debug_onnx import BOUNDARIES


EXPERIMENTS = {
    'affine': {
        'input_name': 'golden_affine_grid',
        'dtype': TensorProto.FLOAT,
        'shape': [1, 2500, 2],
        'producer': ('MatMul_654', 'MatMul', '1356'),
        'consumer': ('Reshape_666', 0),
    },
    'index': {
        'input_name': 'golden_linear_index',
        # TensorRT 8.x has no INT64 execution binding. Keep the runtime
        # boundary explicit INT32, then cast to the audited ONNX Gather index
        # type instead of relying on parser-side implicit conversion.
        'dtype': TensorProto.INT32,
        'shape': [1, 256, 2500],
        'producer': ('Expand_769', 'Expand', '1471'),
        'consumer': ('GatherElements_779', 1),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', choices=sorted(EXPERIMENTS), required=True)
    return parser.parse_args()


def replace_audited_value(model, experiment):
    """Replace one audited consumer and reject any graph-contract drift."""
    spec = EXPERIMENTS[experiment]
    nodes = {node.name: node for node in model.graph.node}
    producer_name, producer_op, old_value = spec['producer']
    producer = nodes.get(producer_name)
    if producer is None or producer.op_type != producer_op:
        raise RuntimeError('{} producer contract changed'.format(producer_name))
    if list(producer.output) != [old_value]:
        raise RuntimeError('{} does not exclusively produce {}'.format(
            producer_name, old_value))

    consumer_name, input_index = spec['consumer']
    consumer = nodes.get(consumer_name)
    if consumer is None or len(consumer.input) <= input_index:
        raise RuntimeError('{} consumer contract changed'.format(consumer_name))
    if consumer.input[input_index] != old_value:
        raise RuntimeError('{} input {} is {}, expected {}'.format(
            consumer_name, input_index, consumer.input[input_index], old_value))

    all_consumers = [(node.name, index) for node in model.graph.node
                     for index, value in enumerate(node.input)
                     if value == old_value]
    if experiment == 'affine' and all_consumers != [(consumer_name, input_index)]:
        raise RuntimeError('{} consumers changed: {}'.format(old_value, all_consumers))

    model.graph.input.append(helper.make_tensor_value_info(
        spec['input_name'], spec['dtype'], spec['shape']))
    replacement_value = spec['input_name']
    if experiment == 'index':
        replacement_value = 'golden_linear_index_int64'
        cast = helper.make_node(
            'Cast', [spec['input_name']], [replacement_value],
            name='Cast_GoldenLinearIndexToInt64', to=TensorProto.INT64)
        consumer_position = list(model.graph.node).index(consumer)
        model.graph.node.insert(consumer_position, cast)
    consumer.input[input_index] = replacement_value
    return {
        'input_name': spec['input_name'],
        'old_value': old_value,
        'consumer': consumer_name,
        'input_index': input_index,
        'replacement_value': replacement_value,
    }


def add_debug_outputs(model):
    produced = {name for node in model.graph.node for name in node.output}
    existing = {item.name for item in model.graph.output}
    for debug_name, (tensor_name, shape) in BOUNDARIES.items():
        if tensor_name not in produced:
            raise RuntimeError('{} source {} is absent'.format(
                debug_name, tensor_name))
        if tensor_name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, shape))


def main():
    args = parse_args()
    model = onnx.load(str(args.input))
    report = replace_audited_value(model, args.experiment)
    add_debug_outputs(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print('Experiment: {}'.format(args.experiment))
    print('Input: {}'.format(report['input_name']))
    print('Replaced {} input {} ({})'.format(
        report['consumer'], report['input_index'], report['old_value']))
    print('Original producer retained only for debug/audit output.')


if __name__ == '__main__':
    main()
