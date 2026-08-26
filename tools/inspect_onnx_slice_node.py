#!/usr/bin/env python
"""Inspect one ONNX Slice and recursively print its producer chains."""

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper

from deployment.onnx_slice_compat import constant_value


def tensor_type_map(model):
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    result = {}
    values = list(model.graph.input) + list(model.graph.output) + \
        list(model.graph.value_info)
    for value in values:
        tensor = value.type.tensor_type
        shape = []
        for dim in tensor.shape.dim:
            shape.append(dim.dim_value if dim.HasField('dim_value') else '?')
        result[value.name] = (
            onnx.TensorProto.DataType.Name(tensor.elem_type), shape)
    return result


def inspect(model, node_name, max_depth=10):
    producers = {output: node for node in model.graph.node
                 for output in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    graph_inputs = {item.name for item in model.graph.input}
    types = tensor_type_map(model)
    matches = [node for node in model.graph.node if node.name == node_name]
    if len(matches) != 1:
        raise ValueError('expected one {}, found {}'.format(node_name,
                                                            len(matches)))
    node = matches[0]

    def describe(name):
        producer = producers.get(name)
        value = constant_value(name, producers, initializers)
        kind = ('initializer' if name in initializers else
                'graph_input' if name in graph_inputs else
                'constant-foldable' if value is not None else 'runtime_tensor')
        display_value = None
        if value is not None:
            display_value = (value.tolist() if value.size <= 16 else
                             '<{} values>'.format(value.size))
        return {
            'name': name,
            'producer': None if producer is None else producer.name,
            'producer_op': None if producer is None else producer.op_type,
            'dtype': types.get(name, ('?', []))[0],
            'shape': types.get(name, ('?', []))[1],
            'kind': kind,
            'constant_value': display_value,
        }

    def chain(name, depth=0, visited=None, depth_limit=None):
        depth_limit = max_depth if depth_limit is None else depth_limit
        visited = set() if visited is None else visited
        prefix = '  ' * depth
        item = describe(name)
        print('{}{} producer={}/{} dtype={} shape={} kind={} value={}'.format(
            prefix, name, item['producer'], item['producer_op'], item['dtype'],
            item['shape'], item['kind'], item['constant_value']))
        if depth >= depth_limit or name in visited:
            return
        visited.add(name)
        producer = producers.get(name)
        if producer is not None:
            for source in producer.input:
                chain(source, depth + 1, visited, depth_limit)

    labels = ('data', 'starts', 'ends', 'axes', 'steps')
    print('Node Name: {}'.format(node.name))
    print('OpType: {}'.format(node.op_type))
    print('Domain: {}'.format(node.domain or 'ai.onnx'))
    for index, (label, name) in enumerate(zip(labels, node.input)):
        print('\nInput[{}] {}:'.format(index, label))
        chain(name, depth_limit=(max_depth if label == 'axes'
                                 else min(3, max_depth)))
    print('\nOutput Tensor: {}'.format(node.output[0]))
    print('Output type/shape: {}'.format(types.get(node.output[0], ('?', []))))
    return {'node': node.name,
            'inputs': {label: describe(name)
                       for label, name in zip(labels, node.input)},
            'output': describe(node.output[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--node', required=True)
    parser.add_argument('--max-depth', type=int, default=10)
    args = parser.parse_args()
    inspect(onnx.load(str(args.onnx)), args.node, args.max_depth)


if __name__ == '__main__':
    main()
