#!/usr/bin/env python
"""Create a 12-node ONNX that isolates the BEVFormer MSDA parser contract."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROLES = (
    [('temporal', layer, 2, 2500, 50, 50, 4) for layer in range(3)] +
    [('spatial', layer, 6, 2500, 15, 25, 8) for layer in range(3)] +
    [('decoder', layer, 1, 900, 50, 50, 4) for layer in range(6)])
HEADS = 8
HEAD_DIM = 32


def create(path):
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    for role, layer, batch, queries, height, width, points in ROLES:
        prefix = '{}_l{}'.format(role, layer)
        value = prefix + '_value'
        locations = prefix + '_sampling_locations'
        weights = prefix + '_attention_weights'
        shapes = prefix + '_spatial_shapes'
        starts = prefix + '_level_start_index'
        output = prefix + '_output'
        inputs.extend([
            helper.make_tensor_value_info(
                value, TensorProto.FLOAT,
                [batch, height * width, HEADS, HEAD_DIM]),
            helper.make_tensor_value_info(
                locations, TensorProto.FLOAT,
                [batch, queries, HEADS, 1, points, 2]),
            helper.make_tensor_value_info(
                weights, TensorProto.FLOAT,
                [batch, queries, HEADS, 1, points]),
        ])
        initializers.extend([
            numpy_helper.from_array(
                np.array([[height, width]], dtype=np.int32), shapes),
            numpy_helper.from_array(np.array([0], dtype=np.int32), starts),
        ])
        nodes.append(helper.make_node(
            'MSDeformableAttention',
            [value, shapes, starts, locations, weights], [output],
            name=prefix + '_MSDeformableAttention', domain='bevformer',
            im2col_step=64))
        outputs.append(helper.make_tensor_value_info(
            output, TensorProto.FLOAT, [batch, queries, HEADS * HEAD_DIM]))
    graph = helper.make_graph(
        nodes, 'bevformer_msda_12_node_parser_test', inputs, outputs,
        initializer=initializers)
    model = helper.make_model(
        graph, producer_name='BEVFormer MSDA parser contract test',
        opset_imports=[helper.make_opsetid('', 13),
                       helper.make_opsetid('bevformer', 1)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path,
                        default=Path('artifacts/msda_12_node_parser_test.onnx'))
    args = parser.parse_args()
    model = create(args.output)
    print('File: {}'.format(args.output))
    print('ONNX checker: PASS')
    print('MSDA nodes: {}'.format(sum(
        node.domain == 'bevformer' and
        node.op_type == 'MSDeformableAttention'
        for node in model.graph.node)))


if __name__ == '__main__':
    main()
