#!/usr/bin/env python
"""Expose alignment/encoder outputs on a production host-affine ONNX copy."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


BOUNDARIES = {
    '1398': [1, 50, 50],
    '1408': [1, 50, 50],
    '1409': [1, 50, 50],
    '1410': [1, 50, 50],
    '1481': [1, 256, 2500],
    '1498': [50, 50, 256],
    '1530': [2500, 1, 256],
    '3370': [1, 2500, 256],
    '4155': [1, 2500, 256],
    '4940': [1, 2500, 256],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def add_validation_outputs(model):
    inputs = {value.name: value for value in model.graph.input}
    affine = inputs.get('affine_grid')
    if affine is None or affine.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError('production affine_grid FP32 input is absent')
    dims = [dim.dim_value for dim in affine.type.tensor_type.shape.dim]
    if dims != [1, 2500, 2]:
        raise RuntimeError('production affine_grid shape changed: {}'.format(dims))
    produced = {name for node in model.graph.node for name in node.output}
    existing = {value.name for value in model.graph.output}
    for tensor_name, shape in BOUNDARIES.items():
        if tensor_name not in produced:
            raise RuntimeError('boundary {} is absent'.format(tensor_name))
        if tensor_name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, shape))
    # TensorRT treats an input as a binding rather than an output binding, so
    # expose an Identity to validate the exact bytes observed by the graph.
    identity_name = 'debug_affine_grid'
    model.graph.node.insert(0, helper.make_node(
        'Identity', ['affine_grid'], [identity_name],
        name='Identity_DebugAffineGrid'))
    model.graph.output.append(helper.make_tensor_value_info(
        identity_name, TensorProto.FLOAT, [1, 2500, 2]))
    onnx.checker.check_model(model)
    return model


def main():
    args = parse_args()
    if args.output.exists():
        raise RuntimeError('refusing to overwrite {}'.format(args.output))
    model = add_validation_outputs(onnx.load(str(args.input)))
    onnx.save(model, str(args.output))
    print('Validation outputs: debug_affine_grid, {}'.format(
        ', '.join(BOUNDARIES)))


if __name__ == '__main__':
    main()
