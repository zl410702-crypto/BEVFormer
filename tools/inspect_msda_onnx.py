#!/usr/bin/env python
"""Inspect and classify BEVFormer MSDeformableAttention ONNX nodes."""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_MODEL = Path(
    'artifacts/bevformer_tiny_stateful_static_sca_chunk640_opset13.onnx')
DOMAIN = 'bevformer'
OP_TYPE = 'MSDeformableAttention'
DTYPE_NAMES = {value: name for name, value in
               onnx.TensorProto.DataType.items()}


ROLE_PATTERNS = (
    ('Temporal Self Attention', re.compile(
        r'encoder\.layers\.(\d+)\.attentions\.0\.attention_weights\.weight')),
    ('Spatial Cross Attention', re.compile(
        r'encoder\.layers\.(\d+)\.attentions\.1\.deformable_attention\.'
        r'attention_weights\.weight')),
    ('Detection Decoder Cross Attention', re.compile(
        r'decoder\.layers\.(\d+)\.attentions\.1\.attention_weights\.weight')),
)


def tensor_from_constant(node, producers=None):
    if node is None:
        return None
    if node.op_type == 'Cast' and producers is not None:
        source = producers.get(node.input[0])
        value = tensor_from_constant(source, producers)
        if value is None:
            return None
        target = next((attribute.i for attribute in node.attribute
                       if attribute.name == 'to'), None)
        if target == onnx.TensorProto.INT32:
            return value.astype(np.int32)
        if target == onnx.TensorProto.INT64:
            return value.astype(np.int64)
        return None
    if node.op_type != 'Constant':
        return None
    for attribute in node.attribute:
        if attribute.name == 'value':
            return numpy_helper.to_array(attribute.t)
    return None


def nearest_attention_weight(value_name, producers, max_depth=24):
    queue = deque([(value_name, 0)])
    visited = set()
    matches = []
    while queue:
        name, depth = queue.popleft()
        if name in visited or depth > max_depth:
            continue
        visited.add(name)
        if name.endswith('attention_weights.weight'):
            matches.append((depth, name))
            continue
        producer = producers.get(name)
        if producer is not None:
            queue.extend((input_name, depth + 1)
                         for input_name in producer.input)
    if not matches:
        raise ValueError('attention_weights weight is not reachable from {}'
                         .format(value_name))
    return min(matches)[1]


def classify_scope(scope):
    for role, pattern in ROLE_PATTERNS:
        match = pattern.search(scope)
        if match:
            return role, int(match.group(1))
    raise ValueError('unrecognized attention scope: {}'.format(scope))


def role_contract(role, spatial_shapes):
    """Return the current BEVFormer-Tiny deployment contract.

    B/Q/P follow the actual Tiny source configuration and its static Stateful
    wrapper. L/S are always derived from the ONNX spatial_shapes Constant.
    """
    levels = int(spatial_shapes.shape[0])
    spatial_size = int(np.prod(spatial_shapes, axis=1).sum())
    common = {'num_heads': 8, 'num_levels': levels,
              'head_dim': 32, 'embed_dims': 256}
    if role == 'Temporal Self Attention':
        common.update(batch=2, num_query=2500, num_points=4,
                      linear_queue_factor=2)
    elif role == 'Spatial Cross Attention':
        common.update(batch=6, num_query=2500, num_points=8,
                      linear_queue_factor=1)
    elif role == 'Detection Decoder Cross Attention':
        common.update(batch=1, num_query=900, num_points=4,
                      linear_queue_factor=1)
    else:
        raise ValueError('unsupported role: {}'.format(role))
    common['spatial_size'] = spatial_size
    return common


def value_info_map(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    values = {}
    for item in list(inferred.graph.input) + list(inferred.graph.output) + \
            list(inferred.graph.value_info):
        tensor = item.type.tensor_type
        values[item.name] = {
            'dtype': DTYPE_NAMES.get(tensor.elem_type,
                                     str(tensor.elem_type)),
            'onnx_shape': [
                dimension.dim_value if dimension.HasField('dim_value')
                else dimension.dim_param if dimension.HasField('dim_param')
                else '?'
                for dimension in tensor.shape.dim],
        }
    for initializer in inferred.graph.initializer:
        values.setdefault(initializer.name, {
            'dtype': DTYPE_NAMES.get(initializer.data_type,
                                     str(initializer.data_type)),
            'onnx_shape': list(initializer.dims),
        })
    return values


def attributes(node):
    result = {}
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.INT:
            result[attribute.name] = attribute.i
        elif attribute.type == onnx.AttributeProto.FLOAT:
            result[attribute.name] = attribute.f
        elif attribute.type == onnx.AttributeProto.STRING:
            result[attribute.name] = attribute.s.decode('utf-8')
        else:
            result[attribute.name] = '<type {}>'.format(attribute.type)
    return result


def validate_index_dtypes(node_name, shape_dtype, index_dtype,
                          require_int32=False):
    if require_int32 and (shape_dtype != 'INT32' or index_dtype != 'INT32'):
        raise ValueError(
            '{} shape/index contract must be INT32, got {}/{}'.format(
                node_name, shape_dtype, index_dtype))


def inspect(model_path, require_int32=False):
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    producers = {output: node for node in model.graph.node
                 for output in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    types = value_info_map(model)
    nodes = []
    for node_index, node in enumerate(model.graph.node):
        if node.domain != DOMAIN or node.op_type != OP_TYPE:
            continue
        if len(node.input) != 5 or len(node.output) != 1:
            raise ValueError('{} has unexpected arity'.format(node.name))
        scope = nearest_attention_weight(node.input[4], producers)
        role, layer = classify_scope(scope)
        shape_node = producers.get(node.input[1])
        spatial_shapes = tensor_from_constant(shape_node, producers)
        if spatial_shapes is None:
            raise ValueError('{} spatial_shapes is not Constant'.format(
                node.name))
        if spatial_shapes.dtype not in (np.dtype(np.int32),
                                        np.dtype(np.int64)) or \
                spatial_shapes.ndim != 2 or \
                spatial_shapes.shape[1] != 2:
            raise ValueError('{} invalid spatial_shapes'.format(node.name))
        contract = role_contract(role, spatial_shapes)
        weight = initializers.get(scope)
        expected_weight_output = (contract['linear_queue_factor'] *
                                  contract['num_heads'] *
                                  contract['num_levels'] *
                                  contract['num_points'])
        if weight is None or weight.dims[0] != expected_weight_output:
            raise ValueError('{} weight/contract mismatch: {} != {}'.format(
                node.name, None if weight is None else weight.dims[0],
                expected_weight_output))
        batch = contract['batch']
        size = contract['spatial_size']
        query = contract['num_query']
        heads = contract['num_heads']
        levels = contract['num_levels']
        points = contract['num_points']
        head_dim = contract['head_dim']
        recovered_shapes = (
            [batch, size, heads, head_dim],
            list(spatial_shapes.shape),
            [levels],
            [batch, query, heads, levels, points, 2],
            [batch, query, heads, levels, points],
        )
        integer_dtype = types.get(node.input[1], {}).get('dtype', 'INT32')
        index_dtype = types.get(node.input[2], {}).get('dtype', integer_dtype)
        validate_index_dtypes(node.name, integer_dtype, index_dtype,
                              require_int32=require_int32)
        expected_dtypes = ('FLOAT', integer_dtype, index_dtype,
                           'FLOAT', 'FLOAT')
        inputs = []
        for position, (name, shape, expected_dtype) in enumerate(zip(
                node.input, recovered_shapes, expected_dtypes)):
            observed = types.get(name, {})
            dtype = observed.get('dtype', expected_dtype)
            if dtype != expected_dtype:
                raise ValueError('{} input {} dtype {} != {}'.format(
                    node.name, position, dtype, expected_dtype))
            inputs.append({
                'position': position,
                'name': name,
                'dtype': dtype,
                'shape': shape,
                'onnx_declared_shape': observed.get('onnx_shape'),
                'shape_source': ('ONNX Constant' if position in (1, 2)
                                 else 'graph scope + Tiny source contract'),
            })
        output_shape = [batch, query, heads * head_dim]
        nodes.append({
            'node_index': node_index,
            'node_name': node.name,
            'role': role,
            'layer': layer,
            'source_scope': scope,
            'inputs': inputs,
            'output': {
                'name': node.output[0], 'dtype': 'FLOAT',
                'shape': output_shape,
                'onnx_declared_shape': types.get(
                    node.output[0], {}).get('onnx_shape'),
                'shape_source': 'MSDA mathematical output contract',
            },
            'attributes': attributes(node),
            'spatial_shapes_value': spatial_shapes.tolist(),
            'num_heads': heads,
            'num_levels': levels,
            'num_points': points,
        })
    if len(nodes) != 12:
        raise ValueError('expected 12 MSDA nodes, found {}'.format(len(nodes)))
    role_counts = Counter(item['role'] for item in nodes)
    expected_counts = Counter({
        'Temporal Self Attention': 3,
        'Spatial Cross Attention': 3,
        'Detection Decoder Cross Attention': 6,
    })
    if role_counts != expected_counts:
        raise ValueError('unexpected role counts: {}'.format(role_counts))
    return {
        'model': str(model_path),
        'onnx_checker': 'PASS',
        'domain': DOMAIN,
        'op_type': OP_TYPE,
        'node_count': len(nodes),
        'int32_contract_required': require_int32,
        'role_counts': dict(role_counts),
        'shape_note': (
            'PyTorch 1.9 leaves custom-op tensor ranks empty in value_info; '
            'FLOAT/INT64 dtypes are read from inferred ONNX types, while '
            'non-Constant shapes are recovered from graph scopes and the '
            'checked BEVFormer-Tiny source contract.'),
        'nodes': nodes,
    }


def print_table(report):
    print('| Node | Role | value | sampling_locations | attention_weights | '
          'H | L | P |')
    print('|---|---|---|---|---|---:|---:|---:|')
    for node in report['nodes']:
        print('| {name} | {role} L{layer} | {value} | {locations} | '
              '{weights} | {heads} | {levels} | {points} |'.format(
                  name=node['node_name'], role=node['role'],
                  layer=node['layer'], value=node['inputs'][0]['shape'],
                  locations=node['inputs'][3]['shape'],
                  weights=node['inputs'][4]['shape'],
                  heads=node['num_heads'], levels=node['num_levels'],
                  points=node['num_points']))


def print_details(report):
    input_labels = ('value', 'spatial_shapes', 'level_start_index',
                    'sampling_locations', 'attention_weights')
    for node in report['nodes']:
        print('\n[{}] {} — {} layer {}'.format(
            node['node_index'], node['node_name'], node['role'],
            node['layer']))
        for label, item in zip(input_labels, node['inputs']):
            print('  input {} {}: name={} dtype={} shape={}'.format(
                item['position'], label, item['name'], item['dtype'],
                item['shape']))
        print('  output: name={} dtype={} shape={}'.format(
            node['output']['name'], node['output']['dtype'],
            node['output']['shape']))
        print('  attributes: {}'.format(node['attributes']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--json', type=Path, dest='json_path')
    parser.add_argument(
        '--require-int32', action='store_true',
        help='Fail unless every spatial_shapes/level_start_index input is INT32.')
    args = parser.parse_args()
    report = inspect(args.model, require_int32=args.require_int32)
    print_details(report)
    print('')
    print_table(report)
    print('\nRole counts: {}'.format(report['role_counts']))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2) + '\n')
        print('JSON: {}'.format(args.json_path))


if __name__ == '__main__':
    main()
