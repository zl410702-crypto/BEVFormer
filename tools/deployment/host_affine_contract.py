"""Production host-affine tensor contract and ONNX graph rewrite."""

from collections import deque
from copy import deepcopy

import onnx
import torch
from onnx import TensorProto, helper

from .grid_sample_rewrite import bev_affine_grid_tensor


AFFINE_INPUT_NAME = 'affine_grid'
AFFINE_SHAPE = [1, 2500, 2]
AFFINE_DTYPE = TensorProto.FLOAT
AFFINE_LAYOUT = 'B,HW,XY'

EXPECTED_AFFINE_PRODUCER = ('MatMul_654', 'MatMul', '1356')
EXPECTED_AFFINE_CONSUMER = ('Reshape_666', 'Reshape', 0)
EXPECTED_SAMPLING_CHAIN = {
    'Round_707': ('Round', '1409'),
    'Round_708': ('Round', '1410'),
    'Clip_733': ('Clip', '1435'),
    'Clip_739': ('Clip', '1441'),
    'GatherElements_779': ('GatherElements', '1481'),
}


def prepare_host_affine_grid(history_chw, angle_degrees, center):
    """Produce the runtime input with the exact deployment PyTorch path."""
    if history_chw.dtype != torch.float32:
        raise TypeError('history_chw must be FP32')
    if angle_degrees.dtype != torch.float32:
        raise TypeError('angle_degrees must be FP32')
    grid = bev_affine_grid_tensor(history_chw, angle_degrees, center)
    if list(grid.shape) != AFFINE_SHAPE:
        raise ValueError('affine grid shape {} != {}'.format(
            list(grid.shape), AFFINE_SHAPE))
    if grid.dtype != torch.float32 or not grid.is_contiguous():
        raise RuntimeError('affine grid must be contiguous FP32')
    return grid


def _msda_contracts(model):
    return [(node.name, list(node.input), list(node.output),
             [(attribute.name, attribute.i) for attribute in node.attribute])
            for node in model.graph.node
            if node.domain == 'bevformer' and
            node.op_type == 'MSDeformableAttention']


def _consumers(model):
    result = {}
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            result.setdefault(value, []).append((node.name, index))
    return result


def _validate_alignment_contract(model):
    nodes = {node.name: node for node in model.graph.node}
    producer_name, producer_op, value_name = EXPECTED_AFFINE_PRODUCER
    producer = nodes.get(producer_name)
    if producer is None or producer.op_type != producer_op or \
            list(producer.output) != [value_name]:
        raise RuntimeError('alignment affine producer contract changed')
    consumer_name, consumer_op, input_index = EXPECTED_AFFINE_CONSUMER
    consumer = nodes.get(consumer_name)
    if consumer is None or consumer.op_type != consumer_op or \
            len(consumer.input) <= input_index or \
            consumer.input[input_index] != value_name:
        raise RuntimeError('alignment affine consumer contract changed')
    if _consumers(model).get(value_name) != [(consumer_name, input_index)]:
        raise RuntimeError('alignment affine tensor consumer set changed')
    for name, (op_type, output) in EXPECTED_SAMPLING_CHAIN.items():
        node = nodes.get(name)
        if node is None or node.op_type != op_type or output not in node.output:
            raise RuntimeError('sampling-chain contract changed at {}'.format(name))
    return consumer, input_index


def _prune_dead_affine_ancestors(model, affine_value):
    """Remove dead nodes only from the replaced affine producer ancestry."""
    producer = {output: node for node in model.graph.node for output in node.output}
    affine_ancestors = set()
    queue = deque([affine_value])
    while queue:
        value = queue.popleft()
        node = producer.get(value)
        if node is None or node.name in affine_ancestors:
            continue
        affine_ancestors.add(node.name)
        queue.extend(node.input)

    needed_nodes = set()
    queue = deque(output.name for output in model.graph.output)
    while queue:
        value = queue.popleft()
        node = producer.get(value)
        if node is None or node.name in needed_nodes:
            continue
        needed_nodes.add(node.name)
        queue.extend(node.input)
    removable = affine_ancestors - needed_nodes
    kept = [node for node in model.graph.node if node.name not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return sorted(removable)


def rewrite_host_affine_input(model):
    """Replace only the audited affine MatMul result; fail on graph drift."""
    model = deepcopy(model)
    if AFFINE_INPUT_NAME in {value.name for value in model.graph.input}:
        raise RuntimeError('{} input already exists'.format(AFFINE_INPUT_NAME))
    before_msda = _msda_contracts(model)
    if len(before_msda) != 12:
        raise RuntimeError('expected 12 MSDA nodes, found {}'.format(
            len(before_msda)))
    consumer, input_index = _validate_alignment_contract(model)
    consumer.input[input_index] = AFFINE_INPUT_NAME
    model.graph.input.append(helper.make_tensor_value_info(
        AFFINE_INPUT_NAME, AFFINE_DTYPE, AFFINE_SHAPE))
    removed = _prune_dead_affine_ancestors(
        model, EXPECTED_AFFINE_PRODUCER[2])
    after_msda = _msda_contracts(model)
    if after_msda != before_msda:
        raise RuntimeError('MSDA node contracts changed')
    onnx.checker.check_model(model)
    return model, {
        'input_name': AFFINE_INPUT_NAME,
        'shape': AFFINE_SHAPE,
        'dtype': 'float32',
        'layout': AFFINE_LAYOUT,
        'producer': EXPECTED_AFFINE_PRODUCER[0],
        'consumer': EXPECTED_AFFINE_CONSUMER[0],
        'removed_dead_nodes': len(removed),
        'removed_node_names': removed,
        'msda_count': len(after_msda),
    }
