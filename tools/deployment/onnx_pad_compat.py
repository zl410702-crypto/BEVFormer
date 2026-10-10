"""TensorRT 8.4 compatibility for fixed-chunk Linear padding.

PyTorch 1.9 exports ``F.pad`` with a runtime-looking pads shape subgraph.
TensorRT 8.4 rejects the two-dimensional shape-tensor reshape in that
subgraph even though the deployment chunk contract is fully static.
"""

import math
from collections import deque

import numpy as np
import onnx
from onnx import numpy_helper

from .onnx_slice_compat import constant_value


def _attribute(node, name):
    return next((item for item in node.attribute if item.name == name), None)


def _constant(name, producers, initializers):
    value = constant_value(name, producers, initializers)
    return None if value is None else np.asarray(value)


def canonicalize_fixed_chunk_pads(model, chunk_size=640, num_query=2500,
                                  expected_count=6):
    """Replace only audited fixed-chunk Pad vectors by INT64 initializers.

    The pass fails closed unless every candidate is the final query slice of
    the fixed deployment GEMM contract.  It intentionally does not fold
    arbitrary runtime-derived Pad values.
    """
    if chunk_size <= 0 or num_query <= 0:
        raise ValueError('chunk_size and num_query must be positive')
    final_start = (math.ceil(num_query / chunk_size) - 1) * chunk_size
    valid_queries = num_query - final_start
    pad_queries = chunk_size - valid_queries
    if pad_queries <= 0:
        raise ValueError('deployment query count does not require padding')

    producers = {output: node for node in model.graph.node
                 for output in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    matches = []
    for node in model.graph.node:
        if node.op_type != 'Pad' or len(node.input) != 3:
            continue
        data = producers.get(node.input[0])
        pads = producers.get(node.input[1])
        if data is None or data.op_type != 'Slice':
            continue
        if len(data.input) != 5:
            raise ValueError('{} data Slice is not five-input'.format(
                node.name))
        values = [_constant(name, producers, initializers)
                  for name in data.input[1:]]
        expected = [np.array([final_start], dtype=np.int64),
                    np.array([final_start + chunk_size], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                    np.array([1], dtype=np.int64)]
        if any(value is None or value.dtype != np.int64 or
               not np.array_equal(value, wanted)
               for value, wanted in zip(values, expected)):
            continue

        mode = _attribute(node, 'mode')
        if mode is None or mode.s != b'constant':
            raise ValueError('{} is not constant-mode Pad'.format(node.name))
        fill = _constant(node.input[2], producers, initializers)
        if fill is None or fill.size != 1 or float(fill.reshape(-1)[0]) != 0.0:
            raise ValueError('{} does not use zero padding'.format(node.name))

        expected_chain = ['Cast', 'Reshape', 'Transpose', 'Slice', 'Reshape']
        cursor = pads
        observed_chain = []
        for op_type in expected_chain:
            if cursor is None or cursor.op_type != op_type:
                raise ValueError('{} pads producer chain changed: {}'.format(
                    node.name, observed_chain))
            observed_chain.append(cursor.op_type)
            cursor = producers.get(cursor.input[0]) if cursor.input else None
        reshape_shape = _constant(pads.input[0], producers, initializers)
        del reshape_shape  # The exact pads data remains deliberately unused.
        inner_reshape = producers[producers[producers[producers[
            pads.input[0]].input[0]].input[0]].input[0]]
        shape_value = _constant(inner_reshape.input[1], producers,
                                initializers)
        if shape_value is None or not np.array_equal(
                shape_value, np.array([-1, 2], dtype=np.int64)):
            raise ValueError('{} expected pads reshape [-1,2]'.format(
                node.name))
        matches.append(node)

    if len(matches) != expected_count:
        raise ValueError('expected {} fixed-chunk Pads, found {}'.format(
            expected_count, len(matches)))

    result = []
    old_pad_values = []
    static_pads = np.array([0, 0, 0, 0, pad_queries, 0], dtype=np.int64)
    for node in matches:
        name = '{}_pads_static'.format(node.name)
        if name in initializers:
            raise ValueError('initializer already exists: {}'.format(name))
        old_pads = node.input[1]
        old_pad_values.append(old_pads)
        model.graph.initializer.append(numpy_helper.from_array(
            static_pads.copy(), name))
        node.input[1] = name
        result.append({'node': node.name, 'old_pads': old_pads,
                       'new_pads': name, 'value': static_pads.tolist()})

    # TensorRT parses dead nodes too.  Remove only dead ancestors of the six
    # replaced pads values; shared/live graph nodes remain protected.
    producers = {output: node for node in model.graph.node
                 for output in node.output}
    pad_ancestors = set()
    queue = deque(old_pad_values)
    while queue:
        value = queue.popleft()
        node = producers.get(value)
        if node is None or node.name in pad_ancestors:
            continue
        pad_ancestors.add(node.name)
        queue.extend(node.input)
    live_nodes = set()
    queue = deque(output.name for output in model.graph.output)
    while queue:
        value = queue.popleft()
        node = producers.get(value)
        if node is None or node.name in live_nodes:
            continue
        live_nodes.add(node.name)
        queue.extend(node.input)
    removed = sorted(pad_ancestors - live_nodes)
    kept = [node for node in model.graph.node if node.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return {
        'chunk_size': chunk_size,
        'num_query': num_query,
        'valid_queries_in_final_chunk': valid_queries,
        'pad_queries': pad_queries,
        'canonicalized': result,
        'removed_dead_nodes': len(removed),
        'removed_node_names': removed,
    }
