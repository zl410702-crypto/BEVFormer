from pathlib import Path
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from deployment.onnx_pad_compat import canonicalize_fixed_chunk_pads  # noqa: E402


def _constant(name, value):
    return helper.make_node(
        'Constant', [], [name], name='Constant_' + name,
        value=numpy_helper.from_array(np.asarray(value)))


def make_model(count=6, start=1920):
    inputs = []
    outputs = []
    nodes = []
    for index in range(count):
        prefix = 'p{}_'.format(index)
        data = prefix + 'data'
        inputs.append(helper.make_tensor_value_info(
            data, TensorProto.FLOAT, [6, 2500, 256]))
        for suffix, value in (
                ('starts', np.array([start], dtype=np.int64)),
                ('ends', np.array([2560], dtype=np.int64)),
                ('axes', np.array([1], dtype=np.int64)),
                ('steps', np.array([1], dtype=np.int64)),
                ('reshape_shape', np.array([-1, 2], dtype=np.int64)),
                ('slice_starts', np.array([-1], dtype=np.int64)),
                ('slice_ends', np.array([-9223372036854775807],
                                         dtype=np.int64)),
                ('slice_axes', np.array([0], dtype=np.int64)),
                ('slice_steps', np.array([-1], dtype=np.int64)),
                ('flat_shape', np.array([-1], dtype=np.int64)),
                ('fill', np.array(0.0, dtype=np.float32))):
            nodes.append(_constant(prefix + suffix, value))
        nodes.extend([
            helper.make_node('Slice', [data, prefix + 'starts',
                                       prefix + 'ends', prefix + 'axes',
                                       prefix + 'steps'], [prefix + 'chunk'],
                             name=prefix + 'ChunkSlice'),
            helper.make_node('Constant', [], [prefix + 'pads_source'],
                             name=prefix + 'PadsSource',
                             value=numpy_helper.from_array(
                                 np.zeros(6, dtype=np.int64))),
            helper.make_node('Reshape', [prefix + 'pads_source',
                                         prefix + 'reshape_shape'],
                             [prefix + 'pads_pairs'],
                             name=prefix + 'PadsReshape'),
            helper.make_node('Slice', [prefix + 'pads_pairs',
                                       prefix + 'slice_starts',
                                       prefix + 'slice_ends',
                                       prefix + 'slice_axes',
                                       prefix + 'slice_steps'],
                             [prefix + 'pads_reverse'],
                             name=prefix + 'PadsSlice'),
            helper.make_node('Transpose', [prefix + 'pads_reverse'],
                             [prefix + 'pads_transpose'], perm=[1, 0],
                             name=prefix + 'PadsTranspose'),
            helper.make_node('Reshape', [prefix + 'pads_transpose',
                                         prefix + 'flat_shape'],
                             [prefix + 'pads_flat'],
                             name=prefix + 'PadsFlat'),
            helper.make_node('Cast', [prefix + 'pads_flat'],
                             [prefix + 'pads'], to=TensorProto.INT64,
                             name=prefix + 'PadsCast'),
            helper.make_node('Pad', [prefix + 'chunk', prefix + 'pads',
                                     prefix + 'fill'], [prefix + 'output'],
                             name='Pad_{}'.format(index), mode='constant'),
        ])
        outputs.append(helper.make_tensor_value_info(
            prefix + 'output', TensorProto.FLOAT, [6, 640, 256]))
    graph = helper.make_graph(nodes, 'fixed_chunk_pad', inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    return model


def test_canonicalizes_six_fixed_chunk_pad_vectors():
    model = make_model()
    report = canonicalize_fixed_chunk_pads(model)
    onnx.checker.check_model(model)
    assert report['pad_queries'] == 60
    assert report['valid_queries_in_final_chunk'] == 580
    assert len(report['canonicalized']) == 6
    assert report['removed_dead_nodes'] > 0
    assert not any(node.name.endswith('PadsReshape')
                   for node in model.graph.node)
    initializers = {item.name: numpy_helper.to_array(item)
                    for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Pad':
            assert initializers[node.input[1]].tolist() == [0, 0, 0, 0, 60, 0]


def test_fails_closed_if_fixed_chunk_slice_contract_drifts():
    with pytest.raises(ValueError, match='expected 6'):
        canonicalize_fixed_chunk_pads(make_model(start=1919))


def test_fails_closed_if_pad_count_drifts():
    with pytest.raises(ValueError, match='expected 6'):
        canonicalize_fixed_chunk_pads(make_model(count=5))
