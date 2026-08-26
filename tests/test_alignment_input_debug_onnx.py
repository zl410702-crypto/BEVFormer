import importlib.util
from pathlib import Path
import sys

import onnx
import pytest
import torch
from onnx import TensorProto, helper


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAPH = load_module('alignment_input_debug',
                    TOOLS / 'create_bevformer_alignment_input_debug_onnx.py')
INPUTS = load_module('alignment_debug_inputs',
                     TOOLS / 'prepare_bevformer_alignment_debug_inputs.py')


def make_model():
    value = helper.make_tensor_value_info('value', TensorProto.FLOAT,
                                          [1, 2500, 3])
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT,
                                         [1, 256, 2500])
    nodes = [
        helper.make_node('MatMul', ['value', 'value'], ['1356'],
                         name='MatMul_654'),
        helper.make_node('Reshape', ['1356', 'value'], ['reshaped'],
                         name='Reshape_666'),
        helper.make_node('Expand', ['value', 'value'], ['1471'],
                         name='Expand_769'),
        helper.make_node('GatherElements', ['data', '1471'], ['1481'],
                         name='GatherElements_779', axis=2),
    ]
    return helper.make_model(helper.make_graph(nodes, 'alignment',
                                               [value, data], []))


@pytest.mark.parametrize('experiment,input_name,consumer,index', [
    ('affine', 'golden_affine_grid', 'Reshape_666', 0),
    ('index', 'golden_linear_index', 'GatherElements_779', 1),
])
def test_replaces_only_audited_consumer(experiment, input_name, consumer, index):
    model = make_model()
    report = GRAPH.replace_audited_value(model, experiment)
    assert report['input_name'] == input_name
    node = next(item for item in model.graph.node if item.name == consumer)
    expected = ('golden_linear_index_int64'
                if experiment == 'index' else input_name)
    assert node.input[index] == expected
    assert input_name in {item.name for item in model.graph.input}
    if experiment == 'index':
        cast = next(item for item in model.graph.node
                    if item.name == 'Cast_GoldenLinearIndexToInt64')
        assert list(cast.input) == [input_name]
        assert list(cast.output) == [expected]
        assert next(attr.i for attr in cast.attribute if attr.name == 'to') == 7


def test_fails_closed_on_graph_drift():
    model = make_model()
    next(node for node in model.graph.node
         if node.name == 'GatherElements_779').input[1] = 'wrong'
    with pytest.raises(RuntimeError, match='expected'):
        GRAPH.replace_audited_value(model, 'index')


def test_linear_index_contract_and_values():
    grid = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]],
                          [[0.0, 0.0], [0.5, -0.5]]]], dtype=torch.float32)
    _, _, _, _, index = INPUTS.make_linear_index(grid, 3, 2, 2)
    assert index.shape == (1, 3, 4)
    assert index.dtype == torch.int64
    assert index[0, 0].tolist() == [0, 3, 0, 1]
    assert torch.equal(index[:, 0], index[:, 1])
