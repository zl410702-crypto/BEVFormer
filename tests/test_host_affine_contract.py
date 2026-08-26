import importlib.util
from pathlib import Path
import sys

import onnx
import pytest
import torch
from onnx import TensorProto, helper


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from deployment.host_affine_contract import (  # noqa: E402
    AFFINE_INPUT_NAME, prepare_host_affine_grid, rewrite_host_affine_input)

VALIDATION_SPEC = importlib.util.spec_from_file_location(
    'host_affine_validation',
    TOOLS / 'create_bevformer_host_affine_validation_onnx.py')
VALIDATION = importlib.util.module_from_spec(VALIDATION_SPEC)
VALIDATION_SPEC.loader.exec_module(VALIDATION)


def make_model(consumer_input='1356', msda_count=12):
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT,
                                         [1, 2500, 3])
    outputs = [helper.make_tensor_value_info('result', TensorProto.FLOAT,
                                             [1, 256, 2500])]
    nodes = [
        helper.make_node('MatMul', ['data', 'data'], ['1356'],
                         name='MatMul_654'),
        helper.make_node('Reshape', [consumer_input, 'data'], ['1368'],
                         name='Reshape_666'),
        helper.make_node('Round', ['1368'], ['1409'], name='Round_707'),
        helper.make_node('Round', ['1368'], ['1410'], name='Round_708'),
        helper.make_node('Clip', ['1409'], ['1435'], name='Clip_733'),
        helper.make_node('Clip', ['1410'], ['1441'], name='Clip_739'),
        helper.make_node('GatherElements', ['data', '1435'], ['1481'],
                         name='GatherElements_779', axis=2),
    ]
    value = '1481'
    for index in range(msda_count):
        output = 'msda_{}'.format(index)
        nodes.append(helper.make_node(
            'MSDeformableAttention', [value] * 5, [output],
            name='MSDA_{}'.format(index), domain='bevformer', im2col_step=64))
        value = output
    nodes.append(helper.make_node('Identity', [value], ['result']))
    graph = helper.make_graph(nodes, 'host_affine', [data], outputs)
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid('bevformer', 1))
    return model


def test_rewrite_adds_static_input_and_preserves_msda_contracts():
    model, report = rewrite_host_affine_input(make_model())
    affine = next(value for value in model.graph.input
                  if value.name == AFFINE_INPUT_NAME)
    assert affine.type.tensor_type.elem_type == TensorProto.FLOAT
    assert [dim.dim_value for dim in affine.type.tensor_type.shape.dim] == \
        [1, 2500, 2]
    reshape = next(node for node in model.graph.node
                   if node.name == 'Reshape_666')
    assert reshape.input[0] == AFFINE_INPUT_NAME
    assert report['msda_count'] == 12
    assert sum(node.domain == 'bevformer' for node in model.graph.node) == 12
    assert report['removed_node_names'] == ['MatMul_654']


def test_rewrite_fails_closed_on_consumer_drift():
    with pytest.raises(RuntimeError, match='consumer contract changed'):
        rewrite_host_affine_input(make_model(consumer_input='data'))


def test_rewrite_fails_closed_on_msda_count_change():
    with pytest.raises(RuntimeError, match='expected 12'):
        rewrite_host_affine_input(make_model(msda_count=11))


def test_runtime_affine_contract_and_dtype_validation():
    history = torch.zeros(256, 50, 50, dtype=torch.float32)
    angle = torch.tensor([-1.0353196], dtype=torch.float32)
    grid = prepare_host_affine_grid(history, angle, [100, 100])
    assert grid.shape == (1, 2500, 2)
    assert grid.dtype == torch.float32
    assert grid.is_contiguous()
    with pytest.raises(TypeError, match='FP32'):
        prepare_host_affine_grid(history.double(), angle, [100, 100])


def test_runtime_affine_is_deterministic():
    history = torch.arange(256 * 50 * 50, dtype=torch.float32).reshape(
        256, 50, 50)
    angle = torch.tensor([-1.0353196], dtype=torch.float32)
    first = prepare_host_affine_grid(history, angle, [100, 100])
    second = prepare_host_affine_grid(history, angle, [100, 100])
    assert torch.equal(first, second)


def test_validation_outputs_require_production_affine_input():
    model, unused_report = rewrite_host_affine_input(make_model())
    produced = {name for node in model.graph.node for name in node.output}
    for tensor_name in VALIDATION.BOUNDARIES:
        if tensor_name in produced:
            continue
        model.graph.node.append(helper.make_node(
            'Identity', ['result'], [tensor_name],
            name='Boundary_{}'.format(tensor_name)))
    validated = VALIDATION.add_validation_outputs(model)
    assert 'debug_affine_grid' in {value.name
                                   for value in validated.graph.output}
    identity = next(node for node in validated.graph.node
                    if node.name == 'Identity_DebugAffineGrid')
    assert list(identity.input) == ['affine_grid']

    without_affine = make_model()
    with pytest.raises(RuntimeError, match='affine_grid'):
        VALIDATION.add_validation_outputs(without_affine)
