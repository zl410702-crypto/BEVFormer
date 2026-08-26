import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


MODULE_PATH = (Path(__file__).parents[1] /
               'tools/deployment/onnx_slice_compat.py')
SPEC = importlib.util.spec_from_file_location('onnx_slice_compat', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_model(axis=0, node_name='Slice_1933'):
    constants = [
        helper.make_node('Constant', [], ['start_scalar'],
                         value=numpy_helper.from_array(
                             np.array(0, dtype=np.int64))),
        helper.make_node('Constant', [], ['end_scalar'],
                         value=numpy_helper.from_array(
                             np.array(1, dtype=np.int64))),
        helper.make_node('Constant', [], ['axis_scalar'],
                         value=numpy_helper.from_array(
                             np.array(axis, dtype=np.int64))),
        helper.make_node('Constant', [], ['unsqueeze_axis'],
                         value=numpy_helper.from_array(
                             np.array([0], dtype=np.int64))),
        helper.make_node('Unsqueeze', ['start_scalar', 'unsqueeze_axis'],
                         ['starts']),
        helper.make_node('Unsqueeze', ['end_scalar', 'unsqueeze_axis'],
                         ['ends']),
        helper.make_node('Unsqueeze', ['axis_scalar', 'unsqueeze_axis'],
                         ['axes']),
        helper.make_node('Constant', [], ['steps'],
                         value=numpy_helper.from_array(
                             np.array([1], dtype=np.int64))),
    ]
    sliced = helper.make_node(
        'Slice', ['data', 'starts', 'ends', 'axes', 'steps'], ['output'],
        name=node_name)
    graph = helper.make_graph(
        constants + [sliced], 'slice_test',
        [helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    return model


def test_constantizes_only_audited_axes_to_initializer():
    model = make_model()
    result = MODULE.canonicalize_slice_axes(
        model, node_name='Slice_1933', expected_axes=[0])
    onnx.checker.check_model(model)
    node = next(node for node in model.graph.node
                if node.name == 'Slice_1933')
    assert result['old_axes'] == 'axes'
    assert node.input[3] == 'Slice_1933_axes_static'
    initializer = next(item for item in model.graph.initializer
                       if item.name == node.input[3])
    assert numpy_helper.to_array(initializer).tolist() == [0]
    assert node.input[:3] == ['data', 'starts', 'ends']
    assert node.input[4] == 'steps'


def test_fails_closed_for_unproven_axis():
    with pytest.raises(ValueError, match='not proven'):
        MODULE.canonicalize_slice_axes(
            make_model(axis=1), node_name='Slice_1933', expected_axes=[0])


def test_constantizes_audited_axis_one_without_changing_other_inputs():
    model = make_model(axis=1, node_name='Slice_2222')
    node = next(node for node in model.graph.node
                if node.name == 'Slice_2222')
    original_non_axes = list(node.input[:3]) + [node.input[4]]
    result = MODULE.canonicalize_slice_axes(
        model, node_name='Slice_2222', expected_axes=[1])
    onnx.checker.check_model(model)
    initializer = next(item for item in model.graph.initializer
                       if item.name == node.input[3])
    assert result['value'] == [1]
    assert numpy_helper.to_array(initializer).tolist() == [1]
    assert list(node.input[:3]) + [node.input[4]] == original_non_axes


def test_general_pass_rewrites_only_foldable_nonconstant_axes():
    model = make_model(axis=1, node_name='Slice_foldable')
    report = MODULE.canonicalize_static_slice_axes(model)
    onnx.checker.check_model(model)
    assert [item['node'] for item in report['canonicalized']] == [
        'Slice_foldable']
    assert report['canonicalized'][0]['value'] == [1]
    assert report['skipped'] == []


def test_general_pass_skips_runtime_axes():
    model = make_model(axis=1, node_name='Slice_runtime')
    node = next(node for node in model.graph.node
                if node.name == 'Slice_runtime')
    model.graph.input.append(helper.make_tensor_value_info(
        'runtime_axes', TensorProto.INT64, [1]))
    node.input[3] = 'runtime_axes'
    report = MODULE.canonicalize_static_slice_axes(model)
    assert report['canonicalized'] == []
    assert report['skipped'][0]['node'] == 'Slice_runtime'
    assert node.input[3] == 'runtime_axes'
