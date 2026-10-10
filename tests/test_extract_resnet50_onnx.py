import importlib.util
from pathlib import Path

import onnx
import pytest


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'tools' / 'deployment' /
               'extract_resnet50_onnx.py')
SPEC = importlib.util.spec_from_file_location('extract_resnet50_onnx',
                                               MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _value(name, shape):
    return onnx.helper.make_tensor_value_info(
        name, onnx.TensorProto.FLOAT, shape)


def _model(weight_name=MODULE.FPN_LATERAL_WEIGHT):
    relu = onnx.helper.make_node('Relu', ['backbone_raw'], ['c5'],
                                 name='backbone_c5_relu')
    lateral = onnx.helper.make_node(
        'Conv', ['c5', weight_name], ['fpn'], name='fpn_lateral')
    graph = onnx.helper.make_graph(
        [relu, lateral], 'boundary_test',
        [_value('images', MODULE.EXPECTED_INPUT_SHAPE),
         _value('backbone_raw', MODULE.EXPECTED_C5_SHAPE)],
        [_value('fpn', [6, 256, 15, 25])],
        value_info=[_value('c5', MODULE.EXPECTED_C5_SHAPE)])
    return onnx.helper.make_model(graph)


def test_finds_c5_from_fpn_consumer():
    assert MODULE.find_resnet50_c5(_model()) == (
        'c5', 'backbone_c5_relu', 'fpn_lateral')


def test_rejects_graph_without_expected_fpn_boundary():
    with pytest.raises(ValueError, match='expected exactly one FPN lateral'):
        MODULE.find_resnet50_c5(_model('different.weight'))


def test_rejects_shape_drift():
    model = _model()
    model.graph.input[0].type.tensor_type.shape.dim[4].dim_value = 801
    with pytest.raises(ValueError, match='unexpected images shape'):
        MODULE.find_resnet50_c5(model)
