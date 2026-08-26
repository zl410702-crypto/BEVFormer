import importlib.util
from pathlib import Path
import sys

import onnx
import pytest
from onnx import TensorProto, helper


MODULE_PATH = (Path(__file__).parents[1] /
               'tools/create_bevformer_host_trig_debug_onnx.py')
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location('host_trig_debug', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_model(cos_consumers=4):
    angle = helper.make_tensor_value_info('angle', TensorProto.FLOAT, [])
    nodes = [helper.make_node('Cos', ['angle'], ['1024'], name='Cos_322'),
             helper.make_node('Sin', ['angle'], ['1025'], name='Sin_323')]
    outputs = []
    for index in range(cos_consumers):
        name = 'cos_out_{}'.format(index)
        nodes.append(helper.make_node('Identity', ['1024'], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))
    for index in range(4):
        name = 'sin_out_{}'.format(index)
        nodes.append(helper.make_node('Identity', ['1025'], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))
    return helper.make_model(helper.make_graph(nodes, 'trig', [angle], outputs))


def test_replaces_only_audited_trig_consumers():
    model = make_model()
    report = MODULE.replace_alignment_trig(model)
    assert report == {'rotation_cos': 4, 'rotation_sin': 4}
    assert {'rotation_cos', 'rotation_sin'} <= {
        value.name for value in model.graph.input}
    assert all('1024' not in node.input and '1025' not in node.input
               for node in model.graph.node
               if node.name not in ('Cos_322', 'Sin_323'))


def test_fails_closed_when_consumer_count_changes():
    with pytest.raises(RuntimeError, match='consumer count'):
        MODULE.replace_alignment_trig(make_model(cos_consumers=3))


def test_arithmetic_boundary_contracts_are_static():
    assert MODULE.ARITHMETIC_BOUNDARIES == {
        '1059': [1, 2, 3],
        '1352': [1, 2500, 3],
        '1353': [1, 3, 2],
        '1355': [1, 3, 2],
    }
