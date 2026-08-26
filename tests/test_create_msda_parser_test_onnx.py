import importlib.util
from collections import Counter
from pathlib import Path

import onnx


SCRIPT = Path(__file__).parents[1] / 'tools/create_msda_parser_test_onnx.py'
SPEC = importlib.util.spec_from_file_location('create_msda_test', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_create_has_exact_12_node_int32_contract(tmp_path):
    path = tmp_path / 'test.onnx'
    model = MODULE.create(path)
    onnx.checker.check_model(model)
    assert path.is_file()
    assert len(model.graph.node) == 12
    assert Counter(node.name.split('_')[0] for node in model.graph.node) == {
        'temporal': 3, 'spatial': 3, 'decoder': 6}
    initializers = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        assert node.domain == 'bevformer'
        assert node.op_type == 'MSDeformableAttention'
        assert len(node.input) == 5
        assert initializers[node.input[1]].data_type == onnx.TensorProto.INT32
        assert initializers[node.input[2]].data_type == onnx.TensorProto.INT32
