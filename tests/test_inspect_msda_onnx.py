import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest


SCRIPT = Path(__file__).parents[1] / 'tools/inspect_msda_onnx.py'
SPEC = importlib.util.spec_from_file_location('inspect_msda_onnx', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('scope,role,layer', [
    ('model.transformer.encoder.layers.2.attentions.0.'
     'attention_weights.weight', 'Temporal Self Attention', 2),
    ('model.transformer.encoder.layers.1.attentions.1.deformable_attention.'
     'attention_weights.weight', 'Spatial Cross Attention', 1),
    ('model.transformer.decoder.layers.5.attentions.1.'
     'attention_weights.weight', 'Detection Decoder Cross Attention', 5),
])
def test_classify_scope(scope, role, layer):
    assert MODULE.classify_scope(scope) == (role, layer)


def test_role_contract_derives_level_and_spatial_size():
    contract = MODULE.role_contract(
        'Spatial Cross Attention', np.array([[15, 25]], dtype=np.int64))
    assert contract == {
        'num_heads': 8, 'num_levels': 1, 'head_dim': 32,
        'embed_dims': 256, 'batch': 6, 'num_query': 2500,
        'num_points': 8, 'linear_queue_factor': 1, 'spatial_size': 375,
    }


def test_unknown_scope_and_role_fail_closed():
    with pytest.raises(ValueError):
        MODULE.classify_scope('unknown.attention_weights.weight')
    with pytest.raises(ValueError):
        MODULE.role_contract('unknown', np.array([[1, 1]], dtype=np.int64))


def test_tensor_from_constant_follows_explicit_int32_cast():
    constant = onnx.helper.make_node(
        'Constant', [], ['shape64'], value=onnx.numpy_helper.from_array(
            np.array([[50, 50]], dtype=np.int64)))
    cast = onnx.helper.make_node(
        'Cast', ['shape64'], ['shape32'], to=onnx.TensorProto.INT32)
    value = MODULE.tensor_from_constant(
        cast, {'shape64': constant, 'shape32': cast})
    assert value.dtype == np.int32
    assert value.tolist() == [[50, 50]]


def test_required_int32_contract_fails_closed():
    MODULE.validate_index_dtypes('node', 'INT32', 'INT32', True)
    with pytest.raises(ValueError, match='must be INT32'):
        MODULE.validate_index_dtypes('node', 'INT64', 'INT32', True)
    with pytest.raises(ValueError, match='must be INT32'):
        MODULE.validate_index_dtypes('node', 'INT32', 'INT64', True)
