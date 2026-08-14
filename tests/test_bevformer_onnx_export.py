import importlib.util
import sys
from pathlib import Path
from unittest import mock

import torch


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from deployment import bevformer_onnx_wrapper as module  # noqa: E402


class FakeGraph:
    def __init__(self):
        self.call = None

    def op(self, name, *inputs, **attributes):
        self.call = (name, inputs, attributes)
        return 'custom-output'


def test_custom_symbolic_has_stable_schema_and_five_tensor_inputs():
    graph = FakeGraph()
    inputs = [object() for _ in range(5)]
    output = module.ms_deformable_attention_symbolic(
        graph, *inputs, 32)
    assert output == 'custom-output'
    assert graph.call[0] == 'bevformer::MSDeformableAttention'
    assert graph.call[1] == tuple(inputs)
    assert graph.call[2] == {'im2col_step_i': 32}


def test_export_overrides_are_restored():
    original_rotate = module.transformer_module.rotate
    original_maximum = torch.maximum
    original_nan_to_num = torch.nan_to_num
    function_class = module.MultiScaleDeformableAttnFunction_fp32
    assert not hasattr(function_class, 'symbolic')
    with module.deployment_export_overrides():
        assert module.transformer_module.rotate is module.bev_rotate_nearest
        assert torch.maximum is module.maximum_opset13
        assert torch.nan_to_num is module.nan_to_num_opset13
        assert hasattr(function_class, 'symbolic')
    assert module.transformer_module.rotate is original_rotate
    assert torch.maximum is original_maximum
    assert torch.nan_to_num is original_nan_to_num
    assert not hasattr(function_class, 'symbolic')


def test_export_overrides_use_exporter_safe_static_can_bus_gate():
    transformer = object.__new__(module.transformer_module.PerceptionTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.use_can_bus = True
    model = torch.nn.Sequential(transformer)

    with module.deployment_export_overrides(model):
        assert transformer.use_can_bus == 1.0
        assert type(transformer.use_can_bus) is float

    assert transformer.use_can_bus is True


def test_maximum_opset13_matches_projection_depth_maximum():
    generator = torch.Generator().manual_seed(0)
    input_a = torch.randn(
        4, 1, 6, 2500, 1, generator=generator, dtype=torch.float32) * 70
    input_a[0, 0, 0, 0, 0] = float('nan')
    input_b = torch.ones_like(input_a) * 1e-5

    expected = module.REFERENCE_MAXIMUM(input_a, input_b)
    actual = module.maximum_opset13(input_a, input_b)
    finite = torch.isfinite(expected)
    absolute_error = (expected[finite] - actual[finite]).abs()

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert absolute_error.max().item() == 0.0
    assert absolute_error.mean().item() == 0.0


def test_nan_to_num_opset13_is_identity_only_for_boolean_mask():
    bev_mask = torch.tensor([True, False], dtype=torch.bool)
    assert module.nan_to_num_opset13(bev_mask) is bev_mask

    values = torch.tensor([float('nan'), float('inf'), -float('inf'), 2.0])
    expected = module.REFERENCE_NAN_TO_NUM(values)
    actual = module.nan_to_num_opset13(values)
    assert torch.equal(actual, expected)


def test_wrapper_stops_at_raw_head_outputs():
    fake_model = mock.MagicMock()
    fake_model.extract_img_feat.return_value = ['features']
    expected = {
        'all_cls_scores': torch.ones(6, 1, 900, 10),
        'all_bbox_preds': torch.ones(6, 1, 900, 10) * 2,
        'bev_embed': torch.ones(2500, 1, 256) * 3,
    }
    fake_model.pts_bbox_head.return_value = expected
    wrapper = module.BEVFormerONNXWrapper(fake_model, {'sample_idx': 'token'})
    outputs = wrapper(torch.ones(1, 6, 3, 4, 5))
    assert outputs == (expected['all_cls_scores'], expected['all_bbox_preds'],
                       expected['bev_embed'])
    fake_model.pts_bbox_head.assert_called_once_with(
        ['features'], [{'sample_idx': 'token'}], prev_bev=None)


def test_environment_blockers_are_reported_without_mutation():
    script = TOOLS / 'export_bevformer_onnx.py'
    spec = importlib.util.spec_from_file_location('export_bevformer_onnx', script)
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    with mock.patch.object(export_module.importlib.util, 'find_spec',
                           return_value=None), mock.patch.object(
                               export_module.torch.cuda, 'is_available',
                               return_value=False):
        blockers = export_module.environment_blockers()
    assert [item['op'] for item in blockers] == [
        'onnx package', 'CUDA runtime']


def test_pytorch_191_constant_folding_is_explicitly_disabled():
    script = TOOLS / 'export_bevformer_onnx.py'
    spec = importlib.util.spec_from_file_location('export_folding_setting', script)
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    assert export_module.ONNX_DO_CONSTANT_FOLDING is False


def test_onnxruntime_validation_is_not_run_when_package_is_missing():
    script = TOOLS / 'export_bevformer_onnx.py'
    spec = importlib.util.spec_from_file_location('export_bevformer_onnx_ort',
                                                  script)
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    with mock.patch.object(export_module.importlib.util, 'find_spec',
                           return_value=None):
        result = export_module.run_onnxruntime(
            Path('unused.onnx'), torch.ones(1), (torch.ones(1),) * 3)
    assert result == {'status': 'NOT RUN',
                      'reason': 'onnxruntime is not installed'}
