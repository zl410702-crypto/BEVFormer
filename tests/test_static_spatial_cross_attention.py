import sys
import importlib.util
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from deployment.bevformer_stateful_onnx_wrapper import (  # noqa: E402
    STATIC_SCA_LINEAR_CHUNK, SpatialCrossAttention,
    disable_tf32_for_validation, fixed_chunk_linear,
    fixed_deformable_linear_override,
    static_spatial_cross_attention_forward,
    static_spatial_cross_attention_override)


class RecordingDeformableAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = None
        self.references = None

    def forward(self, query, key, value, reference_points, spatial_shapes,
                level_start_index):
        del key, value, spatial_shapes, level_start_index
        self.shapes = (tuple(query.shape), tuple(reference_points.shape))
        self.references = reference_points.detach().clone()
        reference_term = reference_points.mean((2, 3), keepdim=False)
        reference_term = reference_term.unsqueeze(-1)
        return query + reference_term


class FakeSpatialAttention(nn.Module):
    def __init__(self, num_cams=2, embed_dims=3):
        super().__init__()
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.deformable_attention = RecordingDeformableAttention()
        self.output_proj = nn.Identity()
        self.dropout = nn.Identity()


def test_static_sca_uses_full_query_capacity_and_safe_invalid_references():
    attention = FakeSpatialAttention()
    query = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    key = torch.zeros(2, 5, 1, 3)
    references = torch.full((2, 1, 4, 2, 2), 0.25)
    mask = torch.zeros(2, 1, 4, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    mask[0, 0, 2, :] = True
    mask[1, 0, 1, 1] = True
    valid_pairs = mask.sum(-1) > 0
    references[~valid_pairs.unsqueeze(-1).unsqueeze(-1).expand_as(
        references)] = float('nan')

    output = static_spatial_cross_attention_forward(
        attention, query, key, key, reference_points_cam=references,
        bev_mask=mask, spatial_shapes=torch.tensor([[1, 5]]),
        level_start_index=torch.tensor([0]))

    assert attention.deformable_attention.shapes == ((2, 4, 3),
                                                      (2, 4, 2, 2))
    fixed_references = attention.deformable_attention.references.reshape(
        1, 2, 4, 2, 2)
    invalid_pairs = ~(mask.sum(-1) > 0).permute(1, 0, 2)
    assert torch.all(fixed_references[invalid_pairs] == 0.5)
    assert torch.isfinite(output).all()
    # Query 3 is invisible in every camera, hence gets residual only.
    assert torch.equal(output[:, 3], query[:, 3])


def test_static_sca_override_is_scoped_and_restores_class_forward():
    attention = object.__new__(SpatialCrossAttention)
    nn.Module.__init__(attention)
    model = nn.Sequential(attention)
    assert 'forward' not in attention.__dict__
    original = attention.forward.__func__

    with static_spatial_cross_attention_override(model):
        assert attention.forward.__func__ is not original

    assert 'forward' not in attention.__dict__
    assert attention.forward.__func__ is original


def test_structural_export_after_validation_failure_is_opt_in():
    script = TOOLS / 'export_bevformer_stateful_onnx.py'
    spec = importlib.util.spec_from_file_location('stateful_export', script)
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    with mock.patch.object(sys, 'argv', ['export']):
        assert export_module.parse_args().allow_validation_failure is False
    with mock.patch.object(
            sys, 'argv', ['export', '--allow-validation-failure']):
        assert export_module.parse_args().allow_validation_failure is True


def test_disable_tf32_context_restores_enabled_state():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with disable_tf32_for_validation():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_disable_tf32_context_restores_after_exception():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            with disable_tf32_for_validation():
                raise RuntimeError('intentional validation failure')
        except RuntimeError as error:
            assert str(error) == 'intentional validation failure'
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_disable_tf32_context_preserves_disabled_state():
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with disable_tf32_for_validation():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_validation_contexts_restore_official_forward_and_tf32_together():
    attention = object.__new__(SpatialCrossAttention)
    nn.Module.__init__(attention)
    model = nn.Sequential(attention)
    original_forward = attention.forward.__func__
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        with disable_tf32_for_validation(), \
                static_spatial_cross_attention_override(model):
            assert attention.forward.__func__ is not original_forward
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert attention.forward.__func__ is original_forward
        assert torch.backends.cuda.matmul.allow_tf32 is original_matmul
        assert torch.backends.cudnn.allow_tf32 is original_cudnn
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_static_sca_override_preserves_default_tf32_during_static_forward():
    attention = object.__new__(SpatialCrossAttention)
    nn.Module.__init__(attention)
    model = nn.Sequential(attention)
    observed = []
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with mock.patch(
                'deployment.bevformer_stateful_onnx_wrapper.'
                'static_spatial_cross_attention_forward',
                side_effect=lambda *args, **kwargs: observed.append((
                    torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32))):
            with static_spatial_cross_attention_override(model):
                attention.forward(None, None, None)
                assert torch.backends.cuda.matmul.allow_tf32 is True
                assert torch.backends.cudnn.allow_tf32 is True
        assert observed == [(True, True)]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


class RecordingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return super().forward(x)


def test_fixed_chunk_2500_uses_four_640_calls_and_trims_padding():
    linear = RecordingLinear(4, 3)
    values = torch.randn(6, 2500, 4)
    output = fixed_chunk_linear(values, linear)
    assert STATIC_SCA_LINEAR_CHUNK == 640
    assert [tuple(item.shape) for item in linear.inputs] == [
        (6, 640, 4), (6, 640, 4), (6, 640, 4), (6, 640, 4)]
    assert torch.count_nonzero(linear.inputs[-1][:, 580:]) == 0
    assert output.shape == (6, 2500, 3)


def test_fixed_chunk_linear_matches_direct_shape_and_values_on_cpu():
    torch.manual_seed(11)
    linear = nn.Linear(5, 7)
    values = torch.randn(2, 19, 5)
    expected = linear(values)
    actual = fixed_chunk_linear(values, linear, chunk_size=8)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_fixed_chunk_padding_is_not_present_in_trimmed_output():
    linear = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    values = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
    output = fixed_chunk_linear(values, linear, chunk_size=6)
    assert output.shape[1] == 10
    assert torch.equal(output, linear(values))


def test_fixed_chunk_rejects_invalid_inputs():
    linear = nn.Linear(2, 1)
    for values, chunk in ((torch.randn(2, 2), 1),
                          (torch.randn(1, 2, 2), 0),
                          (torch.randn(1, 0, 2), 1)):
        try:
            fixed_chunk_linear(values, linear, chunk)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid fixed-chunk input was accepted')


def test_fixed_deformable_linear_override_restores_both_linears():
    deformable = nn.Module()
    deformable.sampling_offsets = nn.Linear(3, 4)
    deformable.attention_weights = nn.Linear(3, 5)
    originals = (deformable.sampling_offsets.forward.__func__,
                 deformable.attention_weights.forward.__func__)
    with fixed_deformable_linear_override(deformable, chunk_size=2):
        assert deformable.sampling_offsets.forward.__func__ is not originals[0]
        assert deformable.attention_weights.forward.__func__ is not originals[1]
    assert deformable.sampling_offsets.forward.__func__ is originals[0]
    assert deformable.attention_weights.forward.__func__ is originals[1]


def test_fixed_deformable_linear_override_restores_after_exception():
    deformable = nn.Module()
    deformable.sampling_offsets = nn.Linear(3, 4)
    deformable.attention_weights = nn.Linear(3, 5)
    originals = (deformable.sampling_offsets.forward.__func__,
                 deformable.attention_weights.forward.__func__)
    try:
        with fixed_deformable_linear_override(deformable, chunk_size=2):
            raise RuntimeError('intentional')
    except RuntimeError:
        pass
    assert deformable.sampling_offsets.forward.__func__ is originals[0]
    assert deformable.attention_weights.forward.__func__ is originals[1]
