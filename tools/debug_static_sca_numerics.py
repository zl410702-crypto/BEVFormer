#!/usr/bin/env python
"""Locate the first Frame-0/Layer-0 static-SCA numerical divergence."""

import json
from pathlib import Path

import numpy as np
import torch

from deployment.bevformer_stateful_onnx_wrapper import (
    SpatialCrossAttention, fixed_deformable_linear_override)
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               prepare_golden_sample)
from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32)


TOKEN = '3e8750f331d7499e9b5123e9eb70f2e2'
ROOT = Path('golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3')
INPUT_ROOT = ROOT / 'frame_000/spatial/layer_0'
OUTPUT = Path('artifacts/static_sca_first_divergence.json')
RTOL = 1e-5
ATOL = 1e-8


def compare_arrays(reference, static):
    reference = reference.detach().cpu().numpy() \
        if isinstance(reference, torch.Tensor) else np.asarray(reference)
    static = static.detach().cpu().numpy() \
        if isinstance(static, torch.Tensor) else np.asarray(static)
    if reference.shape != static.shape:
        raise ValueError('shape mismatch: {} != {}'.format(
            reference.shape, static.shape))
    ref64 = reference.astype(np.float64)
    static64 = static.astype(np.float64)
    difference = np.abs(ref64 - static64)
    unequal = ~(reference == static)
    unequal &= ~(np.isnan(reference) & np.isnan(static))
    mismatch = np.argwhere(unequal)
    result = {
        'shape': list(reference.shape),
        'max_abs': float(difference.max()) if difference.size else 0.0,
        'mean_abs': float(difference.mean()) if difference.size else 0.0,
        'max_rel': float((difference / np.maximum(
            np.abs(ref64), 1e-12)).max()) if difference.size else 0.0,
        'allclose': bool(np.allclose(
            reference, static, rtol=RTOL, atol=ATOL, equal_nan=True)),
        'exact': not bool(mismatch.size),
        'first_mismatching_index': None,
        'reference_value': None,
        'static_value': None,
    }
    if mismatch.size:
        index = tuple(int(item) for item in mismatch[0])
        result['first_mismatching_index'] = list(index)
        result['reference_value'] = float(reference[index])
        result['static_value'] = float(static[index])
    return result


def deformable_forward(module, query, value, references, spatial_shapes,
                       level_start_index):
    batch, num_query, _ = query.shape
    _, num_value, _ = value.shape
    projected_value = module.value_proj(value).view(
        batch, num_value, module.num_heads, -1)
    offsets = module.sampling_offsets(query).view(
        batch, num_query, module.num_heads, module.num_levels,
        module.num_points, 2)
    logits = module.attention_weights(query).view(
        batch, num_query, module.num_heads,
        module.num_levels * module.num_points)
    weights = logits.softmax(-1).view(
        batch, num_query, module.num_heads, module.num_levels,
        module.num_points)
    normalizer = torch.stack(
        [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    anchors = references.shape[2]
    normalized_offsets = offsets / normalizer[
        None, None, None, :, None, :]
    normalized_offsets = normalized_offsets.view(
        batch, num_query, module.num_heads, module.num_levels,
        module.num_points // anchors, anchors, 2)
    locations = references[:, :, None, None, None, :, :] + normalized_offsets
    locations = locations.view(
        batch, num_query, module.num_heads, module.num_levels,
        module.num_points, 2)
    cuda_output = MultiScaleDeformableAttnFunction_fp32.apply(
        projected_value, spatial_shapes, level_start_index,
        locations, weights, module.im2col_step)
    return {
        'sampling_offsets': offsets,
        'attention_logits': logits,
        'attention_weights': weights,
        'sampling_locations': locations,
        'value_after_value_proj': projected_value,
        'ms_deformable_attention_cuda_output': cuda_output,
    }


def compact_path(spatial, inputs):
    query = inputs['query']
    references = inputs['reference_points_cam']
    mask = inputs['bev_mask']
    indexes = [camera_mask[0].sum(-1).nonzero().squeeze(-1)
               for camera_mask in mask]
    max_len = max(len(index) for index in indexes)
    batch, num_query, channels = query.shape
    depth = references.shape[3]
    queries = query.new_zeros(batch, spatial.num_cams, max_len, channels)
    compact_references = references.new_zeros(
        batch, spatial.num_cams, max_len, depth, 2)
    for camera, camera_references in enumerate(references):
        index = indexes[camera]
        queries[0, camera, :len(index)] = query[0, index]
        compact_references[0, camera, :len(index)] = \
            camera_references[0, index]
    flattened_value = inputs['value'].permute(2, 0, 1, 3).reshape(
        batch * spatial.num_cams, inputs['value'].shape[1], channels)
    deform = deformable_forward(
        spatial.deformable_attention,
        queries.reshape(batch * spatial.num_cams, max_len, channels),
        flattened_value,
        compact_references.reshape(
            batch * spatial.num_cams, max_len, depth, 2),
        inputs['spatial_shapes'], inputs['level_start_index'])
    compact_output = deform['ms_deformable_attention_cuda_output'].reshape(
        batch, spatial.num_cams, max_len, channels)
    per_camera = torch.zeros(
        batch, spatial.num_cams, num_query, channels,
        device=query.device, dtype=query.dtype)
    slots_sum = torch.zeros_like(query)
    for camera, index in enumerate(indexes):
        visible = compact_output[:, camera, :len(index)]
        per_camera[:, camera, index] = visible
        slots_sum[:, index] += visible
    valid_mask = (mask.sum(-1) > 0).permute(1, 0, 2)
    count = torch.clamp(valid_mask.permute(0, 2, 1).sum(-1), min=1.0)
    slots = slots_sum / count[..., None]
    projected = spatial.output_proj(slots)
    final = spatial.dropout(projected) + query
    return {
        'input_query': query,
        'query_after_rebatch': queries.reshape(
            batch * spatial.num_cams, max_len, channels),
        **deform,
        'per_camera_output': per_camera,
        'slots_after_camera_average': slots,
        'output_proj_output': projected,
        'final_sca_output': final,
        'indexes': indexes,
        'valid_mask': valid_mask,
    }


def static_path(spatial, inputs):
    query = inputs['query']
    references = inputs['reference_points_cam'].permute(1, 0, 2, 3, 4)
    valid_mask = (inputs['bev_mask'].sum(-1) > 0).permute(1, 0, 2)
    references = torch.where(
        valid_mask[..., None, None], references,
        references.new_tensor(0.5))
    batch, num_query, channels = query.shape
    depth = references.shape[3]
    queries = query[:, None].expand(
        batch, spatial.num_cams, num_query, channels)
    flattened_value = inputs['value'].permute(2, 0, 1, 3).reshape(
        batch * spatial.num_cams, inputs['value'].shape[1], channels)
    with fixed_deformable_linear_override(spatial.deformable_attention):
        deform = deformable_forward(
            spatial.deformable_attention,
            queries.reshape(batch * spatial.num_cams, num_query, channels),
            flattened_value,
            references.reshape(
                batch * spatial.num_cams, num_query, depth, 2),
            inputs['spatial_shapes'], inputs['level_start_index'])
    per_camera_raw = deform['ms_deformable_attention_cuda_output'].reshape(
        batch, spatial.num_cams, num_query, channels)
    per_camera = per_camera_raw * valid_mask[..., None].to(query.dtype)
    slots_sum = torch.zeros_like(query)
    for camera in range(spatial.num_cams):
        slots_sum = slots_sum + per_camera[:, camera]
    count = torch.clamp(
        valid_mask.to(query.dtype).sum(1), min=1.0)
    slots = slots_sum / count[..., None]
    projected = spatial.output_proj(slots)
    final = spatial.dropout(projected) + query
    return {
        'input_query': query,
        'query_after_rebatch': queries.reshape(
            batch * spatial.num_cams, num_query, channels),
        **deform,
        'per_camera_output': per_camera,
        'slots_after_camera_average': slots,
        'output_proj_output': projected,
        'final_sca_output': final,
        'valid_mask': valid_mask,
    }


QUERY_TENSORS = {
    'query_after_rebatch', 'sampling_offsets', 'attention_logits',
    'attention_weights', 'sampling_locations',
    'ms_deformable_attention_cuda_output'}


def select_same_valid_pairs(compact, static, name):
    if name in QUERY_TENSORS:
        compact_values = []
        static_values = []
        for camera, index in enumerate(compact['indexes']):
            length = len(index)
            compact_values.append(compact[name][camera, :length])
            static_values.append(static[name][camera, index])
        return torch.cat(compact_values), torch.cat(static_values)
    if name == 'per_camera_output':
        compact_values = []
        static_values = []
        for camera, index in enumerate(compact['indexes']):
            compact_values.append(compact[name][0, camera, index])
            static_values.append(static[name][0, camera, index])
        return torch.cat(compact_values), torch.cat(static_values)
    return compact[name], static[name]


ORDER = [
    'input_query', 'query_after_rebatch', 'sampling_offsets',
    'attention_logits', 'attention_weights', 'sampling_locations',
    'value_after_value_proj', 'ms_deformable_attention_cuda_output',
    'per_camera_output', 'slots_after_camera_average',
    'output_proj_output', 'final_sca_output']


def run_mode(spatial, inputs, allow_matmul_tf32, allow_cudnn_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
    torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
    with torch.no_grad():
        compact = compact_path(spatial, inputs)
        static = static_path(spatial, inputs)
        torch.cuda.synchronize()
    comparisons = {}
    first_divergence = None
    for name in ORDER:
        reference, actual = select_same_valid_pairs(
            compact, static, name)
        comparisons[name] = compare_arrays(reference, actual)
        mismatch = comparisons[name]['first_mismatching_index']
        if mismatch is not None and (name in QUERY_TENSORS or
                                     name == 'per_camera_output'):
            flat_query = mismatch[0]
            offset = 0
            for camera, index in enumerate(compact['indexes']):
                if flat_query < offset + len(index):
                    local = flat_query - offset
                    comparisons[name]['first_mismatching_camera'] = camera
                    comparisons[name]['first_mismatching_original_query'] = \
                        int(index[local])
                    break
                offset += len(index)
        if first_divergence is None and not comparisons[name]['exact']:
            first_divergence = name
    golden_paths = {
        'query_after_rebatch': 'deformable/query.npy',
        'sampling_offsets': 'deformable/sampling_offsets.npy',
        'attention_weights': 'deformable/attention_weights.npy',
        'sampling_locations': 'deformable/sampling_locations.npy',
        'ms_deformable_attention_cuda_output': 'deformable/output.npy',
        'final_sca_output': 'output.npy',
    }
    golden_comparisons = {}
    golden_first_divergence = None
    for name, relative in golden_paths.items():
        golden = np.load(str(INPUT_ROOT / relative), allow_pickle=False)
        golden_comparisons[name] = compare_arrays(golden, compact[name])
        if (golden_first_divergence is None and
                not golden_comparisons[name]['exact']):
            golden_first_divergence = name
    return {
        'tf32': {
            'cuda_matmul_allow_tf32': allow_matmul_tf32,
            'cudnn_allow_tf32': allow_cudnn_tf32,
        },
        'valid_query_count': [len(index) for index in compact['indexes']],
        'tensor_comparisons': comparisons,
        'first_divergence': first_divergence,
        'official_compact_vs_saved_golden': golden_comparisons,
        'official_compact_vs_saved_golden_first_divergence':
            golden_first_divergence,
    }


def load_inputs():
    names = ('query', 'key', 'value', 'reference_points_cam', 'bev_mask',
             'spatial_shapes', 'level_start_index')
    return {name: torch.from_numpy(np.load(
        str(INPUT_ROOT / '{}.npy'.format(name)), allow_pickle=False)).cuda()
        for name in names}


def main():
    context = prepare_golden_sample(
        TOKEN, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, 0)
    model = context['model'].module
    spatial_modules = [module for module in model.modules()
                       if isinstance(module, SpatialCrossAttention)]
    spatial = spatial_modules[0].eval()
    inputs = load_inputs()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        default = run_mode(spatial, inputs, original_matmul, original_cudnn)
        disabled = run_mode(spatial, inputs, False, False)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
    cuda_inputs = ('sampling_locations', 'attention_weights',
                   'value_after_value_proj')
    cuda_inputs_exact = all(
        default['tensor_comparisons'][name]['exact'] for name in cuda_inputs)
    disabled_all_exact = all(
        item['exact'] for item in
        disabled['tensor_comparisons'].values())
    default_first = default['first_divergence']
    report = {
        'frame': 0,
        'encoder_layer': 0,
        'camera': 'all six cameras; valid pairs gathered by official indexes',
        'valid_query_count': default['valid_query_count'],
        'default_cuda': default,
        'tf32_disabled': disabled,
        'root_cause_evidence': {
            'cuda_input_value_locations_weights_exact': cuda_inputs_exact,
            'default_first_divergence': default_first,
            'sampling_offsets_exact_default': default[
                'tensor_comparisons']['sampling_offsets']['exact'],
            'attention_logits_exact_default': default[
                'tensor_comparisons']['attention_logits']['exact'],
            'all_tensors_exact_with_tf32_disabled': disabled_all_exact,
            'default_compact_matches_saved_golden':
                default[
                    'official_compact_vs_saved_golden_first_divergence'] is None,
            'tf32_disabled_compact_vs_saved_golden_first_divergence':
                disabled[
                    'official_compact_vs_saved_golden_first_divergence'],
            'im2col_step_configured': spatial.deformable_attention.im2col_step,
            'cuda_batch': 6,
            'effective_im2col_step': 6,
        },
        'conclusion': (
            'Fixed chunk=640 under the saved-Golden default TF32 mode makes '
            'all compact/static checkpoints bitwise exact; no numerical '
            'divergence remains.' if default_first is None else
            'FIRST DIVERGENCE = {}. ROOT CAUSE NOT YET PROVEN.'.format(
                default_first)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Output: {}'.format(OUTPUT))


if __name__ == '__main__':
    main()
