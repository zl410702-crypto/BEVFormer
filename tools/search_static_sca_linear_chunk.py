#!/usr/bin/env python
"""Search fixed query chunks that reproduce compact-SCA Linear numerics.

This is an isolated deployment experiment.  It does not patch the model or
change the official SpatialCrossAttention implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from debug_static_sca_numerics import (ATOL, INPUT_ROOT, RTOL, TOKEN,
                                       compare_arrays, load_inputs)
from deployment.bevformer_stateful_onnx_wrapper import SpatialCrossAttention
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               prepare_golden_sample)


OUTPUT = Path('artifacts/static_sca_linear_chunk_search.json')
DEFAULT_CHUNKS = (1, 8, 16, 32, 64, 96, 128, 160, 192, 224, 250, 256,
                  300, 320, 384, 400, 448, 480, 512, 600, 604, 625, 640,
                  768, 800, 1024, 1250, 2500)


def fixed_chunk_linear(values, linear, chunk_size):
    """Apply a Linear in fixed Nq chunks, padding only the final chunk.

    Input and output retain ``[batch_camera, num_query, channels]``.  Every
    Linear invocation has the static shape ``[batch_camera, chunk_size, C]``.
    """
    if values.ndim != 3 or chunk_size <= 0:
        raise ValueError('expected rank-3 values and a positive chunk size')
    pieces = []
    num_query = values.shape[1]
    for start in range(0, num_query, chunk_size):
        piece = values[:, start:start + chunk_size]
        valid = piece.shape[1]
        if valid < chunk_size:
            piece = F.pad(piece, (0, 0, 0, chunk_size - valid))
        pieces.append(linear(piece)[:, :valid])
    return torch.cat(pieces, dim=1)


def compact_queries(query, bev_mask):
    indexes = [camera_mask[0].sum(-1).nonzero().squeeze(-1)
               for camera_mask in bev_mask]
    max_len = max(len(index) for index in indexes)
    result = query.new_zeros(len(indexes), max_len, query.shape[-1])
    for camera, index in enumerate(indexes):
        result[camera, :len(index)] = query[0, index]
    return result, indexes


def gather_valid(values, indexes):
    return torch.cat([values[camera, index]
                      for camera, index in enumerate(indexes)])


def gather_compact(values, indexes):
    return torch.cat([values[camera, :len(index)]
                      for camera, index in enumerate(indexes)])


def load_layer_inputs(frame, layer):
    root = INPUT_ROOT.parents[2] / 'frame_{:03d}/spatial/layer_{}'.format(
        frame, layer)
    names = ('query', 'bev_mask')
    inputs = {name: torch.from_numpy(np.load(
        str(root / '{}.npy'.format(name)), allow_pickle=False)).cuda()
              for name in names}
    return root, inputs


def linear_outputs(module, query):
    batch_camera, num_query, _ = query.shape
    offsets = module.sampling_offsets(query).view(
        batch_camera, num_query, module.num_heads, module.num_levels,
        module.num_points, 2)
    logits = module.attention_weights(query).view(
        batch_camera, num_query, module.num_heads,
        module.num_levels * module.num_points)
    weights = logits.softmax(-1).view(
        batch_camera, num_query, module.num_heads, module.num_levels,
        module.num_points)
    return offsets, logits, weights


def chunked_outputs(module, query, chunk_size):
    batch_camera, num_query, _ = query.shape
    offsets = fixed_chunk_linear(
        query, module.sampling_offsets, chunk_size).view(
            batch_camera, num_query, module.num_heads, module.num_levels,
            module.num_points, 2)
    logits = fixed_chunk_linear(
        query, module.attention_weights, chunk_size).view(
            batch_camera, num_query, module.num_heads,
            module.num_levels * module.num_points)
    weights = logits.softmax(-1).view(
        batch_camera, num_query, module.num_heads, module.num_levels,
        module.num_points)
    return offsets, logits, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', nargs='*', type=int,
                        default=list(DEFAULT_CHUNKS))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--validate-chunk', type=int, default=None,
                        help='also validate this fixed chunk on all 3x3 SCA')
    args = parser.parse_args()

    context = prepare_golden_sample(
        TOKEN, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, 0)
    model = context['model'].module
    spatial = next(module for module in model.modules()
                   if isinstance(module, SpatialCrossAttention)).eval()
    deform = spatial.deformable_attention
    inputs = load_inputs()
    compact, indexes = compact_queries(inputs['query'], inputs['bev_mask'])
    fixed = inputs['query'][:, None].expand(
        1, spatial.num_cams, inputs['query'].shape[1],
        inputs['query'].shape[2]).reshape(
            spatial.num_cams, inputs['query'].shape[1],
            inputs['query'].shape[2])

    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        # Saved Golden was generated with the default CUDA TF32 mode.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.no_grad():
            reference = linear_outputs(deform, compact)
            golden_offsets = torch.from_numpy(np.load(
                str(INPUT_ROOT / 'deformable/sampling_offsets.npy'),
                allow_pickle=False)).cuda()
            golden_weights = torch.from_numpy(np.load(
                str(INPUT_ROOT / 'deformable/attention_weights.npy'),
                allow_pickle=False)).cuda()
            reference_checks = {
                'sampling_offsets': compare_arrays(golden_offsets,
                                                   reference[0]),
                'attention_weights': compare_arrays(golden_weights,
                                                    reference[2]),
            }
            candidates = []
            for chunk_size in args.chunks:
                candidate = chunked_outputs(deform, fixed, chunk_size)
                comparisons = {}
                for name, ref_value, candidate_value in zip(
                        ('sampling_offsets', 'attention_logits',
                         'attention_weights'), reference, candidate):
                    comparisons[name] = compare_arrays(
                        gather_compact(ref_value, indexes),
                        gather_valid(candidate_value, indexes))
                candidates.append({
                    'chunk_size': chunk_size,
                    'gemm_rows_per_call': spatial.num_cams * chunk_size,
                    'chunk_count': int((fixed.shape[1] + chunk_size - 1) /
                                       chunk_size),
                    'last_chunk_valid_queries': int(
                        fixed.shape[1] % chunk_size or chunk_size),
                    'comparisons': comparisons,
                    'bitwise_exact': all(item['exact']
                                         for item in comparisons.values()),
                    'strict_allclose': all(item['allclose']
                                           for item in comparisons.values()),
                })
                torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn

    exact = [item['chunk_size'] for item in candidates
             if item['bitwise_exact']]
    strict = [item['chunk_size'] for item in candidates
              if item['strict_allclose']]
    cross_frame_validation = []
    if args.validate_chunk is not None:
        validation_matmul = torch.backends.cuda.matmul.allow_tf32
        validation_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with torch.no_grad():
                for frame in range(3):
                    for layer in range(3):
                        layer_root, layer_inputs = load_layer_inputs(
                            frame, layer)
                        module = [item for item in model.modules()
                                  if isinstance(item,
                                                SpatialCrossAttention)][layer]
                        compact_query, layer_indexes = compact_queries(
                            layer_inputs['query'], layer_inputs['bev_mask'])
                        fixed_query = layer_inputs['query'][:, None].expand(
                            1, module.num_cams,
                            layer_inputs['query'].shape[1],
                            layer_inputs['query'].shape[2]).reshape(
                            module.num_cams, layer_inputs['query'].shape[1],
                            layer_inputs['query'].shape[2])
                        compact_result = linear_outputs(
                            module.deformable_attention, compact_query)
                        fixed_result = chunked_outputs(
                            module.deformable_attention, fixed_query,
                            args.validate_chunk)
                        names = ('sampling_offsets', 'attention_logits',
                                 'attention_weights')
                        comparisons = {
                            name: compare_arrays(
                                gather_compact(reference, layer_indexes),
                                gather_valid(actual, layer_indexes))
                            for name, reference, actual in zip(
                                names, compact_result, fixed_result)
                        }
                        golden_checks = {
                            'sampling_offsets': compare_arrays(
                                np.load(str(layer_root /
                                            'deformable/sampling_offsets.npy'),
                                        allow_pickle=False), compact_result[0]),
                            'attention_weights': compare_arrays(
                                np.load(str(layer_root /
                                            'deformable/attention_weights.npy'),
                                        allow_pickle=False), compact_result[2]),
                        }
                        cross_frame_validation.append({
                            'frame': frame,
                            'encoder_layer': layer,
                            'visibility_counts': [len(index)
                                                  for index in layer_indexes],
                            'comparisons': comparisons,
                            'compact_vs_saved_golden': golden_checks,
                            'bitwise_exact': all(
                                item['exact']
                                for item in comparisons.values()),
                        })
                        torch.cuda.synchronize()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = validation_matmul
            torch.backends.cudnn.allow_tf32 = validation_cudnn

    report = {
        'scope': 'Frame 0 / Encoder Layer 0 / Spatial Cross Attention',
        'math_mode': 'CUDA matmul TF32 enabled (saved-Golden mode)',
        'input_shape': list(inputs['query'].shape),
        'fixed_per_camera_shape': [1, spatial.num_cams,
                                   inputs['query'].shape[1],
                                   inputs['query'].shape[2]],
        'visibility_counts': [len(index) for index in indexes],
        'compact_max_len': compact.shape[1],
        'rtol': RTOL,
        'atol': ATOL,
        'official_compact_vs_saved_golden': reference_checks,
        'candidates': candidates,
        'bitwise_exact_chunk_sizes': exact,
        'strict_allclose_chunk_sizes': strict,
        # Prefer a conventional 64-aligned deployment capacity over an
        # observed visibility length such as 604.
        'recommended_chunk_size': (640 if 640 in exact else
                                   (exact[0] if exact else None)),
        'cross_frame_validation_chunk_size': args.validate_chunk,
        'cross_frame_validation': cross_frame_validation,
        'runtime_visibility_specialization': False,
        'conclusion': (
            'A fixed chunk reproduces compact-Golden Linear numerics.'
            if exact else
            'No tested fixed chunk reproduces compact-Golden Linear numerics.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Output: {}'.format(args.output))


if __name__ == '__main__':
    main()
