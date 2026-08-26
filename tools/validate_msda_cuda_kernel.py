#!/usr/bin/env python
"""Compare the correctness-first MSDA CUDA kernel with MMCV 1.4.0."""

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


REFERENCE = Path(
    'projects/mmdet3d_plugin/bevformer/modules/'
    'multi_scale_deformable_attn_function.py')
CASES = {
    'temporal_l0': (2, 2500, 50, 50, 4),
    'spatial_l0': (6, 2500, 15, 25, 8),
    'decoder_l0': (1, 900, 50, 50, 4),
}
HEADS = 8
HEAD_DIM = 32


def load_reference():
    spec = importlib.util.spec_from_file_location('msda_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(reference, actual):
    ref = reference.detach().float().cpu().numpy().astype(np.float64)
    got = actual.detach().float().cpu().numpy().astype(np.float64)
    diff = np.abs(ref - got)
    relative = diff / np.maximum(np.abs(ref), 1e-8)
    cosine = float(np.dot(ref.ravel(), got.ravel()) /
                   (np.linalg.norm(ref.ravel()) * np.linalg.norm(got.ravel())))
    return {
        'shape': list(ref.shape),
        'max_abs_error': float(diff.max()),
        'mean_abs_error': float(diff.mean()),
        'max_rel_error': float(relative.max()),
        'cosine_similarity': cosine,
    }


def configure_library(path):
    library = ctypes.CDLL(str(path.resolve()))
    pointer = ctypes.c_void_p
    integers = [ctypes.c_int32] * 7
    signature = [pointer] * 6 + integers + [pointer]
    for name in ('msda_forward_fp32', 'msda_forward_fp16'):
        function = getattr(library, name)
        function.argtypes = signature
        function.restype = ctypes.c_int
    return library


def run_case(library, reference_module, name, dtype):
    batch, queries, height, width, points = CASES[name]
    generator = torch.Generator(device='cuda').manual_seed(20260825)
    value = torch.randn((batch, height * width, HEADS, HEAD_DIM),
                        generator=generator, device='cuda', dtype=dtype)
    locations = torch.rand((batch, queries, HEADS, 1, points, 2),
                           generator=generator, device='cuda', dtype=dtype)
    raw_weights = torch.rand((batch, queries, HEADS, 1, points),
                             generator=generator, device='cuda', dtype=dtype)
    weights = (raw_weights / raw_weights.sum(-1, keepdim=True)).contiguous()
    shapes64 = torch.tensor([[height, width]], dtype=torch.int64, device='cuda')
    starts64 = torch.tensor([0], dtype=torch.int64, device='cuda')
    shapes32 = shapes64.to(torch.int32)
    starts32 = starts64.to(torch.int32)
    # This MMCV 1.4.0 binary has no Half instantiation. For FP16, compare the
    # plugin with MMCV FP32 evaluated from the exact values representable by
    # the half inputs. This isolates kernel precision without inventing a CPU
    # reference with different interpolation semantics.
    function = reference_module.MultiScaleDeformableAttnFunction_fp32
    reference = function.apply(
        value if dtype == torch.float32 else value.float(),
        shapes64, starts64,
        locations if dtype == torch.float32 else locations.float(),
        weights if dtype == torch.float32 else weights.float(), 64)
    output = torch.empty(reference.shape, device='cuda', dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    entry = (library.msda_forward_fp32 if dtype == torch.float32
             else library.msda_forward_fp16)
    status = entry(stream, value.data_ptr(), shapes32.data_ptr(),
                   starts32.data_ptr(), locations.data_ptr(), weights.data_ptr(),
                   batch, height * width, HEADS, HEAD_DIM, 1, queries, points,
                   output.data_ptr())
    if status != 0:
        raise RuntimeError('{} launch returned {}'.format(name, status))
    torch.cuda.synchronize()
    result = metrics(reference, output)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    result['dtype'] = str(dtype).replace('torch.', '')
    result['reference_dtype'] = ('float32' if dtype == torch.float32
                                 else 'float32_from_quantized_fp16_inputs')
    result['threshold'] = tolerance
    result['pass'] = result['max_abs_error'] <= tolerance
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    library = configure_library(args.library)
    reference = load_reference()
    report = {}
    for name in CASES:
        report[name] = {}
        for dtype in (torch.float32, torch.float16):
            result = run_case(library, reference, name, dtype)
            report[name][result['dtype']] = result
            print('{} {}: {}'.format(name, result['dtype'], result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    if not all(item['pass'] for case in report.values()
               for item in case.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
