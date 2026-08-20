#!/usr/bin/env python
"""Validate fixed-Nq deployment SCA against the three-frame Golden dump."""

import json
from pathlib import Path

import numpy as np
import torch

from deployment.bevformer_stateful_onnx_wrapper import (
    BEVFormerStatefulONNXWrapper, SpatialCrossAttention)
from export_bevformer_stateful_onnx import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_FRAME1_TOKEN, DEFAULT_SEQUENCE,
    first_frame_runtime_meta, prepare_runtime_meta, tensor_inputs)
from run_golden_sample import (pipeline_image_tensor, prepare_golden_sample,
                               validate_pipeline_item)


def comparison(reference, actual):
    actual = actual.detach().cpu().numpy()
    difference = np.abs(reference.astype(np.float64) -
                        actual.astype(np.float64))
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    return {
        'shape': list(reference.shape),
        'max_abs': float(difference.max()),
        'mean_abs': float(difference.mean()),
        'max_rel': float((difference / denominator).max()),
        'allclose_rtol_1e-5_atol_1e-8': bool(np.allclose(
            reference, actual, rtol=1e-5, atol=1e-8)),
    }


def main():
    sequence_root = DEFAULT_SEQUENCE
    frames = json.loads((sequence_root / 'sequence.json').read_text())['frames']
    context = prepare_golden_sample(
        DEFAULT_FRAME1_TOKEN, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, 0)
    model = context['model'].module
    report = {
        'validation_math_mode': 'saved-Golden default CUDA/TF32 mode',
        'cuda_matmul_allow_tf32':
            torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    for frame_index, frame in enumerate(frames[:3]):
        dataset_index = context['index'] + frame_index - 1
        info = context['dataset'].data_infos[dataset_index]
        data = context['dataset'][dataset_index]
        _, pipeline_meta = validate_pipeline_item(
            data, info, frame['sample_token'])
        if frame_index == 0:
            runtime_meta = first_frame_runtime_meta(
                pipeline_meta, frame['can_bus_absolute'])
            prev_bev = torch.zeros(2500, 1, 256, device='cuda')
            gate = torch.zeros(1, dtype=torch.float32, device='cuda')
        else:
            runtime_meta = prepare_runtime_meta(
                pipeline_meta, frame['can_bus_absolute'],
                frames[frame_index - 1]['can_bus_absolute'])
            prev_bev = torch.from_numpy(np.load(
                str(sequence_root / 'frame_{:03d}/bev/final_bev.npy'.format(
                    frame_index - 1)), allow_pickle=False)).cuda()
            gate = torch.ones(1, dtype=torch.float32, device='cuda')
        images = pipeline_image_tensor(data).unsqueeze(0).cuda()
        can_bus, shift, lidar2img = tensor_inputs(runtime_meta, model)
        wrapper = BEVFormerStatefulONNXWrapper(model, runtime_meta).cuda().eval()
        spatial_outputs = []
        handles = [module.register_forward_hook(
            lambda unused_module, unused_inputs, output:
            spatial_outputs.append(output.detach().clone()))
            for module in model.modules()
            if isinstance(module, SpatialCrossAttention)]
        try:
            with torch.no_grad():
                outputs = wrapper(images, prev_bev, can_bus, shift,
                                  lidar2img, gate)
        finally:
            for handle in handles:
                handle.remove()
        if (torch.backends.cuda.matmul.allow_tf32 != original_matmul_tf32 or
                torch.backends.cudnn.allow_tf32 != original_cudnn_tf32):
            raise RuntimeError('stateful wrapper leaked TF32 backend state')
        frame_report = {'spatial_layers': {}}
        for layer_index, output in enumerate(spatial_outputs):
            golden = np.load(str(
                sequence_root / 'frame_{:03d}/spatial/layer_{}/output.npy'.format(
                    frame_index, layer_index)), allow_pickle=False)
            frame_report['spatial_layers']['layer_{}'.format(layer_index)] = \
                comparison(golden, output)
        for name, output, relative in (
                ('all_cls_scores', outputs[0], 'detection/cls_scores.npy'),
                ('all_bbox_preds', outputs[1], 'detection/bbox_preds.npy'),
                ('bev_embed', outputs[2], 'bev/final_bev.npy')):
            golden = np.load(str(
                sequence_root / 'frame_{:03d}'.format(frame_index) / relative),
                allow_pickle=False)
            frame_report[name] = comparison(golden, output)
        report['frame_{:03d}'.format(frame_index)] = frame_report
    report['tf32_state_restored'] = True
    report['strict_pass'] = all(
        values['allclose_rtol_1e-5_atol_1e-8']
        for frame_name, frame_report in report.items()
        if frame_name.startswith('frame_')
        for values in list(frame_report['spatial_layers'].values()) + [
            frame_report['all_cls_scores'], frame_report['all_bbox_preds'],
            frame_report['bev_embed']])
    output = Path('artifacts/static_sca_validation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Validation JSON: {}'.format(output))


if __name__ == '__main__':
    main()
