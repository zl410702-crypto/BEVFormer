#!/usr/bin/env python
"""Capture Frame-1 stateful PyTorch BEV encoder-layer Golden boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment.bevformer_stateful_onnx_wrapper import \
    BEVFormerStatefulONNXWrapper
from export_bevformer_stateful_onnx import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_FRAME1_TOKEN, DEFAULT_SEQUENCE,
    prepare_runtime_meta, tensor_inputs)
from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerLayer
from run_golden_sample import prepare_golden_sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frames = json.loads((DEFAULT_SEQUENCE / 'sequence.json').read_text())['frames']
    context = prepare_golden_sample(
        DEFAULT_FRAME1_TOKEN, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, 0)
    model = context['model'].module
    runtime_meta = prepare_runtime_meta(
        context['metas'], frames[1]['can_bus_absolute'],
        frames[0]['can_bus_absolute'])
    images = context['image_tensor'].unsqueeze(0).cuda()
    previous = torch.from_numpy(np.load(
        str(DEFAULT_SEQUENCE / 'frame_000/bev/final_bev.npy'),
        allow_pickle=False)).cuda()
    can_bus, shift, lidar2img = tensor_inputs(runtime_meta, model)
    gate = torch.ones(1, dtype=torch.float32, device='cuda')
    wrapper = BEVFormerStatefulONNXWrapper(model, runtime_meta).cuda().eval()
    captured = []
    handles = [module.register_forward_hook(
        lambda unused_module, unused_inputs, output:
        captured.append(output.detach().cpu().numpy().copy()))
        for module in model.modules() if isinstance(module, BEVFormerLayer)]
    try:
        with torch.no_grad():
            wrapper(images, previous, can_bus, shift, lidar2img, gate)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 3:
        raise RuntimeError('expected 3 encoder layers, captured {}'.format(
            len(captured)))
    if args.output.exists():
        raise RuntimeError('refusing to overwrite {}'.format(args.output))
    args.output.mkdir(parents=True)
    for index, value in enumerate(captured):
        if value.shape != (1, 2500, 256) or value.dtype != np.float32:
            raise RuntimeError('encoder layer {} contract changed'.format(index))
        np.save(str(args.output / 'encoder_l{}.npy'.format(index)), value,
                allow_pickle=False)
        print('encoder_l{} shape={} dtype={}'.format(
            index, value.shape, value.dtype))


if __name__ == '__main__':
    main()
