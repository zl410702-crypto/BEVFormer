#!/usr/bin/env python
"""Trace tensor shapes for one BEVFormer Golden Sample inference.

The tracer wraps methods on the model instance only. It does not modify model
source files or the reference outputs produced by run_golden_sample.py.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               DEFAULT_TOKEN, find_sample_index,
                               import_plugin, pipeline_image_tensor,
                               result_arrays, validate_pipeline_item)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def describe(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device)}
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()
                if key not in ('img_metas',)}
    tensor = getattr(value, 'tensor', None)
    if isinstance(tensor, torch.Tensor):
        return {'type': type(value).__name__, 'tensor': describe(tensor)}
    return {'type': type(value).__name__}


class TraceRecorder:
    def __init__(self):
        self.events = []

    def add(self, name, inputs=None, kwargs=None, output=None, note=None):
        event = {'name': name}
        if inputs is not None:
            event['inputs'] = describe(inputs)
        if kwargs is not None:
            event['kwargs'] = describe(kwargs)
        if output is not None:
            event['output'] = describe(output)
        if note is not None:
            event['note'] = note
        self.events.append(event)

    def wrap(self, obj, method_name, label):
        original = getattr(obj, method_name)

        def traced(*args, **kwargs):
            input_description = describe(args)
            kwargs_description = describe(kwargs)
            output = original(*args, **kwargs)
            self.events.append({
                'name': label,
                'inputs': input_description,
                'kwargs': kwargs_description,
                'output': describe(output),
            })
            return output

        setattr(obj, method_name, traced)


def install_trace(model, recorder):
    head = model.pts_bbox_head
    transformer = head.transformer
    encoder = transformer.encoder

    for obj, method, label in (
            (model, 'forward', 'BEVFormer.forward'),
            (model, 'forward_test', 'BEVFormer.forward_test'),
            (model, 'simple_test', 'BEVFormer.simple_test'),
            (model, 'simple_test_pts', 'BEVFormer.simple_test_pts'),
            (model, 'extract_feat', 'BEVFormer.extract_feat'),
            (model, 'extract_img_feat', 'BEVFormer.extract_img_feat'),
            (model.img_backbone, 'forward', 'ResNet.forward'),
            (model.img_neck, 'forward', 'FPN.forward'),
            (head, 'forward', 'BEVFormerHead.forward'),
            (head, 'get_bboxes', 'BEVFormerHead.get_bboxes'),
            (transformer, 'forward', 'PerceptionTransformer.forward'),
            (transformer, 'get_bev_features',
             'PerceptionTransformer.get_bev_features'),
            (encoder, 'get_reference_points',
             'BEVFormerEncoder.get_reference_points'),
            (encoder, 'point_sampling', 'BEVFormerEncoder.point_sampling'),
            (encoder, 'forward', 'BEVFormerEncoder.forward'),
            (transformer.decoder, 'forward',
             'DetectionTransformerDecoder.forward'),
            (head.bbox_coder, 'decode', 'NMSFreeCoder.decode'),
            (head.bbox_coder, 'decode_single', 'NMSFreeCoder.decode_single')):
        recorder.wrap(obj, method, label)

    for index, layer in enumerate(encoder.layers):
        recorder.wrap(layer, 'forward',
                      'BEVFormerLayer[{}].forward'.format(index))
        temporal = layer.attentions[0]
        spatial = layer.attentions[1]
        deformable = spatial.deformable_attention
        recorder.wrap(temporal, 'forward',
                      'TemporalSelfAttention[{}].forward'.format(index))
        recorder.wrap(temporal.sampling_offsets, 'forward',
                      'TemporalSelfAttention[{}].sampling_offsets'.format(index))
        recorder.wrap(temporal.attention_weights, 'forward',
                      'TemporalSelfAttention[{}].attention_weights'.format(index))
        recorder.wrap(spatial, 'forward',
                      'SpatialCrossAttention[{}].forward'.format(index))
        recorder.wrap(deformable, 'forward',
                      'MSDeformableAttention3D[{}].forward'.format(index))
        recorder.wrap(deformable.sampling_offsets, 'forward',
                      'MSDeformableAttention3D[{}].sampling_offsets'.format(index))
        recorder.wrap(deformable.attention_weights, 'forward',
                      'MSDeformableAttention3D[{}].attention_weights'.format(index))

    for index, branch in enumerate(head.cls_branches):
        recorder.wrap(branch, 'forward', 'cls_branches[{}]'.format(index))
    for index, branch in enumerate(head.reg_branches):
        recorder.wrap(branch, 'forward', 'reg_branches[{}]'.format(index))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for BEVFormer tracing')
    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    set_random_seed(args.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset = build_dataset(cfg.data.test)
    index = find_sample_index(dataset.data_infos, args.token)
    info = dataset.data_infos[index]
    data = dataset[index]
    camera_names, metas = validate_pipeline_item(data, info, args.token)
    pipeline_img = pipeline_image_tensor(data)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    recorder = TraceRecorder()
    recorder.add('Golden Sample pipeline output', output=pipeline_img,
                 note='Unbatched six-camera tensor before MMCV collate')
    recorder.add('Golden Sample metadata', output={
        'dataset_index': index,
        'sample_token': args.token,
        'timestamp': info['timestamp'],
        'camera_names': camera_names,
        'lidar2img': metas['lidar2img'],
        'can_bus': metas['can_bus'],
        'scene_token': metas['scene_token'],
        'prev_bev_before_forward_test': model.prev_frame_info['prev_bev'],
    })
    install_trace(model, recorder)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    batch = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, **batch)
    boxes, scores, labels = result_arrays(outputs[0])
    recorder.add('Final detections', output={
        'boxes_3d': boxes,
        'scores_3d': scores,
        'labels_3d': labels,
        'prev_bev_after_forward_test': model.module.prev_frame_info['prev_bev'],
    })

    output_path = Path(args.output) if args.output else Path(
        'golden_samples') / args.token / 'forward_trace.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'sample_token': args.token,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'events': recorder.events,
    }
    with output_path.open('w') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    print('Wrote {} trace events to {}'.format(
        len(recorder.events), output_path))


if __name__ == '__main__':
    main()
