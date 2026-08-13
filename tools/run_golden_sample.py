#!/usr/bin/env python
"""Run BEVFormer on one nuScenes sample and save a reference result."""

import argparse
import importlib
import json
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


DEFAULT_TOKEN = '3e8750f331d7499e9b5123e9eb70f2e2'
DEFAULT_CONFIG = 'projects/configs/bevformer/bevformer_tiny.py'
DEFAULT_CHECKPOINT = '/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-root', default='golden_samples')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def find_sample_index(data_infos, token):
    """Return the unique dataset index for ``token``."""
    matches = [i for i, info in enumerate(data_infos)
               if info.get('token') == token]
    if len(matches) != 1:
        raise ValueError(
            'Expected exactly one data item for token {!r}, found {}'.format(
                token, len(matches)))
    return matches[0]


def import_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', os.path.dirname(config_path))
    module_path = plugin_dir.rstrip('/').replace('/', '.')
    importlib.import_module(module_path)


def validate_pipeline_item(data, info, token):
    camera_names = sorted(info.get('cams', {}))
    if len(camera_names) != 6:
        raise ValueError('Expected 6 cameras, found {}: {}'.format(
            len(camera_names), camera_names))
    calibration_keys = ('cam_intrinsic', 'sensor2lidar_rotation',
                        'sensor2lidar_translation')
    incomplete_cameras = [
        name for name in camera_names
        if any(key not in info['cams'][name] for key in calibration_keys)
    ]
    if incomplete_cameras:
        raise ValueError('Incomplete camera calibration: {}'.format(
            incomplete_cameras))
    img_metas = data['img_metas']
    if isinstance(img_metas, list):
        if len(img_metas) != 1:
            raise ValueError('Expected one test augmentation')
        img_metas = img_metas[0]
    metas = img_metas.data
    if metas.get('sample_idx') != token:
        raise ValueError('Pipeline metadata token does not match requested token')
    required = ('lidar2img', 'can_bus', 'scene_token')
    missing = [key for key in required if key not in metas]
    if missing:
        raise ValueError('Pipeline metadata is missing: {}'.format(missing))
    if len(metas['lidar2img']) != 6:
        raise ValueError('Expected 6 lidar2img matrices')
    return camera_names, metas


def pipeline_image_tensor(data):
    images = data['img']
    if isinstance(images, list):
        if len(images) != 1:
            raise ValueError('Expected one test augmentation')
        images = images[0]
    return images.data


def result_arrays(result):
    detection = result['pts_bbox']
    boxes = detection['boxes_3d'].tensor.detach().cpu().numpy()
    scores = detection['scores_3d'].detach().cpu().numpy()
    labels = detection['labels_3d'].detach().cpu().numpy()
    if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError('Inconsistent detection output lengths')
    return boxes, scores, labels


def save_reference(output_dir, token, timestamp, boxes, scores, labels,
                   class_names, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / 'detections.npz'
    np.savez(
        str(npz_path),
        sample_token=np.asarray(token),
        timestamp=np.asarray(timestamp, dtype=np.int64),
        boxes_3d=boxes,
        scores_3d=scores,
        labels_3d=labels,
    )

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        label_int = int(label)
        detections.append({
            'score': float(score),
            'label': label_int,
            'class_name': class_names[label_int],
            'box_3d': [float(value) for value in box],
        })
    summary = {
        'sample_token': token,
        'timestamp': int(timestamp),
        'box_layout': ['x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy'],
        'detection_count': len(detections),
        'metadata': metadata,
        'detections': detections,
    }
    json_path = output_dir / 'summary.json'
    with json_path.open('w') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    return npz_path, json_path, summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for BEVFormer inference')
    for path_name, path in (('config', args.config),
                            ('checkpoint', args.checkpoint)):
        if not os.path.isfile(path):
            raise FileNotFoundError('{} not found: {}'.format(path_name, path))

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
    print('Golden sample index={}, timestamp={}, cameras={}'.format(
        index, info['timestamp'], ','.join(camera_names)))
    image_tensor = pipeline_image_tensor(data)
    print('Pipeline image tensor shape={}'.format(tuple(image_tensor.shape)))

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    batch = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, **batch)
    if len(outputs) != 1:
        raise ValueError('Expected one model output, found {}'.format(len(outputs)))

    boxes, scores, labels = result_arrays(outputs[0])
    output_dir = Path(args.output_root) / args.token
    metadata = {
        'dataset_index': index,
        'scene_token': info['scene_token'],
        'camera_names': camera_names,
        'lidar_path': info['lidar_path'],
        'config': args.config,
        'checkpoint': args.checkpoint,
        'seed': args.seed,
        'pipeline_img_shape': list(image_tensor.shape),
        'can_bus_length': len(metas['can_bus']),
    }
    npz_path, json_path, summary = save_reference(
        output_dir, args.token, info['timestamp'], boxes, scores, labels,
        list(model.module.CLASSES), metadata)
    print('Saved {} detections to {} and {}'.format(
        summary['detection_count'], npz_path, json_path))
    for i, detection in enumerate(summary['detections']):
        print('{:03d}: score={:.9f} label={} ({}) box={}'.format(
            i, detection['score'], detection['label'],
            detection['class_name'], detection['box_3d']))


if __name__ == '__main__':
    main()
