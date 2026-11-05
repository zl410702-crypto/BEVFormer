"""Compare the independent model against a TF32-disabled three-frame reference."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .detection import decode_detections
from .model import BEVFormerTiny


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'golden_tensors_fp32/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3'
CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')


def compare(actual, expected):
    actual = actual.detach().float().cpu().numpy()
    if actual.shape != expected.shape:
        raise ValueError('Shape mismatch: {} vs {}'.format(actual.shape, expected.shape))
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {'MaxAE': float(np.max(np.abs(difference))),
            'MAE': float(np.mean(np.abs(difference))),
            'MSE': float(np.mean(difference ** 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'BEVFormer_FP32_Alignment_Metrics.json')
    parser.add_argument('--history-mode', choices=('feedback', 'reference'),
                        default='feedback',
                        help='Pass the model BEV or the previous FP32 reference BEV')
    args = parser.parse_args()
    settings = json.loads((args.reference / 'capture_settings.json').read_text())
    if settings['cudnn_allow_tf32'] or settings['cuda_matmul_allow_tf32']:
        raise ValueError('Reference must have TF32 disabled')
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != settings['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from FP32 reference')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = BEVFormerTiny().cuda().eval()
    load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    state = torch.load(args.checkpoint, map_location='cpu', **load_options)['state_dict']
    model.load_state_dict(state, strict=True)
    captured = {}

    def hook(name, use_input=False):
        def record(module, inputs, output=None):
            value = inputs[0] if use_input else output
            if isinstance(value, (tuple, list)):
                value = value[0]
            captured[name] = value
        return record

    model.img_backbone.register_forward_hook(hook('backbone/c5.npy'))
    model.img_neck.register_forward_hook(hook('fpn/level_0.npy'))
    model.pts_bbox_head.positional_encoding.register_forward_hook(hook('bev/bev_pos.npy'))
    encoder = model.pts_bbox_head.transformer.encoder
    for i, layer in enumerate(encoder.layers):
        temporal = 'temporal/layer_{}/'.format(i)
        spatial = 'spatial/layer_{}/'.format(i)
        layer.attentions[0].register_forward_pre_hook(hook(temporal + 'query.npy', True))
        layer.attentions[0].register_forward_hook(hook(temporal + 'output.npy'))
        layer.attentions[1].register_forward_pre_hook(hook(spatial + 'query.npy', True))
        layer.attentions[1].register_forward_hook(hook(spatial + 'output.npy'))
    decoder = model.pts_bbox_head.transformer.decoder
    for i, layer in enumerate(decoder.layers):
        layer.register_forward_hook(hook('decoder/layer_{}/object_feature.npy'.format(i)))

    reports = {}
    previous_output = None
    for index in range(3):
        frame = 'frame_{:03d}'.format(index)
        base = args.reference / frame
        captured.clear()
        image = np.load(base / 'input/images.npy', allow_pickle=False)
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if digest != settings['input_sha256'][frame]:
            raise ValueError('Input hash mismatch in {}'.format(frame))
        images = torch.from_numpy(image[None].copy()).cuda()
        can_bus = torch.from_numpy(np.load(base / 'input/can_bus.npy')).cuda()
        lidar2img = torch.from_numpy(np.load(base / 'input/lidar2img.npy')).cuda()
        shift = torch.from_numpy(np.load(base / 'temporal/shift.npy')).cuda()
        history = previous_output
        if index and args.history_mode == 'reference':
            previous = args.reference / 'frame_{:03d}'.format(index - 1)
            history = torch.from_numpy(np.load(previous / 'bev/final_bev.npy')).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            output = model(images, can_bus, lidar2img, images.shape[-2:],
                           prev_bev=history, shift=shift)
        captured['bev/bev_query.npy'] = model.pts_bbox_head.bev_embedding.weight
        captured['bev/final_bev.npy'] = output['bev_embed']
        captured['detection/cls_scores.npy'] = output['all_cls_scores']
        captured['detection/bbox_preds.npy'] = output['all_bbox_preds']
        decoded_boxes, decoded_scores, decoded_labels = decode_detections(
            output['all_cls_scores'], output['all_bbox_preds'])[0]
        captured['detection/decoded_boxes.npy'] = decoded_boxes
        captured['detection/decoded_scores.npy'] = decoded_scores
        captured['detection/decoded_labels.npy'] = decoded_labels
        frame_report = {}
        for relative, actual in captured.items():
            expected = np.load(base / relative, allow_pickle=False)
            frame_report[relative] = compare(actual, expected)
            print(frame, relative, frame_report[relative], flush=True)
        reports[frame] = frame_report
        previous_output = output['bev_embed'].detach()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        'reference': str(args.reference),
        'pytorch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'checkpoint_items': len(state), 'stateful_mode': args.history_mode,
        'settings': {'eval': True, 'dtype': 'float32', 'autocast': False,
                     'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                     'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_deterministic': torch.backends.cudnn.deterministic},
        'frames': reports}, indent=2) + '\n')
    print('Report:', args.report)


if __name__ == '__main__':
    main()
