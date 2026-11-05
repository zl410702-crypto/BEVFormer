"""Compare standalone BEVFormer outputs with an existing Golden frame.

Example:
    python -m tools.inference_only.validate_golden_frame --frame 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import BEVFormerTiny


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
FIXTURES = ROOT / 'artifacts/bevformer_tiny_stateful_trt84_runtime'
GOLDEN = ROOT / 'golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3'


def metrics(actual, expected):
    actual = actual.detach().float().cpu().numpy()
    if actual.shape != expected.shape:
        raise ValueError(f'shape mismatch: {actual.shape} vs {expected.shape}')
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return (float(np.max(np.abs(difference))),
            float(np.mean(np.abs(difference))),
            float(np.mean(difference ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--golden', type=Path, default=GOLDEN)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for the full-size frame')
    frame = f'frame_{args.frame:03d}'
    fixture = args.fixtures / frame
    golden = args.golden / frame
    metadata = json.loads((fixture / 'metadata.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = BEVFormerTiny().cuda().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    captured = {}
    model.img_backbone.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('backbone.c5', output[-1]))
    model.img_neck.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('fpn.level_0', output[0]))
    model.pts_bbox_head.positional_encoding.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('bev.bev_pos', output))

    def input_tensor(name):
        spec = metadata['inputs'][name]
        array = np.fromfile(fixture / spec['path'], dtype=np.dtype(spec['dtype']))
        return torch.from_numpy(array.reshape(spec['shape']).copy()).cuda()

    images = input_tensor('images')
    with torch.no_grad():
        output = model(images, input_tensor('can_bus'), input_tensor('lidar2img'),
                       images.shape[-2:], shift=input_tensor('shift'),
                       prev_bev=input_tensor('prev_bev')
                       if input_tensor('use_prev_bev').item() else None)
    captured['bev.bev_query'] = model.pts_bbox_head.bev_embedding.weight
    for name, tensor in captured.items():
        expected = np.load(golden / (name.replace('.', '/') + '.npy'))
        print(f'{name}: MaxAE/MAE/MSE = {metrics(tensor, expected)}')
    for name, tensor in output.items():
        spec = metadata['outputs'][name]
        expected = np.fromfile(fixture / spec['path'], dtype=np.dtype(spec['dtype']))
        expected = expected.reshape(spec['shape'])
        print(f'{name}: MaxAE/MAE/MSE = {metrics(tensor, expected)}')


if __name__ == '__main__':
    main()
