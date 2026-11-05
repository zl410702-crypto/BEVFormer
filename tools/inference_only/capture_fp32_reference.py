"""Capture the original three-frame Golden path with TF32 disabled.

Run with the original BEVFormer environment from the repository root. This
wrapper leaves the original scripts and historical Golden files untouched.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import mmcv
import mmdet
import mmdet3d
import mmseg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import run_golden_sequence as sequence  # noqa: E402


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'golden_tensors_fp32')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth'))
    args = parser.parse_args()
    scene_id = 'fcbccedd61424f1b85dcbf8f897f9754_3e8750f3'
    output = args.output_root / scene_id
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing reference: {}'.format(output))
    if not torch.cuda.is_available():
        raise RuntimeError('Golden capture requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)
    input_hashes = {}
    original_install = sequence.GoldenSequenceRecorder.install

    def install_with_input(self):
        result = original_install(self)

        def capture_input(module, inputs):
            if self.frame is None:
                return
            image = inputs[0].detach().cpu().contiguous().numpy()
            self.dump('input.images', inputs[0], 'input/images.npy')
            input_hashes[self.frame] = hashlib.sha256(image.tobytes()).hexdigest()

        self._handles.append(self.model.img_backbone.register_forward_pre_hook(capture_input))
        return result

    sequence.GoldenSequenceRecorder.install = install_with_input
    sys.argv = ['run_golden_sequence.py', '--num-frames', '3',
                '--checkpoint', str(args.checkpoint),
                '--output-root', str(args.output_root)]
    sequence.main()
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from run_golden_sample import (find_sample_index, import_plugin,
                                   validate_pipeline_item)

    config_path = 'projects/configs/bevformer/bevformer_tiny.py'
    config = Config.fromfile(config_path)
    import_plugin(config, config_path)
    config.data.test.test_mode = True
    dataset = build_dataset(config.data.test)
    frames = json.loads((output / 'sequence.json').read_text())['frames']
    for index, frame in enumerate(frames):
        sample_index = find_sample_index(dataset.data_infos, frame['sample_token'])
        info = dataset.data_infos[sample_index]
        _, metadata = validate_pipeline_item(dataset[sample_index], info,
                                             frame['sample_token'])
        bus = np.asarray(metadata['can_bus'], dtype=np.float64).copy()
        if index:
            previous = np.asarray(frames[index - 1]['can_bus_absolute'],
                                  dtype=np.float64)
            bus[:3] -= previous[:3]
            bus[-1] -= previous[-1]
        else:
            bus[:3] = 0.0
            bus[-1] = 0.0
        input_dir = output / 'frame_{:03d}'.format(index) / 'input'
        np.save(str(input_dir / 'can_bus.npy'), bus.astype(np.float32)[None])
        np.save(str(input_dir / 'lidar2img.npy'),
                np.asarray(metadata['lidar2img'], dtype=np.float32)[None])
    settings = {
        'source': 'tools/run_golden_sequence.py via capture_fp32_reference.py',
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'mmcv': mmcv.__version__, 'mmdet': mmdet.__version__,
        'mmdet3d': mmdet3d.__version__, 'mmseg': mmseg.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'model_eval': True, 'dtype': 'float32',
        'autocast': torch.is_autocast_enabled(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'input_sha256': input_hashes,
    }
    (output / 'capture_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print('Capture settings:', json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
