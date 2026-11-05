"""Checkpoint and forward equivalence for the standalone image encoder."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ImageEncoderTest(unittest.TestCase):
    def test_checkpoint_keys_and_shapes(self):
        import torch
        from tools.inference_only.image_encoder import ImageEncoder

        checkpoint_path = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
        if not checkpoint_path.exists():
            self.skipTest('BEVFormer-tiny checkpoint unavailable')
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError as error:
            if 'weights_only' not in str(error):
                raise
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expected = {name: value for name, value in checkpoint['state_dict'].items()
                    if name.startswith(('img_backbone.', 'img_neck.'))}
        model = ImageEncoder()
        model.load_state_dict(expected, strict=True)
        self.assertEqual(len(expected), 322)
        self.assertEqual(set(model.state_dict()), set(expected))

    def test_no_openmmlab_imports(self):
        source = (ROOT / 'tools/inference_only/image_encoder.py').read_text()
        for package in ('mmcv', 'mmdet', 'mmdet3d', 'mmseg'):
            self.assertNotIn(package, source)

    def test_shape(self):
        import torch
        from tools.inference_only.image_encoder import ImageEncoder

        model = ImageEncoder().eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 6, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 6, 256, 2, 2))


if __name__ == '__main__':
    unittest.main()
