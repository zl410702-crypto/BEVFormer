"""Encoder checkpoint structure and reference geometry checks."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.encoder import BEVFormerEncoder


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.encoder.'


class EncoderTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        state = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in state.items()
                   if key.startswith(PREFIX)}
        encoder = BEVFormerEncoder()
        encoder.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 78)

    def test_geometry_shapes(self):
        encoder = BEVFormerEncoder()
        points = encoder.get_reference_points(2, 3, batch=1)
        lidar2img = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 6, 1, 1)
        projected, mask = encoder.point_sampling(points, lidar2img, (480, 800))
        self.assertEqual(tuple(projected.shape), (6, 1, 6, 4, 2))
        self.assertEqual(tuple(mask.shape), (6, 1, 6, 4))

    def test_three_layers_match_old_model_when_available(self):
        try:
            from mmcv import Config
            from mmdet3d.models import build_model
            from tools.run_golden_sample import import_plugin
        except ImportError:
            self.skipTest('Golden OpenMMLab environment unavailable')
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        config_path = 'projects/configs/bevformer/bevformer_tiny.py'
        config = Config.fromfile(config_path)
        import_plugin(config, config_path)
        config.model.pretrained = None
        config.model.train_cfg = None
        old_model = build_model(config.model, test_cfg=config.get('test_cfg')).eval()
        old_model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['state_dict'],
                                  strict=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        old = old_model.pts_bbox_head.transformer.encoder.to(device).eval()
        new = BEVFormerEncoder().to(device).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        matrix = torch.tensor([[3., 0, 0, 400], [0, 3., 0, 240],
                               [0, 0, 0, 1], [0, 0, 0, 1]])
        lidar2img = matrix.reshape(1, 1, 4, 4).repeat(1, 6, 1, 1).to(device)
        metadata = {'lidar2img': lidar2img[0].cpu().numpy(),
                    'img_shape': [(480, 800, 3)] * 6}
        torch.manual_seed(59)
        query = torch.randn(6, 1, 256, device=device)
        key = torch.randn(6, 9, 1, 256, device=device)
        position = torch.randn_like(query)
        previous = torch.randn_like(query)
        spatial_shapes = torch.tensor([[3, 3]], device=device)
        level_starts = torch.tensor([0], device=device)
        shift = torch.tensor([[0.01, -0.02]], device=device)
        with torch.no_grad():
            expected = old(query, key, key, bev_h=2, bev_w=3, bev_pos=position,
                           spatial_shapes=spatial_shapes, level_start_index=level_starts,
                           prev_bev=previous, shift=shift, img_metas=[metadata])
            actual = new(query, key, key, 2, 3, position, spatial_shapes,
                         level_starts, lidar2img, (480, 800), prev_bev=previous,
                         shift=shift)
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
