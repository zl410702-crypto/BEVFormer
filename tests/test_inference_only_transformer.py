"""Transformer checkpoint and host motion math checks."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.transformer import (
    PerceptionTransformer, compute_shift, rotate_bev_nearest)


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.'


class TransformerTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        state = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in state.items()
                   if key.startswith(PREFIX)}
        transformer = PerceptionTransformer()
        transformer.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 220)

    def test_zero_motion_shift(self):
        can_bus = torch.zeros(1, 18)
        shift = compute_shift(can_bus, (0.512, 0.512), 50, 50)
        torch.testing.assert_close(shift, torch.zeros(1, 2))

    def test_zero_rotation(self):
        features = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        rotated = rotate_bev_nearest(features, torch.tensor([0.]), center=(100, 100))
        torch.testing.assert_close(rotated, features)

    def test_historical_forward_matches_old_model_when_available(self):
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
        old = old_model.pts_bbox_head.transformer.to(device).eval()
        new = PerceptionTransformer().to(device).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        branches = old_model.pts_bbox_head.reg_branches.to(device).eval()
        projection = torch.tensor([[3., 0, 0, 400], [0, 3., 0, 240],
                                   [0, 0, 0, 1], [0, 0, 0, 1]])
        lidar2img = projection.reshape(1, 1, 4, 4).repeat(1, 6, 1, 1).to(device)
        can_bus = torch.zeros(1, 18, device=device)
        can_bus[0, 0] = 0.5
        can_bus[0, 1] = 0.2
        can_bus[0, -2] = 0.1
        can_bus[0, -1] = 17.25
        metadata = {'lidar2img': lidar2img[0].cpu().numpy(),
                    'img_shape': [(480, 800, 3)] * 6,
                    'can_bus': [float(value) for value in can_bus[0].cpu()]}
        torch.manual_seed(67)
        features = [torch.randn(1, 6, 256, 3, 3, device=device)]
        bev_queries = torch.randn(6, 256, device=device)
        object_queries = torch.randn(7, 512, device=device)
        position = torch.randn(1, 256, 2, 3, device=device)
        history = torch.randn(6, 1, 256, device=device)
        with torch.no_grad():
            expected = old(features, bev_queries, object_queries, 2, 3,
                           bev_pos=position, reg_branches=branches,
                           prev_bev=history.clone(), img_metas=[metadata])
            actual = new(features, bev_queries, object_queries, 2, 3,
                         position, can_bus, lidar2img, (480, 800),
                         reg_branches=branches, prev_bev=history.clone())
        thresholds = (1e-4, 3e-3, 0.0, 4e-5)
        for reference, output, threshold in zip(expected, actual, thresholds):
            self.assertLessEqual((output - reference).abs().max().item(), threshold)


if __name__ == '__main__':
    unittest.main()
