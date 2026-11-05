"""Decoder structure and layer-level equivalence against Golden implementation."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.decoder import DetectionTransformerDecoder


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.decoder.'


class DecoderTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        state = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in state.items()
                   if key.startswith(PREFIX)}
        decoder = DetectionTransformerDecoder()
        decoder.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 132)
        self.assertEqual(len(decoder.layers), 6)

    def test_one_layer_matches_old_model_when_available(self):
        try:
            from mmcv import Config
            from mmdet3d.models import build_model
            from tools.run_golden_sample import import_plugin
        except ImportError:
            self.skipTest('Golden OpenMMLab environment unavailable')
        config_path = 'projects/configs/bevformer/bevformer_tiny.py'
        config = Config.fromfile(config_path)
        import_plugin(config, config_path)
        config.model.pretrained = None
        config.model.train_cfg = None
        old_model = build_model(config.model, test_cfg=config.get('test_cfg')).eval()
        old = old_model.pts_bbox_head.transformer.decoder.layers[0]
        new = DetectionTransformerDecoder().layers[0].eval()
        new.load_state_dict(old.state_dict(), strict=True)
        torch.manual_seed(43)
        query = torch.randn(7, 1, 256)
        value = torch.randn(9, 1, 256)
        query_pos = torch.randn_like(query)
        references = torch.rand(1, 7, 1, 2)
        shapes = torch.tensor([[3, 3]])
        starts = torch.tensor([0])
        with torch.no_grad():
            expected = old(query, value, value, query_pos=query_pos,
                           reference_points=references, spatial_shapes=shapes,
                           level_start_index=starts)
            actual = new(query, value, value, query_pos=query_pos,
                         reference_points=references, spatial_shapes=shapes,
                         level_start_index=starts)
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)

    def test_six_layers_with_refinement_when_available(self):
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
        old = old_model.pts_bbox_head.transformer.decoder.to(device).eval()
        new = DetectionTransformerDecoder().to(device).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        branches = old_model.pts_bbox_head.reg_branches.to(device).eval()
        torch.manual_seed(53)
        query = torch.randn(7, 1, 256, device=device)
        value = torch.randn(9, 1, 256, device=device)
        query_pos = torch.randn_like(query)
        references = torch.rand(1, 7, 3, device=device)
        shapes = torch.tensor([[3, 3]], device=device)
        starts = torch.tensor([0], device=device)
        with torch.no_grad():
            expected = old(query, value, value, query_pos=query_pos,
                           reference_points=references, reg_branches=branches,
                           spatial_shapes=shapes, level_start_index=starts)
            actual = new(query, value, value, query_pos=query_pos,
                         reference_points=references, reg_branches=branches,
                         spatial_shapes=shapes, level_start_index=starts)
        self.assertLessEqual((actual[0] - expected[0]).abs().max().item(), 2e-3)
        self.assertLessEqual((actual[1] - expected[1]).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
