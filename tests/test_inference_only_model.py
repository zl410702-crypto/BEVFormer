"""Full standalone BEVFormer-tiny checkpoint and module structure tests."""

import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from tools.inference_only.model import BEVFormerTiny


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')


class BEVFormerTinyTest(unittest.TestCase):
    def test_complete_checkpoint_strict_load(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        state = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        model = BEVFormerTiny().eval()
        model.load_state_dict(state, strict=True)
        self.assertEqual(len(state), 643)
        self.assertEqual(set(model.state_dict()), set(state))
        self.assertEqual(len(list(model.named_parameters())), 484)
        self.assertEqual(len(list(model.named_buffers())), 159)

    def test_positional_encoding_shape(self):
        model = BEVFormerTiny().eval()
        with torch.no_grad():
            positions = model.pts_bbox_head.positional_encoding(torch.zeros(1, 50, 50))
        self.assertEqual(tuple(positions.shape), (1, 256, 50, 50))

    def test_head_matches_old_model_when_available(self):
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
        state = torch.load(CHECKPOINT, map_location='cpu')['state_dict']
        old_model = build_model(config.model, test_cfg=config.get('test_cfg')).eval()
        old_model.load_state_dict(state, strict=True)
        new_model = BEVFormerTiny().eval()
        new_model.load_state_dict(state, strict=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        old = old_model.pts_bbox_head.to(device).eval()
        new = new_model.pts_bbox_head.to(device).eval()
        for head in (old, new):
            head.bev_h, head.bev_w = 2, 3
            head.bev_embedding = nn.Embedding.from_pretrained(
                head.bev_embedding.weight[:6].detach().clone(), freeze=False).to(device)
            head.query_embedding = nn.Embedding.from_pretrained(
                head.query_embedding.weight[:7].detach().clone(), freeze=False).to(device)
        matrix = torch.tensor([[3., 0, 0, 400], [0, 3., 0, 240],
                               [0, 0, 0, 1], [0, 0, 0, 1]])
        lidar2img = matrix.reshape(1, 1, 4, 4).repeat(1, 6, 1, 1).to(device)
        can_bus = torch.zeros(1, 18, device=device)
        metadata = {'lidar2img': lidar2img[0].cpu().numpy(),
                    'img_shape': [(480, 800, 3)] * 6, 'can_bus': [0.] * 18}
        torch.manual_seed(71)
        features = (torch.randn(1, 6, 256, 3, 3, device=device),)
        with torch.no_grad():
            expected = old(features, [metadata])
            actual = new(features, can_bus, lidar2img, (480, 800))
        for name in ('all_cls_scores', 'all_bbox_preds', 'bev_embed'):
            self.assertLessEqual((actual[name] - expected[name]).abs().max().item(), 1e-3)


if __name__ == '__main__':
    unittest.main()
