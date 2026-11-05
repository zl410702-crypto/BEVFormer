"""SCA checkpoint structure and old-model numerical equivalence."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.spatial_cross_attention import SpatialCrossAttention


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.encoder.layers.0.attentions.1.'


class SpatialCrossAttentionTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        weights = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in weights.items()
                   if key.startswith(PREFIX)}
        module = SpatialCrossAttention()
        module.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 8)

    def test_forward_matches_old_module_when_available(self):
        try:
            from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
                SpatialCrossAttention as OldSpatialCrossAttention)
        except ImportError:
            self.skipTest('Golden OpenMMLab environment unavailable')
        torch.manual_seed(31)
        old = OldSpatialCrossAttention(deformable_attention=dict(
            type='MSDeformableAttention3D', embed_dims=256, num_points=8,
            num_levels=1)).eval()
        new = SpatialCrossAttention().eval()
        new.load_state_dict(old.state_dict(), strict=True)
        query = torch.randn(1, 9, 256)
        key = torch.randn(6, 9, 1, 256)
        references = torch.rand(6, 1, 9, 4, 2)
        mask = torch.rand(6, 1, 9, 4) > 0.4
        shapes = torch.tensor([[3, 3]])
        starts = torch.tensor([0])
        with torch.no_grad():
            expected = old(query, key, key, reference_points_cam=references,
                           bev_mask=mask, spatial_shapes=shapes,
                           level_start_index=starts)
            actual = new(query, key, key, reference_points_cam=references,
                         bev_mask=mask, spatial_shapes=shapes,
                         level_start_index=starts)
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
