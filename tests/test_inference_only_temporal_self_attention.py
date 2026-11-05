"""Temporal attention structural and old-model numeric checks."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.temporal_self_attention import TemporalSelfAttention


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.encoder.layers.0.attentions.0.'


class TemporalSelfAttentionTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        weights = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in weights.items()
                   if key.startswith(PREFIX)}
        module = TemporalSelfAttention()
        module.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 8)

    def test_forward_matches_old_module_when_available(self):
        try:
            from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import (
                TemporalSelfAttention as OldTemporalSelfAttention)
        except ImportError:
            self.skipTest('Golden OpenMMLab environment unavailable')
        torch.manual_seed(23)
        old = OldTemporalSelfAttention(num_levels=1).eval()
        new = TemporalSelfAttention().eval()
        new.load_state_dict(old.state_dict(), strict=True)
        query = torch.randn(1, 4, 256)
        value = torch.randn(2, 4, 256)
        references = torch.rand(2, 4, 1, 2)
        shapes = torch.tensor([[2, 2]])
        starts = torch.tensor([0])
        with torch.no_grad():
            expected = old(query, value=value, reference_points=references,
                           spatial_shapes=shapes, level_start_index=starts)
            actual = new(query, value=value, reference_points=references,
                         spatial_shapes=shapes, level_start_index=starts)
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
