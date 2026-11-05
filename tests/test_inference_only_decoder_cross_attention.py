"""Decoder cross-attention checkpoint and old-model comparison."""

import inspect
import unittest
from pathlib import Path

import torch

from tools.inference_only.decoder_cross_attention import CustomMSDeformableAttention


CHECKPOINT = Path('/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth')
PREFIX = 'pts_bbox_head.transformer.decoder.layers.0.attentions.1.'


class DecoderCrossAttentionTest(unittest.TestCase):
    def test_checkpoint_subtree(self):
        if not CHECKPOINT.exists():
            self.skipTest('checkpoint unavailable')
        load_options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
        weights = torch.load(CHECKPOINT, map_location='cpu', **load_options)['state_dict']
        subtree = {key[len(PREFIX):]: value for key, value in weights.items()
                   if key.startswith(PREFIX)}
        module = CustomMSDeformableAttention()
        module.load_state_dict(subtree, strict=True)
        self.assertEqual(len(subtree), 8)

    def test_forward_matches_old_module_when_available(self):
        try:
            from projects.mmdet3d_plugin.bevformer.modules.decoder import (
                CustomMSDeformableAttention as OldCustomMSDeformableAttention)
        except ImportError:
            self.skipTest('Golden OpenMMLab environment unavailable')
        torch.manual_seed(37)
        old = OldCustomMSDeformableAttention(num_levels=1).eval()
        new = CustomMSDeformableAttention().eval()
        new.load_state_dict(old.state_dict(), strict=True)
        query = torch.randn(7, 1, 256)
        value = torch.randn(9, 1, 256)
        shapes = torch.tensor([[3, 3]])
        starts = torch.tensor([0])
        with torch.no_grad():
            for coordinate_dims in (2, 4):
                references = torch.rand(1, 7, 1, coordinate_dims)
                expected = old(query, value=value, reference_points=references,
                               spatial_shapes=shapes, level_start_index=starts)
                actual = new(query, value=value, reference_points=references,
                             spatial_shapes=shapes, level_start_index=starts)
                self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
