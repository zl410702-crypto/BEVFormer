"""Independent CPU contract tests for multi-scale deformable attention."""

import unittest

import torch

from tools.inference_only.deformable_attention import multi_scale_deformable_attention


class DeformableAttentionTest(unittest.TestCase):
    def test_cuda_extension_matches_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        try:
            from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import (
                MultiScaleDeformableAttnFunction_fp32)
        except ImportError:
            self.skipTest('Golden MMCV CUDA Extension unavailable')
        torch.manual_seed(121)
        for batch, queries, heads, channels, shapes, points in (
            (2, 19, 8, 32, ((5, 7),), 4),
            (6, 23, 8, 32, ((3, 5),), 8),
            (1, 17, 8, 32, ((4, 4),), 4),
            (2, 7, 4, 16, ((2, 3), (1, 4)), 3),
        ):
            starts = [0]
            for height, width in shapes[:-1]:
                starts.append(starts[-1] + height * width)
            value = torch.randn(batch, sum(height * width for height, width in shapes),
                                heads, channels, device='cuda')
            locations = torch.rand(batch, queries, heads, len(shapes), points, 2,
                                   device='cuda') * 1.5 - 0.25
            weights = torch.randn(batch, queries, heads, len(shapes), points,
                                  device='cuda').softmax(-1)
            spatial_shapes = torch.tensor(shapes, device='cuda')
            level_starts = torch.tensor(starts, device='cuda')
            with torch.no_grad():
                expected = MultiScaleDeformableAttnFunction_fp32.apply(
                    value, spatial_shapes, level_starts, locations, weights, 64)
                actual = multi_scale_deformable_attention(
                    value, spatial_shapes, level_starts, locations, weights)
            error = (actual - expected).abs()
            self.assertLessEqual(error.max().item(), 1e-6)

    def test_tsa_sca_decoder_shapes(self):
        for batch, queries, heads, channels, shapes, points in (
            (2, 7, 8, 32, ((2, 3),), 4),
            (6, 5, 8, 32, ((3, 4),), 8),
            (1, 9, 8, 32, ((2, 2), (1, 3)), 4),
        ):
            starts = [0]
            for height, width in shapes[:-1]:
                starts.append(starts[-1] + height * width)
            torch.manual_seed(7)
            value = torch.randn(batch, sum(height * width for height, width in shapes),
                                heads, channels)
            locations = torch.rand(batch, queries, heads, len(shapes), points, 2)
            weights = torch.rand(batch, queries, heads, len(shapes), points)
            output = multi_scale_deformable_attention(
                value, torch.tensor(shapes), torch.tensor(starts), locations, weights)
            self.assertEqual(tuple(output.shape), (batch, queries, heads * channels))
            self.assertTrue(torch.isfinite(output).all())

    def test_center_and_out_of_bounds(self):
        value = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1)
        locations = torch.tensor([[[[[[0.25, 0.25], [0.75, 0.75], [2., 2.]]]]]])
        weights = torch.tensor([[[[[0.5, 0.5, 1.]]]]])
        output = multi_scale_deformable_attention(
            value, torch.tensor([[2, 2]]), torch.tensor([0]), locations, weights)
        torch.testing.assert_close(output, torch.tensor([[[2.5]]]))

    def test_invalid_level_start_index(self):
        with self.assertRaisesRegex(ValueError, 'level_start_index'):
            multi_scale_deformable_attention(
                torch.zeros(1, 2, 1, 1), torch.tensor([[1, 1], [1, 1]]),
                torch.tensor([0, 0]), torch.zeros(1, 1, 1, 2, 1, 2),
                torch.ones(1, 1, 1, 2, 1))


if __name__ == '__main__':
    unittest.main()
