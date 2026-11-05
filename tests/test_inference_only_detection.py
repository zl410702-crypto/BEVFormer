"""Check standalone detection decoding against the original Golden tensors."""

import unittest
from pathlib import Path

import numpy as np
import torch

from tools.inference_only.detection import decode_detections


ROOT = (Path(__file__).resolve().parents[1] / 'golden_tensors_fp32'
        / 'fcbccedd61424f1b85dcbf8f897f9754_3e8750f3')


class DetectionDecodeTest(unittest.TestCase):
    def test_three_frame_golden_decode(self):
        if not ROOT.exists():
            self.skipTest('FP32 reference unavailable')
        for index in range(3):
            frame = ROOT / 'frame_{:03d}'.format(index) / 'detection'
            classes = torch.from_numpy(np.load(frame / 'cls_scores.npy'))
            boxes = torch.from_numpy(np.load(frame / 'bbox_preds.npy'))
            decoded = decode_detections(classes, boxes)[0]
            for name, tensor in zip(('boxes', 'scores', 'labels'), decoded):
                expected = np.load(frame / ('decoded_' + name + '.npy'))
                self.assertEqual(tensor.shape, expected.shape)
                self.assertLessEqual(np.max(np.abs(tensor.numpy() - expected)), 1e-6)


if __name__ == '__main__':
    unittest.main()
