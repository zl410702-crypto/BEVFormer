#!/usr/bin/env python
"""Compare production host-affine TRT outputs with Frame-1 PyTorch Golden."""

import argparse
import json
from pathlib import Path

import numpy as np


SHAPES = {
    'debug_affine_grid': [1, 2500, 2],
    '1398': [1, 50, 50],
    '1408': [1, 50, 50],
    '1409': [1, 50, 50],
    '1410': [1, 50, 50],
    '1481': [1, 256, 2500],
    '1498': [50, 50, 256],
    '1530': [2500, 1, 256],
    '3370': [1, 2500, 256],
    '4155': [1, 2500, 256],
    '4940': [1, 2500, 256],
    'bev_embed': [2500, 1, 256],
    'all_cls_scores': [6, 1, 900, 10],
    'all_bbox_preds': [6, 1, 900, 10],
}


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError('shape mismatch {} != {}'.format(
            reference.shape, actual.shape))
    difference = np.abs(reference.astype(np.float64) -
                        actual.astype(np.float64))
    denominator = (np.linalg.norm(reference.ravel().astype(np.float64)) *
                   np.linalg.norm(actual.ravel().astype(np.float64)))
    return {
        'shape': list(reference.shape),
        'max_abs': float(difference.max()),
        'mean_abs': float(difference.mean()),
        'rmse': float(np.sqrt(np.mean(difference * difference))),
        'cosine': float(np.dot(reference.ravel().astype(np.float64),
                               actual.ravel().astype(np.float64)) / denominator),
        'mismatch_count': int(np.count_nonzero(reference != actual)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--alignment-reference', type=Path, required=True)
    parser.add_argument('--encoder-reference', type=Path, required=True)
    parser.add_argument('--golden-frame', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    alignment = args.alignment_reference
    encoder = args.encoder_reference
    golden = args.golden_frame
    references = {
        'debug_affine_grid': np.load(alignment / 'affine_grid.npy'),
        '1398': np.load(alignment / 'x_pixel.npy'),
        '1408': np.load(alignment / 'y_pixel.npy'),
        '1409': np.load(alignment / 'x_rounded.npy'),
        '1410': np.load(alignment / 'y_rounded.npy'),
        '1481': np.load(alignment / 'gathered.npy'),
        # ONNX tensor 1498 is the HWC transpose of the captured NCHW output.
        '1498': np.load(alignment / 'rotated_nchw.npy')[0].transpose(1, 2, 0),
        '1530': np.load(golden / 'temporal/prev_bev_aligned.npy'),
        '3370': np.load(encoder / 'encoder_l0.npy'),
        '4155': np.load(encoder / 'encoder_l1.npy'),
        '4940': np.load(encoder / 'encoder_l2.npy'),
        'bev_embed': np.load(golden / 'bev/final_bev.npy'),
        'all_cls_scores': np.load(golden / 'detection/cls_scores.npy'),
        'all_bbox_preds': np.load(golden / 'detection/bbox_preds.npy'),
    }
    report = {}
    for name, reference in references.items():
        actual = np.fromfile(str(args.outputs / (name + '.bin')), np.float32)
        actual = actual.reshape(SHAPES[name])
        reference = np.asarray(reference, dtype=np.float32).reshape(SHAPES[name])
        report[name] = metrics(reference, actual)
    report['round_mismatch_total'] = (
        report['1409']['mismatch_count'] + report['1410']['mismatch_count'])
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
