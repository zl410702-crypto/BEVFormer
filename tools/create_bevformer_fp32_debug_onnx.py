#!/usr/bin/env python
"""Add audited BEV encoder boundary outputs to a separate debug ONNX."""

import argparse
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


BOUNDARIES = {
    'debug_alignment_angle_radians': ('1023', []),
    'debug_alignment_cosine': ('1024', []),
    'debug_alignment_sine': ('1025', []),
    'debug_alignment_grid_flat': ('1356', [1, 2500, 2]),
    'debug_alignment_x': ('1398', [1, 50, 50]),
    'debug_alignment_y': ('1408', [1, 50, 50]),
    'debug_alignment_x_rounded': ('1409', [1, 50, 50]),
    'debug_alignment_y_rounded': ('1410', [1, 50, 50]),
    'debug_alignment_x_clipped': ('1435', [1, 50, 50]),
    'debug_alignment_y_clipped': ('1441', [1, 50, 50]),
    'debug_alignment_gathered': ('1481', [1, 256, 2500]),
    'debug_alignment_rotated_hwc': ('1498', [50, 50, 256]),
    'debug_aligned_prev_bev': ('1530', [2500, 1, 256]),
    'debug_temporal_l0_msda': ('2764', [2, 2500, 256]),
    'debug_temporal_l0_output': ('2782', [1, 2500, 256]),
    'debug_spatial_l0_msda': ('3293', [6, 2500, 256]),
    'debug_spatial_l0_output': ('3340', [1, 2500, 256]),
    'debug_encoder_l0_output': ('3370', [1, 2500, 256]),
    'debug_encoder_l1_output': ('4155', [1, 2500, 256]),
    'debug_encoder_l2_output': ('4940', [1, 2500, 256]),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    shutil.copyfile(str(args.input), str(args.output))
    model = onnx.load(str(args.output))
    produced = {name for node in model.graph.node for name in node.output}
    existing = {item.name for item in model.graph.output}
    for debug_name, (tensor_name, shape) in BOUNDARIES.items():
        if tensor_name not in produced:
            raise RuntimeError('{} source tensor {} is absent'.format(
                debug_name, tensor_name))
        if tensor_name in existing:
            raise RuntimeError('{} is already a graph output'.format(
                tensor_name))
        # Keep the real ONNX value name so TensorRT exposes the exact tensor.
        model.graph.output.append(helper.make_tensor_value_info(
            tensor_name, TensorProto.FLOAT, shape))
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print('Debug outputs:')
    for debug_name, value in BOUNDARIES.items():
        print('{}: {} {}'.format(debug_name, value[0], value[1]))


if __name__ == '__main__':
    main()
