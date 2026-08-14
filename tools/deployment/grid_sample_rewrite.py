"""ONNX-opset-13-friendly rewrite for tiny BEV temporal NEAREST sampling."""

import torch
import torch.nn as nn
from torchvision.transforms.functional import _get_inverse_affine_matrix


def bev_grid_sample_nearest(input_tensor, grid):
    """Match grid_sample(..., nearest, zeros, align_corners=False).

    This intentionally implements only the mode used inside torchvision 0.10.1
    by the BEVFormer tiny temporal prev-BEV rotation.
    """
    if input_tensor.dim() != 4:
        raise ValueError('input_tensor must be 4D NCHW')
    if grid.dim() != 4 or grid.size(-1) != 2:
        raise ValueError('grid must have shape [N, Hout, Wout, 2]')
    if input_tensor.size(0) != grid.size(0):
        raise ValueError('input and grid batch dimensions must match')
    if not input_tensor.is_floating_point() or not grid.is_floating_point():
        raise TypeError('input_tensor and grid must be floating point')

    batch, channels, height, width = input_tensor.shape
    output_height, output_width = grid.shape[1:3]
    x = ((grid[..., 0] + 1.0) * float(width) - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * float(height) - 1.0) * 0.5
    x_nearest = torch.round(x)
    y_nearest = torch.round(y)
    valid = ((x_nearest >= 0.0) & (x_nearest < float(width)) &
             (y_nearest >= 0.0) & (y_nearest < float(height)))

    x_index = x_nearest.clamp(0.0, float(width - 1)).to(torch.long)
    y_index = y_nearest.clamp(0.0, float(height - 1)).to(torch.long)
    linear_index = (y_index * width + x_index).reshape(batch, 1, -1)
    linear_index = linear_index.expand(batch, channels,
                                       output_height * output_width)
    flattened = input_tensor.reshape(batch, channels, height * width)
    sampled = torch.gather(flattened, 2, linear_index)
    sampled = sampled.reshape(batch, channels, output_height, output_width)
    return sampled * valid.unsqueeze(1).to(dtype=input_tensor.dtype)


class GridSampleNearestRewriteWrapper(nn.Module):
    def forward(self, input_tensor, grid):
        return bev_grid_sample_nearest(input_tensor, grid)


def bev_rotate_nearest(input_tensor, angle, center):
    """Match the fixed-size torchvision 0.10.1 temporal tensor rotation."""
    if input_tensor.dim() != 3:
        raise ValueError('BEV rotation input must have shape [C, H, W]')
    channels, height, width = input_tensor.shape
    del channels
    center_f = [float(center[0]) - width * 0.5,
                float(center[1]) - height * 0.5]
    matrix = _get_inverse_affine_matrix(
        center_f, -float(angle), [0.0, 0.0], 1.0, [0.0, 0.0])
    theta = input_tensor.new_tensor(matrix).reshape(1, 2, 3)
    base_grid = input_tensor.new_empty(1, height, width, 3)
    x_grid = torch.linspace(-width * 0.5 + 0.5,
                            width * 0.5 - 0.5,
                            steps=width, device=input_tensor.device,
                            dtype=input_tensor.dtype)
    y_grid = torch.linspace(-height * 0.5 + 0.5,
                            height * 0.5 - 0.5,
                            steps=height, device=input_tensor.device,
                            dtype=input_tensor.dtype).unsqueeze(-1)
    base_grid[..., 0].copy_(x_grid)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1.0)
    scale = input_tensor.new_tensor([0.5 * width, 0.5 * height])
    grid = base_grid.reshape(1, height * width, 3).bmm(
        theta.transpose(1, 2) / scale)
    grid = grid.reshape(1, height, width, 2)
    return bev_grid_sample_nearest(input_tensor.unsqueeze(0), grid).squeeze(0)
