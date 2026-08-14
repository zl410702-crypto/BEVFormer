"""Deployment-only BEVFormer wrapper ending at raw detection-head outputs."""

from contextlib import contextmanager
import importlib

import torch
import torch.nn as nn

from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32)

from .grid_sample_rewrite import bev_rotate_nearest


CUSTOM_DOMAIN = 'bevformer'
CUSTOM_OP_TYPE = 'MSDeformableAttention'
transformer_module = importlib.import_module(
    'projects.mmdet3d_plugin.bevformer.modules.transformer')
REFERENCE_MAXIMUM = torch.maximum
REFERENCE_NAN_TO_NUM = torch.nan_to_num


def maximum_opset13(input_a, input_b):
    """Implement floating-point maximum with opset-13 exportable operators.

    The BEVFormer call has two equally shaped float32 tensors and a finite
    epsilon RHS. Keep explicit NaN propagation so this also matches
    ``torch.maximum`` if projection input A is unexpectedly NaN.
    """
    selected = torch.where(input_a > input_b, input_a, input_b)
    return torch.where(torch.isnan(input_a), input_a, selected)


def nan_to_num_opset13(input_tensor, nan=0.0, posinf=None, neginf=None):
    """Remove a redundant nan_to_num from the boolean projection mask."""
    if input_tensor.dtype == torch.bool:
        return input_tensor
    return REFERENCE_NAN_TO_NUM(
        input_tensor, nan=nan, posinf=posinf, neginf=neginf)


def ms_deformable_attention_symbolic(
        graph, value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights, im2col_step):
    """Emit one stable custom schema for all BEVFormer deformable attention."""
    try:
        step = int(im2col_step)
    except (TypeError, ValueError):
        step = 64
    return graph.op(
        '{}::{}'.format(CUSTOM_DOMAIN, CUSTOM_OP_TYPE),
        value, spatial_shapes, level_start_index, sampling_locations,
        attention_weights, im2col_step_i=step)


@contextmanager
def deployment_export_overrides(model=None):
    """Install deployment behavior temporarily without changing Reference code."""
    function_class = MultiScaleDeformableAttnFunction_fp32
    had_symbolic = hasattr(function_class, 'symbolic')
    original_symbolic = getattr(function_class, 'symbolic', None)
    original_rotate = transformer_module.rotate
    original_maximum = torch.maximum
    original_nan_to_num = torch.nan_to_num
    can_bus_gates = []
    if model is not None:
        for child in model.modules():
            if (isinstance(child, transformer_module.PerceptionTransformer)
                    and isinstance(child.use_can_bus, bool)):
                can_bus_gates.append((child, child.use_can_bus))
                # PyTorch 1.9's opset-13 exporter cannot lower
                # aten::mul(Tensor, bool). This static deployment gate is
                # mathematically identical as a Python 0/1 scalar.
                child.use_can_bus = float(child.use_can_bus)
    function_class.symbolic = staticmethod(ms_deformable_attention_symbolic)
    transformer_module.rotate = bev_rotate_nearest
    torch.maximum = maximum_opset13
    torch.nan_to_num = nan_to_num_opset13
    try:
        yield
    finally:
        for child, use_can_bus in can_bus_gates:
            child.use_can_bus = use_can_bus
        torch.nan_to_num = original_nan_to_num
        torch.maximum = original_maximum
        transformer_module.rotate = original_rotate
        if had_symbolic:
            function_class.symbolic = original_symbolic
        else:
            delattr(function_class, 'symbolic')


class BEVFormerONNXWrapper(nn.Module):
    """Export image inference through raw cls/reg predictions and BEV output."""

    def __init__(self, model, img_meta, use_prev_bev=False):
        super().__init__()
        self.model = model
        self.img_meta = img_meta
        self.use_prev_bev = use_prev_bev

    def forward(self, images, prev_bev=None):
        if self.use_prev_bev and prev_bev is None:
            raise ValueError('prev_bev is required when use_prev_bev=True')
        if not self.use_prev_bev:
            prev_bev = None
        # Reference extract_img_feat uses squeeze_ for B=1. Keep export inputs
        # immutable so reference evaluation and tracing can reuse them safely.
        images = images.clone()
        with deployment_export_overrides(self.model):
            image_features = self.model.extract_img_feat(
                images, [self.img_meta], len_queue=None)
            outputs = self.model.pts_bbox_head(
                image_features, [self.img_meta], prev_bev=prev_bev)
        return (outputs['all_cls_scores'], outputs['all_bbox_preds'],
                outputs['bev_embed'])
