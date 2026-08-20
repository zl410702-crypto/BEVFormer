"""Stateful single-frame BEVFormer ONNX wrapper with runtime alignment data."""

from contextlib import contextmanager
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bevformer_onnx_wrapper import deployment_export_overrides
from .grid_sample_rewrite import bev_rotate_nearest_tensor
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
    SpatialCrossAttention)


STATIC_SCA_LINEAR_CHUNK = 640


def fixed_chunk_linear(x, linear, chunk_size=STATIC_SCA_LINEAR_CHUNK):
    """Run a deployment Linear with a fixed query-capacity per invocation.

    ``chunk_size`` is a deployment GEMM contract.  It is independent of
    camera visibility and never derives from ``bev_mask`` or compact indexes.
    """
    if x.ndim != 3:
        raise ValueError('fixed-chunk Linear expects [Bcam,Nq,C] input')
    if chunk_size <= 0:
        raise ValueError('fixed-chunk Linear requires chunk_size > 0')
    num_query = x.shape[1]
    if num_query == 0:
        raise ValueError('fixed-chunk Linear requires Nq > 0')
    outputs = []
    for start in range(0, num_query, chunk_size):
        chunk = x[:, start:start + chunk_size]
        valid_queries = chunk.shape[1]
        if valid_queries < chunk_size:
            chunk = F.pad(
                chunk, (0, 0, 0, chunk_size - valid_queries), value=0.0)
        outputs.append(linear(chunk)[:, :valid_queries])
    return torch.cat(outputs, dim=1)


@contextmanager
def fixed_deformable_linear_override(
        deformable_attention, chunk_size=STATIC_SCA_LINEAR_CHUNK):
    """Temporarily chunk only offsets/weights Linear calls."""
    originals = []
    for linear_name in ('sampling_offsets', 'attention_weights'):
        linear = getattr(deformable_attention, linear_name, None)
        if linear is None:
            continue
        originals.append((linear, 'forward' in linear.__dict__,
                          linear.__dict__.get('forward')))
        original_forward = linear.forward

        def wrapped(this, x, _forward=original_forward):
            return fixed_chunk_linear(x, _forward, chunk_size)

        linear.forward = types.MethodType(wrapped, linear)
    try:
        yield
    finally:
        for linear, had_instance_forward, original in originals:
            if had_instance_forward:
                linear.forward = original
            else:
                del linear.forward


@contextmanager
def disable_tf32_for_validation():
    """Use strict FP32 math temporarily for static-SCA validation/export."""
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def static_spatial_cross_attention_forward(
        attention, query, key, value, residual=None, query_pos=None,
        key_padding_mask=None, reference_points=None, spatial_shapes=None,
        reference_points_cam=None, bev_mask=None, level_start_index=None,
        flag='encoder', **kwargs):
    """Fixed-capacity deployment path for Spatial Cross Attention.

    Unlike the reference implementation, this path never compacts visible
    queries with ``nonzero``. Every camera processes all ``num_query`` BEV
    queries; invalid camera/query pairs receive a safe normalized reference
    point and are masked before the camera reduction.
    """
    del key_padding_mask, reference_points, flag
    if key is None:
        key = query
    if value is None:
        value = key
    inp_residual = query if residual is None else residual
    if query_pos is not None:
        query = query + query_pos

    batch, num_query, channels = query.shape
    if channels != attention.embed_dims:
        raise ValueError('query channels do not match embed_dims')
    if reference_points_cam is None or bev_mask is None:
        raise ValueError('reference_points_cam and bev_mask are required')

    # Reference layouts are [Ncam,B,Nq,D,2] and [Ncam,B,Nq,D].
    references_fixed = reference_points_cam.permute(1, 0, 2, 3, 4)
    valid_mask = (bev_mask.sum(-1) > 0).permute(1, 0, 2)
    valid_references = valid_mask.unsqueeze(-1).unsqueeze(-1)
    safe_coordinate = references_fixed.new_tensor(0.5)
    references_fixed = torch.where(
        valid_references, references_fixed, safe_coordinate)
    queries_fixed = query.unsqueeze(1).expand(
        batch, attention.num_cams, num_query, channels)

    num_cams, feature_length, key_batch, key_channels = key.shape
    if num_cams != attention.num_cams or key_batch != batch:
        raise ValueError('camera feature layout does not match query batch')
    key = key.permute(2, 0, 1, 3).reshape(
        batch * attention.num_cams, feature_length, key_channels)
    value = value.permute(2, 0, 1, 3).reshape(
        batch * attention.num_cams, feature_length, key_channels)
    depth = references_fixed.shape[3]
    with fixed_deformable_linear_override(attention.deformable_attention):
        camera_features = attention.deformable_attention(
            query=queries_fixed.reshape(
                batch * attention.num_cams, num_query, channels),
            key=key, value=value,
            reference_points=references_fixed.reshape(
                batch * attention.num_cams, num_query, depth, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).reshape(
                batch, attention.num_cams, num_query, channels)

    mask = valid_mask.unsqueeze(-1).to(camera_features.dtype)
    camera_features = camera_features * mask
    # Preserve the reference camera accumulation order. A vectorized reduction
    # can choose a different floating-point addition tree and amplify tiny
    # differences through later encoder layers.
    slots = torch.zeros_like(query)
    for camera_index in range(attention.num_cams):
        slots = slots + camera_features[:, camera_index]
    count = torch.clamp(mask.sum(1), min=1.0)
    slots = slots / count
    slots = attention.output_proj(slots)
    return attention.dropout(slots) + inp_residual


@contextmanager
def static_spatial_cross_attention_override(model):
    """Temporarily enable fixed-Nq SCA only for deployment/export."""
    originals = []
    for child in model.modules():
        if isinstance(child, SpatialCrossAttention):
            originals.append((child, 'forward' in child.__dict__,
                              child.__dict__.get('forward')))

            def wrapped(this, *args, **kwargs):
                return static_spatial_cross_attention_forward(
                    this, *args, **kwargs)

            child.forward = types.MethodType(wrapped, child)
    try:
        yield
    finally:
        for child, had_instance_forward, original in originals:
            if had_instance_forward:
                child.forward = original
            else:
                del child.forward


def point_sampling_tensor(encoder, reference_points, lidar2img,
                          image_height, image_width):
    """Tensor-input equivalent of ``BEVFormerEncoder.point_sampling``."""
    reference_points = reference_points.clone()
    pc_range = encoder.pc_range
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.permute(1, 0, 2, 3)
    depth, batch, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(
        depth, batch, 1, num_query, 4).repeat(
            1, 1, num_cam, 1, 1).unsqueeze(-1)
    matrices = lidar2img.view(1, batch, num_cam, 1, 4, 4).repeat(
        depth, 1, 1, num_query, 1, 1)
    projected = torch.matmul(
        matrices.to(torch.float32),
        reference_points.to(torch.float32)).squeeze(-1)
    epsilon = projected.new_tensor(1e-5)
    bev_mask = projected[..., 2:3] > epsilon
    denominator = torch.where(projected[..., 2:3] > epsilon,
                              projected[..., 2:3], epsilon)
    projected = projected[..., 0:2] / denominator
    projected[..., 0] /= float(image_width)
    projected[..., 1] /= float(image_height)
    bev_mask = (bev_mask & (projected[..., 1:2] > 0.0)
                & (projected[..., 1:2] < 1.0)
                & (projected[..., 0:1] < 1.0)
                & (projected[..., 0:1] > 0.0))
    projected = projected.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    return projected, bev_mask


def stateful_get_bev_features(
        transformer, mlvl_feats, bev_queries, bev_h, bev_w,
        runtime_can_bus, runtime_shift, runtime_lidar2img, use_prev_bev,
        grid_length=(0.512, 0.512), bev_pos=None, prev_bev=None, **kwargs):
    """Tensor-runtime variant of ``PerceptionTransformer.get_bev_features``."""
    if prev_bev is None:
        raise ValueError('stateful wrapper requires a prev_bev tensor')
    batch = mlvl_feats[0].size(0)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, batch, 1)
    bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
    gate = use_prev_bev.reshape(batch, 1, 1).to(bev_queries.dtype)

    if prev_bev.shape[1] == bev_h * bev_w:
        prev_bev = prev_bev.permute(1, 0, 2)
    aligned = []
    for index in range(batch):
        history = prev_bev[:, index].reshape(
            bev_h, bev_w, -1).permute(2, 0, 1)
        if transformer.rotate_prev_bev:
            history = bev_rotate_nearest_tensor(
                history, runtime_can_bus[index, -1:],
                center=transformer.rotate_center)
        aligned.append(history.permute(1, 2, 0).reshape(bev_h * bev_w, -1))
    aligned = torch.stack(aligned, 1)

    can_bus_embedding = transformer.can_bus_mlp(runtime_can_bus)[None, :, :]
    bev_queries = bev_queries + can_bus_embedding * transformer.use_can_bus
    # Gate=0 exactly reproduces the official no-history current/current queue.
    aligned = aligned * gate.permute(1, 0, 2) + \
        bev_queries * (1.0 - gate.permute(1, 0, 2))
    shift = runtime_shift * gate.reshape(batch, 1)

    feat_flatten = []
    spatial_shapes = []
    for level, feat in enumerate(mlvl_feats):
        _, _, _, height, width = feat.shape
        flattened = feat.flatten(3).permute(1, 0, 3, 2)
        if transformer.use_cams_embeds:
            flattened = flattened + transformer.cams_embeds[
                :, None, None, :].to(flattened.dtype)
        flattened = flattened + transformer.level_embeds[
            None, None, level:level + 1, :].to(flattened.dtype)
        feat_flatten.append(flattened)
        spatial_shapes.append((height, width))
    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=bev_pos.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                   spatial_shapes.prod(1).cumsum(0)[:-1]))
    feat_flatten = feat_flatten.permute(0, 2, 1, 3)

    encoder = transformer.encoder
    ref_3d = encoder.get_reference_points(
        bev_h, bev_w, encoder.pc_range[5] - encoder.pc_range[2],
        encoder.num_points_in_pillar, dim='3d', bs=batch,
        device=bev_queries.device, dtype=bev_queries.dtype)
    ref_2d = encoder.get_reference_points(
        bev_h, bev_w, dim='2d', bs=batch, device=bev_queries.device,
        dtype=bev_queries.dtype)
    image_shape = kwargs['img_metas'][0]['img_shape'][0]
    reference_points_cam, bev_mask = point_sampling_tensor(
        encoder, ref_3d, runtime_lidar2img,
        image_height=image_shape[0], image_width=image_shape[1])
    shifted_ref_2d = ref_2d + shift[:, None, None, :]
    current_query = bev_queries.permute(1, 0, 2)
    current_pos = bev_pos.permute(1, 0, 2)
    _, length, num_bev_level, _ = ref_2d.shape
    history_value = aligned.permute(1, 0, 2)
    temporal_value = torch.stack(
        [history_value, current_query], 1).reshape(batch * 2, length, -1)
    hybrid_ref_2d = torch.stack(
        [shifted_ref_2d, ref_2d], 1).reshape(
            batch * 2, length, num_bev_level, 2)
    output = current_query
    for layer in encoder.layers:
        no_history_value = torch.stack(
            [current_query, current_query], 1).reshape(
                batch * 2, length, -1)
        layer_value = temporal_value * gate.reshape(1, 1, 1) + \
            no_history_value * (1.0 - gate.reshape(1, 1, 1))
        output = layer(
            current_query, feat_flatten, feat_flatten,
            bev_pos=current_pos, ref_2d=hybrid_ref_2d, ref_3d=ref_3d,
            bev_h=bev_h, bev_w=bev_w, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points_cam=reference_points_cam, bev_mask=bev_mask,
            prev_bev=layer_value, **kwargs)
        current_query = output
    return output


@contextmanager
def stateful_bev_override(transformer, can_bus, shift, lidar2img, gate):
    """Temporarily install the tensor-runtime BEV path on one model instance."""
    original = transformer.get_bev_features

    def wrapped(this, mlvl_feats, bev_queries, bev_h, bev_w,
                grid_length=(0.512, 0.512), bev_pos=None, prev_bev=None,
                **kwargs):
        return stateful_get_bev_features(
            this, mlvl_feats, bev_queries, bev_h, bev_w,
            can_bus, shift, lidar2img, gate, grid_length=grid_length,
            bev_pos=bev_pos, prev_bev=prev_bev, **kwargs)

    transformer.get_bev_features = types.MethodType(wrapped, transformer)
    try:
        yield
    finally:
        transformer.get_bev_features = original


class BEVFormerStatefulONNXWrapper(nn.Module):
    """One-frame interface with explicit temporal state and runtime geometry."""

    def __init__(self, model, img_meta):
        super().__init__()
        self.model = model
        self.img_meta = img_meta

    def forward(self, images, prev_bev, can_bus, shift, lidar2img,
                use_prev_bev):
        images = images.clone()
        transformer = self.model.pts_bbox_head.transformer
        with deployment_export_overrides(self.model), \
                static_spatial_cross_attention_override(self.model), \
                stateful_bev_override(
                    transformer, can_bus, shift, lidar2img, use_prev_bev):
            image_features = self.model.extract_img_feat(
                images, [self.img_meta], len_queue=None)
            outputs = self.model.pts_bbox_head(
                image_features, [self.img_meta], prev_bev=prev_bev)
        return (outputs['all_cls_scores'], outputs['all_bbox_preds'],
                outputs['bev_embed'])
