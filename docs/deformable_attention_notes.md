# BEVFormer tiny deformable-attention tensor notes

These notes combine the repository implementation with dumps under
`golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3`.

## BEV and projection path

```text
BEV query [2500,256]
 -> BEVFormerEncoder.get_reference_points
 -> reference_points_3d [1,4,2500,3]
 -> BEVFormerEncoder.point_sampling (lidar2img projection)
 -> reference_points_cam [6,1,2500,4,2]
 -> bev_mask [6,1,2500,4]
 -> SpatialCrossAttention camera-wise rebatching
 -> MSDeformableAttention3D
```

The source is `projects/mmdet3d_plugin/bevformer/modules/encoder.py`, class
`BEVFormerEncoder`. `get_reference_points(dim='3d')` creates four normalized
height anchors for every cell of the 50x50 BEV plane. `point_sampling`
denormalizes them into `pc_range`, makes homogeneous points, multiplies by each
camera's `lidar2img`, divides by depth and image dimensions, and returns valid
normalized image coordinates plus the visibility mask. The 2D temporal grid is
separate: `reference_points_2d` is `[1,2500,1,2]` and represents normalized BEV
cell centers. These tensors must never share one ambiguous filename.

`projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`, class
`SpatialCrossAttention.forward`, selects visible queries per camera. In this
run the padded rebatched query is `[6,604,256]`; flattened one-level FPN
key/value is `[6,375,256]`, with `spatial_shapes=[[15,25]]` and
`level_start_index=[0]`.

## MSDeformableAttention3D

In class `MSDeformableAttention3D.forward`:

```text
query [6,604,256]
 -> sampling_offsets Linear
 -> [6,604,8 heads,1 level,8 points,2]

query -> attention_weights Linear -> softmax
 -> [6,604,8,1,8]

reference_points_cam + offsets / [width,height]
 -> sampling_locations [6,604,8,1,8,2]
 -> MMCV MultiScaleDeformableAttnFunction (bilinear sampling and weighting)
 -> output [6,604,256]
```

There are four projected Z anchors and eight total sampling points. The source
reshapes offsets so each anchor receives two points, then flattens anchors back
to eight locations. Dumps contain operator inputs, learned parameters,
reconstructed sampling locations using the exact source formula, and output;
the CUDA kernel was not modified and `sampled_features` is intentionally not
captured.

## Temporal deformable attention

The source is
`projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`, class
`TemporalSelfAttention.forward`. Its current query is `[1,2500,256]`; effective
value is `[2,2500,256]` (history/current, or current/current on Frame 0).
The module concatenates the first value slice and current query to 512 channels,
then predicts offsets `[2,2500,8,1,4,2]` and weights `[2,2500,8,1,4]`.
Locations are the appropriate shifted/unshifted `[2,2500,1,2]` reference grid
plus normalized offsets. The CUDA operator outputs both queue entries, which
the module averages before output projection and residual addition; final
layer output is `[1,2500,256]`.

## Decoder and box layout

`DetectionTransformerDecoder.forward` produces six object-feature tensors,
each `[900,1,256]`; initial references are `[1,900,3]` and refined references
are `[6,1,900,3]`. `BEVFormerHead.forward` emits complete layer stacks:
classification `[6,1,900,10]` and regression `[6,1,900,10]`.

`projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`, class
`NMSFreeCoder.decode_single`, uses last-layer sigmoid class scores and top-k,
then calls `projects/mmdet3d_plugin/core/bbox/util.py::denormalize_bbox`.
The regression layout before decode is
`[cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy]`; decoded boxes are
`[x,y,z,w,l,h,yaw,vx,vy]`, observed shape `[300,9]`.
