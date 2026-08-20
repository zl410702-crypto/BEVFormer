# Golden Tensor Boundary List

This list freezes 30 deployment-oriented boundaries selected from the 280
three-frame debug tensors in sequence
`fcbccedd61424f1b85dcbf8f897f9754_3e8750f3`. The primary stateful reference is
Frame 1. A shape mismatch is a failure: comparison must not silently transpose
or reshape tensors. Any required layout conversion must be explicit and
recorded in the ONNX/TRT comparison report.

Priority meanings: **P0** must align before deployment proceeds; **P1** locates
the source of a P0 failure; **P2** is advanced or end-to-end validation.

## Boundary table

| ID | Tensor Name | Module | Semantic | Golden Shape | Layout | Dtype | Frame | Golden Path | PyTorch Source | ONNX Boundary | TRT Boundary | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G01 | `backbone_c5` | Backbone | ResNet stage-4 camera features before FPN | `[6,2048,15,25]` | `[Ncam,C,H,W]` | float32 | All | `frame_001/backbone/c5.npy` | `mmdet ResNet.forward`; `BEVFormer.extract_img_feat` | Optional | Optional | P1 |
| G02 | `fpn_level_0` | FPN | Only FPN level consumed by tiny encoder | `[6,256,15,25]` | `[Ncam,C,H,W]` | float32 | All | `frame_001/fpn/level_0.npy` | `mmcv FPN.forward`; `BEVFormer.extract_img_feat` | Required | Required | P0 |
| G03 | `bev_query_embedding` | BEV Input | Learned unbatched 50x50 BEV queries | `[2500,256]` | `[Nbev,C]` | float32 | All | `frame_001/bev/bev_query.npy` | `bevformer_head.py`; `BEVFormerHead.forward` | Optional | Optional | P1 |
| G04 | `bev_positional_encoding` | BEV Input | Positional encoding before flatten/permute | `[1,256,50,50]` | `[B,C,Hbev,Wbev]` | float32 | All | `frame_001/bev/bev_pos.npy` | `BEVFormerHead.forward` | Optional | Optional | P1 |
| G05 | `reference_points_2d` | BEV Encoder | Normalized BEV cell centers for temporal attention | `[1,2500,1,2]` | `[B,Nbev,L,XY]` | float32 | All | `frame_001/bev/reference_points_2d.npy` | `encoder.py`; `BEVFormerEncoder.get_reference_points` | Optional | Optional | P1 |
| G06 | `reference_points_3d` | BEV Encoder | Four normalized pillar anchors per BEV cell | `[1,4,2500,3]` | `[B,D,Nbev,XYZ]` | float32 | All | `frame_001/bev/reference_points_3d.npy` | `encoder.py`; `BEVFormerEncoder.get_reference_points` | Optional | Optional | P1 |
| G07 | `prev_bev_stored` | Runtime State | Raw prior encoder output saved before next-frame alignment | `[2500,1,256]` | `[Nbev,B,C]` | float32 | Frame 1/2 only | `frame_001/temporal/prev_bev_stored.npy` | `bevformer.py`; `BEVFormer.forward_test` | Required | Required | P0 |
| G08 | `prev_bev_aligned` | Temporal Alignment | Historical BEV after rotation at encoder boundary | `[2500,1,256]` | `[Nbev,B,C]` | float32 | Frame 1/2 only | `frame_001/temporal/prev_bev_aligned.npy` | `transformer.py`; `PerceptionTransformer.get_bev_features` | Alignment-boundary dependent | Alignment-boundary dependent | P0 |
| G09 | `temporal_layer0_query` | Temporal Self Attention | Layer-0 current BEV query after batching | `[1,2500,256]` | `[B,Nbev,C]` | float32 | All | `frame_001/temporal/layer_0/query.npy` | `encoder.py`; `BEVFormerLayer.forward` | Required | Required | P0 |
| G10 | `temporal_layer0_effective_value` | Temporal Self Attention | History/current queue actually consumed by TSA | `[2,2500,256]` | `[B×Q,Nbev,C]` | float32 | All | `frame_001/temporal/layer_0/value_effective.npy` | `temporal_self_attention.py`; `TemporalSelfAttention.forward` | Required | Required | P0 |
| G11 | `temporal_layer0_reference_points` | Temporal Self Attention | Shifted history plus current BEV reference grids | `[2,2500,1,2]` | `[B×Q,Nbev,L,XY]` | float32 | All | `frame_001/temporal/layer_0/reference_points.npy` | `encoder.py`; `BEVFormerEncoder.forward` | Required | Required | P0 |
| G12 | `temporal_layer0_sampling_offsets` | Temporal Deformable Attention | Learned TSA offsets after queue reshape | `[2,2500,8,1,4,2]` | `[B×Q,Nbev,H,L,P,XY]` | float32 | All | `frame_001/temporal/layer_0/sampling_offsets.npy` | `TemporalSelfAttention.forward` | Optional | Optional | P1 |
| G13 | `temporal_layer0_attention_weights` | Temporal Deformable Attention | Softmax TSA weights after queue reshape | `[2,2500,8,1,4]` | `[B×Q,Nbev,H,L,P]` | float32 | All | `frame_001/temporal/layer_0/attention_weights.npy` | `TemporalSelfAttention.forward` | Optional | Optional | P1 |
| G14 | `temporal_layer0_output` | Temporal Self Attention | Layer-0 temporal result after projection and residual | `[1,2500,256]` | `[B,Nbev,C]` | float32 | All | `frame_001/temporal/layer_0/output.npy` | `TemporalSelfAttention.forward` | Required | Required | P0 |
| G15 | `spatial_layer0_flattened_value` | Spatial Cross Attention | Camera/FPN value before camera rebatching | `[6,375,1,256]` | `[Ncam,Nvalue,B,C]` | float32 | All | `frame_001/spatial/layer_0/value.npy` | `transformer.py`; `PerceptionTransformer.get_bev_features` | Required | Required | P0 |
| G16 | `reference_points_cam` | Camera Projection | Projected normalized image points for four heights | `[6,1,2500,4,2]` | `[Ncam,B,Nbev,D,XY]` | float32 | All | `frame_001/bev/reference_points_cam.npy` | `encoder.py`; `BEVFormerEncoder.point_sampling` | Required | Required | P0 |
| G17 | `bev_visibility_mask` | Camera Projection | Visibility of projected BEV anchors | `[6,1,2500,4]` | `[Ncam,B,Nbev,D]` | bool | All | `frame_001/bev/bev_mask.npy` | `BEVFormerEncoder.point_sampling` | Optional | Optional | P1 |
| G18 | `spatial_layer0_rebatch_query` | MSDeformableAttention3D | Visible queries padded and flattened per camera; `604` is the observed `Nquery_max` | `[6,604,256]` | `[B×Ncam,Nquery_max_observed,C]` | float32 | All | `frame_001/spatial/layer_0/deformable/query.npy` | `spatial_cross_attention.py`; `SpatialCrossAttention.forward` | Required | Required | P0 |
| G19 | `spatial_layer0_sampling_offsets` | MSDeformableAttention3D | Learned spatial plugin offsets | `[6,604,8,1,8,2]` | `[B×Ncam,Nquery_max,H,L,P,XY]` | float32 | All | `frame_001/spatial/layer_0/deformable/sampling_offsets.npy` | `MSDeformableAttention3D.forward` | Optional | Optional | P1 |
| G20 | `spatial_layer0_attention_weights` | MSDeformableAttention3D | Softmax spatial plugin weights | `[6,604,8,1,8]` | `[B×Ncam,Nquery_max,H,L,P]` | float32 | All | `frame_001/spatial/layer_0/deformable/attention_weights.npy` | `MSDeformableAttention3D.forward` | Optional | Optional | P1 |
| G21 | `spatial_layer0_plugin_output` | MSDeformableAttention3D | Camera-rebatched deformable operator output | `[6,604,256]` | `[B×Ncam,Nquery_max,C]` | float32 | All | `frame_001/spatial/layer_0/deformable/output.npy` | `MSDeformableAttention3D.forward` | Required | Required | P0 |
| G22 | `spatial_layer0_output` | Spatial Cross Attention | Scatter/average/project/residual result on full BEV grid | `[1,2500,256]` | `[B,Nbev,C]` | float32 | All | `frame_001/spatial/layer_0/output.npy` | `SpatialCrossAttention.forward` | Optional | Optional | P1 |
| G23 | `final_bev` | BEV Encoder | Final encoder output and next-frame state source | `[2500,1,256]` | `[Nbev,B,C]` | float32 | All | `frame_001/bev/final_bev.npy` | `BEVFormerHead.forward`; `PerceptionTransformer.forward` | Required | Required | P0 |
| G24 | `decoder_layer0_output` | Detection Decoder | First decoder layer object features | `[900,1,256]` | `[Nobject,B,C]` | float32 | All | `frame_001/decoder/layer_0/object_feature.npy` | `decoder.py`; `DetectionTransformerDecoder.forward` | Optional | Optional | P1 |
| G25 | `decoder_layer5_output` | Detection Decoder | Last decoder layer object features | `[900,1,256]` | `[Nobject,B,C]` | float32 | All | `frame_001/decoder/layer_5/object_feature.npy` | `DetectionTransformerDecoder.forward` | Required | Required | P0 |
| G26 | `all_decoder_cls_scores` | Detection Head | Class logits from all six decoder layers | `[6,1,900,10]` | `[Ldec,B,Nobject,Nclass]` | float32 | All | `frame_001/detection/cls_scores.npy` | `bevformer_head.py`; `BEVFormerHead.forward` | Required | Required | P0 |
| G27 | `all_decoder_bbox_preds` | Detection Head | Ten-code regression outputs from all decoder layers | `[6,1,900,10]` | `[Ldec,B,Nobject,Ncode]` | float32 | All | `frame_001/detection/bbox_preds.npy` | `BEVFormerHead.forward` | Required | Required | P0 |
| G28 | `decoded_boxes` | Detection Decode | Top-k decoded `[x,y,z,w,l,h,yaw,vx,vy]` boxes | `[300,9]` | `[Ndetection,box9]` | float32 | All | `frame_001/detection/decoded_boxes.npy` | `nms_free_coder.py`; `NMSFreeCoder.decode_single` | Not required | End-to-End only | P2 |
| G29 | `decoded_scores` | Detection Decode | Top-k sigmoid detection scores | `[300]` | `[Ndetection]` | float32 | All | `frame_001/detection/decoded_scores.npy` | `NMSFreeCoder.decode_single` | Not required | End-to-End only | P2 |
| G30 | `decoded_labels` | Detection Decode | Top-k class indices | `[300]` | `[Ndetection]` | int64 | All | `frame_001/detection/decoded_labels.npy` | `NMSFreeCoder.decode_single` | Not required | End-to-End only | P2 |

`Q` in `B×Q` means the temporal queue length (2), not query count. `H` means
attention heads, `L` feature levels, `P` sampling points, and `D` pillar height
anchors. In particular, BEV query is `[Nbev,C]`, temporal tensors use
`[B,Nbev,C]`, while final BEV and decoder features use `[N,B,C]` layouts.

## Temporal Runtime State Boundaries

```text
Frame t final_bev [Nbev,B,C] (G23)
        |
        v       exact state assignment
Runtime State / Frame t+1 prev_bev_stored [Nbev,B,C] (G07)
        |
        v       torchvision nearest rotation
prev_bev_aligned [Nbev,B,C] (G08)
        |
        v       permute + stack with current query
temporal effective value [B*queue,Nbev,C] (G10)
        |
        v
Temporal Self Attention output [B,Nbev,C] (G14)
```

The Golden experiment verified both transitions:

```text
frame_000 final_bev == frame_001 prev_bev_stored
frame_001 final_bev == frame_002 prev_bev_stored
MAE=0, MaxAE=0, RMSE=0, Cosine=1
```

G08 is after rotation/alignment and is not expected to equal G23/G07. Frame 0
has no G07/G08. It remains useful for checking the official no-history branch,
where current query is duplicated to form G10, but Frame 1 is the primary
deployment reference. G08 remains P0, but its ONNX/TRT interface requirement
depends on the final deployment partition: if rotation/alignment is inside the
exported graph or engine, expose/compare G08 there; if alignment is an external
runtime component, compare G08 at that component boundary instead. This list
does not prematurely require G08 to be an engine input or output.

## Deformable Attention Plugin Boundaries

For Temporal Deformable Attention, core plugin inputs are G09 query, G10 value,
and G11 reference points; G14 is the fused module output. G12/G13 are P1 debug
boundaries for distinguishing linear/softmax error from deformable sampling
error. The existing `sampling_locations.npy` remains available as deep debug,
but is derived from G11/G12 and is not frozen as a long-term boundary.

For Spatial MSDeformableAttention3D, G18 is the rebatched query, G15 supplies
the upstream flattened camera value (the plugin-local reshaped value is
`[6,375,256]`), G16 supplies projection coordinates before visibility
rebatching, and G21 is the direct plugin output. G19/G20 are P1. The operator
also receives constant `spatial_shapes=[[15,25]]` and `level_start_index=[0]`;
these are validated artifacts but not numeric compare boundaries for this
one-level tiny model. G22 catches errors in scatter, camera averaging, output
projection, and residual after the plugin.

For G18, `604` is the **observed** `Nquery_max` in all three frames of this
Golden sequence, not a universal static model dimension. `SpatialCrossAttention`
derives it from the maximum number of visible BEV queries after applying each
camera's `bev_mask`, then pads during camera rebatching. Different calibration,
image visibility, or input data can therefore change this dimension. The same
observed padded-query axis propagates into G19, G20, and G21. Deployment code
must either support that dynamic dimension or document and validate an explicit
fixed-capacity/padding contract.

## P0 Deployment Alignment Boundaries

First-pass PyTorch/ONNX/TRT FP32/TRT FP16 comparison uses exactly these 15:

```text
G02 fpn_level_0
G07 prev_bev_stored
G08 prev_bev_aligned
G09 temporal_layer0_query
G10 temporal_layer0_effective_value
G11 temporal_layer0_reference_points
G14 temporal_layer0_output
G15 spatial_layer0_flattened_value
G16 reference_points_cam
G18 spatial_layer0_rebatch_query
G21 spatial_layer0_plugin_output
G23 final_bev
G25 decoder_layer5_output
G26 all_decoder_cls_scores
G27 all_decoder_bbox_preds
```

If a P0 fails, expand comparison to adjacent P1 boundaries. G28-G30 validate
the complete detection pipeline only when decode/post-processing runs outside
the TensorRT engine.

## Validation and stability

All 30 selected paths exist, load with `np.load(..., allow_pickle=False)`, and
match the source manifest shape/dtype. Every boundary marked All Frames has
identical shape and dtype in Frames 0, 1, and 2. G07/G08 exist only in Frames 1
and 2 and are stable between those frames. No selected boundary changes shape
across its applicable frames.
