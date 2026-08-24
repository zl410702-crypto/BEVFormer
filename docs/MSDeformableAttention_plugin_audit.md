# MSDeformableAttention TensorRT Plugin Audit — Phase 1

## Scope and result

This phase audits the current Stateful static-SCA ONNX only. It does not add
TensorRT source, build an engine, or implement a CUDA kernel.

Audited model:

```text
artifacts/bevformer_tiny_stateful_static_sca_chunk640_opset13.onnx
ai.onnx opset: 13
custom domain: bevformer opset 1
custom op: bevformer::MSDeformableAttention
node count: 12
```

Feasibility conclusion: **YES**. One TensorRT 8.6.4
`MSDeformableAttentionPlugin` class can serve all 12 nodes as 12 plugin layer
instances. The plugin boundary starts after sampling locations and attention
weights have already been calculated, so the different query/reference-point
generation semantics do not change the custom operator's mathematics.

This conclusion is limited to the audited FP32 ONNX contract. It does not yet
claim TensorRT parser, engine, CUDA, or FP16 readiness.

## Implementation provenance and call chains

### [Reference Repo] Temporal Self Attention

```text
projects/mmdet3d_plugin/bevformer/modules/transformer.py
PerceptionTransformer.get_bev_features
  ↓
projects/mmdet3d_plugin/bevformer/modules/encoder.py
BEVFormerEncoder.forward → BEVFormerLayer.forward
  ↓ attentions[0]
projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
TemporalSelfAttention.forward
  ↓
projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
MultiScaleDeformableAttnFunction_fp32.apply/forward
  ↓
mmcv._ext.ms_deform_attn_forward
```

`TemporalSelfAttention` constructs query features by concatenating the first
history queue value and current query. Its two-element BEV queue is folded into
the custom op batch dimension. After the custom op, the two queue outputs are
reshaped and averaged outside the plugin boundary.

### [Reference Repo] Spatial Cross Attention

```text
BEVFormerEncoder.forward → BEVFormerLayer.forward
  ↓ attentions[1]
projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
SpatialCrossAttention.forward
  ↓
MSDeformableAttention3D.forward
  ↓
MultiScaleDeformableAttnFunction_fp32.apply/forward
  ↓
mmcv._ext.ms_deform_attn_forward
```

The reference implementation compacts visible BEV queries per camera before
`MSDeformableAttention3D` and scatters/averages camera results afterward.

### [Reference Repo] Detection Decoder Cross Attention

```text
projects/mmdet3d_plugin/bevformer/modules/transformer.py
PerceptionTransformer.forward
  ↓
projects/mmdet3d_plugin/bevformer/modules/decoder.py
DetectionTransformerDecoder.forward
  ↓ decoder layer attentions[1]
CustomMSDeformableAttention.forward
  ↓
MultiScaleDeformableAttnFunction_fp32.apply/forward
  ↓
mmcv._ext.ms_deform_attn_forward
```

### [Official implementation] MMCV 1.4.0

The loaded extension belongs to MMCV 1.4.0 in the export environment:

```text
/data/conda_envs/bevformer_export/lib/python3.8/site-packages/mmcv/
  ops/multi_scale_deform_attn.py
  ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu
  ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
```

The forward entry computes `im2col_step_ = min(batch, im2col_step)`, then calls
`ms_deformable_im2col_cuda`. The forward CUDA kernel maps one logical thread to
one `(batch, query, head, channel)` output element, loops over levels and
sampling points, performs normalized-coordinate bilinear interpolation, applies
the attention weight, and accumulates one scalar output. This is the preferred
kernel source for the later implementation phase; no kernel is copied or
modified in phase 1.

### [Current deployment/export code]

```text
tools/deployment/bevformer_stateful_onnx_wrapper.py
BEVFormerStatefulONNXWrapper / stateful_get_bev_features
  ↓ deployment-only state, alignment and fixed-Nq Spatial SCA

tools/deployment/bevformer_onnx_wrapper.py
deployment_export_overrides
  ↓ temporarily installs
ms_deformable_attention_symbolic
  ↓ emits
bevformer::MSDeformableAttention(
    value, spatial_shapes, level_start_index,
    sampling_locations, attention_weights,
    im2col_step=64)
```

The export-only static Spatial SCA uses all 2500 queries and a runtime
visibility mask. Its two Linear layers use fixed chunk 640, but those Linear
operations are upstream of the custom op and do not alter the five-input plugin
schema.

## Twelve-node ONNX audit

PyTorch 1.9 does not attach an ONNX type/shape function to this custom domain,
so standard shape inference records FLOAT/INT64 input dtypes but leaves most
custom-op input ranks empty. The inspection script therefore reports both the
raw `onnx_declared_shape` and a recovered shape. Recovery uses the nearest
attention-weight initializer scope, actual ONNX `spatial_shapes` Constant, and
the checked Tiny configuration/static wrapper contract. It fails rather than
guessing if the scope, dtype, arity, role count, or Linear output dimension does
not match.

All nodes have `im2col_step=64` and exactly five inputs in this order:

```text
value, spatial_shapes, level_start_index,
sampling_locations, attention_weights
```

| Index / Node | Role | Value | Locations | Weights | Output | H/L/P |
|---|---|---|---|---|---|---|
| 2060 `MSDeformableAttention_2060` | Temporal L0 | `[2,2500,8,32]` | `[2,2500,8,1,4,2]` | `[2,2500,8,1,4]` | `[2,2500,256]` | 8/1/4 |
| 2587 `MSDeformableAttention_2587` | Spatial L0 | `[6,375,8,32]` | `[6,2500,8,1,8,2]` | `[6,2500,8,1,8]` | `[6,2500,256]` | 8/1/8 |
| 2840 `MSDeformableAttention_2840` | Temporal L1 | `[2,2500,8,32]` | `[2,2500,8,1,4,2]` | `[2,2500,8,1,4]` | `[2,2500,256]` | 8/1/4 |
| 3367 `MSDeformableAttention_3367` | Spatial L1 | `[6,375,8,32]` | `[6,2500,8,1,8,2]` | `[6,2500,8,1,8]` | `[6,2500,256]` | 8/1/8 |
| 3620 `MSDeformableAttention_3620` | Temporal L2 | `[2,2500,8,32]` | `[2,2500,8,1,4,2]` | `[2,2500,8,1,4]` | `[2,2500,256]` | 8/1/4 |
| 4147 `MSDeformableAttention_4147` | Spatial L2 | `[6,375,8,32]` | `[6,2500,8,1,8,2]` | `[6,2500,8,1,8]` | `[6,2500,256]` | 8/1/8 |
| 4493 `MSDeformableAttention_4493` | Decoder L0 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |
| 4967 `MSDeformableAttention_4967` | Decoder L1 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |
| 5441 `MSDeformableAttention_5441` | Decoder L2 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |
| 5915 `MSDeformableAttention_5915` | Decoder L3 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |
| 6389 `MSDeformableAttention_6389` | Decoder L4 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |
| 6863 `MSDeformableAttention_6863` | Decoder L5 | `[1,2500,8,32]` | `[1,900,8,1,4,2]` | `[1,900,8,1,4]` | `[1,900,256]` | 8/1/4 |

Exact ONNX value names are machine-readable in
`artifacts/msda_onnx_audit.json`. Shared Constants are expected: all Spatial
nodes use `spatial_shapes=[[15,25]]`, `level_start_index=[0]`; Temporal and
Decoder use `[[50,50]]`, `[0]`.

Role counts are source-scope verified, not assumed from ordering:

```text
Temporal Self Attention:           3
Spatial Cross Attention:           3
Detection Decoder Cross Attention: 6
```

## Role semantics around the common boundary

### Temporal

- Query source: current BEV query concatenated with the history queue feature
  before the offsets/weights Linear layers.
- Value source: projected history and current BEV queue, with queue folded into
  `B=2`.
- Sampling location source: shifted/current 2-D BEV reference points plus
  learned normalized offsets.
- Output destination: reshape queue dimension, average history/current, then
  output projection and residual outside the custom op.

### Spatial

- Query source: current BEV query expanded to six cameras by the deployment
  static-Nq path, `Q=2500` for every camera.
- Value source: projected single-scale C5/FPN camera features; `15×25=375`
  values per camera and camera folded into `B=6`.
- Sampling location source: 3-D BEV anchors projected into each camera, safe
  invalid coordinates, plus learned offsets; eight total points.
- Output destination: reshape to six cameras, apply visibility mask, sum and
  divide by visible-camera count, then output projection and residual.

### Decoder

- Query source: 900 object-query features after decoder self-attention.
- Value source: final BEV encoder feature, `50×50=2500`, with `B=1`.
- Sampling location source: decoder reference points plus learned offsets.
- Output destination: decoder cross-attention output projection, residual,
  FFN, and iterative reference refinement outside the custom op.

The semantic differences are all outside the custom boundary. At the boundary,
all three roles use normalized `(x,y)` locations, identical level flattening,
the same bilinear sampling rule, and the same head-concatenated output layout.

## Unified TensorRT Plugin contract

Use one `IPluginV2DynamicExt` class and creator:

```text
MSDeformableAttentionPlugin
  ├─ 3 Temporal layer instances
  ├─ 3 Spatial layer instances
  └─ 6 Decoder layer instances
```

Symbols:

```text
B = effective batch (queue/camera may be folded here)
S = sum_l(H_l × W_l)
Q = query count
H = attention heads
L = feature levels
P = sampling points per level
D = channel dimension per head
C = H × D
```

| Position | Tensor | FP32 phase dtype | Rank / layout | Meaning |
|---:|---|---|---|---|
| 0 | `value` | FLOAT | 4, `[B,S,H,D]` | Flattened projected feature values, levels concatenated along S |
| 1 | `spatial_shapes` | INT64 in current ONNX | 2, `[L,2]` | `(height,width)` for each level |
| 2 | `level_start_index` | INT64 in current ONNX | 1, `[L]` | Start offset of each level in S |
| 3 | `sampling_locations` | FLOAT | 6, `[B,Q,H,L,P,2]` | Normalized `(x,y)` sample coordinates |
| 4 | `attention_weights` | FLOAT | 5, `[B,Q,H,L,P]` | Per-location weights, already softmax-normalized upstream |
| output 0 | `output` | FLOAT | 3, `[B,Q,H×D]` | Per-head sampled sums concatenated over H |

Required runtime/configuration checks for the later plugin:

```text
rank(value)=4
rank(spatial_shapes)=2 and spatial_shapes[-1]=2
rank(level_start_index)=1
rank(sampling_locations)=6 and last dimension=2
rank(attention_weights)=5
sum(height_l*width_l)=S
matching B/Q/H/L/P across inputs
output C=H*D
im2col_step attribute=64
```

TensorRT 8.6.4 compatibility note: the ONNX contract carries INT64 shape
inputs. Whether the parser/plugin boundary preserves INT64 or requires an
explicit, audited INT32 conversion must be decided in the implementation
phase; it must not be silently assumed.

## Mathematical contract

For output element `(b,q,h,d)`:

```text
output[b,q,h,d] =
  Σ over level l
  Σ over point p
    attention_weights[b,q,h,l,p]
    × bilinear(
        value[b, level_start_index[l]:, h, d],
        sampling_locations[b,q,h,l,p,:],
        spatial_shapes[l])
```

The normalized location `(x,y)` becomes:

```text
w_im = x * width  - 0.5
h_im = y * height - 0.5
```

If the point is inside the kernel's accepted border interval, four neighboring
feature values are bilinearly interpolated. The weighted values are accumulated
over `L×P`. Heads are not reduced together; `[H,D]` is flattened/concatenated
to `C=H×D` in the output.

## Phase-1 decision and next gate

```text
12-node audit: PASS
classification: PASS (3 Temporal + 3 Spatial + 6 Decoder)
single-plugin mathematical feasibility: YES
TensorRT source: NOT STARTED by design
CUDA implementation: NOT STARTED by design
FP32 engine/compare: NOT STARTED
FP16: NOT STARTED
```

Before implementation, review and approve:

1. one dynamic plugin class with 12 instances;
2. the five-input contract above;
3. explicit handling of ONNX INT64 shape tensors in TensorRT 8.6.4;
4. reuse/minimal adaptation of the MMCV 1.4.0 forward CUDA kernel.

