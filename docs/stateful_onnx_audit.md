# BEVFormer Tiny Stateful Single-Frame ONNX Audit

## Export scope and source chain

The legacy static exporter is `tools/export_bevformer_onnx.py`. It constructs
`tools/deployment/bevformer_onnx_wrapper.py::BEVFormerONNXWrapper` with
`use_prev_bev=False`, so its `forward` forces `prev_bev=None`. The shared custom
symbolic is `ms_deformable_attention_symbolic` in that wrapper module and emits
`bevformer::MSDeformableAttention` without changing its five-input schema.

The stateful exporter is `tools/export_bevformer_stateful_onnx.py`. It uses
`tools/deployment/bevformer_stateful_onnx_wrapper.py::BEVFormerStatefulONNXWrapper.forward`
and a model-instance-only override of
`PerceptionTransformer.get_bev_features`. Production BEVFormer sources and the
legacy ONNX artifact are unchanged.

## Runtime interface

All shapes are static for the current deployment baseline.

| Input | Shape | Dtype | Layout / unit | Meaning |
|---|---:|---|---|---|
| `images` | `[1,6,3,480,800]` | float32 | `[B,Ncam,C,H,W]` | Current six-camera images |
| `prev_bev` | `[2500,1,256]` | float32 | `[Nbev,B,C]` | Raw previous `bev_embed`; identical to output layout |
| `can_bus` | `[1,18]` | float32 | `[B,18]` | Metadata after official `forward_test` delta processing; `0:3` translation delta, `-2` current ego heading in radians, `-1` rotation delta in degrees |
| `shift` | `[1,2]` | float32 | `[B,XY]` | `[shift_x,shift_y]`, computed on the host with the exact reference formula |
| `lidar2img` | `[1,6,4,4]` | float32 | `[B,Ncam,4,4]` | Per-frame camera projection matrices |
| `use_prev_bev` | `[1]` | float32 | `[B]`, exactly 0 or 1 | Arithmetic state gate; no ONNX control-flow branch |

| Output | Shape | Dtype | Layout |
|---|---:|---|---|
| `all_cls_scores` | `[6,1,900,10]` | float32 | `[Ldec,B,Nobject,Nclass]` |
| `all_bbox_preds` | `[6,1,900,10]` | float32 | `[Ldec,B,Nobject,Ncode]` |
| `bev_embed` | `[2500,1,256]` | float32 | `[Nbev,B,C]` |

`bev_embed` can be passed directly as the next invocation's `prev_bev`; no
implicit transpose is allowed. `shift` is a host input because the reference
formula uses `atan2`, while PyTorch 1.9/opset 13 is a fragile place to reproduce
that calculation. It is not a learned value. Host code must use the equations
from `PerceptionTransformer.get_bev_features` and the delta-form `can_bus`.

## Frame 0 initialization

The unified interface uses:

```text
prev_bev     = zeros([2500,1,256])
use_prev_bev = [0.0]
can_bus[0:3] = 0
can_bus[-1]  = 0 degrees
shift        = [[0,0]]
```

The gate does more than zero the history. At each encoder layer it selects the
official no-history effective queue `[current_layer_query,current_layer_query]`
and unshifted reference points. This is necessary because merely supplying a
zero history is not equivalent to `prev_bev=None`. The experiment verified the
gate strategy against the original `None` path with zero error for all three
outputs.

For Frame 1+, pass the prior `bev_embed`, set the gate to 1, provide delta-form
motion and host-computed shift, and provide the current frame's `lidar2img`.

## PyTorch validation

Stateful wrapper with Frame 1 images, Frame 0 Golden final BEV, and real Frame 1
motion versus Golden Frame 1:

| Tensor | Shape | MAE | MaxAE | RMSE | Cosine |
|---|---:|---:|---:|---:|---:|
| `bev_embed` | `[2500,1,256]` | 0 | 0 | 0 | 1 |
| `all_cls_scores` | `[6,1,900,10]` | 0 | 0 | 0 | 1 |
| `all_bbox_preds` | `[6,1,900,10]` | 0 | 0 | 0 | 0.9999999999999999 |

Real Frame 1 history versus zero history with `use_prev_bev=1` proves that
history affects outputs:

| Tensor | MAE | MaxAE | RMSE | Cosine |
|---|---:|---:|---:|---:|
| `bev_embed` | 0.35504960 | 2.54637070 | 0.45440000 | 0.82054811 |
| `all_cls_scores` | 0.56155904 | 8.60607862 | 0.76911825 | 0.99394234 |
| `all_bbox_preds` | 0.28503741 | 9.93623400 | 0.55889686 | 0.99880423 |

Frame 0 gate=0 versus official `prev_bev=None` has MAE/MaxAE/RMSE 0 for all
three outputs.

## Temporal graph dependency

`prev_bev` is not an unused interface decoration. Its direct consumer is ONNX
`Gather_301`. A shortest graph path to the first temporal custom operator
contains:

```text
prev_bev
 -> Gather -> Reshape -> Transpose
 -> Reshape -> GatherElements            # nearest rotate sampling
 -> Reshape -> Mul -> Squeeze -> Transpose
 -> Reshape -> Unsqueeze -> Concat        # history/current queue
 -> gate Mul/Add
 -> value projection MatMul/Add/Reshape
 -> bevformer::MSDeformableAttention_2061
```

The first custom node's value, sampling-location, and attention-weight inputs
all have upstream dependency on `prev_bev`; its value input also depends on
current `images`. This proves the effective queue is history/current rather
than current/current for gate=1. `prev_bev` reaches all graph outputs and is
upstream of all 12 custom nodes. The sensitivity experiment supplies the
independent numerical proof.

## Alignment implementation

The PyTorch reference operation is torchvision tensor NEAREST rotation. The
stateful wrapper keeps the angle as `can_bus[:,-1]` in **degrees** and exports:

```text
rotation_delta
 -> degree-to-radian Mul
 -> Sin + Cos
 -> affine matrix Stack/Reshape/Transpose
 -> MatMul with fixed 50x50 base grid
 -> coordinate Round + Clip + Cast
 -> GatherElements from prev_bev
 -> validity-mask Mul
```

There is no ONNX `GridSample` or `AffineGrid`. This is the repository's
opset-13-compatible nearest-grid equivalent. The graph contains 1 `Sin`, 1
`Cos`, 2 `Round`, 59 `Clip`, and 1 `GatherElements`; the latter is directly
downstream of both `prev_bev` and `can_bus`. Separate unit tests compare the
tensor-angle implementation to torchvision at zero, the real Frame 1 angle,
and an additional nontrivial angle with exact equality.

## Deformable attention audit

There are exactly 12 custom nodes in execution order:

```text
3 x Temporal Self Attention
3 x Spatial MSDeformableAttention3D
6 x Detection Decoder deformable attention
```

Every node has:

```text
domain: bevformer
op_type: MSDeformableAttention
inputs: value, spatial_shapes, level_start_index,
        sampling_locations, attention_weights
attribute: im2col_step = 64
```

The schema is unchanged from the legacy model. Node indices/names are trace
artifacts and are recorded in the generated audit JSON, not treated as a
stable API.

## Static spatial rebatch limitation

The graph is intentionally static. `SpatialCrossAttention.forward` uses Python
`len`/iteration while tracing camera visibility rebatching. Consequently the
observed Frame 1 `Nquery_max=604` is embedded in the traced allocation. It is
also 604 for the three Golden frames, but is not a universal model semantic.
Supporting arbitrary visibility maxima requires a separately designed
fixed-capacity contract or a future dynamic rewrite; this export does not
claim that 604 generalizes beyond the audited baseline.

## Golden boundary mapping

Natural graph I/O boundaries are G07 `prev_bev_stored` (`prev_bev` input), G23
`final_bev` (`bev_embed` output), G26 `all_decoder_cls_scores`, and G27
`all_decoder_bbox_preds`. G08-G21 and G25 are internal tensors. They remain
available as future debug outputs but were deliberately not promoted to the
production interface during this task.

## Artifact audit

```text
file: artifacts/bevformer_tiny_stateful_opset13.onnx
size: 134,955,322 bytes
sha256: ec453ff17e2a3453d3bd27a761f02b506c29e6c7187e22be68ed0a12342bb994
IR version: 6
producer: pytorch 1.9
ai.onnx opset: 13
bevformer domain opset: 1
node count: 14,956
onnx.checker: PASS
shape inference: PASS (14,894 value_info entries)
GridSample nodes: 0
custom nodes: 12
```

The export environment has no `onnxruntime`; whole-graph ORT execution was not
attempted. No TensorRT command, build, plugin compilation, or inference was
performed.
