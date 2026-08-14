# BEVFormer Tiny ONNX Opset-13 Export

## Export Boundary

The first static export ends at the detection head raw outputs:

```text
images [1, 6, 3, 480, 800]
  -> ResNet-50
  -> FPN
  -> BEV encoder (50 x 50)
  -> six-layer detection decoder (900 object queries)
  -> classification / regression branches
  -> all_cls_scores [6, 1, 900, 10]
  -> all_bbox_preds [6, 1, 900, 10]
  -> bev_embed [2500, 1, 256]
```

`NMSFreeCoder.decode`, TopK, `atan2`, range filtering, and final boxes are not
part of the graph. Batch, camera count, image size, BEV size, and query count
are static in this stage. Golden metadata is frozen as export-time constants.

## Deployment Isolation

The wrapper is implemented in:

```text
tools/deployment/bevformer_onnx_wrapper.py
```

It calls the existing backbone, neck, transformer, and head, but returns before
postprocess. When a temporal input is enabled, the module-level torchvision
rotation reference is temporarily redirected to the numerically verified
NEAREST deployment rewrite. The override is restored in `finally`; no Reference
source file is changed.

The Golden token is the first scene frame, so this particular static export has
`prev_bev=None` and does not execute rotation. The wrapper keeps a separate
`use_prev_bev` mode for a later temporal-input export rather than substituting a
zero tensor for the Reference `None` semantics.

## Deformable Attention Custom Node

All three callers use the same
`MultiScaleDeformableAttnFunction_fp32.apply` signature:

```text
value
value_spatial_shapes
value_level_start_index
sampling_locations
attention_weights
im2col_step
```

The deployment symbolic emits one schema:

```text
domain:  bevformer
op_type: MSDeformableAttention
inputs:  value, spatial_shapes, level_start_index,
         sampling_locations, attention_weights
attr:    im2col_step
```

A successful Golden graph must contain 12 such nodes: three temporal attention,
three encoder spatial deformable attention, and six decoder deformable
cross-attention calls.

## Command

```bash
python tools/export_bevformer_onnx.py \
  --token 3e8750f331d7499e9b5123e9eb70f2e2 \
  --opset 13 \
  --output artifacts/bevformer_tiny_opset13.onnx
```

## Current Result

```text
ONNX export status: FAIL
Exporter started: NO
ONNX output: NOT GENERATED
ONNX checker: NOT RUN
GridSample remaining: NOT GENERATED
MSDeformableAttention custom nodes: NOT GENERATED
ONNX Runtime numerical validation: NOT RUN
```

The current Reference environment has neither `onnx` nor `onnxruntime`, and
CUDA is unavailable in the current execution context. No separate ONNX export
environment exists on this host. Per the environment-safety constraint, no
package was installed and PyTorch/MMCV/CUDA were not changed.

For the isolated Python 3.8 export environment, start evaluation with
`onnx==1.10.2`: it is contemporary with the PyTorch 1.9 generation and its
schema catalog includes opset 13 (ONNX 1.8 already introduced opset 13, while
1.10.2 supports through opset 15). Pin a protobuf version compatible with that
ONNX release rather than inheriting a modern protobuf blindly. This is an
environment proposal only; no package was installed in the Reference env.

## Tests

```bash
pytest -q \
  tests/test_bevformer_onnx_export.py \
  tests/test_grid_sample_rewrite.py \
  tests/test_profile_golden_sample_ops.py \
  tests/test_run_golden_sample.py
```

Current result:

```text
33 passed
```

## Success Checks Implemented

After export, the tool will:

1. run `onnx.checker.check_model`;
2. require default ONNX opset 13;
3. reject `GridSample`, `grid_sampler`, or similarly named nodes;
4. require exactly 12 `bevformer::MSDeformableAttention` nodes;
5. write `artifacts/bevformer_tiny_opset13_ops.json` with domain/op/count;
6. reference `docs/operator_compatibility.yaml` for follow-up review.

ONNX Runtime comparison remains optional and is reported as `NOT RUN` when the
package is absent. TensorRT is outside this stage.

## Next Gate

Create an isolated environment that preserves PyTorch 1.9.1/MMCV/CUDA ABI,
adds a compatible ONNX Python package, provides CUDA, and reruns the command.
The first actual exporter exception must be added to the blocker report before
any further rewrite is considered.
