# BEVFormer Temporal Rotation Rewrite

## 1. Problem

The non-first-frame temporal BEV rotation in the current BEVFormer tiny Golden
Reference uses `torchvision.transforms.functional.rotate`. With torchvision
0.10.1, this tensor rotation ultimately executes:

```text
aten::grid_sampler
aten::grid_sampler_2d
```

PyTorch 1.9.1 does not provide a standard ONNX opset-13 exporter path for this
grid sampler. This document concerns the current tiny Golden Reference path in
`transformer.py`; it does not concern the separate bilinear call in
`transformerV2.py`.

## 2. Reference Path

Source:

```text
projects/mmdet3d_plugin/bevformer/modules/transformer.py
PerceptionTransformer.get_bev_features()
```

The temporal layout transition is:

```text
prev_bev [2500, 1, 256]
  -> reshape / permute
rotate input [256, 50, 50]
  -> torchvision.transforms.functional.rotate
internal grid_sample input [1, 256, 50, 50]
grid [1, 50, 50, 2]
internal output [1, 256, 50, 50]
  -> squeeze / permute / reshape
rotated prev_bev [2500, 1, 256]
```

The actual parameters are:

```text
interpolation = NEAREST
padding_mode = zeros
align_corners = False
expand = False
fill = None
```

`torchvision.rotate` supplies the batch dimension before its internal grid
sampler and removes it again on return.

## 3. Why Rewrite Is Required

The deployment target is:

```text
PyTorch 1.9.1 -> ONNX opset 13
```

The `grid_sampler` used internally by this version of `torchvision.rotate`
cannot be directly emitted as a standard opset-13 graph. The deployment path
therefore replaces the sampler with equivalent basic tensor operations:

```text
torchvision.rotate internal grid sampler
  -> deployment-only NEAREST tensor implementation
```

The Reference path remains unchanged and continues to use
`torchvision.rotate`.

## 4. Rewrite Implementation

Implementation:

```text
tools/deployment/grid_sample_rewrite.py
```

The rewrite intentionally supports only the semantics required by the current
BEVFormer tiny temporal rotation:

- 4D NCHW input;
- NEAREST interpolation;
- zeros padding;
- `align_corners=False`;
- floating-point input and grid.

It is not a general `grid_sample` implementation and it is not a bilinear
rewrite. Its implementation uses basic tensor operations including:

```text
round
clamp
cast
gather
reshape
comparison
mul
```

For `align_corners=False`, normalized coordinates are converted with:

```text
x = ((grid_x + 1) * width  - 1) / 2
y = ((grid_y + 1) * height - 1) / 2
```

Rounded indices are clamped only to make `gather` safe. A mask computed from
the original rounded indices restores zeros-padding behavior for out-of-range
samples.

## 5. Real Temporal Fixture

The fixture was captured by running two consecutive samples from the same
scene through one model instance:

```text
first token:  3e8750f331d7499e9b5123e9eb70f2e2
second token: 3950bd41f74548429c0f7700ff3d8269
timestamp:    1533151604048025
relative yaw: -1.0353196091174368 deg
```

The files currently exist at:

```text
golden_samples/3950bd41f74548429c0f7700ff3d8269/
  temporal_grid_sample_reference.pt
  temporal_grid_sample_reference.json
```

The fixture stores the real internal grid-sampler input, torchvision-generated
grid, reference output, sample metadata, runtime operator names, and rewrite
error metrics. Its recorded dtype is `torch.float32`.

## 6. Numerical Validation

The real temporal fixture comparison produced:

```text
Reference shape: (1, 256, 50, 50)
Rewrite shape:   (1, 256, 50, 50)

Max abs error:      0.0
Mean abs error:     0.0
Max relative error: 0.0

Numerical validation: PASS
```

The rewrite is exactly equal to torchvision's real NEAREST temporal case for
this fixture.

## 7. Tests

The focused suite is:

```text
tests/test_grid_sample_rewrite.py
```

It covers random tensors, integer and subpixel positions, all four image
boundaries, completely out-of-range samples, normalized coordinates around
`-1` and `+1`, multiple channels, multiple batches, validation failures, the
real temporal fixture, and the no-ONNX environment status.

The latest executed related-suite result is:

```text
27 passed
```

Run the following to reproduce it:

```bash
pytest -q \
  tests/test_grid_sample_rewrite.py \
  tests/test_profile_golden_sample_ops.py \
  tests/test_run_golden_sample.py
```

## 8. ONNX Status

```text
Numerical rewrite:              VERIFIED
ONNX opset13 export:            PENDING
ONNX graph no-GridSample check: PENDING
```

The frozen Reference environment does not contain the `onnx` package. It is
intentionally not installed into that environment.

The next stage must use an independent Export Environment to perform:

```text
rewrite
  -> torch.onnx.export(opset_version=13)
  -> inspect ONNX graph nodes
  -> confirm no GridSample, grid_sampler, or custom grid-sample node remains
```

## 9. Reference Safety

`projects/mmdet3d_plugin/bevformer/modules/transformer.py` was not modified.
Reference inference still executes `torchvision.rotate`; the rewrite is an
independent deployment-only module and is not imported by the Reference model.
The existing Golden Reference output is unaffected.

## 10. Deployment Decision

```text
Temporal rotation blocker status

Runtime problem:     torchvision.rotate -> grid_sampler
Deployment strategy: REWRITE
Numerical status:    VERIFIED
ONNX status:         PENDING
TensorRT status:     NOT PART OF THIS STAGE
```

## 11. Next Step

1. Create an independent ONNX Export Environment.
2. Export the minimal rewrite wrapper with ONNX opset 13.
3. Confirm the graph contains no `GridSample` or `grid_sampler` node.
4. Begin full BEVFormer ONNX export.
5. Handle `MultiScaleDeformableAttnFunction_fp32` with a custom ONNX node and
   the later TensorRT plugin path.
