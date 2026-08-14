# BEVFormer ONNX13 / TensorRT 8.4.12 Compatibility

> Generated from the real Golden Sample runtime inventory. A runtime operator is not automatically an ONNX node, and CUDA/library kernels are treated as implementation evidence rather than separate blockers.

## 1. Deployment Baseline

| Item | Value |
| --- | --- |
| Golden token | `3e8750f331d7499e9b5123e9eb70f2e2` |
| PyTorch / profiler CUDA | `1.9.1+cu111` / `11.1` |
| ONNX target | opset 13 |
| TensorRT target | 8.4.12.5 |
| PONY runtime | CUDA 11.4; cuDNN 8.4.1; aarch64 / sm_87 |
| Plugin API | `IPluginV2DynamicExt` |

Runtime classification (all 237 profiler keys):

- PyTorch ATen operator: 111
- MMCV / custom operator: 1
- CUDA kernel: 70
- cuDNN / CUTLASS / GEMM implementation kernel: 23
- Other runtime implementation detail: 32

Only 45 observed, deployment-relevant runtime operators are retained below. Shared tensor operators can appear in several modules; the global count remains name-deduplicated.

Module assignment evidence combines event shapes, the same-token `forward_trace.json`, repository call sites, and the documented call chain. This profiler export contains no stack/scope/parent timestamps, so shared ops are not assigned an invented exclusive owner.

## 2. Backbone

### Backbone.ResNet

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Backbone.ResNet | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Backbone.ResNet | aten::batch_norm | BatchNormalization | SUPPORTED | SUPPORTED | NATIVE | Eval-mode frozen BatchNorm; parser may fold constants. |
| Backbone.ResNet | aten::conv2d | Conv | SUPPORTED | SUPPORTED | NATIVE | Semantic convolution entry; cudnn_convolution and CUDA kernels are implementation details. |
| Backbone.ResNet | aten::max_pool2d | MaxPool | SUPPORTED | SUPPORTED | NATIVE | Indices are not consumed by the ResNet inference path. |
| Backbone.ResNet | aten::mul | Mul | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| Backbone.ResNet | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| Backbone.ResNet | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Backbone.ResNet | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |

Conclusion: Native 6; Conditional 2; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

## 3. Neck

### Neck.FPN

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Neck.FPN | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Neck.FPN | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Neck.FPN | aten::conv2d | Conv | SUPPORTED | SUPPORTED | NATIVE | Semantic convolution entry; cudnn_convolution and CUDA kernels are implementation details. |
| Neck.FPN | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| Neck.FPN | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| Neck.FPN | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Neck.FPN | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |

Conclusion: Native 5; Conditional 2; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

## 4. BEV Encoder

### BEVEncoder.TemporalSelfAttention

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| BEVEncoder.TemporalSelfAttention | MultiScaleDeformableAttnFunction_fp32 | Custom domain MSDeformableAttention | CUSTOM | CUSTOM | PLUGIN | No standard ONNX op captures this fused sampling/reduction; target API is IPluginV2DynamicExt. |
| BEVEncoder.TemporalSelfAttention | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| BEVEncoder.TemporalSelfAttention | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| BEVEncoder.TemporalSelfAttention | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| BEVEncoder.TemporalSelfAttention | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| BEVEncoder.TemporalSelfAttention | aten::grid_sampler *(latent)* | GridSample (introduced in ONNX opset 16) | UNSUPPORTED | UNKNOWN | REWRITE | Latent rotate-prev-BEV path; not executed because this Golden Sample is the first scene frame. |
| BEVEncoder.TemporalSelfAttention | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| BEVEncoder.TemporalSelfAttention | aten::mean | ReduceMean | SUPPORTED | SUPPORTED | NATIVE | Temporal queue reduction. |
| BEVEncoder.TemporalSelfAttention | aten::mul | Mul | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| BEVEncoder.TemporalSelfAttention | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| BEVEncoder.TemporalSelfAttention | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| BEVEncoder.TemporalSelfAttention | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| BEVEncoder.TemporalSelfAttention | aten::softmax | Softmax | SUPPORTED | CONDITIONAL | CHECK | Verify axis and dynamic-rank lowering in the exported graph. |
| BEVEncoder.TemporalSelfAttention | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 9; Conditional 3; Unsupported 0; Unknown 0; Plugin candidates 1; Risk **HIGH**.

### BEVEncoder.SpatialCrossAttention

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| BEVEncoder.SpatialCrossAttention | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| BEVEncoder.SpatialCrossAttention | aten::bitwise_and | And | CONDITIONAL | CONDITIONAL | CHECK | Boolean mask vs integer bitwise semantics depend on dtype. |
| BEVEncoder.SpatialCrossAttention | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| BEVEncoder.SpatialCrossAttention | aten::clamp | Clip | SUPPORTED | CONDITIONAL | CHECK | Verify scalar/tensor bound lowering and dtype. |
| BEVEncoder.SpatialCrossAttention | aten::cumsum | CumSum | SUPPORTED | CONDITIONAL | CHECK | Used to build level_start_index; TRT parser support/dtype must be tested. |
| BEVEncoder.SpatialCrossAttention | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| BEVEncoder.SpatialCrossAttention | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| BEVEncoder.SpatialCrossAttention | aten::embedding | Gather | SUPPORTED | CONDITIONAL | CHECK | Constant embedding weights and INT index type must parse correctly. |
| BEVEncoder.SpatialCrossAttention | aten::gt | Greater | SUPPORTED | CONDITIONAL | CHECK | Used in camera validity and optional score filtering. |
| BEVEncoder.SpatialCrossAttention | aten::index | Gather / GatherND / NonZero + GatherND | CONDITIONAL | CONDITIONAL | CHECK | Advanced and boolean indexing have multiple lowerings; SCA has dynamic valid-query indexing. |
| BEVEncoder.SpatialCrossAttention | aten::index_put_ | ScatterND / ScatterElements | CONDITIONAL | UNKNOWN | CHECK | Accumulating writes and duplicate indices may not match plain ScatterND semantics. |
| BEVEncoder.SpatialCrossAttention | aten::index_select | Gather | SUPPORTED | CONDITIONAL | CHECK | Check INT32/INT64 indices on the target parser. |
| BEVEncoder.SpatialCrossAttention | aten::matmul | MatMul | SUPPORTED | SUPPORTED | NATIVE | Includes projection and attention matrix multiplication. |
| BEVEncoder.SpatialCrossAttention | aten::maximum | Max | SUPPORTED | CONDITIONAL | CHECK | Dynamic max_len construction may become shape-tensor logic. |
| BEVEncoder.SpatialCrossAttention | aten::mul | Mul | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| BEVEncoder.SpatialCrossAttention | aten::nonzero | NonZero | SUPPORTED | CONDITIONAL | CHECK | Produces data-dependent output shape; a central TensorRT dynamic-shape risk. |
| BEVEncoder.SpatialCrossAttention | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| BEVEncoder.SpatialCrossAttention | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| BEVEncoder.SpatialCrossAttention | aten::select | Gather / Slice + Squeeze | CONDITIONAL | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. |
| BEVEncoder.SpatialCrossAttention | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| BEVEncoder.SpatialCrossAttention | aten::stack | Unsqueeze + Concat | SUPPORTED | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. |
| BEVEncoder.SpatialCrossAttention | aten::sub | Sub | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| BEVEncoder.SpatialCrossAttention | aten::sum | ReduceSum | SUPPORTED | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. |
| BEVEncoder.SpatialCrossAttention | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 9; Conditional 14; Unsupported 0; Unknown 1; Plugin candidates 0; Risk **MEDIUM**.

### BEVEncoder.MSDeformableAttention3D

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| BEVEncoder.MSDeformableAttention3D | MultiScaleDeformableAttnFunction_fp32 | Custom domain MSDeformableAttention | CUSTOM | CUSTOM | PLUGIN | No standard ONNX op captures this fused sampling/reduction; target API is IPluginV2DynamicExt. |
| BEVEncoder.MSDeformableAttention3D | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| BEVEncoder.MSDeformableAttention3D | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| BEVEncoder.MSDeformableAttention3D | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| BEVEncoder.MSDeformableAttention3D | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| BEVEncoder.MSDeformableAttention3D | aten::softmax | Softmax | SUPPORTED | CONDITIONAL | CHECK | Verify axis and dynamic-rank lowering in the exported graph. |
| BEVEncoder.MSDeformableAttention3D | aten::sum | ReduceSum | SUPPORTED | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. |

Conclusion: Native 3; Conditional 3; Unsupported 0; Unknown 0; Plugin candidates 1; Risk **HIGH**.

### BEVEncoder.FFN

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| BEVEncoder.FFN | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| BEVEncoder.FFN | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| BEVEncoder.FFN | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| BEVEncoder.FFN | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| BEVEncoder.FFN | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |

Conclusion: Native 4; Conditional 1; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

### BEVEncoder.LayerNorm

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| BEVEncoder.LayerNorm | aten::layer_norm | ReduceMean + Sub + Pow + Add + Sqrt + Div + Mul + Add | CONDITIONAL | CONDITIONAL | CHECK | Opset 13 has no standard LayerNormalization op; exporter decomposition and parser fusion must be verified. |

Conclusion: Native 0; Conditional 1; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

## 5. Detection Decoder

### Decoder.ObjectSelfAttention

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder.ObjectSelfAttention | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Decoder.ObjectSelfAttention | aten::bmm | MatMul | SUPPORTED | SUPPORTED | NATIVE | Batched MatMul lowering. |
| Decoder.ObjectSelfAttention | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Decoder.ObjectSelfAttention | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| Decoder.ObjectSelfAttention | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| Decoder.ObjectSelfAttention | aten::embedding | Gather | SUPPORTED | CONDITIONAL | CHECK | Constant embedding weights and INT index type must parse correctly. |
| Decoder.ObjectSelfAttention | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Decoder.ObjectSelfAttention | aten::matmul | MatMul | SUPPORTED | SUPPORTED | NATIVE | Includes projection and attention matrix multiplication. |
| Decoder.ObjectSelfAttention | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| Decoder.ObjectSelfAttention | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Decoder.ObjectSelfAttention | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| Decoder.ObjectSelfAttention | aten::softmax | Softmax | SUPPORTED | CONDITIONAL | CHECK | Verify axis and dynamic-rank lowering in the exported graph. |
| Decoder.ObjectSelfAttention | aten::split | Split | SUPPORTED | CONDITIONAL | CHECK | Static split sizes expected but parser must confirm. |
| Decoder.ObjectSelfAttention | aten::sum | ReduceSum | SUPPORTED | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. |
| Decoder.ObjectSelfAttention | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 9; Conditional 6; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

### Decoder.DeformableCrossAttention

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder.DeformableCrossAttention | MultiScaleDeformableAttnFunction_fp32 | Custom domain MSDeformableAttention | CUSTOM | CUSTOM | PLUGIN | No standard ONNX op captures this fused sampling/reduction; target API is IPluginV2DynamicExt. |
| Decoder.DeformableCrossAttention | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Decoder.DeformableCrossAttention | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| Decoder.DeformableCrossAttention | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Decoder.DeformableCrossAttention | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| Decoder.DeformableCrossAttention | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Decoder.DeformableCrossAttention | aten::softmax | Softmax | SUPPORTED | CONDITIONAL | CHECK | Verify axis and dynamic-rank lowering in the exported graph. |
| Decoder.DeformableCrossAttention | aten::sum | ReduceSum | SUPPORTED | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. |
| Decoder.DeformableCrossAttention | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 5; Conditional 3; Unsupported 0; Unknown 0; Plugin candidates 1; Risk **HIGH**.

### Decoder.FFN

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder.FFN | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Decoder.FFN | aten::dropout | Identity (evaluation mode) | SUPPORTED | SUPPORTED | NATIVE | Model is eval(); deployment graph should eliminate dropout. |
| Decoder.FFN | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Decoder.FFN | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| Decoder.FFN | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |

Conclusion: Native 4; Conditional 1; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

### Decoder.LayerNorm

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder.LayerNorm | aten::layer_norm | ReduceMean + Sub + Pow + Add + Sqrt + Div + Mul + Add | CONDITIONAL | CONDITIONAL | CHECK | Opset 13 has no standard LayerNormalization op; exporter decomposition and parser fusion must be verified. |

Conclusion: Native 0; Conditional 1; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

### Decoder.ReferenceRefinement

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder.ReferenceRefinement | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Decoder.ReferenceRefinement | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Decoder.ReferenceRefinement | aten::clamp | Clip | SUPPORTED | CONDITIONAL | CHECK | Verify scalar/tensor bound lowering and dtype. |
| Decoder.ReferenceRefinement | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| Decoder.ReferenceRefinement | aten::index | Gather / GatherND / NonZero + GatherND | CONDITIONAL | CONDITIONAL | CHECK | Advanced and boolean indexing have multiple lowerings; SCA has dynamic valid-query indexing. |
| Decoder.ReferenceRefinement | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Decoder.ReferenceRefinement | aten::log | Log | SUPPORTED | SUPPORTED | NATIVE | Part of inverse-sigmoid/reference refinement arithmetic. |
| Decoder.ReferenceRefinement | aten::matmul | MatMul | SUPPORTED | SUPPORTED | NATIVE | Includes projection and attention matrix multiplication. |
| Decoder.ReferenceRefinement | aten::mul | Mul | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| Decoder.ReferenceRefinement | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Decoder.ReferenceRefinement | aten::select | Gather / Slice + Squeeze | CONDITIONAL | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. |
| Decoder.ReferenceRefinement | aten::sigmoid | Sigmoid | SUPPORTED | SUPPORTED | NATIVE | Postprocess class scores and normalized reference/box coordinates. |
| Decoder.ReferenceRefinement | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| Decoder.ReferenceRefinement | aten::split | Split | SUPPORTED | CONDITIONAL | CHECK | Static split sizes expected but parser must confirm. |
| Decoder.ReferenceRefinement | aten::stack | Unsqueeze + Concat | SUPPORTED | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. |
| Decoder.ReferenceRefinement | aten::sub | Sub | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |

Conclusion: Native 9; Conditional 7; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

## 6. Detection Head

### Head.Cls

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Head.Cls | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Head.Cls | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Head.Cls | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Head.Cls | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| Head.Cls | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| Head.Cls | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Head.Cls | aten::select | Gather / Slice + Squeeze | CONDITIONAL | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. |
| Head.Cls | aten::sigmoid | Sigmoid | SUPPORTED | SUPPORTED | NATIVE | Postprocess class scores and normalized reference/box coordinates. |
| Head.Cls | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| Head.Cls | aten::stack | Unsqueeze + Concat | SUPPORTED | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. |
| Head.Cls | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 7; Conditional 4; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

### Head.Reg

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Head.Reg | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Head.Reg | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Head.Reg | aten::linear | Gemm / MatMul + Add | SUPPORTED | SUPPORTED | NATIVE | Exact lowering depends on rank and exporter folding. |
| Head.Reg | aten::permute | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant permutation expected. |
| Head.Reg | aten::relu | Relu | SUPPORTED | SUPPORTED | NATIVE | In-place variants are exporter aliases, not distinct deployment ops. |
| Head.Reg | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Head.Reg | aten::select | Gather / Slice + Squeeze | CONDITIONAL | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. |
| Head.Reg | aten::sigmoid | Sigmoid | SUPPORTED | SUPPORTED | NATIVE | Postprocess class scores and normalized reference/box coordinates. |
| Head.Reg | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| Head.Reg | aten::stack | Unsqueeze + Concat | SUPPORTED | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. |
| Head.Reg | aten::transpose | Transpose | SUPPORTED | SUPPORTED | NATIVE | Constant two-axis swap. |

Conclusion: Native 7; Conditional 4; Unsupported 0; Unknown 0; Plugin candidates 0; Risk **LOW**.

## 7. Postprocess

### Postprocess

| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | TensorRT 8.4.12 Status | Action | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Postprocess | aten::add | Add | SUPPORTED | SUPPORTED | NATIVE | Covers residual and bias addition; in-place form is not a separate ONNX op. |
| Postprocess | aten::all | ReduceMin / Cast or ReduceSum comparison | CONDITIONAL | CONDITIONAL | CPP_POSTPROCESS | Boolean reduction lowering varies; post-center-range filtering is a good C++ boundary. |
| Postprocess | aten::atan2 | No direct opset-13 Atan2; quadrant-correct Atan-based subgraph required | UNSUPPORTED | UNKNOWN | CPP_POSTPROCESS | Keep yaw decode in C++ or implement a verified quadrant-correct rewrite. |
| Postprocess | aten::bitwise_and | And | CONDITIONAL | CONDITIONAL | CHECK | Boolean mask vs integer bitwise semantics depend on dtype. |
| Postprocess | aten::cat | Concat | SUPPORTED | SUPPORTED | NATIVE | Axis must be constant. |
| Postprocess | aten::div | Div | SUPPORTED | SUPPORTED | NATIVE | Check integer floor division separately. |
| Postprocess | aten::exp | Exp | SUPPORTED | SUPPORTED | NATIVE | Used by bbox dimension decode. |
| Postprocess | aten::floor_divide | Div + Floor / integer Div | CONDITIONAL | CONDITIONAL | CPP_POSTPROCESS | Integer bbox index decode semantics and dtype need validation. |
| Postprocess | aten::ge | GreaterOrEqual | SUPPORTED | CONDITIONAL | CPP_POSTPROCESS | Center-range filtering can remain outside the TensorRT engine. |
| Postprocess | aten::gt | Greater | SUPPORTED | CONDITIONAL | CHECK | Used in camera validity and optional score filtering. |
| Postprocess | aten::index | Gather / GatherND / NonZero + GatherND | CONDITIONAL | CONDITIONAL | CHECK | Advanced and boolean indexing have multiple lowerings; SCA has dynamic valid-query indexing. |
| Postprocess | aten::index_select | Gather | SUPPORTED | CONDITIONAL | CHECK | Check INT32/INT64 indices on the target parser. |
| Postprocess | aten::le | LessOrEqual | SUPPORTED | CONDITIONAL | CPP_POSTPROCESS | Center-range filtering can remain outside the TensorRT engine. |
| Postprocess | aten::mul | Mul | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| Postprocess | aten::nonzero | NonZero | SUPPORTED | CONDITIONAL | CHECK | Produces data-dependent output shape; a central TensorRT dynamic-shape risk. |
| Postprocess | aten::remainder | Mod | SUPPORTED | CONDITIONAL | CPP_POSTPROCESS | Integer label decode; C++ postprocess avoids parser integer corner cases. |
| Postprocess | aten::reshape | Reshape | SUPPORTED | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. |
| Postprocess | aten::select | Gather / Slice + Squeeze | CONDITIONAL | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. |
| Postprocess | aten::sigmoid | Sigmoid | SUPPORTED | SUPPORTED | NATIVE | Postprocess class scores and normalized reference/box coordinates. |
| Postprocess | aten::slice | Slice | SUPPORTED | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. |
| Postprocess | aten::split | Split | SUPPORTED | CONDITIONAL | CHECK | Static split sizes expected but parser must confirm. |
| Postprocess | aten::stack | Unsqueeze + Concat | SUPPORTED | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. |
| Postprocess | aten::sub | Sub | SUPPORTED | SUPPORTED | NATIVE | Basic elementwise arithmetic. |
| Postprocess | aten::sum | ReduceSum | SUPPORTED | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. |
| Postprocess | aten::topk | TopK-11 | SUPPORTED | CONDITIONAL | CPP_POSTPROCESS | K=300 over 9000 scores is static here; validate parser limits and index dtype, or keep in C++. |

Conclusion: Native 7; Conditional 17; Unsupported 1; Unknown 0; Plugin candidates 0; Risk **MEDIUM**.

## 8. Blocker Summary

| Module(s) | Runtime Op | Effective Status | Action | Reason / evidence |
| --- | --- | --- | --- | --- |
| Postprocess | aten::all | CONDITIONAL | CPP_POSTPROCESS | Boolean reduction lowering varies; post-center-range filtering is a good C++ boundary. Evidence: Observed twice in NMSFreeCoder center-range mask. |
| BEVEncoder.SpatialCrossAttention, Postprocess | aten::bitwise_and | CONDITIONAL | CHECK | Boolean mask vs integer bitwise semantics depend on dtype. Evidence: Observed 9 calls. |
| BEVEncoder.SpatialCrossAttention, Decoder.ReferenceRefinement | aten::clamp | CONDITIONAL | CHECK | Verify scalar/tensor bound lowering and dtype. Evidence: Observed 114 calls. |
| BEVEncoder.SpatialCrossAttention | aten::cumsum | CONDITIONAL | CHECK | Used to build level_start_index; TRT parser support/dtype must be tested. Evidence: Observed once. |
| BEVEncoder.SpatialCrossAttention, Decoder.ObjectSelfAttention | aten::embedding | CONDITIONAL | CHECK | Constant embedding weights and INT index type must parse correctly. Evidence: Observed twice for learned BEV/object positional embeddings. |
| Postprocess | aten::floor_divide | CONDITIONAL | CPP_POSTPROCESS | Integer bbox index decode semantics and dtype need validation. Evidence: Observed once for bbox_index = topk_index // 10. |
| Postprocess | aten::ge | CONDITIONAL | CPP_POSTPROCESS | Center-range filtering can remain outside the TensorRT engine. Evidence: Observed twice. |
| BEVEncoder.SpatialCrossAttention, Postprocess | aten::gt | CONDITIONAL | CHECK | Used in camera validity and optional score filtering. Evidence: Observed 12 calls. |
| BEVEncoder.SpatialCrossAttention, Decoder.ReferenceRefinement, Postprocess | aten::index | CONDITIONAL | CHECK | Advanced and boolean indexing have multiple lowerings; SCA has dynamic valid-query indexing. Evidence: 58 calls; shapes include [2500,256], [2500,4,2], [900,10], [300,9]. |
| BEVEncoder.SpatialCrossAttention, Postprocess | aten::index_select | CONDITIONAL | CHECK | Check INT32/INT64 indices on the target parser. Evidence: Observed twice. |
| BEVEncoder.LayerNorm, Decoder.LayerNorm | aten::layer_norm | CONDITIONAL | CHECK | Opset 13 has no standard LayerNormalization op; exporter decomposition and parser fusion must be verified. Evidence: Observed 40 calls; encoder/decoder norms plus head positional path. |
| Postprocess | aten::le | CONDITIONAL | CPP_POSTPROCESS | Center-range filtering can remain outside the TensorRT engine. Evidence: Observed twice. |
| BEVEncoder.SpatialCrossAttention | aten::maximum | CONDITIONAL | CHECK | Dynamic max_len construction may become shape-tensor logic. Evidence: Observed once while deriving SCA rebatch length. |
| BEVEncoder.SpatialCrossAttention, Postprocess | aten::nonzero | CONDITIONAL | CHECK | Produces data-dependent output shape; a central TensorRT dynamic-shape risk. Evidence: Observed 21 calls while selecting valid camera queries and filtered boxes. |
| Postprocess | aten::remainder | CONDITIONAL | CPP_POSTPROCESS | Integer label decode; C++ postprocess avoids parser integer corner cases. Evidence: Observed once for class label = topk_index % 10. |
| Backbone.ResNet, Neck.FPN, BEVEncoder.TemporalSelfAttention, BEVEncoder.SpatialCrossAttention, BEVEncoder.MSDeformableAttention3D, BEVEncoder.FFN, Decoder.ObjectSelfAttention, Decoder.DeformableCrossAttention, Decoder.FFN, Decoder.ReferenceRefinement, Head.Cls, Head.Reg, Postprocess | aten::reshape | CONDITIONAL | CHECK | Static shapes are routine; dynamic max_len and camera-valid-query dimensions need profiles. Evidence: Observed 101 calls plus view/flatten aliases. |
| BEVEncoder.SpatialCrossAttention, Decoder.ReferenceRefinement, Head.Cls, Head.Reg, Postprocess | aten::select | CONDITIONAL | CHECK | Mapping depends on index form and exporter canonicalization. Evidence: Observed 325 calls. |
| Backbone.ResNet, Neck.FPN, BEVEncoder.TemporalSelfAttention, BEVEncoder.SpatialCrossAttention, Decoder.ObjectSelfAttention, Decoder.ReferenceRefinement, Head.Cls, Head.Reg, Postprocess | aten::slice | CONDITIONAL | CHECK | Dynamic bounds and negative axes require parser validation. Evidence: Observed 366 calls. |
| BEVEncoder.TemporalSelfAttention, BEVEncoder.MSDeformableAttention3D, Decoder.ObjectSelfAttention, Decoder.DeformableCrossAttention | aten::softmax | CONDITIONAL | CHECK | Verify axis and dynamic-rank lowering in the exported graph. Evidence: Observed 18 calls. |
| Decoder.ObjectSelfAttention, Decoder.ReferenceRefinement, Postprocess | aten::split | CONDITIONAL | CHECK | Static split sizes expected but parser must confirm. Evidence: Observed 13 calls. |
| BEVEncoder.SpatialCrossAttention, Decoder.ReferenceRefinement, Head.Cls, Head.Reg, Postprocess | aten::stack | CONDITIONAL | CHECK | Exporter decomposes stack; validate shape tensors. Evidence: Observed 22 calls. |
| BEVEncoder.SpatialCrossAttention, BEVEncoder.MSDeformableAttention3D, Decoder.ObjectSelfAttention, Decoder.DeformableCrossAttention, Postprocess | aten::sum | CONDITIONAL | CHECK | Axes and keepdims may be constants; shape-tensor reductions require parser validation. Evidence: Observed 42 calls. |
| Postprocess | aten::topk | CONDITIONAL | CPP_POSTPROCESS | K=300 over 9000 scores is static here; validate parser limits and index dtype, or keep in C++. Evidence: Observed once with input [9000]. |
| BEVEncoder.TemporalSelfAttention, BEVEncoder.MSDeformableAttention3D, Decoder.DeformableCrossAttention | MultiScaleDeformableAttnFunction_fp32 | CUSTOM | PLUGIN | No standard ONNX op captures this fused sampling/reduction; target API is IPluginV2DynamicExt. Evidence: 12 calls and 12 ms_deformable_im2col_gpu_kernel launches = 3 TSA + 3 SCA + 6 decoder layers. |
| BEVEncoder.SpatialCrossAttention | aten::index_put_ | UNKNOWN | CHECK | Accumulating writes and duplicate indices may not match plain ScatterND semantics. Evidence: Observed 18 calls; SCA scatter-add style slot accumulation in source. |
| Postprocess | aten::atan2 | UNSUPPORTED | CPP_POSTPROCESS | Keep yaw decode in C++ or implement a verified quadrant-correct rewrite. Evidence: Observed once with [300,1] inputs in denormalize_bbox. |
| BEVEncoder.TemporalSelfAttention | aten::grid_sampler *(latent)* | UNSUPPORTED | REWRITE | Latent rotate-prev-BEV path; not executed because this Golden Sample is the first scene frame. Evidence: Absent from current profile; transformer.py uses F.grid_sample only when prev_bev exists. |

### Special-path findings

- `MultiScaleDeformableAttnFunction_fp32` executed 12 times in detailed events. The profile also contains 12 `ms_deformable_im2col_gpu_kernel` launches; these are one custom primitive used by 3 TSA, 3 encoder SCA, and 6 decoder cross-attention layers—not 12 ONNX blockers.
- `aten::grid_sampler` was not observed. The Golden token is a first scene frame (`prev_bev=None`), so historical BEV rotation is latent. ONNX `GridSample` is newer than opset 13; a multi-frame export needs a rewrite or separately scoped plugin decision.
- SCA scatter-style accumulation appears as `aten::index_put_`; duplicate-index accumulation semantics must be checked after export.
- Postprocess `TopK`, integer index decode, boolean filtering, and `atan2` are recommended as a C++ boundary for this baseline.

### Module-level summary

| Module group | Native | Conditional | Plugin candidates | Risk |
| --- | ---: | ---: | ---: | --- |
| Backbone | 6 | 2 | 0 | LOW |
| Neck | 5 | 2 | 0 | LOW |
| BEV Encoder | 12 | 16 | 1 | HIGH |
| Detection Decoder | 14 | 11 | 1 | HIGH |
| Detection Head | 7 | 4 | 0 | LOW |
| Postprocess | 7 | 17 | 0 | MEDIUM |

## 9. Evidence and Confirmation Gates

- ONNX mappings are checked against the ONNX operator catalog for opset availability; `GridSample` is not an opset-13 operator.
- TensorRT status is deliberately conditional where this repository has no TensorRT 8.4.12 parser/build evidence. The archived NVIDIA 8.4 support matrix and developer guide are database evidence links.
- Required next gate: export the exact model at opset 13 and inspect the graph, especially indexing/scatter/nonzero/dynamic reshape and LayerNorm decomposition.
- Required final gate: run TensorRT 8.4.12.5 parser and engine build on aarch64/sm_87 with representative shape profiles. No status in this report substitutes for that parser/build test.

