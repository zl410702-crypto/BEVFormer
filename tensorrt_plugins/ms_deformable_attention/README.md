# BEVFormer MSDeformableAttention TensorRT Plugin

Correctness-first unified TensorRT plugin for the 12
`bevformer::MSDeformableAttention` nodes in BEVFormer Tiny:

- Temporal Self Attention: 3 instances
- Spatial Cross Attention: 3 instances
- Detection Decoder Cross Attention: 6 instances

The deployment target is NVIDIA Orin, DriveOS 6.0.5, CUDA 11.4 and
TensorRT 8.4.12.5. The implementation uses `IPluginV2DynamicExt`; it does not
use TensorRT 10 APIs.

## Operator contract

The plugin type and version are `MSDeformableAttention` and `1`. The ONNX
parser looks up that type with an empty plugin namespace. Input order is fixed:

| # | Input | Type | Shape | Meaning |
|---:|---|---|---|---|
| 0 | `value` | FP32/FP16 | `[B,S,H,D]` | Flattened levels, split by head |
| 1 | `spatial_shapes` | INT32 | `[L,2]` | Level `(height,width)` pairs |
| 2 | `level_start_index` | INT32 | `[L]` | Level offsets in `S` |
| 3 | `sampling_locations` | same as value | `[B,Q,H,L,P,2]` | Normalized `(x,y)` coordinates |
| 4 | `attention_weights` | same as value | `[B,Q,H,L,P]` | Weights calculated upstream |

Output is `[B,Q,H*D]`, with the same floating type as `value`. Mixed floating
types, INT64 shape/index inputs, and non-linear tensor formats are rejected.
Sampling-offset prediction, attention-weight Linear/Softmax, and output
projection are outside this plugin.

For every `(b,q,h,d)`, one CUDA thread loops over levels and points. A location
is mapped exactly as in MMCV 1.4.0:

```text
w = x * level_width  - 0.5
h = y * level_height - 0.5
```

The four in-bounds neighbors are bilinearly interpolated; out-of-bounds
neighbors contribute zero. The interpolated value is multiplied by the
corresponding attention weight and accumulated over `L` and `P`. The current
kernel uses 1024 threads per block and no workspace. FP16 inputs use FP32
interpolation/accumulation and are cast to FP16 at output.

`im2col_step=64` remains an ONNX attribute for schema compatibility but is not
a mathematical parameter. MMCV evaluates `min(batch, im2col_step)` and merely
partitions the effective batch. Current effective batches are 1, 2, and 6, so
the plugin launches the full batch once.

## Build

The CMake version check deliberately requires the PONY_ASW board contract,
TensorRT 8.4.12.5. `TENSORRT_ROOT` must contain `include/` and `lib/`.

```bash
cmake -S tensorrt_plugins/ms_deformable_attention \
  -B build/msda \
  -DTENSORRT_ROOT=/path/to/TensorRT-8.4.12.5 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-11.4/bin/nvcc
cmake --build build/msda -j
```

Outputs include `libms_deformable_attention_plugin.so` and the FP32-only
`build_bevformer_fp32_engine` audit tool. Enable contract logging with
`-DBEVFORMER_MSDA_DEBUG=ON`. `BUILD.pony` documents the equivalent PONY_ASW
`cuda_library` integration; it is a template rather than an active BUILD file.

## Tests

Create and parse the isolated 12-node contract model:

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/create_msda_parser_test_onnx.py

build/msda/build_bevformer_fp32_engine \
  artifacts/msda_12_node_parser_test.onnx \
  build/msda/libms_deformable_attention_plugin.so \
  artifacts/msda_12_node_fp32.engine
```

CUDA-vs-MMCV validation uses the test bridge and checks the actual Temporal L0,
Spatial L0 and Decoder L0 shapes in both precisions. The installed MMCV binary
does not instantiate its Half forward kernel, so FP16 is compared with MMCV
FP32 evaluated from the exact quantized FP16 input values.

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/validate_msda_cuda_kernel.py \
  --library /path/to/libmsda_kernel_test.so \
  --json artifacts/msda_cuda_kernel_validation.json
```

## Known limitations

- No INT8 support.
- No performance tuning, vectorized half2 path, shared-memory tiling, fusion,
  or custom `im2col_step` batching.
- The isolated 12-node model parses and builds on the available x86 TensorRT
  8.5.3 preflight environment. The exact 8.4.12.5 parser/build must run on the
  Orin target because TensorRT engines are not portable across versions or
  architectures.
- The full BEVFormer ONNX currently reaches a separate parser blocker at
  `Slice_1933`: dynamic Slice axes are unsupported by the available host
  parser. This occurs before the first MSDA node and is not a plugin failure.
