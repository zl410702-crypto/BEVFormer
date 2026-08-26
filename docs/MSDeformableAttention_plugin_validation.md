# Unified MSDeformableAttention Plugin Validation

## Target and scope

```text
Target: NVIDIA Orin / DriveOS 6.0.5
CUDA: 11.4
TensorRT: 8.4.12.5
Plugin API: IPluginV2DynamicExt
Plugin type/version/namespace: MSDeformableAttention / 1 / ""
```

The production BEVFormer inference path and the five-input ONNX custom-op
contract are unchanged. Export-only Cast nodes make both index inputs INT32.

## CUDA numerical validation

Deterministic seed `20260825` was used. FP32 compares directly with the MMCV
1.4.0 CUDA forward. The installed MMCV extension has no Half instantiation, so
FP16 compares with MMCV FP32 evaluated from the exact quantized FP16 inputs.

| Role | Dtype | Shape | MaxAE | MeanAE | Max relative | Cosine | Result |
|---|---|---|---:|---:|---:|---:|---|
| Temporal L0 | FP32 | `[2,2500,256]` | 0 | 0 | 0 | 1 | PASS |
| Spatial L0 | FP32 | `[6,2500,256]` | 0 | 0 | 0 | 1 | PASS |
| Decoder L0 | FP32 | `[1,900,256]` | 0 | 0 | 0 | ~1 | PASS |
| Temporal L0 | FP16 | `[2,2500,256]` | 9.6750e-4 | 5.2945e-5 | 1.0* | 0.999999979 | PASS |
| Spatial L0 | FP16 | `[6,2500,256]` | 4.8792e-4 | 3.8173e-5 | 5.3214e-2 | 0.999999978 | PASS |
| Decoder L0 | FP16 | `[1,900,256]` | 5.6958e-4 | 5.2893e-5 | 1.2591e-2 | 0.999999979 | PASS |

`*` The maximum relative error is dominated by a reference value close to
zero; MaxAE remains below the agreed 1e-2 FP16 threshold.

Machine-readable results are in the local ignored artifact
`artifacts/msda_cuda_kernel_validation.json`.

## Serialization

The plugin has no persistent parameters, so its serialized payload is zero
bytes. A created instance and a deserialized instance were enqueued with the
same FP32 device inputs. Outputs were bitwise identical: **PASS**.

## Parser integration

`artifacts/msda_12_node_parser_test.onnx` contains the actual three Temporal,
three Spatial and six Decoder static contracts. ONNX checker passed. On the
available x86 TensorRT 8.5.3 preflight environment, the parser logged 12
successful `MSDeformableAttention` plugin creations, the network contained
exactly 12 `PluginV2` layers, and the isolated FP32 engine built successfully
(57,245 bytes): **PASS (host preflight)**.

The same sources compile against PONY_ASW's TensorRT 8.4.12.5 headers. Exact
parser/build execution remains pending on Orin because the cached 8.4.12.5
libraries are aarch64 and TensorRT engines are not portable.

The full BEVFormer ONNX was attempted in the host preflight environment. It
stops at `Slice_1933` because that parser does not support dynamic Slice axes.
This occurs before the first MSDA custom node:

```text
Plugin parser integration: PASS on isolated 12-node contract
Whole-model parser: FAIL/BLOCKED by unrelated dynamic Slice axes
Full engine build: attempted, not completed
```

## Build and test commands

Board build:

```bash
cmake -S tensorrt_plugins/ms_deformable_attention -B build/msda \
  -DTENSORRT_ROOT=/path/to/TensorRT-8.4.12.5 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-11.4/bin/nvcc
cmake --build build/msda -j
```

Repository tests:

```bash
/data/conda_envs/bevformer_export/bin/python -m pytest -q \
  tests/test_create_msda_parser_test_onnx.py \
  tests/test_bevformer_onnx_export.py \
  tests/test_inspect_msda_onnx.py \
  tests/test_grid_sample_rewrite.py
```

Observed result: `39 passed, 1 skipped`.
