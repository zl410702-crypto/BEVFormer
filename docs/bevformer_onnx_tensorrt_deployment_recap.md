# BEVFormer ONNX 与 TensorRT FP32 部署工作回顾

## 1. 当前结论

当前已经完成以下 PC 侧部署闭环：

```text
3-Frame Stateful PyTorch Golden
        ↓
Stateful Single-Frame ONNX
        ↓
Static-Nq Spatial Cross Attention
        ↓
TensorRT 8.4 Compatibility Cleanup
        ↓
Unified MSDeformableAttention Plugin
        ↓
PC TensorRT Parser PASS
        ↓
PC TensorRT FP32 Engine Build PASS
        ↓
Frame 0 / Frame 1 Runtime
        ↓
Host Affine Grid 修复 Stateful alignment
```

相关代码提交：

```text
96c8959 feat: add BEVFormer TensorRT FP32 deployment path
```

目标板端环境：

```text
NVIDIA Orin
DriveOS 6.0.5
CUDA 11.4
TensorRT 8.4.12.5
```

PC 预验证环境：

```text
RTX 4060 Laptop
TensorRT 8.5.3
```

PC 生成的 TensorRT engine 仅用于 host validation，不能复制到 Orin。Orin 必须使用板端 TensorRT 8.4.12.5 重新构建 engine。

## 2. Stateful PyTorch Golden

首先建立了真实连续三帧 Golden Reference：

```text
Frame 0: 3e8750f331d7499e9b5123e9eb70f2e2
Frame 1: 3950bd41f74548429c0f7700ff3d8269
Frame 2: c5f58c19249d4137ae063b0e9ecd8b8e
```

验证结果：

```text
Frame0 final_bev == Frame1 prev_bev_stored
Frame1 final_bev == Frame2 prev_bev_stored
```

因此 Stateful runtime 的核心状态传递为：

```text
bev_embed_t
        ↓
prev_bev_(t+1)
```

不能在每帧重新把 `prev_bev` 设为 `None`。

运行命令：

```bash
/data/conda_envs/bevformer/bin/python \
  tools/run_golden_sequence.py \
  --token 3e8750f331d7499e9b5123e9eb70f2e2 \
  --num-frames 3
```

Golden 数据目录：

```text
golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3/
```

Tensor 比较：

```bash
/data/conda_envs/bevformer/bin/python tools/tensor_compare.py A.npy B.npy
```

## 3. 从首帧 ONNX 升级为 Stateful ONNX

旧 ONNX `artifacts/bevformer_tiny_opset13.onnx` 只有 `images` 输入，导出时使用：

```text
prev_bev=None
use_prev_bev=False
```

所以旧模型只能表示首帧计算。新的 Stateful ONNX 接口为：

| Input | Shape | Dtype | 语义 |
| --- | --- | --- | --- |
| `images` | `[1,6,3,480,800]` | FP32 | 当前帧六相机图像 |
| `prev_bev` | `[2500,1,256]` | FP32 | 上一帧 BEV state |
| `can_bus` | `[1,18]` | FP32 | 官方 test-time delta 处理后的 ego-motion |
| `shift` | `[1,2]` | FP32 | BEV translation shift |
| `lidar2img` | `[1,6,4,4]` | FP32 | 当前帧相机投影矩阵 |
| `use_prev_bev` | `[1]` | FP32 | 首帧/历史帧 gate |

输出：

| Output | Shape | Layout |
| --- | --- | --- |
| `all_cls_scores` | `[6,1,900,10]` | decoder layer, batch, query, class |
| `all_bbox_preds` | `[6,1,900,10]` | decoder layer, batch, query, box |
| `bev_embed` | `[2500,1,256]` | `[Nbev,B,C]` |

`bev_embed` 与下一帧 `prev_bev` 的 shape/layout 完全一致。

导出命令：

```bash
cd /home/eric/workspace/bevformer/BEVFormer

/data/conda_envs/bevformer_export/bin/python \
  tools/export_bevformer_stateful_onnx.py \
  --output artifacts/bevformer_tiny_stateful_opset13.onnx
```

Exporter 在真正导出前验证：

```text
Frame 1 wrapper vs Golden
real prev_bev vs zero prev_bev sensitivity
Frame 0 use_prev_bev=0 vs official prev_bev=None
```

默认保护规则：

```text
Golden validation FAIL → 停止导出
```

`--allow-validation-failure` 只允许生成结构调试 candidate，不能用来宣称数值部署已经通过。

## 4. ONNX MSDeformableAttention Custom Operator

MMCV deformable attention 被导出为：

```text
domain  = bevformer
op_type = MSDeformableAttention
version = 1
```

即：

```text
bevformer::MSDeformableAttention
```

Custom op 输入顺序固定为：

```text
0 value
1 spatial_shapes
2 level_start_index
3 sampling_locations
4 attention_weights
```

数学 contract：

```text
value:                 [B,S,H,D]
spatial_shapes:        [L,2]
level_start_index:     [L]
sampling_locations:    [B,Q,H,L,P,2]
attention_weights:     [B,Q,H,L,P]
output:                [B,Q,H*D]
```

坐标顺序为 `(x,y)`，归一化坐标到 feature map 的映射遵循 MMCV：

```text
x_feature = x_normalized * width  - 0.5
y_feature = y_normalized * height - 0.5
```

Plugin 内部完成：

```text
sampling location
        ↓
four-neighbor bilinear interpolation
        ↓
multiply attention weight
        ↓
sum over levels and sampling points
        ↓
concatenate heads
```

以下运算在 Plugin 外：

```text
sampling_offsets Linear
attention_weights Linear
Softmax
output projection
```

## 5. 为什么 ONNX 中有 12 个 MSDA

实际审计结果：

```text
Temporal Self Attention:            3
Spatial Cross Attention:            3
Detection Decoder Cross Attention:  6
Total:                             12
```

Tiny 模型有三个 encoder layer，每层包含一个 Temporal MSDA 和一个 Spatial MSDA；六个 decoder layer 每层包含一个 Decoder MSDA。

代表性 contract：

```text
Temporal:
  value               [2,2500,8,32]
  sampling_locations  [2,2500,8,1,4,2]
  attention_weights   [2,2500,8,1,4]
  output              [2,2500,256]

Spatial:
  value               [6,375,8,32]
  sampling_locations  [6,2500,8,1,8,2]
  attention_weights   [6,2500,8,1,8]
  output              [6,2500,256]

Decoder:
  value               [1,2500,8,32]
  sampling_locations  [1,900,8,1,4,2]
  attention_weights   [1,900,8,1,4]
  output              [1,900,256]
```

三类 attention 的 query/reference 来源不同，但进入 custom op 后数学 contract 相同。因此最终使用：

```text
one MSDeformableAttentionPlugin class
        +
12 TensorRT Plugin layer instances
```

审计命令：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/inspect_msda_onnx.py \
  artifacts/bevformer_tiny_stateful_static_sca_chunk640_int32_opset13.onnx \
  --require-int32 \
  --json artifacts/msda_onnx_audit_int32.json
```

## 6. Spatial Cross Attention 的 trace-specialized 604

官方 Spatial Cross Attention 根据 camera visibility 做 compact rebatch：

```python
index_query_per_img = mask.sum(-1).nonzero()
max_len = max(visible_query_count)
```

导出样本中观察到 `Nquery_max=604`。PyTorch trace 会把 visibility count、camera query count 和 `max_len=604` 固化进 ONNX，使图的 shape 依赖导出帧。

deployment-only static SCA 改成：

```text
query [1,2500,256]
    ↓ camera expand
[1,6,2500,256]
    ↓ flatten camera
[6,2500,256]
```

所有 camera 保留完整 2500 个 BEV query，visibility 变成 runtime mask：

```text
valid_mask [1,6,2500]
```

invalid reference point 在进入 MSDA 前替换为安全坐标，防止 invalid sampling 产生 NaN。MSDA 输出后执行：

```text
camera_features × valid_mask
        ↓
sum(camera)
        ↓
divide by visible-camera count
```

最终：

```text
Nq contract:                 2500
visibility:                  runtime mask
trace-specialized 604:       removed
visibility-derived NonZero:  removed
```

普通官方 `SpatialCrossAttention.forward()` 未修改，static path 只位于 deployment wrapper。

## 7. Fixed Linear Chunk 640

Static SCA 将两个 Linear 的输入从 compact query length 改成固定 `Nq=2500`。在默认 CUDA/TF32 math mode 下，不同 GEMM `M` 维会产生 shape-dependent rounding difference。

中间 tensor 检查确认 first divergence 位于：

```text
attention_logits before softmax
```

不是 MSDA CUDA kernel。

最终确定固定部署 GEMM contract：

```text
STATIC_SCA_LINEAR_CHUNK = 640
```

执行方式：

```text
chunk 0: 640
chunk 1: 640
chunk 2: 640
chunk 3: 580 valid + 60 zero padding
```

每块均以 `[6,640,256]` 执行 Linear，最后一块 trim 60 个 padding，再 concat 回 2500 query。

只应用于：

```text
sampling_offsets Linear
attention_weights Linear
```

未修改：

```text
value_proj
output_proj
MSDA CUDA kernel
sampling_locations formula
visibility mask
camera reduction
```

640 是固定 deployment GEMM execution contract，不来源于 visibility，也不依赖 trace 中的 604。

## 8. INT64 到 INT32 Plugin Contract

ONNX exporter 容易将 `spatial_shapes` 和 `level_start_index` 导出为 INT64。TensorRT Plugin contract 明确要求：

```text
spatial_shapes:       INT32
level_start_index:    INT32
```

Exporter 在 custom op boundary 显式插入 Cast，不依赖 TensorRT parser 做隐式类型转换。最终 12 个 Plugin node 都满足：

```text
value:                  FP32/FP16
sampling_locations:     FP32/FP16
attention_weights:      FP32/FP16
spatial_shapes:         INT32
level_start_index:      INT32
```

## 9. TensorRT 8.4 Slice Compatibility

Whole-model parser 曾遇到：

```text
Slice_1933
Slice_2222
其他同类 Slice

Assertion failed: axes.allValuesKnown()
This version of TensorRT does not support dynamic axes.
```

这些节点的 axis 语义并非真正动态，而是旧 PyTorch exporter 将固定 axis 表示为：

```text
Constant scalar
    ↓
Unsqueeze
    ↓
Slice.axes
```

TensorRT 8.4 parser 没有将该 producer chain 识别为 build-time weights。

兼容 pass 对能够严格静态求值的 axes 执行：

```text
constant-preserving producer chain
        ↓
INT64 initializer
        ↓
Slice.axes
```

只允许以下静态 producer：

```text
Initializer
Constant
Cast(Constant)
Unsqueeze(Constant)
Concat(constants)
```

如果依赖 Graph Input、runtime Shape、NonZero、Where 或 data-dependent Gather，则跳过，不硬编码。

检查指定 Slice：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/inspect_onnx_slice_node.py \
  --onnx artifacts/bevformer_tiny_stateful_trt84_compat_opset13.onnx \
  --node Slice_2222 \
  --max-depth 10
```

运行测试：

```bash
/data/conda_envs/bevformer_export/bin/python -m pytest \
  tests/test_onnx_slice_compat.py -v
```

最终兼容模型：

```text
artifacts/bevformer_tiny_stateful_trt84_compat_opset13.onnx
```

状态：

```text
ONNX checker:    PASS
shape inference: PASS
MSDA nodes:      12
```

## 10. Unified MSDA TensorRT Plugin

Plugin 位于：

```text
tensorrt_plugins/ms_deformable_attention/
```

| 文件 | 作用 |
| --- | --- |
| `ms_deformable_attention_plugin.h/.cpp` | `IPluginV2DynamicExt` 与 PluginCreator |
| `ms_deformable_attention_kernel.h/.cu` | correctness-first CUDA kernel |
| `CMakeLists.txt` | CMake 构建 |
| `BUILD.pony` | PONY_ASW/板端集成参考 |
| `build_bevformer_fp32_engine.cpp` | whole parser 与 FP32 engine builder |
| `validation/msda_kernel_test_bridge.cpp` | CUDA kernel Python 测试桥 |
| `validation/test_plugin_serialization.cpp` | 序列化测试 |
| `validation/run_bevformer_engine.cpp` | TensorRT runtime runner |

Plugin API：

```text
IPluginV2DynamicExt
Plugin type: MSDeformableAttention
Version: 1
Namespace: ""
```

CUDA 采用 one-thread-per-output-channel：

```text
one thread → one (batch, query, head, channel) output scalar
```

每个 thread 循环所有 level 和 sampling point，执行 bilinear sampling、weight multiply 和累加。

第一版优先正确性，没有使用 shared-memory tiling、half2、warp reduction、kernel fusion 或 INT8。

`im2col_step=64` 保留为 ONNX schema attribute，但不是数学 contract。MMCV wrapper 使用：

```cpp
im2col_step_ = min(batch, im2col_step)
```

当前 effective batch 为 Temporal=2、Spatial=6、Decoder=1，因此 Plugin 直接处理完整 effective batch。

## 11. Plugin 构建与验证

目标板端构建：

```bash
cmake \
  -S tensorrt_plugins/ms_deformable_attention \
  -B build/msda \
  -DTENSORRT_ROOT=/path/to/TensorRT-8.4.12.5 \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-11.4/bin/nvcc

cmake --build build/msda -j
```

主要输出：

```text
build/msda/libms_deformable_attention_plugin.so
build/msda/build_bevformer_fp32_engine
```

CUDA numerical validation：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/validate_msda_cuda_kernel.py \
  --library /path/to/libmsda_kernel_test.so \
  --json artifacts/msda_cuda_kernel_validation.json
```

结果：

| Node | FP32 MaxAE | Result |
| --- | ---: | --- |
| Temporal L0 | 0 | PASS |
| Spatial L0 | 0 | PASS |
| Decoder L0 | 0 | PASS |

Plugin FP16 kernel 也做过独立正确性验证：

```text
Temporal MaxAE: 9.6750e-4
Spatial MaxAE:  4.8792e-4
Decoder MaxAE:  5.6958e-4
```

但完整模型当前仍以 FP32 为主，尚未进入完整 FP16 engine acceptance。

Serialization 验证：

```text
create → serialize → deserialize → enqueue
```

输出 bitwise identical，结果 PASS。

## 12. 12-Node Parser Test

生成隔离的 12-node custom-op ONNX：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/create_msda_parser_test_onnx.py
```

Parser/build：

```bash
build/msda/build_bevformer_fp32_engine \
  artifacts/msda_12_node_parser_test.onnx \
  build/msda/libms_deformable_attention_plugin.so \
  artifacts/msda_12_node_fp32.engine
```

预期：

```text
Parser: PASS
PluginV2 layers: 12
FP32 engine: PASS
```

隔离模型用于证明 Plugin Creator 注册、ONNX domain/op_type、12 个 Plugin contract 和 engine serialization 正确。它与 whole-model parser 是两个不同层次的验证。

## 13. Whole-Model FP32 Engine

兼容模型构建命令：

```bash
build/msda/build_bevformer_fp32_engine \
  artifacts/bevformer_tiny_stateful_trt84_compat_opset13.onnx \
  build/msda/libms_deformable_attention_plugin.so \
  artifacts/bevformer_tiny_stateful_pc_trt853_fp32.engine
```

Builder 配置：

```text
Explicit Batch
FP32
Workspace = 4 GiB
PluginV2 layer count = 12
No FP16
No INT8
```

PC 结果：

```text
Whole parser:       PASS
Plugin instances:   12
FP32 engine build:  PASS
FP32 runtime:       PASS
```

`artifacts/bevformer_tiny_stateful_pc_trt853_fp32.engine` 是 TensorRT 8.5.3/x86/RTX 4060 host-validation engine，不能部署到 Orin。

## 14. Frame 1 Numerical Divergence

原始完整 engine 的 Frame 0：

```text
bev_embed MaxAE ≈ 0.0071
```

Frame 1：

```text
bev_embed MaxAE ≈ 0.46
```

Frame 1 分别使用 TRT Frame0 prev_bev 和 Golden prev_bev，误差都约为 0.46，因此排除了 Frame0 state error propagation 是主要根因。

逐级 debug 定位结果：

```text
Frame1 inputs                    EXACT
rotation angle                   EXACT
theta.transpose / scale          EXACT
MatMul_654 affine output         first numerical divergence
Round_707 / Round_708            first discontinuous divergence
GatherElements                   selects different BEV cells
aligned_prev_bev                 MaxAE 2.638653
final bev_embed                  MaxAE ≈ 0.46
```

根因链：

```text
TensorRT affine MatMul FP32 arithmetic difference
        ↓
affine grid MaxAE ≈ 3.81e-4
        ↓
coordinates cross half-integer / validity boundaries
        ↓
31 rounded-index mismatches + 2 validity-mask mismatches
        ↓
different BEV cells selected
        ↓
aligned_prev_bev large error
```

这不是 MSDA Plugin 错误。

## 15. Host Trig 和 Host Index 实验

Host trig 实验把 PyTorch 计算的 cos/sin 作为 TensorRT 输入。虽然 trig 输入 bitwise exact，但 TensorRT 内部 `MatMul_654` 的输出仍存在差异，因此 Round mismatch 仍存在。问题不只来自 TensorRT Cos/Sin，而是 affine MatMul/坐标算术的 FP32 execution difference。

Host integer-index 实验也没有作为生产方案，因为只传 linear index 不能表达并行的 validity mask；index 相同时，TensorRT 内部 validity 判断仍可能不同。

## 16. Host Affine Grid Production Contract

最终生产方案把 PyTorch-equivalent affine grid 放到 host 生成：

```text
affine_grid [1,2500,2] FP32
```

TensorRT 继续执行：

```text
affine_grid
    ↓
pixel coordinate conversion
    ↓
Round
    ↓
validity mask
    ↓
Clip
    ↓
linear index
    ↓
GatherElements
    ↓
aligned prev_bev
```

Contract：

| Field | Value |
| --- | --- |
| Name | `affine_grid` |
| Shape | `[1,2500,2]` |
| Dtype | FP32 |
| Layout | `[B,HW,XY]`, XY order is x then y |
| Producer | Host ego-motion preprocessing |
| Consumer | ONNX alignment sampling subgraph |
| Static | Current Tiny uses fixed 50×50 BEV |
| Ownership | Host buffer remains valid until async H2D completes |

生成 runtime input：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/prepare_bevformer_host_affine_runtime.py \
  --runtime-frame <frame_001_runtime_fixture_dir> \
  --output <output_dir>
```

输出：

```text
affine_grid.bin
affine_grid_metadata.json
```

## 17. Host Affine ONNX 与 Engine

生成 production candidate：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/export_bevformer_host_affine_onnx.py \
  --input artifacts/bevformer_tiny_stateful_trt84_compat_opset13.onnx \
  --output artifacts/bevformer_tiny_stateful_fp32_host_affine.onnx \
  --report artifacts/bevformer_tiny_stateful_fp32_host_affine.audit.json
```

Graph 改造：

```text
Before:
base_grid + theta/scale
        ↓
MatMul_654
        ↓
tensor 1356
        ↓
Reshape_666
        ↓
sampling chain

After:
affine_grid runtime input
        ↓
Reshape_666
        ↓
existing coordinate/Round/mask/Clip/Gather chain
```

改造会验证 producer、唯一 consumer、shape/dtype、downstream topology 和 MSDA count；graph drift 时直接失败。

ONNX 信息：

```text
Path: artifacts/bevformer_tiny_stateful_fp32_host_affine.onnx
Size: 134,782,839 bytes
SHA256: 21d972d852158b25a35f0af00aafb93f2c9993cccc3fa70b15b57f27d95aed03
Node count: 11,665
MSDA count: 12
ONNX checker: PASS
Shape inference: PASS
```

新增 graph input：

```text
images
prev_bev
can_bus
shift
lidar2img
use_prev_bev
affine_grid
```

构建 Host Affine FP32 engine：

```bash
build/msda/build_bevformer_fp32_engine \
  artifacts/bevformer_tiny_stateful_fp32_host_affine.onnx \
  build/msda/libms_deformable_attention_plugin.so \
  artifacts/bevformer_tiny_stateful_pc_trt853_fp32_host_affine.engine
```

PC engine 大小：

```text
141,295,140 bytes
```

状态：

```text
Parser: PASS
PluginV2 layers: 12
FP32 build: PASS
Runtime: PASS
```

## 18. Host Affine Numerical Results

| Tensor | MaxAE | Result |
| --- | ---: | --- |
| `affine_grid` input | 0 | EXACT |
| rounded x | 0 | EXACT |
| rounded y | 0 | EXACT |
| GatherElements | 0 | EXACT |
| rotated BEV | 0 | EXACT |
| aligned_prev_bev | 0 | EXACT |
| Encoder L0 | 0.00413465 | FP32 close |
| Encoder L1 | 0.00653470 | FP32 close |
| Encoder L2 | 0.00579101 | FP32 close |
| `bev_embed` | 0.00579101 | FP32 close |
| `all_cls_scores` | 0.103784 | FP32 output difference |
| `all_bbox_preds` | 0.154463 | FP32 output difference |

结果：

```text
Round mismatch total: 0
Gather mismatch: 0
aligned_prev_bev: bitwise exact
```

Host Affine 将 Frame 1 `bev_embed MaxAE` 从约 0.46 降低到约 0.00579。

比较命令：

```bash
/data/conda_envs/bevformer_export/bin/python \
  tools/compare_bevformer_host_affine.py \
  --outputs <trt_debug_outputs> \
  --alignment-reference <alignment_golden_dir> \
  --encoder-reference <encoder_golden_dir> \
  --golden-frame golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3/frame_001 \
  --report artifacts/bevformer_tiny_stateful_fp32_host_affine_validation.json
```

## 19. 常用工具速查

| Tool | Purpose |
| --- | --- |
| `tools/run_golden_sequence.py` | 生成三帧 Stateful PyTorch Golden |
| `tools/tensor_compare.py` | 比较两个 `.npy` |
| `tools/export_bevformer_stateful_onnx.py` | 导出 Stateful ONNX |
| `tools/inspect_msda_onnx.py` | 审计 12 个 MSDA node |
| `tools/create_msda_parser_test_onnx.py` | 创建 12-node 隔离 parser 模型 |
| `tools/validate_msda_cuda_kernel.py` | Plugin CUDA vs MMCV 验证 |
| `tools/inspect_onnx_slice_node.py` | 追踪 Slice 五输入和 axes producer |
| `tools/deployment/onnx_slice_compat.py` | 静态 Slice axes canonicalization |
| `build/msda/build_bevformer_fp32_engine` | Parser + FP32 engine build |
| `validation/run_bevformer_engine.cpp` | TensorRT runtime runner |
| `tools/prepare_bevformer_host_affine_runtime.py` | 生成 `affine_grid` binding |
| `tools/export_bevformer_host_affine_onnx.py` | 生成 Host Affine ONNX |
| `tools/compare_bevformer_host_affine.py` | Host Affine boundary compare |

## 20. 当前 Readiness

```text
Stateful PyTorch Golden:                    PASS
Stateful ONNX export:                      PASS
Static SCA Nq=2500:                        PASS
Trace-specialized 604:                     REMOVED
Visibility NonZero:                        REMOVED
Fixed Linear chunk=640:                    IMPLEMENTED
MSDA unified Plugin:                       PASS
12-node Plugin parser:                     PASS
Whole-model PC parser:                     PASS
PC FP32 engine build:                      PASS
PC FP32 runtime:                           PASS
Frame1 alignment root cause:               PROVEN
Host Affine contract:                      IMPLEMENTED
Round/Gather/aligned_prev_bev:             EXACT
Full FP32 output comparison:               COMPLETED
FP16 whole-model deployment:               NOT STARTED
Orin TensorRT 8.4.12.5 engine build:       REQUIRED ON BOARD
```

下一步建议：

```text
同步代码和 Host Affine runtime preparation 到 Orin
        ↓
使用 CUDA 11.4 / TensorRT 8.4.12.5 编译 Plugin
        ↓
在 Orin 本机重新 build engine
        ↓
运行 Frame 0 / Frame 1
        ↓
复核 affine_grid、alignment、BEV、CLS、BBox
```

## 21. 相关文档

- `docs/MSDeformableAttention_plugin_audit.md`
- `docs/MSDeformableAttention_plugin_validation.md`
- `docs/host_affine_grid_contract.md`
- `tensorrt_plugins/ms_deformable_attention/README.md`
