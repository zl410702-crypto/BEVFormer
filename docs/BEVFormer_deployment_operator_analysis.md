# BEVFormer Deployment Operator Analysis

## 1. 目标

本文档记录 BEVFormer 在部署到 PONY Orin / TensorRT 前的算子级兼容性分析流程。

当前目标不是直接生成 TensorRT Engine，而是先回答：

1. BEVFormer 真实 inference 路径执行了哪些 operator？
2. 这些 operator 分别属于哪个模块？
3. PyTorch operator 是否能映射到 ONNX opset 13？
4. ONNX operator 是否能被 TensorRT 8.4.12.5 解析和执行？
5. 不支持的部分应该选择：Native / Rewrite / TensorRT Plugin / C++ CUDA Postprocess。

整体流程：

```text
Golden Sample
    ↓
真实 BEVFormer inference
    ↓
torch.profiler
    ↓
runtime operator inventory
    ↓
按模型模块分类
    ↓
PyTorch → ONNX opset13 → TensorRT 8.4.12
    ↓
Compatibility Report
    ↓
Native / Rewrite / Plugin / C++ Postprocess
```

## 2. 当前部署基线

### 2.1 BEVFormer Reference Environment

```text
Python:        3.8.20
PyTorch:       1.9.1+cu111
CUDA Toolkit:  11.1
cuDNN:         8.0.5
```

该环境已经稳定运行 Golden Sample，应保持冻结，作为数值 Reference。

Golden Sample：

```text
token:
3e8750f331d7499e9b5123e9eb70f2e2
```

Reference 输出：

```text
golden_samples/
└── 3e8750f331d7499e9b5123e9eb70f2e2/
    └── detections.npz
```

### 2.2 PONY Orin Build Baseline

当前 PONY 工程静态 build dependency 高可信指向：

```text
Target:          NVIDIA Drive AGX Orin
Architecture:    aarch64
GPU Arch:        sm_87
DriveOS config:  6.0.5
TensorRT:        8.4.12.5
CUDA:            11.4 family
CUDA nvcc:       11.4.160
cuDNN:           8.4.1
Compiler config: Clang 10
C++ standard:    C++17
Plugin API:      IPluginV2DynamicExt
ONNX target:     opset 13
```

注意：TensorRT 8.4.12.5 当前属于 **PONY Orin build baseline**，尚未通过真实板端 runtime shell 做最终现场确认。

## 3. BEVFormer 模块划分

```text
BEVFormer
├── 1. Backbone
│   └── ResNet
├── 2. Neck
│   └── FPN
├── 3. BEV Encoder
│   ├── TemporalSelfAttention
│   ├── SpatialCrossAttention
│   ├── MSDeformableAttention3D
│   ├── FFN
│   └── LayerNorm
├── 4. Detection Decoder
│   ├── Object Self-Attention
│   ├── Deformable Cross Attention
│   ├── FFN / LayerNorm
│   └── Reference Refinement
├── 5. Detection Head
│   ├── cls branch
│   └── reg branch
└── 6. Postprocess
    ├── sigmoid
    ├── TopK
    ├── bbox decode
    └── range/filter
```

## 4. Runtime Operator Profiling

### 4.1 Profiler 工具

已有工具：

```text
tools/profile_golden_sample_ops.py
```

该脚本基于真实 Golden Sample inference，使用 `torch.profiler.profile` 记录模型真实 forward 中执行的 operator。

主要配置：

```python
activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]
record_shapes=True
```

Profiler 区间只覆盖模型 inference forward，不包含模型初始化、checkpoint load、dataset 初始化和图像读取。

### 4.2 运行方式

```bash
cd /home/eric/workspace/bevformer/BEVFormer
conda activate bevformer

python tools/profile_golden_sample_ops.py \
  --token 3e8750f331d7499e9b5123e9eb70f2e2 \
  --row-limit 200
```

成功后生成：

```text
golden_samples/3e8750f331d7499e9b5123e9eb70f2e2/
├── operator_profile.json
├── operator_profile.csv
└── operator_events.json
```

### 4.3 Profiler 原始结果

真实 profiling 得到：

```text
Unique profiler keys: 237
```

这 **不代表有 237 个 ONNX operator**。其中包含大量底层 CUDA / cuDNN / CUTLASS 实现 kernel，例如：

```text
cutlass::Kernel<...>
xmma_new::gemm::kernel<...>
cudnn::...
at::native::unrolled_elementwise_kernel<...>
```

部署分析真正关注的是：

```text
aten::*
MMCV/custom operator
特殊 CUDA operator
```

## 5. Operator Compatibility Analysis

### 5.1 分析工具

已有：

```text
tools/analyze_operator_compatibility.py
```

兼容性数据库：

```text
docs/operator_compatibility.yaml
```

最终报告：

```text
docs/BEVFormer_operator_compatibility.md
```

### 5.2 运行方式

先运行 profiler：

```bash
python tools/profile_golden_sample_ops.py \
  --token 3e8750f331d7499e9b5123e9eb70f2e2 \
  --row-limit 200
```

再运行：

```bash
python tools/analyze_operator_compatibility.py
```

整体工具链：

```text
Golden Sample
↓
profile_golden_sample_ops.py
↓
operator_profile.json + operator_events.json
↓
analyze_operator_compatibility.py
+
docs/operator_compatibility.yaml
↓
docs/BEVFormer_operator_compatibility.md
```

## 6. 当前分析结果

237 个 profiler key 经清洗后：

```text
全局唯一 deployment-relevant operators: 45
Module/operator relations:             158
```

全局状态：

```text
Supported:          19
Conditional:        23
Unsupported:         1
Custom:              1
Unknown:             1
Plugin candidates:   1
```

状态定义：

```text
SUPPORTED
CONDITIONAL
UNSUPPORTED
CUSTOM
UNKNOWN
```

处理方式：

```text
NATIVE
CHECK
REWRITE
PLUGIN
CPP_POSTPROCESS
```

## 7. 模块级风险结论

### 7.1 Backbone — LOW

主要是 Conv、BatchNorm、ReLU、Pooling、Add 等标准算子。

```text
Risk: LOW
```

### 7.2 FPN — LOW

主要包括 Conv、Add、Reshape、feature formatting。

```text
Risk: LOW
```

### 7.3 BEV Encoder — HIGH

主要风险：

```text
TemporalSelfAttention
SpatialCrossAttention
MSDeformableAttention3D
dynamic valid query
NonZero
index / index_put
scatter-style accumulation
historical BEV rotation
```

```text
Risk: HIGH
```

### 7.4 Detection Decoder — HIGH

主要风险：

```text
Deformable Cross Attention
reference refinement
dynamic indexing
```

```text
Risk: HIGH
```

### 7.5 Detection Head — LOW

主要是 Linear、classification branch、regression branch。

```text
Risk: LOW
```

### 7.6 Postprocess — MEDIUM

包括：

```text
Sigmoid
TopK
index decode
atan2
range filter
bbox decode
```

部分逻辑建议放在 TensorRT 外部：

```text
TensorRT
↓
raw cls / reg output
↓
C++ / CUDA postprocess
↓
3D boxes
```

```text
Risk: MEDIUM
```

## 8. 当前明确的 Plugin Candidate

真实 profiler 已确认执行：

```text
MultiScaleDeformableAttnFunction_fp32
```

底层 CUDA kernel：

```text
ms_deformable_im2col_gpu_kernel<float>
```

一次 Golden Sample forward 中执行：

```text
TemporalSelfAttention:             3
Encoder MSDeformableAttention3D:  3
Decoder deformable cross-attn:    6
Total:                            12
```

注意：12 次执行不代表 12 个不同 Plugin，本质上是同一个 MultiScaleDeformableAttention primitive 在不同模块中重复调用。

当前最明确的部署方向：

```text
PyTorch/MMCV custom op
↓
ONNX custom node
↓
TensorRT PluginCreator
↓
IPluginV2DynamicExt
↓
CUDA kernel
```

即：

```text
MSDeformableAttention → PLUGIN candidate
```

## 9. 当前 Conditional Operators

以下 operator 尚不能仅通过静态文档判断最终 TensorRT 兼容性：

```text
aten::index
aten::index_select
aten::index_put_
NonZero
ScatterND
ScatterElements
dynamic Reshape
Slice
Split
Stack
LayerNorm decomposition
Softmax axis
CumSum
TopK
boolean mask / reduction
INT32 / INT64 conversion
```

这些应继续通过：

```text
实际 ONNX export
↓
TensorRT 8.4.12 parser
↓
engine build
```

进行验证。

不能简单认为：

```text
ONNX schema 中存在 = TensorRT 一定支持
```

## 10. 两个特殊问题

### 10.1 atan2

当前建议将 bbox decode 中的 `atan2` 放入 C++ postprocess，而不是优先塞进 TensorRT。

### 10.2 grid_sample

当前 Golden Sample 是 scene 首帧，因此真实 profiling 没有覆盖 `prev_bev rotation`。

潜在 temporal 路径中可能使用 `grid_sample`。标准 ONNX `GridSample` 在更高 opset 才提供标准表达，因此 opset 13 下属于潜在 blocker。

当前状态：

```text
latent UNSUPPORTED / REWRITE candidate
```

后续必须使用非首帧 temporal Golden Sample 再验证。

## 11. Golden Reference 一致性

兼容性分析没有修改 Golden Reference。

当前 `detections.npz` SHA-256：

```text
a904a128e7b166b1459686deb1bbbe0d2f072cc8a7188ecc2c51ad94353d8c60
```

## 12. 当前完成状态

已完成：

```text
[x] 模型整体 forward 流程梳理
[x] Golden Sample
[x] Forward module trace
[x] Runtime operator profiling
[x] Operator 清洗
[x] 模块化 operator 分类
[x] ONNX13 / TensorRT 8.4.12 第一版兼容性分析
[x] Plugin candidate 初步确认
```

尚未完成：

```text
[ ] 实际 ONNX opset13 export
[ ] ONNX graph operator verification
[ ] TensorRT 8.4.12 parser test
[ ] TensorRT engine build
[ ] MSDeformableAttention Plugin
[ ] temporal second-frame Golden Sample
[ ] ONNX vs PyTorch 数值对齐
[ ] TensorRT vs PyTorch 数值对齐
[ ] PONY C++ pipeline integration
```

## 13. 下一步

下一阶段不需要继续扩展 profiler，进入真实 ONNX export：

```text
BEVFormer
↓
第一次 opset13 export
↓
记录 exporter failure
↓
得到真实 ONNX blocker
↓
逐项处理：
Native / Rewrite / Plugin / C++ Postprocess
```

重点验证：

```text
1. MSDeformableAttention custom export
2. index / index_put
3. NonZero
4. Scatter
5. dynamic shape operations
6. LayerNorm decomposition
7. INT64 / INT32
8. TopK
```

第一次 export 的目标不是“一次成功”，而是把当前 `CONDITIONAL / UNKNOWN` 状态进一步收敛成真实的：

```text
SUPPORTED / REWRITE / PLUGIN / CPP_POSTPROCESS
```

## 14. 方法论总结

以后拿到其他模型，也可以复用同一套流程：

```text
1. 理解模型流程
2. 划分模块
3. 建立 Golden Sample
4. Profiler 获取真实 runtime operator
5. 清除 CUDA/cuDNN implementation noise
6. operator 归属具体模块
7. PyTorch → ONNX 映射
8. ONNX → TensorRT compatibility
9. 找 blocker
10. 决定 Native / Rewrite / Plugin / C++
11. ONNX 数值对齐
12. TensorRT 数值对齐
13. 集成目标 C++ 工程
```

核心分析对象：

```text
Module
→ Runtime Operator
→ ONNX13 Mapping
→ TensorRT 8.4.12 Support
→ Deployment Action
```

核心原则：不仅要问“ONNX opset 13 有没有这个算子”，还要继续确认“TensorRT 8.4.12.5 能不能执行这个具体 operator 和当前使用方式”。
