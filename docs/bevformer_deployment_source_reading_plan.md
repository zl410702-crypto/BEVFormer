# BEVFormer 推理与部署源码阅读计划

## 1. 学习目标

本计划用于系统理清当前仓库中已经打通的部署链路：

```text
BEVFormer Tiny PyTorch FP32 inference
        ↓
3-frame Stateful Golden Reference
        ↓
Stateful ONNX export
        ↓
MSDeformableAttention TensorRT Plugin
        ↓
PC TensorRT FP32 engine build and runtime
        ↓
后续 Orin / PONY C++ 集成
```

当前第一目标不是继续开发新功能，而是能够独立回答：

- 一帧图像在 PyTorch 中经过了哪些对象和函数？
- `prev_bev` 如何跨帧保存、对齐和重新使用？
- Golden Tensor 在哪里截取，代表什么计算边界？
- ONNX wrapper 为什么存在，修改了哪些部署接口？
- 12 个自定义 MSDeformableAttention 节点如何由一个 Plugin 实现？
- PC 上如何从 ONNX 构建 Engine，并通过 C++ 绑定输入输出？
- 哪些 PC 代码以后可以迁移到 Orin/PONY，哪些必须重新实现？

---

## 2. 当前基线必须先分清

当前已经完成并验证的主线使用：

```text
Config:
projects/configs/bevformer/bevformer_tiny.py

Model type:
BEVFormer

Golden precision:
PyTorch FP32

Deployment baseline:
ONNX FP32 interface + TensorRT FP32 engine
```

它不是 `bevformer_tiny_fp16.py`。以下三个概念是独立维度：

| 概念 | 含义 | 当前阶段 |
|---|---|---|
| BEVFormer Tiny | 较小的模型结构：50×50 BEV、3 层 Encoder、ResNet50 | 当前模型 |
| PyTorch FP16 配置 | MMCV 混合精度训练、loss scaling、特殊 temporal runner | 暂不作为阅读主线 |
| TensorRT FP16 | Engine 内部选择 FP16 layer/plugin tactic | FP32 闭环理解后再学习 |

检查项：

- [ ] 能解释 Tiny 与 FP16 为什么不是同一个概念
- [ ] 能在 Golden `sequence.json` 中找到实际 config
- [ ] 能说明为什么当前先学习 FP32，再研究 FP16

---

## 3. 总体阅读方法

每个阶段不要一次逐行读完整文件，按下面四轮进行：

1. **调用轮**：只记录从哪个函数进入、调用了谁、返回到哪里。
2. **Shape 轮**：只记录关键 Tensor 的 shape、dtype 和 layout。
3. **状态轮**：只记录跨函数、跨帧保存的数据。
4. **验证轮**：使用 Golden、ONNX audit 或 C++ 输出验证自己的理解。

建议为每个函数记录：

```text
File:
Class:
Function:
Input:
Output:
Calls:
State changed:
Python/PyTorch/MMCV/BEVFormer mechanism:
```

---

# Stage 1 — 原始 PyTorch 推理

## 1.1 模型配置与对象构建

主文件：

```text
projects/configs/bevformer/bevformer_tiny.py
```

阅读目标：只确认模型组成，不先研究训练 loss。

- [ ] 找到 `model = dict(...)`
- [ ] 找到 `type='BEVFormer'`
- [ ] 确认 Backbone 是 ResNet50
- [ ] 确认 FPN 输出 level 数量
- [ ] 确认 `bev_h=50`、`bev_w=50`
- [ ] 确认 embedding dimension 是 256
- [ ] 确认 Encoder 是 3 层
- [ ] 确认 Decoder 是 6 层
- [ ] 确认 object query 数量是 900

需要形成的结构图：

```text
Config dict
  ↓ OpenMMLab Registry/build_from_cfg
BEVFormer
  ├── ResNet50
  ├── FPN
  └── BEVFormerHead
       └── PerceptionTransformer
            ├── BEVFormerEncoder × 3
            └── DetectionTransformerDecoder × 6
```

完成标准：

- [ ] 能从配置项找到对应 Python class
- [ ] 能解释配置中的 `type` 如何选择 class
- [ ] 能说明 config 是对象构造说明，不是实际 forward 过程

## 1.2 Detector 推理入口

主文件：

```text
projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
```

建议阅读顺序：

```text
BEVFormer.__init__
BEVFormer.forward
BEVFormer.forward_test
BEVFormer.simple_test
BEVFormer.extract_feat
BEVFormer.extract_img_feat
BEVFormer.simple_test_pts
```

主要调用链：

```text
model(...)
  ↓ torch.nn.Module.__call__
BEVFormer.forward(...)
  ↓ return_loss=False
BEVFormer.forward_test(...)
  ↓
BEVFormer.simple_test(...)
  ├── extract_feat()
  └── simple_test_pts()
```

重点状态：

```text
prev_frame_info
├── prev_bev
├── scene_token
├── prev_pos
└── prev_angle
```

检查项：

- [ ] 找到 scene 变化时重置 `prev_bev` 的代码
- [ ] 找到 `can_bus[:3]` translation delta 的计算
- [ ] 找到 `can_bus[-1]` rotation delta 的计算
- [ ] 找到当前帧 `bev_embed` 保存回 `prev_frame_info` 的位置
- [ ] 能解释 Frame 0、Frame 1 的行为为什么不同

## 1.3 Detection Head 与 Transformer

主文件：

```text
projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
projects/mmdet3d_plugin/bevformer/modules/transformer.py
```

重点函数：

```text
BEVFormerHead.forward
PerceptionTransformer.get_bev_features
PerceptionTransformer.forward
```

数据流：

```text
FPN multi-level features
        ↓
BEV query + BEV positional encoding
        ↓
get_bev_features()
        ↓
BEV Encoder
        ↓
bev_embed
        ↓
Detection Decoder
        ↓
all_cls_scores + all_bbox_preds
```

检查项：

- [ ] 找到 BEV query embedding 的来源
- [ ] 找到 BEV positional encoding 的来源
- [ ] 区分 `get_bev_features()` 与完整 `forward()`
- [ ] 找到 `prev_bev` 传入 Transformer 的位置
- [ ] 找到 Decoder 接收 `bev_embed` 的位置

## 1.4 Encoder、Temporal、Spatial 和 Decoder

主文件：

```text
projects/mmdet3d_plugin/bevformer/modules/encoder.py
projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
projects/mmdet3d_plugin/bevformer/modules/decoder.py
projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
```

单个 Encoder Layer：

```text
BEV query
  ↓ Temporal Self Attention
Norm
  ↓ Spatial Cross Attention
Norm
  ↓ FFN
Norm
  ↓ Encoder layer output
```

建议只追踪以下内容：

- [ ] `BEVFormerEncoder.point_sampling()` 如何生成 camera reference points
- [ ] Temporal Attention 的 query 和历史 value 如何组合
- [ ] Spatial Cross Attention 如何使用 camera feature
- [ ] MSDeformableAttention 的五个输入分别是什么
- [ ] 三层 Encoder 如何依次更新 query
- [ ] 六层 Decoder 如何输出分类与框回归

## 1.5 Stage 1 Shape 清单

- [ ] `images`: `[1, 6, 3, 480, 800]`
- [ ] `bev_query`: `[2500, 256]`
- [ ] `prev_bev`: `[2500, 1, 256]`
- [ ] `bev_embed`: `[2500, 1, 256]`
- [ ] `object_query`: 900 个 query，确认实际 layout
- [ ] `all_cls_scores`: `[6, 1, 900, 10]`
- [ ] `all_bbox_preds`: `[6, 1, 900, 10]`

Stage 1 完成标准：

- [ ] 能手画一帧完整 PyTorch 推理调用链
- [ ] 能手画 Frame 0 → Frame 1 的 `prev_bev` 状态链
- [ ] 能区分 Detector、Head、Transformer、Encoder、Attention、Decoder

---

# Stage 2 — 3-Frame Stateful Golden Reference

## 2.1 单 sample 准备

主文件：

```text
tools/run_golden_sample.py
```

建议阅读：

```text
parse_args
import_plugin
prepare_golden_sample
validate_pipeline_item
pipeline_image_tensor
run_model_inference
result_arrays
```

检查项：

- [ ] 找到 Config 加载入口
- [ ] 找到 plugin module 导入入口
- [ ] 找到 dataset 和 model 构建入口
- [ ] 找到 checkpoint 加载入口
- [ ] 找到 sample token 定位方法
- [ ] 找到真正调用模型的位置

## 2.2 连续三帧入口

主文件：

```text
tools/run_golden_sequence.py
```

重点函数：

```text
consecutive_indices
print_sequence
main
```

检查项：

- [ ] 确认三帧来自同一个 scene
- [ ] 确认 sample 使用 nuScenes `prev/next` 链
- [ ] 确认 timestamp 严格递增
- [ ] 确认只创建一个 model instance
- [ ] 确认每帧调用官方 model inference
- [ ] 确认没有手工写 `prev_bev = last_output`

状态链：

```text
同一个 model instance
        ↓
Frame 0 forward_test
        ↓ save prev_frame_info.prev_bev
Frame 1 forward_test
        ↓ reuse and update
Frame 2 forward_test
```

## 2.3 Tensor 记录与比较

主文件：

```text
tools/golden_sequence_recorder.py
tools/tensor_dump.py
tools/tensor_compare.py
```

数据流：

```text
module hook
  ↓ capture Tensor
detach → cpu → numpy
  ↓
.npy + tensor_manifest.json
  ↓
tensor_compare.py
```

检查项：

- [ ] 找到 hook 的安装和恢复位置
- [ ] 确认 dump 使用 `detach().cpu().numpy()`
- [ ] 区分 input hook、output hook 和源码内 instrumentation
- [ ] 找到 `final_bev` dump point
- [ ] 找到 `prev_bev_stored` dump point
- [ ] 验证 Frame t `final_bev` 等于 Frame t+1 `prev_bev_stored`

参考文档：

```text
docs/temporal_bev_notes.md
docs/deformable_attention_notes.md
docs/golden_tensor_boundaries.md
```

Stage 2 完成标准：

- [ ] 能说明 Golden 是官方推理加观测，不是另一套推理实现
- [ ] 能为任意一个 `.npy` 找到对应 PyTorch tensor 和源码位置
- [ ] 能使用 `tensor_compare.py` 比较两个 Tensor

---

# Stage 3 — Stateful ONNX 导出

## 3.1 导出总入口

主文件：

```text
tools/export_bevformer_stateful_onnx.py
```

建议阅读顺序：

```text
parse_args
prepare_runtime_meta
first_frame_runtime_meta
compute_shift
Wrapper validation
Frame 0 initialization validation
torch.onnx.export
ONNX checker and graph audit
```

检查项：

- [ ] 确认默认 config 和 checkpoint
- [ ] 确认 Frame 1 使用 Frame 0 Golden `final_bev`
- [ ] 确认 export 前先比较 wrapper 与 Golden
- [ ] 确认 real/zero `prev_bev` sensitivity check
- [ ] 确认 Frame 0 的 `use_prev_bev` gate
- [ ] 找到 `torch.onnx.export(...)`
- [ ] 找到 opset 和 input/output name 定义

## 3.2 Stateful Wrapper

主文件：

```text
tools/deployment/bevformer_stateful_onnx_wrapper.py
```

核心思想：将 PyTorch 对象内部的隐式状态转换成 ONNX 显式输入输出。

```text
PyTorch implicit runtime state
prev_frame_info dict
        ↓
ONNX explicit contract
images
prev_bev
can_bus
shift
lidar2img
use_prev_bev
affine_grid（host-affine production variant）
        ↓
all_cls_scores
all_bbox_preds
bev_embed
```

检查项：

- [ ] 找到 wrapper 的 `forward()`
- [ ] 逐个记录 input shape、dtype、layout 和语义
- [ ] 确认 `prev_bev` 输入与 `bev_embed` 输出 layout 相同
- [ ] 确认 Frame 0 与 Frame 1+ 共用同一个接口
- [ ] 列出 wrapper 与官方 `forward_test()` 的职责差异

## 3.3 Deployment-only 改写

目录：

```text
tools/deployment/
```

重点文件：

| 文件 | 作用 |
|---|---|
| `bevformer_onnx_wrapper.py` | 基础 ONNX wrapper、custom symbolic、deployment override |
| `bevformer_stateful_onnx_wrapper.py` | Stateful 单帧 Tensor 接口 |
| `grid_sample_rewrite.py` | 将 previous-BEV rotation 改写成可导出采样图 |
| `onnx_slice_compat.py` | 将可严格静态求值的 Slice axes 规范为 TRT 8.4 可解析常量 |
| `host_affine_contract.py` | 定义 host 侧 affine-grid 输入契约 |

必须区分：

```text
projects/mmdet3d_plugin/...
→ 官方/常规 PyTorch inference

tools/deployment/...
→ deployment/export-only adaptation
```

检查项：

- [ ] 确认 deployment override 使用 context 安装并恢复
- [ ] 确认 normal inference 默认行为没有改变
- [ ] 理解 static SCA 为什么固定 Nq=2500
- [ ] 理解 fixed Linear chunk=640 的来源和作用范围
- [ ] 理解为什么移除了 visibility-derived `604`
- [ ] 理解 Slice compatibility pass 为什么必须 fail-closed

## 3.4 Host Affine Production Contract

相关文件：

```text
tools/export_bevformer_host_affine_onnx.py
tools/prepare_bevformer_host_affine_runtime.py
tools/deployment/host_affine_contract.py
tools/deployment/grid_sample_rewrite.py
docs/host_affine_grid_contract.md
```

问题链：

```text
TensorRT affine MatMul 微小误差
        ↓
坐标跨过 half-integer 边界
        ↓
Round 得到不同 index
        ↓
Gather 不同 BEV cell
        ↓
aligned_prev_bev 明显不一致
```

最终契约：

```text
Host computes affine_grid [1,2500,2] FP32
        ↓ TensorRT input
coordinate scaling
        ↓
Round + validity mask + Clip
        ↓
GatherElements
        ↓
aligned_prev_bev
```

检查项：

- [ ] 理解为什么只把 affine-grid MatMul 移到 host
- [ ] 理解为什么没有把 integer index 单独作为 host contract
- [ ] 确认 Round、valid mask、Clip、Gather 仍在 Engine 内
- [ ] 能说明 host affine input 的 shape、dtype 和 layout

Stage 3 完成标准：

- [ ] 能画出 Stateful ONNX 的全部 graph inputs/outputs
- [ ] 能解释每一项 deployment-only 修改解决的问题
- [ ] 能区分模型语义修改与等价部署改写

---

# Stage 4 — MSDeformableAttention TensorRT Plugin

目录：

```text
tensorrt_plugins/ms_deformable_attention/
```

## 4.1 先读 Operator Contract

首先阅读：

```text
tensorrt_plugins/ms_deformable_attention/README.md
docs/MSDeformableAttention_plugin_audit.md
docs/MSDeformableAttention_plugin_validation.md
```

五输入一输出：

```text
value
spatial_shapes
level_start_index
sampling_locations
attention_weights
        ↓
MSDeformableAttention
        ↓
output [B,Q,H*D]
```

12 个实例分类：

```text
Temporal Self Attention × 3
Spatial Cross Attention × 3
Detection Decoder Cross Attention × 6
```

检查项：

- [ ] 能解释为何语义位置不同但数学 operator contract 相同
- [ ] 能说明 `spatial_shapes` 和 `level_start_index` 为什么使用 INT32
- [ ] 能说明 sampling coordinate 的 x/y 顺序
- [ ] 能说明 Plugin 不包含 offset Linear、softmax 和 output projection

## 4.2 Plugin API

先读头文件：

```text
tensorrt_plugins/ms_deformable_attention/ms_deformable_attention_plugin.h
```

再读实现：

```text
tensorrt_plugins/ms_deformable_attention/ms_deformable_attention_plugin.cpp
```

阅读顺序：

```text
getPluginType / getPluginVersion
getOutputDimensions
supportsFormatCombination
configurePlugin
enqueue
serialize / deserialize
clone
PluginCreator
REGISTER_TENSORRT_PLUGIN
```

调用关系：

```text
ONNX parser sees bevformer::MSDeformableAttention
        ↓
Plugin Registry finds PluginCreator
        ↓
createPlugin / deserializePlugin
        ↓
MSDeformableAttentionPlugin instance
        ↓ runtime enqueue()
        ↓ CUDA kernel
```

检查项：

- [ ] 知道 Plugin type/version 必须匹配 parser 查找契约
- [ ] 知道一个 C++ Plugin class 对应 12 个 layer instance
- [ ] 能解释 `getOutputDimensions()` 如何从输入推导 `[B,Q,H*D]`
- [ ] 能解释 `supportsFormatCombination()` 如何约束 dtype
- [ ] 能解释 `enqueue()` 的输入来自 TensorRT device buffer

## 4.3 CUDA Kernel

文件：

```text
tensorrt_plugins/ms_deformable_attention/ms_deformable_attention_kernel.h
tensorrt_plugins/ms_deformable_attention/ms_deformable_attention_kernel.cu
```

第一遍只理解正确性数据流：

```text
一个 output element (b,q,h,c)
        ↓
遍历 level l
        ↓
遍历 sampling point p
        ↓
normalized coordinate → feature coordinate
        ↓
读取四个邻近 feature
        ↓
bilinear interpolation
        ↓
乘 attention weight 并累加
        ↓
write output
```

检查项：

- [ ] 找到 kernel launch 入口
- [ ] 找到 thread index 到 `(b,q,h,c)` 的映射
- [ ] 找到 normalized coordinate 到 pixel coordinate 的公式
- [ ] 找到 out-of-bound 处理
- [ ] 找到双线性插值四个权重
- [ ] 确认没有误把 attention 前后的 Linear 放进 Plugin

## 4.4 Plugin 验证

相关文件：

```text
tensorrt_plugins/ms_deformable_attention/validation/msda_kernel_test_bridge.cpp
tensorrt_plugins/ms_deformable_attention/validation/test_plugin_serialization.cpp
tools/inspect_msda_onnx.py
```

检查项：

- [ ] 理解 isolated kernel test 如何与 PyTorch/MMCV reference 比较
- [ ] 理解 serialization/deserialization 为什么需要单独测试
- [ ] 能在 audit 中找到 12 个 custom node 的 shape
- [ ] 区分 isolated Plugin PASS 与 whole-engine PASS

Stage 4 完成标准：

- [ ] 能从 ONNX custom node 一直追到 CUDA kernel
- [ ] 能说明 Plugin 每个输入的 shape、dtype、layout 和语义
- [ ] 能说明 PluginCreator、Plugin instance、CUDA kernel 的分工

---

# Stage 5 — PC TensorRT Engine 构建与运行

## 5.1 Engine Builder

文件：

```text
tensorrt_plugins/ms_deformable_attention/build_bevformer_fp32_engine.cpp
```

阅读顺序：

```text
main
  ↓ Plugin registration/load
createInferBuilder
  ↓
createNetworkV2
  ↓
createParser
  ↓
parseFromFile
  ↓
createBuilderConfig
  ↓
buildSerializedNetwork
  ↓
write .engine
```

检查项：

- [ ] 找到 ONNX 路径和 Engine 输出路径
- [ ] 找到 TensorRT logger
- [ ] 找到 Plugin 注册或加载过程
- [ ] 找到 FP32 build 配置
- [ ] 找到 workspace 配置
- [ ] 区分 parser PASS 与 engine build PASS
- [ ] 知道 PC Engine 不能直接部署到 Orin

## 5.2 Engine Runtime

文件：

```text
tensorrt_plugins/ms_deformable_attention/validation/run_bevformer_engine.cpp
```

阅读顺序：

```text
read engine bytes
  ↓
deserializeCudaEngine
  ↓
createExecutionContext
  ↓
inspect bindings
  ↓
allocate host/device buffers
  ↓
copy input Host → Device
  ↓
executeV2
  ↓
copy output Device → Host
  ↓
save/compare results
```

检查项：

- [ ] 列出全部 input/output binding
- [ ] 为每个 binding 记录 dtype、shape 和字节数
- [ ] 找到 CUDA buffer 分配和释放位置
- [ ] 找到 host-to-device copy
- [ ] 找到 `executeV2()`
- [ ] 找到 device-to-host copy
- [ ] 找到 Frame 0 `bev_embed` 作为 Frame 1 `prev_bev` 的运行逻辑
- [ ] 找到 `affine_grid` binding 的输入位置

## 5.3 PC 与 Orin 的边界

当前环境：

```text
PC precheck:
RTX 4060 Laptop
TensorRT 8.5.3

Target board:
NVIDIA Orin
DriveOS 6.0.5
CUDA 11.4
TensorRT 8.4.12.5
```

必须记住：

```text
ONNX 可以作为跨平台模型交换文件
Plugin source 可以针对目标环境重新编译
PC 生成的 TensorRT Engine 不应直接复制到 Orin
Orin 必须使用板端 TensorRT/CUDA 重新 build Engine
```

Stage 5 完成标准：

- [ ] 能解释 ONNX parser、Builder、Engine、ExecutionContext 的区别
- [ ] 能画出 C++ runtime 的内存流
- [ ] 能说明为何 PC Engine 不能作为 Orin 正式产物
- [ ] 能说明 Plugin `.so` 在 parse/build/runtime 中的作用

---

# Stage 6 — PONY C++ 集成前准备

这一阶段只做接口映射，不立即修改 PONY。

## 6.1 需要从当前仓库提取的接口

建立一张最终 runtime contract 表：

| Binding | Shape | Dtype | Producer | Consumer | Stateful |
|---|---|---|---|---|---|
| images | `[1,6,3,480,800]` | FP32 | 图像预处理 | TensorRT Engine | No |
| prev_bev | `[2500,1,256]` | FP32 | Runtime state | TensorRT Engine | Yes |
| can_bus | 按当前 ONNX 实际值 | FP32 | ego-motion preparation | Engine | No |
| shift | `[1,2]` | FP32 | host motion calculation | Engine | No |
| lidar2img | `[1,6,4,4]` | FP32 | calibration | Engine | No |
| use_prev_bev | `[1]` | FP32 | scene/state manager | Engine | Yes |
| affine_grid | `[1,2500,2]` | FP32 | host alignment preparation | Engine | No |
| bev_embed | `[2500,1,256]` | FP32 | Engine | next-frame state | Yes |
| all_cls_scores | `[6,1,900,10]` | FP32 | Engine | decode/postprocess | No |
| all_bbox_preds | `[6,1,900,10]` | FP32 | Engine | decode/postprocess | No |

注意：表中 `can_bus` 最终 shape 应再次以 production ONNX audit 为准。

## 6.2 PONY 侧需要寻找的模块

- [ ] 相机图像获取与六相机同步
- [ ] Resize/Normalize/Pad 和 Tensor layout
- [ ] CUDA stream 和 device memory 管理
- [ ] TensorRT Engine loader
- [ ] Plugin `.so` 构建和注册方式
- [ ] ego pose / can bus 数据来源
- [ ] lidar-to-image calibration 来源
- [ ] scene reset 或 sequence reset 信号
- [ ] 跨帧 `prev_bev` device buffer 生命周期
- [ ] detection decode 和下游消息格式

## 6.3 建议的 PONY 模块边界

```text
CameraPreprocessor
        ↓ images
MotionPreprocessor
        ├── can_bus
        ├── shift
        ├── lidar2img
        └── affine_grid
        ↓
BEVFormerTensorRTRunner
        ├── Engine/Context
        ├── input/output buffers
        └── enqueue
        ↓
BEVFormerStateManager
        ├── prev_bev
        ├── scene/sequence identity
        └── use_prev_bev
        ↓
DetectionPostprocessor
```

检查项：

- [ ] 能把现有 Python input preparation 映射到 PONY C++ producer
- [ ] 能定义首帧、连续帧、scene reset 三种状态转换
- [ ] 能说明哪些 Tensor 应一直保留在 GPU
- [ ] 能说明哪些数据来自车辆平台而不是模型内部

---

# 7. 推荐跟踪顺序

## Milestone 1 — 能看懂 PyTorch 单帧推理

- [ ] 完成 Stage 1.1～1.4
- [ ] 手画调用链
- [ ] 手写关键 Tensor shape 表
- [ ] 选择 Frame 0，用断点跟完一次 `forward_test()`

## Milestone 2 — 能看懂 Stateful temporal 行为

- [ ] 完成 Stage 1 temporal state 检查
- [ ] 完成 Stage 2
- [ ] 对照 Frame 0/1/2 的 `sequence.json`
- [ ] 验证两组 `final_bev → prev_bev_stored`

## Milestone 3 — 能解释 ONNX 接口为什么这样设计

- [ ] 完成 Stage 3.1～3.3
- [ ] 列出 Stateful ONNX inputs/outputs
- [ ] 区分 runtime variable、model constant 和 implicit Python state
- [ ] 对每项 export-only rewrite 写一句原因

## Milestone 4 — 能理解 Host Affine 修复

- [ ] 完成 Stage 3.4
- [ ] 能解释连续小误差为何会变成离散 Gather 大误差
- [ ] 能说明 host affine contract 为什么完整且最小

## Milestone 5 — 能理解 Plugin

- [ ] 完成 Stage 4
- [ ] 手写 MSDA 五输入一输出 contract
- [ ] 从一个 ONNX node 追到 `enqueue()` 和 CUDA kernel

## Milestone 6 — 能理解 PC Engine 生命周期

- [ ] 完成 Stage 5
- [ ] 手画 build 和 runtime 两条不同调用链
- [ ] 能说明 bindings、buffer、stream 和 execution context

## Milestone 7 — 准备进入 PONY 集成

- [ ] 完成 Stage 6 runtime contract
- [ ] 在 PONY 仓库找到对应模块
- [ ] 形成 PC validation runner → PONY runtime 的差异表
- [ ] 确认 Orin 侧 Plugin 和 Engine 重建方案

---

# 8. 每阶段建议产出

为了避免“读过但没有留下理解”，每个阶段完成后保留一个简短产物：

| Stage | 建议产物 |
|---|---|
| PyTorch inference | 一张函数调用链 + Tensor shape 表 |
| Golden | 一张三帧 state transition 图 |
| ONNX export | 一张 inputs/outputs 与 Python state 映射表 |
| Plugin | 一张五输入一输出 contract + C++ 调用链 |
| PC Engine | 一张 build/runtime 流程图 + binding 表 |
| PONY integration | 一张模块职责与数据所有权表 |

---

# 9. 最终自测

完成本计划后，应能不依赖 Codex 回答：

- [ ] `model(...)` 为什么会进入 `forward()`？
- [ ] `forward_test()` 在哪里保存 temporal state？
- [ ] Frame 0 没有历史 BEV 时如何执行？
- [ ] Frame 1 的 translation 和 rotation delta 在哪里计算？
- [ ] `bev_embed` 为什么既是输出又是下一帧输入？
- [ ] Golden hook 为什么不会改变模型计算？
- [ ] ONNX 为什么不能直接依赖 `prev_frame_info` Python dict？
- [ ] `use_prev_bev` 为什么需要成为显式 runtime input？
- [ ] static SCA 为什么使用 Nq=2500，而不是 traced 604？
- [ ] ONNX 中为什么有 12 个 MSDeformableAttention 节点？
- [ ] 为什么它们可以共享一个 Plugin class？
- [ ] PluginCreator 和 Plugin instance 有什么区别？
- [ ] `enqueue()` 与 CUDA kernel 分别负责什么？
- [ ] parser PASS 为什么不等于 Engine runtime 数值 PASS？
- [ ] host affine-grid 解决了什么问题？
- [ ] PC Engine 为什么不能直接复制到 Orin？
- [ ] PONY 中谁负责保存和 reset `prev_bev`？

---

# 10. 当前阶段暂不展开

在完成上述 FP32 阅读主线前，暂不深入：

- PyTorch `bevformer_tiny_fp16.py` 训练实现；
- TensorRT FP16 tactic 和误差验收；
- INT8 量化；
- Plugin 性能优化；
- CUDA kernel fusion；
- Orin 性能 profiling；
- PONY production code 修改。

正确顺序是：

```text
先理解已经验证的 FP32 闭环
        ↓
再完成 PONY C++ 接口映射
        ↓
再讨论 Orin build 和 FP16
```
