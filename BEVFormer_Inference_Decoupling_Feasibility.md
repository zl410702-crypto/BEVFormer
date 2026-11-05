# BEVFormer-tiny 推理模型解耦可行性分析

> 只读调查结论；未改模型源码、Checkpoint 或环境，未运行新模型、PTQ、ONNX 或 TensorRT。完整 643 项权重清单见同目录 `BEVFormer_Checkpoint_Inventory.md`。本报告中的「可直接加载」仅指结构可能性，不代表数值等价。

## 结论

独立 PyTorch inference-only 模型**可行但属于中高风险迁移**。建议采用「B→C」渐进路线：先在独立环境建立保留原 `state_dict` 层级的薄兼容模型并逐层对齐，再移除训练/Registry/Builder 依赖。不要原地升级旧环境，也不要在 FP32 对齐之前引入 ModelOpt。旧模型在当前 Golden 环境以 `strict=True` 载入原 Checkpoint 成功，这是迁移的结构基线，而不是新模型的验证结果。

## 1. OpenMMLab 推理依赖矩阵

| File / Class·Function | Dependency / Import | 实际作用及推理必要性 | 替代与难度 / 风险 |
|---|---|---|---|
| `tools/run_golden_sample.py:143` `prepare_golden_sample` | `mmcv.Config`, `MMDataParallel`, `collate`, `mmcv.runner.load_checkpoint`; `mmdet.set_random_seed`; `mmdet3d.build_dataset/build_model` | Golden 入口、权重加载、DataContainer 解包；**仅数据/构建层**，不是网络计算 | 独立配置常量 + 明确 dataset/preprocess + `torch.load`；中。必须复现六相机顺序与 metadata |
| `bevformer/detectors/bevformer.py:21` `BEVFormer` | `mmdet.DETECTORS`, `mmdet3d.MVXTwoStageDetector`, MMCV AMP 装饰器 | 父类构造 `pts_bbox_head`、`img_backbone`、`img_neck`，并提供 `with_img_neck`；**推理构建必需** | 显式 `nn.Module` 构造并保留三个属性名；中。漏建模块会使 strict-load 失败 |
| `bevformer/detectors/bevformer.py:67` `extract_img_feat` / `:236` `forward_test` | 上述父类属性、`GridMask`、`bbox3d2result` | 图像 reshape、Backbone/FPN、scene reset、CAN delta、结果封装；**推理必需** | 原逻辑逐语义迁移；中。`squeeze_`、metadata 原地改写和 scene 边界敏感 |
| `bevformer/dense_heads/bevformer_head.py:17` `BEVFormerHead` | `mmdet.DETRHead`, `HEADS`; `mmdet3d.build_bbox_coder`; MMCV `Linear`/AMP | `DETRHead.__init__` 建 `loss_cls`、`positional_encoding`、`transformer` 并调用子类 `_init_layers`；**构建与前向必需**，Loss 非前向必需 | 显式重建属性、LayerNorm/Linear/Embedding/分支；高。`cls_out_channels` 由 sigmoid loss 设为 10，不能盲设为 11 |
| `bevformer/dense_heads/bevformer_head.py:118` `forward` / `:483` `get_bboxes` | `mmdet.inverse_sigmoid`; `mmdet3d` 3D box 类型 | BEV query、分类/回归 refinement、bbox decode；**前向和结果解码必需** | 算式可用 PyTorch；bbox 类型需保留或独立 decode；中高。坐标范围、角度/速度顺序需核对 |
| `bevformer/modules/transformer.py:27` `PerceptionTransformer` | MMCV `BaseModule`, `build_transformer_layer_sequence`, 初始化与 AMP；`mmdet.TRANSFORMER` | 构建 encoder/decoder，生成 BEV 与 object queries；**推理必需** | 显式 `ModuleList`；高。初始权重加载后可忽略初始化结果，但模块拓扑不能改 |
| `bevformer/modules/encoder.py:25` `BEVFormerEncoder` / `:243` `BEVFormerLayer` | MMCV `TransformerLayerSequence`, `BaseModule`, attention/FFN builder | 3 层 encoder、reference points、投影与 operation order；**推理必需** | 显式层序列与 FFN；高。mask、几何、norm 顺序易偏差 |
| `bevformer/modules/decoder.py:53` `DetectionTransformerDecoder` / `:133` `CustomMSDeformableAttention` | MMCV `TransformerLayerSequence`, `MultiheadAttention`/FFN/Registry, `mmcv.ops` fallback | 6 层 decoder、self/cross attention、iterative reference refinement；**推理必需** | `nn.MultiheadAttention` 可能保留同名 `in_proj_*`，但 mask、batch_first、dropout、residual 必须对齐；高 |
| `bevformer/modules/temporal_self_attention.py:26` `TemporalSelfAttention` | MMCV `BaseModule`、初始化、CUDA op/fallback | 历史 BEV 的 TSA 与 `value_proj`；**推理必需** | 算子边界保留；其余 PyTorch 迁移；高 |
| `bevformer/modules/spatial_cross_attention.py:32` `SpatialCrossAttention` / `:179` `MSDeformableAttention3D` | MMCV `BaseModule`、attention builder、CUDA op/fallback | 相机可见 query 聚合与 SCA；**推理必需** | 原式迁移，特别核对 visible-query compaction；高 |
| `bevformer/modules/multi_scale_deformable_attn_function.py:12` `MultiScaleDeformableAttnFunction_fp32.forward` | `mmcv.utils.ext_loader` → `_ext.ms_deform_attn_forward` | 真正的 CUDA forward；**推理必需** | PyTorch 2.8 下重新编译/替代算子；最高风险。不可复用旧 ABI 的 `.so` |
| `bevformer/modules/custom_base_transformer_layer.py:38` `MyCustomBaseTransformerLayer` | MMCV `BaseModule`, `build_attention`, `build_feedforward_network`, norm builder | 注册 `attentions`、`ffns`、`norms` 的顺序与 forward residual；**推理必需** | 薄兼容或显式构造；高。参数名由层级与 FFN 内部结构决定 |
| `core/bbox/coders/nms_free_coder.py:10` `NMSFreeCoder` + `core/bbox/util.py` | `mmdet.BaseBBoxCoder`/Registry，PyTorch tensor ops | sigmoid/topk、反归一化、范围筛选；**最终检测解码必需**，原始 logits/box/bev 输出不依赖 | 可独立实现；中。阈值与返回 box 类型必须一致 |
| `datasets/nuscenes_dataset.py:19` `CustomNuScenesDataset`、`datasets/pipelines/*` | `mmdet3d.NuScenesDataset`/3D box，MMCV image ops/DataContainer，`mmdet.PIPELINES` | nuScenes 读取、camera 标定、预处理与 metadata；**Golden/Calibration 输入必需，但可移出网络** | 独立只读 loader/preprocess；中高。图像 normalize/pad/resize 和 camera 顺序必须逐项对照 |
| `models/utils/grid_mask.py:70` `GridMask` | PyTorch `nn.Module` | eval 路径是否实际启用须核对；`extract_img_feat` 会依 `use_grid_mask` 调用 | 可复用纯 PyTorch 实现；低，但要验证 `eval()` 行为 |
| `bevformer/dense_heads/bevformer_head.py:215`–`:482` Loss/target 方法，`bevformer/detectors/bevformer.py:111`–`:235` 训练方法 | `mmdet` Loss/assigner/sampler、MMCV 分布式辅助 | **训练专用**，独立推理模型不需要 | 删除训练调用而非伪造 Loss；低到中，须保留 `cls_out_channels` 构造语义 |
| `tools/train.py:28`、`bevformer/apis/train.py:8` | `mmseg` 版本记录 / `train_segmentor` | **训练/记录专用，推理不需要** | 不进入独立推理环境；低 |

这里的路径均相对 `BEVFormer/projects/mmdet3d_plugin/`（`tools/` 相对 `BEVFormer/`）。其他 `core/` 中大量 assigner、loss、evaluation 及 `datasets/` 的 GT 处理未逐文件搬迁：它们不在已核对的 Tiny 推理前向链上；是否被插件初始化的 import side effect 触及，仍需在新环境做 import smoke test。`models/` 下其余 DD3D 等模块亦不能仅凭 import 数量判为 Tiny 推理必需。全树初筛共有 363 条 OpenMMLab import/类/注册匹配；矩阵聚焦实际入口和所用 Tiny config。

## 2. Checkpoint 结构与加载判定

只读 `torch.load(..., map_location='cpu')` 确认顶层为 `meta`、`state_dict`、`optimizer`；`state_dict` 是 643 项 `OrderedDict`，**无 `module.` 前缀**。`meta` 含 config、框架版本、CLASSES、epoch 等；训练 optimizer 不属于推理权重。文件为现有路径 `/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth`，未修改。

| 前缀 | state_dict 项 | 元素数 | 主要结构 |
|---|---:|---:|---|
| `img_backbone` | 318 | 23,561,205 | ResNet50 conv/BN；含 53 个 `num_batches_tracked` INT64 Buffer |
| `img_neck` | 4 | 1,114,624 | FPN `lateral_convs.0.conv` 与 `fpn_convs.0.conv` |
| `pts_bbox_head` | 321 | 8,898,501 | BEV/Query Embedding、3 层 Encoder、6 层 Decoder、6 组 cls/reg branches |
| **合计** | **643** | **33,574,330** | **484 个 Parameter tensor / 33,521,157 元素；159 个 Buffer tensor / 53,173 元素** |

这些 Parameter/Buffer 数来自在旧环境按 `tools/run_golden_sample.py:143` 的配置/Builder 路径构造模型、对**原始 Checkpoint** 执行 `load_state_dict(strict=True)` 后的 `named_parameters()`/`named_buffers()`；结果为 `All keys matched successfully`。原模型含 6 个 Decoder layer，未发现共享 Parameter object。BN 的 `running_mean`、`running_var` 与 `num_batches_tracked` 均在 159 个 Buffer 内；新模型应保持 `track_running_stats` 和 eval 模式，不能只复制可训练参数。`pts_bbox_head.code_weights` 在源码中被包装为 `requires_grad=False` 的 `nn.Parameter`，不是 Buffer，见 `bevformer/dense_heads/bevformer_head.py:62`。

六组 `cls_branches.0..5`、`reg_branches.0..5` 因 `with_box_refine=True` 而通过 deepcopy 独立构建，见 `bevformer/dense_heads/bevformer_head.py:69`；Decoder 6 层由配置 `projects/configs/bevformer/bevformer_tiny.py:118` 和运行时检查共同确认。`DETRHead.__init__` 先构建 positional encoding/transformer 再调用子类 `_init_layers`，旧框架源码位于 `/data/conda_envs/bevformer/lib/python3.8/site-packages/mmdet/models/dense_heads/detr_head.py:51`。`MVXTwoStageDetector.__init__` 构建三个顶层子模块，见 `/data/conda_envs/bevformer/lib/python3.8/site-packages/mmdet3d-0.17.1-py3.8-linux-x86_64.egg/mmdet3d/models/detectors/mvx_two_stage.py:23`。因此新模型仅“顶层名字相同”远远不够；必须在内部保持具体容器层级和算子属性。

**直接加载可行性：** 若新模型精确复刻 643 项的名称、shape、dtype 及 Buffer 注册，则可望 `strict=True` 加载；目前只验证了**旧模型** strict-load，未验证新模型。未发现必须 Key Mapping 的证据。若以后需要映射，必须生成完整逐键原→新映射，并检查一对一、形状、dtype、别名、缺失/意外键；不能修改原始 `.pth`。`strict=True` 也不验证 ResNet stride、BN eps、FFN residual、attention mask、几何公式或 Stateful 逻辑。

## 3. 推理计算与数据流

入口是 `tools/run_golden_sample.py:143` `prepare_golden_sample` → `:191` `run_model_inference` → `bevformer/detectors/bevformer.py:143` `forward` → `:236` `forward_test` → `:283` `simple_test`。`forward_test` 按 scene token 清空 `prev_bev`，把绝对 `can_bus[:3]` 和 `[-1]` 改为相邻帧 delta，并保存新 `bev_embed` 与绝对 pose；这部分是推理语义，不属于训练框架可随意删除的包装。

已由源码/配置确认的核心 shape：输入六相机为 `[B,6,3,H,W]`（导出基线使用 `[1,6,3,480,800]`，原始样本形状以实际 pipeline 为准）；`extract_img_feat` 将相机合并到 batch，再经 ResNet50 C5 与单层 FPN，恢复为 `[B,6,256,Hf,Wf]`，见 `bevformer/detectors/bevformer.py:67`。BEV 为 `50×50=2500` queries，Embedding `[2500,256]`；object queries 为 900，Embedding `[900,512]`，见 `bevformer/dense_heads/bevformer_head.py:69`、`:118`。Transformer 在 `bevformer/modules/transformer.py:104`、`:203` 连接 3 层 Encoder 与 6 层 Decoder；`bevformer/modules/decoder.py:66` 做逐层 reference refinement。Head 返回 `bev_embed [2500,B,256]`、`all_cls_scores [6,B,900,10]`、`all_bbox_preds [6,B,900,10]`；最后一维由原 Checkpoint 的 `cls_branches.0.6.weight [10,256]` 和 `reg_branches.0.4.weight [10,256]` 及 Head forward 核对，源码 `bevformer_head.py:118` 的 9 维注释并非此配置的真实 code size。解码路径为 `bevformer_head.py:483` → `core/bbox/coders/nms_free_coder.py:40` → `mmdet3d.bbox3d2result`；它与原始网络输出可分离验证。

`tools/deployment/bevformer_stateful_onnx_wrapper.py:329` 已提供显式 `images, prev_bev, can_bus, shift, lidar2img, use_prev_bev` 边界，可作为独立模型的**接口参照**，但其 forward 仍调用旧 `model.extract_img_feat`/Head，不能视为已解耦模型。`tools/run_golden_sequence.py:81` 及现有 `golden_tensors/` 提供三帧状态验证资产。

## 4. 三种方案比较

| 方案 | 改造范围 / 难度 | 权重复用 | Golden 对齐风险 | ModelOpt 风险 | 维护成本 |
|---|---|---|---|---|---|
| A：旧 OpenMMLab 适配 PyTorch 2.8 | **高**：旧 MMCV `_ext` 重编译；MMCV 1.4、mmdet、mmdet3d、AMP/Builder/API 连锁适配 | 原类结构最易保留，理论上直接加载 | 中：框架行为升级可能改变 forward | 高：旧库与新 Torch ABI/API 未验证 | 高，持续背负旧框架 |
| B：最小兼容层 | **中高**：保留 BEVFormer 核心文件，替换 Registry/Builder、BaseModule、TransformerLayerSequence/FFN、Detector/Head 构建薄层 | 目标为全 643 键不映射 | 中高：隐式父类行为与命名必须精确复刻 | 中高：CUDA 算子仍独立解决 | 中，shim 边界需测试 |
| C：纯 PyTorch 推理模型 | **高**：ResNet50、FPN、Head、Encoder/Decoder、位置编码、decode、Stateful 全链显式实现 | 可保持同名，但完全重写时键差异风险最高 | 高，数值细节最多 | 解耦后最低；初期开发风险高 | 长期最低 |

**推荐 B→C。** 先以最少替代件保留既有数学逻辑与 module 命名，严格对齐后再逐段替换为纯 PyTorch。A 虽保留权重结构，却把最难控制的兼容性问题扩大到整个旧框架；直接 C 则一次改变太多，误差难定位。不可把 torchvision ResNet50/FPN、`nn.MultiheadAttention` 视为“直接可替换”：它们是可用实现材料，但 stride、BN/Conv 封装、权重名、residual/attention mask 需要逐层核对。`models/utils/grid_mask.py:70` 和 `core/bbox/util.py` 中的张量计算更接近可直接复用；MSDA 的 Python forward 逻辑可参考，但 CUDA Extension 必须针对新环境解决。

## 5. 最小迁移架构与验证顺序

建议新代码放在独立 `tools/inference_only/`（待方案批准后再决定），不改旧 `projects/` 和 Golden 入口。顶层显式定义 `img_backbone`、`img_neck`、`pts_bbox_head`；Head 内保持 `positional_encoding`、`transformer.encoder.layers`、`transformer.decoder.layers`、`cls_branches`、`reg_branches`、BEV/Query Embedding 原路径。数据适配器输出六相机 tensor、`lidar2img`、CAN/pose 与 scene token；状态适配器仅负责 reset/delta/`prev_bev`；网络核心返回原始 cls/bbox/bev，后处理独立运行。这样既保持 strict-load 目标，又为 ModelOpt 和显式 ONNX 接口提供清晰边界。

1. 冻结原始权重与配置的 hash，使用完整清单核对 643 项名称/shape/dtype；不改原文件。
2. 构造最小骨架，先在 CPU 执行 `load_state_dict(strict=True)`；若失败，按逐键差异修正结构，只有有证据时才引入完整 key mapping。
3. 在可用 GPU/算子环境中逐段比较 ResNet50 C5、FPN、BEV query/position、Encoder 每层、Decoder 每层、cls/reg 分支；使用 `golden_tensors/` 和 `tools/tensor_dump.py` 的 tensor 清单。
4. 比较单帧原始网络输出，再对齐 `NMSFreeCoder` 的 box/score/label；不能只看最终检测框。
5. 按 `tools/run_golden_sequence.py:81` 比较三帧：Frame0 无历史、Frame1/2 真 `prev_bev`、scene reset；分开记录单步与反馈误差。
6. 全部 FP32 对齐后才尝试 ModelOpt 0.41.0；随后才讨论 Q/DQ ONNX 和 TensorRT。

验收须预先固定每层 MaxAE/MAE/MSE、shape/dtype、检测匹配及 Stateful 误差阈值；本轮不虚构阈值。当前 `nvidia-smi` 无法连接驱动，GPU 前向、MMCV 新环境 Extension、性能均**待验证**。最先实施的单一任务建议是：**只建立新模型骨架并验证 643 项 `strict=True` 无映射加载**，随后单独审查 diff，再进入任何 forward 迁移。
