# BEVFormer Python Usage Index

## 1. 用途与扫描边界

本文不重新讲 Python，只把 `docs/python_for_ai_source_reading.md` 中的知识点映射到当前仓库真实代码。

扫描范围：

```text
projects/mmdet3d_plugin/bevformer/**/*.py
tools/deployment/**/*.py
tools/export_bevformer_stateful_onnx.py
```

为补充 `__getitem__`、pipeline callable 和 property，额外查看：

```text
projects/mmdet3d_plugin/datasets/**/*.py
projects/mmdet3d_plugin/dd3d/structures/**/*.py
projects/mmdet3d_plugin/models/utils/**/*.py
```

层级边界：

```text
[Python Language]       class, self, **kwargs, decorator, __call__
[Python Standard Library] deepcopy, Path, contextmanager
[PyTorch]               Tensor, nn.Module.__call__, forward, no_grad
[MMCV/MMDetection]      Registry, BaseModule, auto_fp16, DETRHead
[BEVFormer]             detector/head/transformer/prev_bev semantics
```

## 2. Object、reference 与 mutation

### 位置 1：Temporal metadata 原地修改前保存副本

| Field | Value |
| --- | --- |
| Python Knowledge | reference / mutation / deepcopy |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])`<br>`img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']` |
| Purpose | 保存 absolute pose，同时把当前 metadata 改为相对 translation delta |

### 位置 2：训练 history metadata 独立副本

| Field | Value |
| --- | --- |
| Python Knowledge | deepcopy / nested mutable object |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_train` |
| Representative Code | `prev_img_metas = copy.deepcopy(img_metas)` |
| Purpose | History queue 处理不能污染当前帧 metadata |

### 位置 3：复制 attention 配置

| Field | Value |
| --- | --- |
| Python Knowledge | list comprehension / deepcopy |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py` |
| Class | `MyCustomBaseTransformerLayer` |
| Function | `__init__` |
| Representative Code | `attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]` |
| Purpose | 为多个 attention 创建互不共享的嵌套配置 |

## 3. List、tuple、dict 与 nested access

### 位置 1：跨帧状态 dict

| Field | Value |
| --- | --- |
| Python Knowledge | dict / mutable state / self attribute |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `__init__` |
| Representative Code | `self.prev_frame_info = {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}` |
| Purpose | 同一个 detector instance 保存 temporal runtime state |

### 位置 2：test-time metadata 多层访问

| Field | Value |
| --- | --- |
| Python Knowledge | nested list / dict / indexing |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `img_metas[0][0]['scene_token']`<br>`img_metas[0][0]['can_bus'][-1]` |
| Purpose | 从 augmentation→batch→sample metadata 读取 scene 与 ego rotation |

### 位置 3：Head 输出 dict

| Field | Value |
| --- | --- |
| Python Knowledge | dict construction / key access |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | `forward` |
| Representative Code | `outs = {'bev_embed': bev_embed, 'all_cls_scores': outputs_classes, 'all_bbox_preds': outputs_coords, ...}` |
| Purpose | 用稳定 key 返回 encoder state 与 detection predictions |

### 位置 4：多值 tuple 返回和解包

| Field | Value |
| --- | --- |
| Python Knowledge | tuple return / unpacking |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `simple_test` |
| Representative Code | `new_prev_bev, bbox_pts = self.simple_test_pts(...)`<br>`return new_prev_bev, bbox_list` |
| Purpose | 同时传回下一帧 state 与检测结果 |

## 4. Indexing 与 slicing

### 位置 1：训练序列切分 history/current

| Field | Value |
| --- | --- |
| Python Knowledge | slicing / ellipsis |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_train` |
| Representative Code | `prev_img = img[:, :-1, ...]`<br>`img = img[:, -1, ...]` |
| Purpose | 前面 frame 作为 history，最后 frame 作为 current input |

### 位置 2：BBox channel slicing

| Field | Value |
| --- | --- |
| Python Knowledge | tensor slicing / half-open interval / ellipsis |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | `forward` |
| Representative Code | `tmp[..., 0:2] += reference[..., 0:2]`<br>`tmp[..., 4:5] += reference[..., 2:3]` |
| Purpose | 选择并 refinement bbox 的特定坐标 channel |

### 位置 3：Fixed chunk slicing

| Field | Value |
| --- | --- |
| Python Knowledge | slicing / range / shape attribute |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | module-level helper |
| Function | `fixed_chunk_linear` |
| Representative Code | `chunk = x[:, start:start + chunk_size]`<br>`outputs.append(linear(chunk)[:, :valid_queries])` |
| Purpose | 固定 640 query GEMM contract，并 trim 最后一块 padding |

## 5. Function arguments 与 `*args/**kwargs`

### 位置 1：Detector train/test dispatch

| Field | Value |
| --- | --- |
| Python Knowledge | default argument / `**kwargs` packing and unpacking |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward` |
| Representative Code | `def forward(self, return_loss=True, **kwargs):`<br>`return self.forward_test(**kwargs)` |
| Purpose | 用 return_loss 分流，透明转发 img/img_metas 等参数 |

### 位置 2：Head 继承参数转发

| Field | Value |
| --- | --- |
| Python Knowledge | `*args` / `**kwargs` / super |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | `__init__` |
| Representative Code | `def __init__(self, *args, ..., **kwargs):`<br>`super(...).__init__(*args, transformer=transformer, **kwargs)` |
| Purpose | 将 DETR 通用配置交给父类，只在子类消费 BEVFormer 特有参数 |

### 位置 3：Transformer metadata 转发

| Field | Value |
| --- | --- |
| Python Knowledge | keyword arguments / `**kwargs` |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `get_bev_features` |
| Representative Code | `def get_bev_features(..., **kwargs):`<br>`[each['can_bus'] for each in kwargs['img_metas']]` |
| Purpose | 从通用 transformer API 中取得 per-sample runtime metadata |

### 位置 4：Encoder 参数继续下传

| Field | Value |
| --- | --- |
| Python Knowledge | `*args` / keyword-only-like named parameters / `**kwargs` |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | `BEVFormerEncoder` |
| Function | `forward` |
| Representative Code | `def forward(self, bev_query, key, value, *args, ..., **kwargs):`<br>`layer(..., **kwargs)` |
| Purpose | 通用 sequence API 转发 img_metas 和 attention-specific arguments |

### 位置 5：Deployment wrapper 保持官方签名

| Field | Value |
| --- | --- |
| Python Knowledge | wrapper / `*args` / `**kwargs` forwarding |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | closure inside override |
| Function | `wrapped` |
| Representative Code | `def wrapped(this, *args, **kwargs): return static_spatial_cross_attention_forward(this, *args, **kwargs)` |
| Purpose | 临时替换 forward，同时不丢失官方调用参数 |

## 6. Class、self 与 attribute

### 位置 1：Detector object

| Field | Value |
| --- | --- |
| Python Knowledge | class / instance / self / `__init__` |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `__init__` |
| Representative Code | `self.video_test_mode = video_test_mode`<br>`self.prev_frame_info = {...}` |
| Purpose | 保存 detector 配置和 temporal state |

### 位置 2：Head object

| Field | Value |
| --- | --- |
| Python Knowledge | instance attributes / method |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | `__init__`, `_init_layers` |
| Representative Code | `self.bev_h = bev_h`<br>`self.bev_embedding = nn.Embedding(...)` |
| Purpose | 保存 BEV shape 并注册 learnable submodules |

### 位置 3：Stateful ONNX wrapper object

| Field | Value |
| --- | --- |
| Python Knowledge | class / instance attribute / method |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | `BEVFormerStatefulONNXWrapper` |
| Function | `__init__`, `forward` |
| Representative Code | `self.model = model`<br>`self.img_meta = img_meta` |
| Purpose | 将原模型与静态 metadata 封装成显式 stateful tensor interface |

## 7. Inheritance 与 `super()`

### 位置 1：Detector inheritance

| Field | Value |
| --- | --- |
| Python Knowledge | inheritance / super |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer(MVXTwoStageDetector)` |
| Function | `__init__` |
| Representative Code | `super(BEVFormer, self).__init__(..., img_backbone, ..., pts_bbox_head, ...)` |
| Purpose | 父类构建并保存 backbone、neck、head 等公共 detector 子模块 |

### 位置 2：Head inheritance

| Field | Value |
| --- | --- |
| Python Knowledge | inheritance / override / super |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead(DETRHead)` |
| Function | `__init__`, `forward` |
| Representative Code | `class BEVFormerHead(DETRHead):` |
| Purpose | 继承 DETR loss、assigner、sampler 和公共 head state，override forward |

### 位置 3：Transformer inheritance

| Field | Value |
| --- | --- |
| Python Knowledge | inheritance / BaseModule initialization |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer(BaseModule)` |
| Function | `__init__` |
| Representative Code | `super(PerceptionTransformer, self).__init__(**kwargs)` |
| Purpose | 使用 MMCV BaseModule 的 parameter/init_cfg/module 行为 |

### 位置 4：Encoder sequence inheritance

| Field | Value |
| --- | --- |
| Python Knowledge | inheritance / inherited attributes |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | `BEVFormerEncoder(TransformerLayerSequence)` |
| Function | `__init__`, `forward` |
| Representative Code | `super(BEVFormerEncoder, self).__init__(*args, **kwargs)`<br>`for layer in self.layers:` |
| Purpose | 父类根据 config 构建并保存 encoder layers |

### 位置 5：Deployment PyTorch Module

| Field | Value |
| --- | --- |
| Python Knowledge | inheritance / inherited callable behavior |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | `BEVFormerStatefulONNXWrapper(nn.Module)` |
| Function | `__init__` |
| Representative Code | `super().__init__()` |
| Purpose | 注册 wrapper 为 PyTorch module，并继承 `nn.Module.__call__` |

## 8. Callable、`__call__` 与 method binding

### 位置 1：Head module call

| Field | Value |
| --- | --- |
| Python Knowledge | callable object / inherited `__call__` |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | caller `BEVFormer`; actual object `BEVFormerHead` |
| Function | `simple_test_pts` → `BEVFormerHead.forward` |
| Representative Code | `outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)` |
| Purpose | 经 PyTorch `nn.Module.__call__` 执行 head forward |

### 位置 2：Transformer module call

| Field | Value |
| --- | --- |
| Python Knowledge | callable object / dynamic dispatch |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | caller `BEVFormerHead`; actual object `PerceptionTransformer` |
| Function | `forward` → `PerceptionTransformer.forward` |
| Representative Code | `outputs = self.transformer(mlvl_feats, bev_queries, object_query_embeds, ...)` |
| Purpose | 运行 encoder 和 decoder |

### 位置 3：Encoder layer call

| Field | Value |
| --- | --- |
| Python Knowledge | callable PyTorch child module |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | caller `BEVFormerEncoder`; actual object `BEVFormerLayer` |
| Function | `forward` |
| Representative Code | `output = layer(bev_query, key, value, ...)` |
| Purpose | 每个 layer 依次执行 temporal attention、spatial attention、FFN/norm |

### 位置 4：Dataset `__getitem__`

| Field | Value |
| --- | --- |
| Python Knowledge | special method / indexing protocol |
| File Path | `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py` |
| Class | `CustomNuScenesDataset` |
| Function | `__getitem__` |
| Representative Code | `def __getitem__(self, idx): ...` |
| Purpose | `dataset[index]` 触发 sample preparation，而不是普通 list lookup |

## 9. Control flow 与工程循环工具

### 位置 1：Scene reset

| Field | Value |
| --- | --- |
| Python Knowledge | if / comparison / dict access |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:` |
| Purpose | scene 变化时清空 previous BEV |

### 位置 2：FPN level enumerate

| Field | Value |
| --- | --- |
| Python Knowledge | for / enumerate / tuple unpacking |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `get_bev_features` |
| Representative Code | `for lvl, feat in enumerate(mlvl_feats):` |
| Purpose | 同时获得 FPN level index 和 feature Tensor |

### 位置 3：Detection result zip

| Field | Value |
| --- | --- |
| Python Knowledge | zip / tuple unpacking / mutation |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `simple_test` |
| Representative Code | `for result_dict, pts_bbox in zip(bbox_list, bbox_pts):` |
| Purpose | 将同一 batch index 的 detection 写入对应 result dict |

### 位置 4：Camera accumulation range

| Field | Value |
| --- | --- |
| Python Knowledge | range / loop / assignment |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | module helper |
| Function | `static_spatial_cross_attention_forward` |
| Representative Code | `for camera_index in range(attention.num_cams): slots = slots + camera_features[:, camera_index]` |
| Purpose | 固定 camera accumulation order，避免不同浮点加法树 |

## 10. Comprehension、append 与 setdefault

### 位置 1：History timestamp metadata

| Field | Value |
| --- | --- |
| Python Knowledge | list comprehension / nested indexing |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `obtain_history_bev` |
| Representative Code | `img_metas = [each[i] for each in img_metas_list]` |
| Purpose | 为 batch 中每个 sample 抽取第 i 个历史 timestamp metadata |

### 位置 2：Decoded bbox comprehension

| Field | Value |
| --- | --- |
| Python Knowledge | list comprehension / tuple unpacking |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `simple_test_pts` |
| Representative Code | `[bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]` |
| Purpose | 将每个 batch item 的 bbox tuple 转成标准 result dict |

### 位置 3：ONNX consumer adjacency map

| Field | Value |
| --- | --- |
| Python Knowledge | dict.setdefault / mutable list.append |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | `descendants` |
| Representative Code | `consumers.setdefault(name, []).append(index)` |
| Purpose | 一个 tensor name 对应多个 consumer node index |

### 位置 4：Golden failure dict comprehension

| Field | Value |
| --- | --- |
| Python Knowledge | dict comprehension / conditional filter |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | `assert_close_report` |
| Representative Code | `{name: values for name, values in report.items() if values['MaxAE'] > atol}` |
| Purpose | 只保留超过数值阈值的 outputs |

## 11. Decorators 与 Registry

### 位置 1：Detector registration

| Field | Value |
| --- | --- |
| Python Knowledge | class decorator |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | class definition |
| Representative Code | `@DETECTORS.register_module()` |
| Purpose | `[MMCV/MMDetection]` 把字符串 `BEVFormer` 映射到 class object |

### 位置 2：Head registration

| Field | Value |
| --- | --- |
| Python Knowledge | class decorator |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | class definition |
| Representative Code | `@HEADS.register_module()` |
| Purpose | 允许 config 通过 `type='BEVFormerHead'` 构建 head |

### 位置 3：Attention registration

| Field | Value |
| --- | --- |
| Python Knowledge | class decorator |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` |
| Class | `SpatialCrossAttention`, `MSDeformableAttention3D` |
| Function | class definitions |
| Representative Code | `@ATTENTION.register_module()` |
| Purpose | MMCV transformer builder 可从配置创建 attention object |

### 位置 4：Precision method decorator

| Field | Value |
| --- | --- |
| Python Knowledge | decorator with arguments / wrapper |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | `BEVFormerEncoder` |
| Function | `point_sampling` |
| Representative Code | `@force_fp32(apply_to=('reference_points', 'img_metas'))` |
| Purpose | `[MMCV]` 在 method 入口保证关键投影计算使用 FP32 |

### 位置 5：Contextmanager decorator

| Field | Value |
| --- | --- |
| Python Knowledge | generator decorator / closure / try-finally |
| File Path | `tools/deployment/bevformer_onnx_wrapper.py` |
| Class | none |
| Function | `deployment_export_overrides` |
| Representative Code | `@contextmanager` |
| Purpose | `[Python Standard Library]` 把 yield-based helper 变为 `with` context |

## 12. Registry 与 build_from_cfg

### 位置 1：Attention builder

| Field | Value |
| --- | --- |
| Python Knowledge | dict / class object / decorator registry |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` |
| Class | `SpatialCrossAttention` |
| Function | `__init__` |
| Representative Code | `self.deformable_attention = build_attention(deformable_attention)` |
| Purpose | `[MMCV]` 从 config dict 的 type 字符串创建 attention instance |

### 位置 2：Transformer sequence builder

| Field | Value |
| --- | --- |
| Python Knowledge | dict / Registry / dynamic class construction |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `__init__` |
| Representative Code | `self.encoder = build_transformer_layer_sequence(encoder)` |
| Purpose | `[MMCV]` 根据嵌套 config 构建 encoder 和其 layers |

### 位置 3：显式 `build_from_cfg`

| Field | Value |
| --- | --- |
| Python Knowledge | config dict / keyword construction |
| File Path | `projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py` |
| Class | none |
| Function | `custom hook` setup |
| Representative Code | `hook = build_from_cfg(hook_cfg, HOOKS)` |
| Purpose | `[MMCV]` 从 hook config 和 registry 创建 hook instance |

### 位置 4：Dataset builder

| Field | Value |
| --- | --- |
| Python Knowledge | config dict / Registry / default arguments |
| File Path | `projects/mmdet3d_plugin/datasets/builder.py` |
| Class | none |
| Function | dataset build path |
| Representative Code | `dataset = build_from_cfg(cfg, DATASETS, default_args)` |
| Purpose | 从 dataset config 创建 dataset instance |

## 13. Import 与 relative import

### 位置 1：第三方 BaseModule

| Field | Value |
| --- | --- |
| Python Knowledge | absolute import / external package namespace |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | module scope |
| Function | import section |
| Representative Code | `from mmcv.runner.base_module import BaseModule` |
| Purpose | 引用安装环境 MMCV class，不是当前仓库定义 |

### 位置 2：当前 package relative import

| Field | Value |
| --- | --- |
| Python Knowledge | relative import |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | module scope |
| Function | import section |
| Representative Code | `from .temporal_self_attention import TemporalSelfAttention` |
| Purpose | 从同一 `modules` package 引入 BEVFormer 自定义 class |

### 位置 3：Deployment relative import

| Field | Value |
| --- | --- |
| Python Knowledge | package-relative import |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | module scope |
| Function | import section |
| Representative Code | `from .grid_sample_rewrite import bev_rotate_nearest_tensor` |
| Purpose | 复用同一 deployment package 内的 rotate rewrite |

## 14. Exception、assert 与 fail-closed validation

### 位置 1：Input type validation

| Field | Value |
| --- | --- |
| Python Knowledge | isinstance / raise |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `if not isinstance(var, list): raise TypeError(...)` |
| Purpose | test API 输入结构错误时立即报告 |

### 位置 2：Attention shape contract

| Field | Value |
| --- | --- |
| Python Knowledge | assert / boolean expression |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` |
| Class | `TemporalSelfAttention` |
| Function | `forward` |
| Representative Code | `assert self.num_bev_queue == 2` |
| Purpose | 固定 Temporal history/current queue 数量 |

### 位置 3：Export Golden gate

| Field | Value |
| --- | --- |
| Python Knowledge | dict comprehension / raise RuntimeError |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | `assert_close_report` |
| Representative Code | `if failed: raise RuntimeError('Wrapper validation failed: ...')` |
| Purpose | 数值不一致时默认拒绝导出 deployment model |

### 位置 4：Static SCA argument validation

| Field | Value |
| --- | --- |
| Python Knowledge | condition / raise ValueError |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `fixed_chunk_linear` |
| Representative Code | `if x.ndim != 3: raise ValueError(...)` |
| Purpose | deployment helper 对错误 rank fail closed |

## 15. Context manager 与 try/finally

### 位置 1：ONNX symbolic override

| Field | Value |
| --- | --- |
| Python Knowledge | contextmanager / try-finally / attribute restoration |
| File Path | `tools/deployment/bevformer_onnx_wrapper.py` |
| Class | none |
| Function | `deployment_export_overrides` |
| Representative Code | `try: yield`<br>`finally: ... function_class.symbolic = original_symbolic` |
| Purpose | 导出期间临时安装 symbolic/rotate rewrite，退出后恢复 Reference 行为 |

### 位置 2：Static Spatial Cross Attention override

| Field | Value |
| --- | --- |
| Python Knowledge | contextmanager / bound method patch / finally |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `static_spatial_cross_attention_override` |
| Representative Code | `child.forward = types.MethodType(wrapped, child)`<br>`finally: ... restore/delete forward` |
| Purpose | 仅 deployment/export 使用 fixed-Nq SCA，普通 inference 不变 |

### 位置 3：Strict TF32 validation context

| Field | Value |
| --- | --- |
| Python Knowledge | save/restore state / try-finally |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `disable_tf32_for_validation` |
| Representative Code | `original_matmul = ...allow_tf32`<br>`finally: ...allow_tf32 = original_matmul` |
| Purpose | 临时验证 math mode，不永久改变全局 PyTorch 状态 |

### 位置 4：组合多个 context

| Field | Value |
| --- | --- |
| Python Knowledge | multiple context managers / reverse-order cleanup |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | `BEVFormerStatefulONNXWrapper` |
| Function | `forward` |
| Representative Code | `with deployment_export_overrides(...), static_spatial_cross_attention_override(...), stateful_bev_override(...):` |
| Purpose | 在同一 forward 内组合 symbolic、SCA 和 stateful tensor override |

## 16. `isinstance`、`hasattr` 与 `getattr`

### 位置 1：Backbone output normalization

| Field | Value |
| --- | --- |
| Python Knowledge | isinstance / dict/list conversion |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `extract_img_feat` |
| Representative Code | `if isinstance(img_feats, dict): img_feats = list(img_feats.values())` |
| Purpose | 统一不同 backbone 的 output container |

### 位置 2：Attention-specific initialization

| Field | Value |
| --- | --- |
| Python Knowledge | isinstance / inheritance-aware type check |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `init_weights` |
| Representative Code | `if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention):` |
| Purpose | 只对指定 attention 子类执行专用初始化 |

### 位置 3：Symbolic attribute inspection

| Field | Value |
| --- | --- |
| Python Knowledge | hasattr / getattr / dynamic attribute name |
| File Path | `tools/deployment/bevformer_onnx_wrapper.py` |
| Class | none |
| Function | `deployment_export_overrides` |
| Representative Code | `had_symbolic = hasattr(function_class, 'symbolic')`<br>`original_symbolic = getattr(function_class, 'symbolic', None)` |
| Purpose | 区分原 class 有/无 symbolic 两种状态，退出时精确恢复 |

### 位置 4：按名字访问 Linear

| Field | Value |
| --- | --- |
| Python Knowledge | getattr / loop over attribute names |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `fixed_deformable_linear_override` |
| Representative Code | `linear = getattr(deformable_attention, linear_name, None)` |
| Purpose | 同一逻辑处理 sampling_offsets 与 attention_weights 两个 Linear |

## 17. Closure、lambda 与 method patch

### 位置 1：Closure 捕获 original forward

| Field | Value |
| --- | --- |
| Python Knowledge | nested function / closure / default capture |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `fixed_deformable_linear_override` |
| Representative Code | `def wrapped(this, x, _forward=original_forward): return fixed_chunk_linear(x, _forward, chunk_size)` |
| Purpose | 每个 Linear wrapper 固定捕获对应原 forward，避免循环 late binding |

### 位置 2：Stateful transformer method patch

| Field | Value |
| --- | --- |
| Python Knowledge | closure / bound method / types.MethodType |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class | none |
| Function | `stateful_bev_override` |
| Representative Code | `transformer.get_bev_features = types.MethodType(wrapped, transformer)` |
| Purpose | 临时把 runtime tensor inputs 捕获进 transformer method |

### 位置 3：Debug hook lambda

| Field | Value |
| --- | --- |
| Python Knowledge | lambda / callback / default capture |
| File Path | `tools/golden_sequence_recorder.py` |
| Class | recorder helper |
| Function | hook registration |
| Representative Code | `lambda m, i, o, index=index: self._linear_outputs.__setitem__(index, o)` |
| Purpose | 每个 layer hook 固定保存到自己的 index，避免 closure late binding |

## 18. Tensor object 与 layout methods

### 位置 1：BEV query batch expansion

| Field | Value |
| --- | --- |
| Python Knowledge | object method / tuple arguments |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `get_bev_features` |
| Representative Code | `bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)` |
| Purpose | `[PyTorch] [N,C] → [N,B,C]` |

### 位置 2：BEV positional layout

| Field | Value |
| --- | --- |
| Python Knowledge | chained method calls |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `get_bev_features` |
| Representative Code | `bev_pos = bev_pos.flatten(2).permute(2, 0, 1)` |
| Purpose | `[PyTorch] [B,C,H,W] → [HW,B,C]` |

### 位置 3：Temporal queue stack

| Field | Value |
| --- | --- |
| Python Knowledge | list literal / function call / reshape |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | `BEVFormerEncoder` |
| Function | `forward` |
| Representative Code | `torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)` |
| Purpose | `[PyTorch]` 新建 history/current queue，再折入 effective batch |

### 位置 4：Attention weights contiguous layout

| Field | Value |
| --- | --- |
| Python Knowledge | method chaining / attribute shape |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` |
| Class | `TemporalSelfAttention` |
| Function | `forward` |
| Representative Code | `attention_weights.permute(...).reshape(...).contiguous()` |
| Purpose | `[PyTorch]` 形成 MMCV CUDA op 需要的连续 queue-first layout |

### 位置 5：Golden NumPy conversion

| Field | Value |
| --- | --- |
| Python Knowledge | chained methods / returned object types |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | `compare_triplet` |
| Representative Code | `output.detach().cpu().numpy()` |
| Purpose | `[PyTorch]` 从 graph 分离、移到 CPU、转为 NumPy 做 Golden compare |

## 19. `staticmethod`、property 与 special methods

### 位置 1：Static reference-point generator

| Field | Value |
| --- | --- |
| Python Knowledge | staticmethod decorator |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class | `BEVFormerEncoder` |
| Function | `get_reference_points` |
| Representative Code | `@staticmethod`<br>`def get_reference_points(H, W, ...):` |
| Purpose | 生成结果只依赖显式参数，不读取 encoder instance self |

### 位置 2：Custom autograd static forward

| Field | Value |
| --- | --- |
| Python Knowledge | staticmethod / special framework calling convention |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py` |
| Class | `MultiScaleDeformableAttnFunction_fp32` |
| Function | `forward`, `backward` |
| Representative Code | `@staticmethod`<br>`def forward(ctx, value, ...):` |
| Purpose | `[PyTorch]` `.apply()` 将 Tensor 交给 MMCV CUDA extension |

### 位置 3：DD3D property

| Field | Value |
| --- | --- |
| Python Knowledge | property decorator / attribute-style method |
| File Path | `projects/mmdet3d_plugin/dd3d/structures/image_list.py` |
| Class | `ImageList` |
| Function | property getters |
| Representative Code | `@property` |
| Purpose | 调用处通过 attribute 形式读取派生结构信息 |

## 20. `pathlib.Path` 与文件 context

### 位置 1：Export paths

| Field | Value |
| --- | --- |
| Python Knowledge | Path object / overloaded `/` / method call |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | module defaults / `main` |
| Representative Code | `DEFAULT_OUTPUT = Path('artifacts/...onnx')`<br>`args.sequence_root / 'sequence.json'` |
| Purpose | `[Python Standard Library]` 组合 Golden、ONNX 和 audit 路径 |

### 位置 2：Audit JSON context

| Field | Value |
| --- | --- |
| Python Knowledge | Path method / `with open` context manager |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class | none |
| Function | `main` |
| Representative Code | `with audit_path.open('w') as stream:` |
| Purpose | 文件无论正常或异常路径都正确关闭 |

## 21. 总索引

| # | Python Knowledge | File | Class | Function | Usage |
| ---: | --- | --- | --- | --- | --- |
| 1 | reference/deepcopy | `detectors/bevformer.py` | BEVFormer | forward_test | 保存 absolute pose 后原地计算 delta |
| 2 | deepcopy | `detectors/bevformer.py` | BEVFormer | forward_train | 隔离 history metadata |
| 3 | deepcopy/comprehension | `modules/custom_base_transformer_layer.py` | MyCustomBaseTransformerLayer | `__init__` | 独立复制 attention cfg |
| 4 | dict/self | `detectors/bevformer.py` | BEVFormer | `__init__` | temporal runtime state |
| 5 | nested access | `detectors/bevformer.py` | BEVFormer | forward_test | augmentation/batch/sample metadata |
| 6 | dict output | `dense_heads/bevformer_head.py` | BEVFormerHead | forward | 命名检测输出 |
| 7 | tuple unpacking | `detectors/bevformer.py` | BEVFormer | simple_test | state 与 detection 多返回值 |
| 8 | slicing | `detectors/bevformer.py` | BEVFormer | forward_train | history/current frame split |
| 9 | ellipsis/slice | `dense_heads/bevformer_head.py` | BEVFormerHead | forward | bbox channel refinement |
| 10 | slice/range | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | fixed_chunk_linear | 640-query chunks |
| 11 | `**kwargs` | `detectors/bevformer.py` | BEVFormer | forward | train/test input forwarding |
| 12 | `*args/**kwargs` | `dense_heads/bevformer_head.py` | BEVFormerHead | `__init__` | parent config forwarding |
| 13 | `**kwargs` | `modules/transformer.py` | PerceptionTransformer | get_bev_features | img_metas access |
| 14 | `*args/**kwargs` | `modules/encoder.py` | BEVFormerEncoder | forward | layer parameter forwarding |
| 15 | wrapper args | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | wrapped | signature preservation |
| 16 | class/self | `detectors/bevformer.py` | BEVFormer | `__init__` | detector state |
| 17 | attributes | `dense_heads/bevformer_head.py` | BEVFormerHead | `_init_layers` | register embeddings/branches |
| 18 | class/self | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | BEVFormerStatefulONNXWrapper | `__init__` | model/meta wrapper |
| 19 | inheritance/super | `detectors/bevformer.py` | BEVFormer | `__init__` | inherit detector components |
| 20 | inheritance/super | `dense_heads/bevformer_head.py` | BEVFormerHead | `__init__` | inherit DETR head state |
| 21 | inheritance | `modules/transformer.py` | PerceptionTransformer | `__init__` | MMCV BaseModule |
| 22 | inheritance | `modules/encoder.py` | BEVFormerEncoder | `__init__` | TransformerLayerSequence |
| 23 | nn.Module inheritance | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | BEVFormerStatefulONNXWrapper | `__init__` | inherited callable dispatch |
| 24 | callable | `detectors/bevformer.py` | BEVFormer → BEVFormerHead | simple_test_pts | head module call |
| 25 | callable | `dense_heads/bevformer_head.py` | BEVFormerHead → PerceptionTransformer | forward | transformer module call |
| 26 | callable | `modules/encoder.py` | BEVFormerEncoder → BEVFormerLayer | forward | layer module call |
| 27 | `__getitem__` | `datasets/nuscenes_dataset.py` | CustomNuScenesDataset | `__getitem__` | dataset indexing |
| 28 | if | `detectors/bevformer.py` | BEVFormer | forward_test | scene reset |
| 29 | enumerate | `modules/transformer.py` | PerceptionTransformer | get_bev_features | FPN level/index |
| 30 | zip | `detectors/bevformer.py` | BEVFormer | simple_test | batch result pairing |
| 31 | range | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | static SCA | camera accumulation |
| 32 | list comprehension | `detectors/bevformer.py` | BEVFormer | obtain_history_bev | timestamp metadata |
| 33 | list comprehension | `detectors/bevformer.py` | BEVFormer | simple_test_pts | bbox conversion |
| 34 | setdefault/append | `tools/export_bevformer_stateful_onnx.py` | — | descendants | ONNX consumer map |
| 35 | dict comprehension | `tools/export_bevformer_stateful_onnx.py` | — | assert_close_report | failure filtering |
| 36 | decorator/Registry | `detectors/bevformer.py` | BEVFormer | class definition | detector registration |
| 37 | decorator/Registry | `dense_heads/bevformer_head.py` | BEVFormerHead | class definition | head registration |
| 38 | decorator/Registry | `modules/spatial_cross_attention.py` | SpatialCrossAttention | class definition | attention registration |
| 39 | precision decorator | `modules/encoder.py` | BEVFormerEncoder | point_sampling | force FP32 input |
| 40 | contextmanager | `tools/deployment/bevformer_onnx_wrapper.py` | — | deployment_export_overrides | temporary symbolic patch |
| 41 | build_attention | `modules/spatial_cross_attention.py` | SpatialCrossAttention | `__init__` | config→attention object |
| 42 | sequence builder | `modules/transformer.py` | PerceptionTransformer | `__init__` | config→encoder/decoder |
| 43 | build_from_cfg | `bevformer/apis/mmdet_train.py` | — | hook setup | config→hook object |
| 44 | relative import | `modules/transformer.py` | — | module scope | local attention class |
| 45 | raise/isinstance | `detectors/bevformer.py` | BEVFormer | forward_test | API validation |
| 46 | assert | `modules/temporal_self_attention.py` | TemporalSelfAttention | forward | queue contract |
| 47 | raise | `tools/export_bevformer_stateful_onnx.py` | — | assert_close_report | Golden export gate |
| 48 | try/finally | `tools/deployment/bevformer_onnx_wrapper.py` | — | deployment_export_overrides | restore official behavior |
| 49 | context manager | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | static SCA override | deployment-only patch |
| 50 | hasattr/getattr | `tools/deployment/bevformer_onnx_wrapper.py` | — | deployment_export_overrides | inspect symbolic state |
| 51 | getattr | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | fixed linear override | access two named Linear modules |
| 52 | closure | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | fixed linear override | capture original forward |
| 53 | MethodType | `tools/deployment/bevformer_stateful_onnx_wrapper.py` | — | stateful_bev_override | bind temporary method |
| 54 | Tensor methods | `modules/transformer.py` | PerceptionTransformer | get_bev_features | BEV layouts |
| 55 | stack/reshape | `modules/encoder.py` | BEVFormerEncoder | forward | temporal queue |
| 56 | contiguous | `modules/temporal_self_attention.py` | TemporalSelfAttention | forward | CUDA input layout |
| 57 | detach/cpu/numpy | `tools/export_bevformer_stateful_onnx.py` | — | compare_triplet | Golden conversion |
| 58 | staticmethod | `modules/encoder.py` | BEVFormerEncoder | get_reference_points | instance-independent helper |
| 59 | property | `dd3d/structures/image_list.py` | ImageList | property getter | attribute-style derived state |
| 60 | Path/with | `tools/export_bevformer_stateful_onnx.py` | — | main | artifact paths/file cleanup |

## 22. 推荐使用方式

1. 在主文档学完一个 Python 章节。
2. 在本索引总表搜索对应 knowledge。
3. 打开 2–4 个真实位置，只回答 Python 问题：对象类型、参数绑定、引用、返回结构。
4. 暂时不要求解释 attention 或 geometry 数学。
5. 最后沿 `model → forward_test → head → transformer → encoder` 综合练习。
