# 读懂当前 BEVFormer 工程所需的 Python / PyTorch 基础知识清单

## 1. 这份指南解决什么问题

这不是一份完整 Python 教程。目标是让具备模型与部署高层知识、但 Python 基础较弱的读者，能够独立追踪当前仓库中的：

```text
BEVFormer inference
→ temporal state
→ BEV encoder
→ attention
→ detection head
→ ONNX deployment wrapper
```

阅读时先回答三个问题：

1. 当前变量是什么对象、由哪个 class 创建？
2. `object(...)` 最终调用哪个 `forward()`？
3. Tensor 当前 shape/layout 是什么，下一行如何改变它？

只要这三个问题能够持续回答，绝大多数 BEVFormer inference 源码就能读下去。

## 2. 扫描范围与统计

重点扫描范围：

```text
projects/mmdet3d_plugin/bevformer/**/*.py
tools/deployment/**/*.py
tools/export_bevformer_stateful_onnx.py
```

核心 BEVFormer 与 deployment 范围的 AST 统计快照：

| 机制 | 数量 | 阅读意义 |
| --- | ---: | --- |
| Python 文件 | 32 | 核心模型与 deployment 范围 |
| 带 `**kwargs` 的函数定义 | 39 | 参数经多层 Transformer 转发 |
| 调用时展开 `**kwargs` | 41 | 同一批 metadata/attention 参数继续向下传 |
| Decorator | 56 | Registry、FP16/FP32、autograd、context manager |
| 显式继承 class | 28 | 行为大量来自 PyTorch/MMCV/MMDetection 父类 |
| `super()` 调用 | 26 | 父类负责注册子模块和构建公共层 |
| List comprehension | 46 | 批量构造 metadata、layer 和结果 |
| Dict comprehension | 7 | audit/report 与配置处理 |
| `assert` | 41 | shape、配置和 attention contract 检查 |
| `raise` | 47 | fail-closed validation |
| `try` | 10 | 版本兼容、临时 override 恢复 |

上述统计不是整个仓库所有数据转换脚本的语法统计，而是面向“阅读 BEVFormer inference 和 ONNX deployment”的有效范围。

## 3. 推荐阅读入口

不要从所有文件随机开始。建议按下面顺序打开：

1. `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
2. `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
3. `projects/mmdet3d_plugin/bevformer/modules/transformer.py`
4. `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
5. `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`
6. `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`
7. `projects/mmdet3d_plugin/bevformer/modules/decoder.py`
8. `tools/deployment/bevformer_stateful_onnx_wrapper.py`
9. `tools/export_bevformer_stateful_onnx.py`

## 4. Level A：必须马上掌握

Level A 是阅读调用链和 Tensor layout 的最低门槛。不掌握这些内容，通常会在 detector/head/transformer 的前三层调用中断掉。

### A1. `class`、object、`self`、`__init__`

| 字段 | 内容 |
| --- | --- |
| Concept | `class` 定义对象模板；object 是 class 的实例；`self` 是当前实例；`__init__` 初始化实例状态 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `BEVFormer.__init__` |
| 原始片段 | `class BEVFormer(MVXTwoStageDetector):`<br>`def __init__(..., video_test_mode=False):`<br>`self.video_test_mode = video_test_mode` |
| Python 语法 | `self.xxx` 把值保存在当前 detector object 中，之后其他 method 通过同一个 `self` 读取 |
| BEVFormer 作用 | 保存 backbone、neck、head，以及跨帧 `prev_frame_info` 等运行时状态 |
| 不懂会在哪里断掉 | 看不懂 `self.pts_bbox_head`、`self.prev_frame_info` 为什么能跨 method/跨 frame 使用 |

需要建立的心智模型：

```text
BEVFormer class
        ↓ 实例化
model object
        ├─ model.img_backbone
        ├─ model.img_neck
        ├─ model.pts_bbox_head
        └─ model.prev_frame_info
```

### A2. Inheritance 与 `super()`

| 字段 | 内容 |
| --- | --- |
| Concept | 子类继承父类的方法和属性；`super()` 调用父类实现 |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class / Function | `BEVFormerHead(DETRHead).__init__` |
| 原始片段 | `class BEVFormerHead(DETRHead):`<br>`super(BEVFormerHead, self).__init__(*args, transformer=transformer, **kwargs)` |
| Python 语法 | `BEVFormerHead` 是 `DETRHead` 的子类；父类初始化完成分类 loss、assigner、sampler、transformer 等公共成员 |
| BEVFormer 作用 | BEVFormer 只实现差异部分，DETR 通用能力来自 MMDetection 父类 |
| 不懂会在哪里断掉 | 在本文件里找不到 `self.num_query`、`self.loss_cls`、`self.assigner` 时会误以为代码缺失 |

阅读规则：如果 `self.xxx` 在当前 `__init__` 没赋值，立刻检查父类、builder 或 registry 创建过程。

### A3. `nn.Module`、`forward()` 与 `module(...)`

| 字段 | 内容 |
| --- | --- |
| Concept | PyTorch 模型组件继承 `nn.Module`；调用 module object 时由 `nn.Module.__call__()` 间接执行 `forward()` |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class / Function | `BEVFormerStatefulONNXWrapper(nn.Module)` / `forward` |
| 原始片段 | `class BEVFormerStatefulONNXWrapper(nn.Module):`<br>`outputs = wrapper1(...)` |
| Python/PyTorch 语法 | `wrapper1(...)` 不是直接查找自定义 `__call__`；它调用继承的 `nn.Module.__call__`，后者再调用 `wrapper1.forward(...)` |
| BEVFormer 作用 | PyTorch hooks、autocast、module wrapper 和 tracing 都依赖 `__call__` 入口 |
| 不懂会在哪里断掉 | 会一直寻找 `BEVFormerStatefulONNXWrapper.__call__`，或者错误地用 `forward()` 绕过 hooks |

核心等价关系仅用于理解：

```text
module(inputs)
≈ nn.Module.__call__(module, inputs)
→ module.forward(inputs)
```

不要在真实代码中为了“更直接”把 `module(...)` 改成 `module.forward(...)`。

### A4. Positional、keyword、default arguments

| 字段 | 内容 |
| --- | --- |
| Concept | 位置参数按顺序匹配；关键字参数按名字匹配；默认参数可省略 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `simple_test_pts` |
| 原始片段 | `def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):`<br>`self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)` |
| Python 语法 | `x`、`img_metas` 是 positional；`prev_bev=...` 是 keyword；没提供时默认 `None` |
| BEVFormer 作用 | 明确历史 BEV 是可选状态，`rescale` 是测试选项 |
| 不懂会在哪里断掉 | 无法判断一次调用中的第三个 Tensor 到底匹配哪个参数，也容易混淆 `prev_bev` 与 `rescale` |

### A5. `*args` 与 `**kwargs`

| 字段 | 内容 |
| --- | --- |
| Concept | `*args` 收集/展开额外位置参数；`**kwargs` 收集/展开额外关键字参数 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class / Function | `BEVFormerEncoder.forward` |
| 原始片段 | `def forward(self, bev_query, key, value, *args, ..., **kwargs):`<br>`reference_points_cam, bev_mask = self.point_sampling(..., kwargs['img_metas'])` |
| Python 语法 | 未显式列出的 keyword 被装入 dict `kwargs`；调用下层时 `**kwargs` 再把 dict 展开为关键字参数 |
| BEVFormer 作用 | `img_metas`、attention mask、reference point 等参数穿过多层通用 Transformer API |
| 不懂会在哪里断掉 | 会认为 `img_metas` 没有传入，或者不知道它为何突然出现在 encoder 内 |

追踪 `**kwargs` 时不要只看当前函数签名，要沿调用链搜索：

```text
forward_test(**kwargs)
→ simple_test(..., **kwargs)
→ transformer(..., **kwargs)
→ encoder(..., **kwargs)
→ layer attention(..., **kwargs)
```

### A6. List、tuple、dict

| 字段 | 内容 |
| --- | --- |
| Concept | list 可变且有顺序；tuple 通常表达固定结构；dict 用 key 访问字段 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `BEVFormer.__init__`, `simple_test` |
| 原始片段 | `self.prev_frame_info = {'prev_bev': None, 'scene_token': None, ...}`<br>`return new_prev_bev, bbox_list` |
| Python 语法 | `{...}` 是 dict；`return a, b` 实际返回二元素 tuple；`bbox_list` 是可变 list |
| BEVFormer 作用 | dict 保存 temporal state/metadata/output；tuple 返回多个 Tensor；list 保存 batch/level/frame 数据 |
| 不懂会在哪里断掉 | 看不懂 `img_metas[0][0]['can_bus']` 的嵌套结构，或无法解包 `new_prev_bev, bbox_results` |

### A7. Indexing 与 slicing

| 字段 | 内容 |
| --- | --- |
| Concept | `x[i]` 取一个位置；`x[a:b]` 取半开区间；`...` 保留未写出的维度 |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class / Function | `BEVFormerHead.forward` |
| 原始片段 | `tmp[..., 0:2] += reference[..., 0:2]`<br>`tmp[..., 4:5] += reference[..., 2:3]` |
| Python/PyTorch 语法 | `...` 代表前面所有维；`0:2` 包含 0、1，不包含 2；`4:5` 保留长度为 1 的维度 |
| BEVFormer 作用 | 对 bbox 的 xy 和 z 分量分别做 reference refinement |
| 不懂会在哪里断掉 | 无法判断修改的是 bbox 哪些 channel；可能误以为 `4:5` 等于整数索引 `4` |

### A8. `if`、`for`、`enumerate()`、`zip()`

| 字段 | 内容 |
| --- | --- |
| Concept | 条件分支、循环、同时获得索引和值、并行遍历多个序列 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `simple_test`, `forward_test` |
| 原始片段 | `for result_dict, pts_bbox in zip(bbox_list, bbox_pts):`<br>`if scene_token != self.prev_frame_info['scene_token']:` |
| Python 语法 | `zip` 每次返回同位置的一对值；`if` 根据 scene 是否变化选择 reset |
| BEVFormer 作用 | 将每个 batch item 的 3D detection 放入结果 dict；scene 切换时清空历史状态 |
| 不懂会在哪里断掉 | 看不懂 state reset 或 batch 结果如何一一对应 |

`enumerate` 的真实例子：

```python
for lvl, feat in enumerate(mlvl_feats):
```

这里 `lvl` 是 FPN level index，`feat` 是该 level Tensor。

### A9. Tensor、shape、dtype、device

| 字段 | 内容 |
| --- | --- |
| Concept | Tensor 同时具有数值、维度、元素类型和所在设备 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | `PerceptionTransformer.get_bev_features` |
| 原始片段 | `bs = mlvl_feats[0].size(0)`<br>`device=bev_query.device, dtype=bev_query.dtype` |
| PyTorch 语法 | `.shape/.size()` 读取维度；`.dtype` 是 FP32/FP16 等；`.device` 是 CPU/CUDA |
| BEVFormer 作用 | 新 reference points 必须与 query 位于相同 device，并使用兼容 dtype |
| 不懂会在哪里断掉 | 无法定位 shape mismatch、CPU/CUDA mismatch 或 FP16/FP32 隐式转换 |

阅读每个 Tensor 时建议在纸上固定记录：

```text
name | shape | layout | dtype | device
```

### A10. `reshape()` / `view()`

| 字段 | 内容 |
| --- | --- |
| Concept | 在元素总数不变时改变 Tensor 维度解释；`view` 通常要求兼容内存布局，`reshape` 必要时可复制 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` |
| Class / Function | `TemporalSelfAttention.forward` |
| 原始片段 | `value = value.reshape(bs*self.num_bev_queue, num_value, self.num_heads, -1)` |
| PyTorch 语法 | 把 channel 拆成 `num_heads × head_dim`，`-1` 由 PyTorch 自动推导 |
| BEVFormer 作用 | 将 Temporal queue 折入 batch，并为 deformable attention 拆 head |
| 不懂会在哪里断掉 | 会把 effective batch=2 误认为真实 batch=2，无法理解 Plugin 的 `[B,S,H,D]` contract |

### A11. `permute()` / `transpose()`

| 字段 | 内容 |
| --- | --- |
| Concept | 改变维度顺序；`transpose(a,b)` 只交换两个维，`permute(...)` 指定全部维顺序 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | `get_bev_features` |
| 原始片段 | `bev_pos = bev_pos.flatten(2).permute(2, 0, 1)`<br>`feat = feat.flatten(3).permute(1, 0, 3, 2)` |
| PyTorch 语法 | 数据值不变，但 layout 从 `[B,C,H,W]` 变为 `[HW,B,C]` 等 |
| BEVFormer 作用 | 在 batch-first、sequence-first、camera-first contract 间转换 |
| 不懂会在哪里断掉 | 最常见的阅读错误：把 `[B,N,C]` 当成 `[N,B,C]`，后续 compare 全部错位 |

### A12. `unsqueeze()` / `squeeze()`

| 字段 | 内容 |
| --- | --- |
| Concept | 增加或删除长度为 1 的维度 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | `get_bev_features` |
| 原始片段 | `bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)` |
| PyTorch 语法 | `[N,C] → [N,1,C]`，再复制 batch 维得到 `[N,B,C]` |
| BEVFormer 作用 | 将模型共享的 BEV embedding 扩展到当前 batch |
| 不懂会在哪里断掉 | 无法理解为何 query 从二维 embedding 变成三维 runtime Tensor |

### A13. `cat()` / `stack()`

| 字段 | 内容 |
| --- | --- |
| Concept | `cat` 沿已有维拼接；`stack` 新增一维后组合 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class / Function | `BEVFormerEncoder.forward` |
| 原始片段 | `prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(...)`<br>`feat_flatten = torch.cat(feat_flatten, 2)` |
| PyTorch 语法 | `stack` 创建 history/current queue 维；`cat` 合并多个 feature level 的 flattened values |
| BEVFormer 作用 | Temporal attention 形成两帧 queue；Spatial attention 合并多尺度特征 |
| 不懂会在哪里断掉 | 无法分清“两项 queue”与“沿空间维拼接”，会误读 `value` 的来源 |

### A14. `matmul()` / `bmm()`

| 字段 | 内容 |
| --- | --- |
| Concept | Tensor 矩阵乘；`bmm` 明确处理三维 batched matrix multiply |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py`、`tools/deployment/grid_sample_rewrite.py` |
| Class / Function | `point_sampling`、`bev_rotate_nearest_tensor` |
| 原始片段 | `reference_points_cam = torch.matmul(lidar2img, reference_points)`<br>`base_grid.reshape(...).bmm(theta.transpose(1, 2) / scale)` |
| PyTorch 语法 | 按 batch 对相机投影矩阵/affine matrix 与坐标做乘法 |
| BEVFormer 作用 | 生成 camera reference points，以及 previous-BEV rotation grid |
| 不懂会在哪里断掉 | 无法理解 `lidar2img` 投影和 Host Affine contract；也看不出 shape 中哪两维参与矩阵乘 |

### A15. Linear 与 `nn.Sequential`

| 字段 | 内容 |
| --- | --- |
| Concept | `nn.Linear` 对最后一维做仿射变换；`nn.Sequential` 按顺序调用多个 module |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | `PerceptionTransformer.init_layers` |
| 原始片段 | `self.reference_points = nn.Linear(self.embed_dims, 3)`<br>`self.can_bus_mlp = nn.Sequential(nn.Linear(...), nn.ReLU(...), ...)` |
| PyTorch 语法 | Linear 可接受前面任意 batch/query 维，只变换最后的 feature 维 |
| BEVFormer 作用 | 从 query 预测 reference point；将 18 维 CAN bus 编码到 256 维 |
| 不懂会在哪里断掉 | 会误以为 Linear 只能处理二维矩阵，无法理解 `[B,Q,C]` 输入 |

### A16. `.cuda()`、`.eval()`、`.train()`、`torch.no_grad()`

| 字段 | 内容 |
| --- | --- |
| Concept | `.cuda()` 移到 GPU；eval/train 切换 module mode；`no_grad` 禁止构建反向图 |
| File Path | `tools/export_bevformer_stateful_onnx.py`、`projects/.../detectors/bevformer.py` |
| Class / Function | `main`、`obtain_history_bev` |
| 原始片段 | `wrapper1 = ... .cuda().eval()`<br>`with torch.no_grad():` |
| PyTorch 语法 | mode 与 gradient tracking 是两套独立状态；`.eval()` 不等于 `no_grad()` |
| BEVFormer 作用 | 固定 dropout/norm inference 行为，节省 history inference 显存 |
| 不懂会在哪里断掉 | 会认为 `.eval()` 自动关闭梯度，或忽略 dropout 导致 Golden 不稳定 |

### A17. `detach().cpu().numpy()`

| 字段 | 内容 |
| --- | --- |
| Concept | 从 autograd graph 分离，移到 CPU，再共享/转换为 NumPy 表示 |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class / Function | `compare_triplet` |
| 原始片段 | `output.detach().cpu().numpy()` |
| PyTorch 语法 | CUDA Tensor 不能直接 `.numpy()`；requires-grad Tensor 应先 detach |
| BEVFormer 作用 | 将 PyTorch wrapper 输出交给 NumPy Golden compare |
| 不懂会在哪里断掉 | 会遇到 CUDA/gradient 相关异常，或误以为这是模型内计算 |

### A18. `.contiguous()`

| 字段 | 内容 |
| --- | --- |
| Concept | 把可能只是 stride view 的 Tensor 变成连续内存布局 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` |
| Class / Function | `TemporalSelfAttention.forward` |
| 原始片段 | `.reshape(...).contiguous()` |
| PyTorch 语法 | `permute` 后 logical shape 正确，但物理 stride 常不连续；`contiguous` 必要时复制 |
| BEVFormer 作用 | 给 MMCV CUDA op/ONNX/TensorRT boundary 提供稳定连续布局 |
| 不懂会在哪里断掉 | 只看 shape 不看 stride，难以解释 `view` 失败或 C++ binding layout 问题 |

## 5. Level B：读 BEVFormer 经常遇到

### B1. List comprehension

| 字段 | 内容 |
| --- | --- |
| Concept | 用一行循环生成 list |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `obtain_history_bev` |
| 原始片段 | `img_metas = [each[i] for each in img_metas_list]` |
| 语法意义 | 对每个 batch metadata 取第 `i` 帧，组成新 list |
| 实际作用 | 从 `[batch][time]` metadata 取当前 history timestamp |
| 不懂的断点 | 无法辨认 batch/time 两层嵌套，容易把 `each[i]` 理解为 Tensor channel |

### B2. Dict comprehension

| 字段 | 内容 |
| --- | --- |
| Concept | 用循环生成 dict |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class / Function | `assert_close_report` |
| 原始片段 | `{name: values for name, values in report.items() if values['MaxAE'] > atol}` |
| 语法意义 | 只保留 MaxAE 超限的项目 |
| 实际作用 | 构造 Golden validation failure report |
| 不懂的断点 | 看不懂为什么 `failed` 只含部分 output |

### B3. Set

| 字段 | 内容 |
| --- | --- |
| Concept | 无序、不重复元素集合，适合成员检查与集合运算 |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class / Function | `audit_onnx`, `descendants` |
| 原始片段 | `expected_inputs = {'images', 'prev_bev', ...}`<br>`bool(values & output_names)` |
| 语法意义 | `{...}` 在没有 key/value 时是 set；`&` 求交集 |
| 实际作用 | 检查 graph inputs 集合完全相同，以及 input 是否能到达 output |
| 不懂的断点 | 会把 set 当 dict，无法理解 `values & output_names` |

### B4. Decorator

| 字段 | 内容 |
| --- | --- |
| Concept | `@decorator` 在 class/function 定义完成后包装或登记它 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `BEVFormer`, `extract_feat` |
| 原始片段 | `@DETECTORS.register_module()`<br>`@auto_fp16(apply_to=('img'))` |
| 语法意义 | 前者把 class 注册到 registry；后者包装 method 并处理指定参数精度 |
| 实际作用 | 配置文件可以用字符串创建 detector；MMCV 控制混合精度 |
| 不懂的断点 | 明明没有显式 new class 却能构建模型；或参数 dtype 在函数入口发生变化却找不到代码 |

### B5. Import 与 relative import

| 字段 | 内容 |
| --- | --- |
| Concept | import 引入外部名字；以 `.` 开头的是当前 package 内相对导入 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | module imports |
| 原始片段 | `from mmcv.runner.base_module import BaseModule`<br>`from .temporal_self_attention import TemporalSelfAttention` |
| 语法意义 | 第一行来自安装环境；第二行相对当前 `modules` package |
| 实际作用 | 区分 MMCV 提供的框架能力和 BEVFormer 自定义实现 |
| 不懂的断点 | 会在仓库里寻找 `BaseModule` 找不到，或不知道 `.` 指哪个目录 |

### B6. `with` 与 context manager

| 字段 | 内容 |
| --- | --- |
| Concept | context manager 在进入/退出代码块时执行成对操作，即使异常也执行退出清理 |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class / Function | `main` |
| 原始片段 | `with torch.no_grad(), deployment_export_overrides():` |
| 语法意义 | 同时进入两个 context，退出时按相反顺序恢复 |
| 实际作用 | 关闭梯度并临时安装 ONNX/deployment override，导出后恢复官方模型行为 |
| 不懂的断点 | 会以为 override 是永久修改，或忽略异常路径仍会恢复 hook/symbolic |

### B7. `try` / `except` / `finally`

| 字段 | 内容 |
| --- | --- |
| Concept | 捕获异常并执行兼容分支；`finally` 无论成功失败都会运行 |
| File Path | `tools/deployment/bevformer_onnx_wrapper.py` |
| Class / Function | `deployment_export_overrides` |
| 原始片段 | `try: yield`<br>`finally: ... restore original_symbolic ...` |
| 语法意义 | context 内异常不会阻止状态恢复；是否继续抛出取决于是否捕获 |
| 实际作用 | 临时 symbolic、forward patch 在导出后完整恢复 |
| 不懂的断点 | 调试失败时会误判普通 PyTorch 已被 deployment patch 污染 |

### B8. `raise` 与 `assert`

| 字段 | 内容 |
| --- | --- |
| Concept | `raise` 主动抛异常；`assert condition` 检查开发期不变量 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` |
| Class / Function | `TemporalSelfAttention.forward` |
| 原始片段 | `assert self.num_bev_queue == 2`<br>`raise ValueError(...)` |
| 语法意义 | 条件不满足立即中断，避免错误 Tensor 继续传播 |
| 实际作用 | 固定 temporal queue 和 head/channel contract |
| 不懂的断点 | 会把异常当作 CUDA 随机问题，而忽略它正在指出明确 shape/config 不变量 |

### B9. `deepcopy`

| 字段 | 内容 |
| --- | --- |
| Concept | 递归复制嵌套 object，避免修改副本时污染原对象 |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `forward_test` |
| 原始片段 | `tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])` |
| 语法意义 | `tmp_pos` 拥有独立数据，而不是指向即将被原地减法修改的 view/reference |
| 实际作用 | 保存 absolute pose，同时把当前 metadata 改成 relative delta |
| 不懂的断点 | 会不理解为什么普通赋值可能导致 prev_pos 跟着当前 can_bus 一起变化 |

### B10. `isinstance()`

| 字段 | 内容 |
| --- | --- |
| Concept | 检查 object 是否属于某 class 或其子类 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class / Function | `init_weights` |
| 原始片段 | `if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention):` |
| 语法意义 | 对 module tree 中特定 attention 类型执行专用初始化 |
| 实际作用 | 不同 attention class 使用不同 init method |
| 不懂的断点 | 看不懂同一循环为何只处理部分子模块 |

### B11. `hasattr()` / `getattr()`

| 字段 | 内容 |
| --- | --- |
| Concept | 动态检查/读取 object attribute；`getattr(obj,name,default)` 可提供默认值 |
| File Path | `tools/deployment/bevformer_onnx_wrapper.py` |
| Class / Function | `deployment_export_overrides` |
| 原始片段 | `had_symbolic = hasattr(function_class, 'symbolic')`<br>`original_symbolic = getattr(function_class, 'symbolic', None)` |
| 语法意义 | attribute 名在 runtime 决定，不必写死成 `obj.symbolic` |
| 实际作用 | 安装 symbolic 前记录原状态，退出时精确恢复 |
| 不懂的断点 | 无法理解 deployment patch 如何兼容“原来有/没有 symbolic”两种环境 |

### B12. Lambda

| 字段 | 内容 |
| --- | --- |
| Concept | 小型匿名函数，常用于排序 key 或 hook callback |
| File Path | `tools/golden_sequence_recorder.py` |
| Class / Function | hook registration |
| 原始片段 | `lambda m, i, o, index=index: self._linear_outputs.__setitem__(index, o)` |
| 语法意义 | 创建临时 callback；`index=index` 把当前循环 index 固定在默认参数中 |
| 实际作用 | 每个 Linear hook 把输出保存到对应 layer index |
| 不懂的断点 | 容易遇到 Python closure late binding：所有 hook 看似都写入最后一个 index |

核心 BEVFormer/deployment 32 文件中 AST 未发现 lambda；它主要出现在 Golden/debug 工具中，所以优先级低于 `**kwargs` 和 decorator。

### B13. `pathlib.Path`

| 字段 | 内容 |
| --- | --- |
| Concept | 面向对象的文件路径 API；`/` 用于拼路径，不是除法 |
| File Path | `tools/export_bevformer_stateful_onnx.py` |
| Class / Function | module defaults / `main` |
| 原始片段 | `DEFAULT_OUTPUT = Path('artifacts/...onnx')`<br>`args.sequence_root / 'sequence.json'` |
| 语法意义 | Path object 支持 `.exists()`、`.read_text()`、`.parent`、`.stem` 等 |
| 实际作用 | 管理 Golden、ONNX、audit JSON 和 runtime fixture 路径 |
| 不懂的断点 | 会把 `/` 当数值运算，或混淆 Path 与字符串 API |

### B14. Property

| 字段 | 内容 |
| --- | --- |
| Concept | `@property` 让无参数 method 以 attribute 形式读取 |
| File Path | `projects/mmdet3d_plugin/dd3d/structures/image_list.py` |
| Class / Function | `ImageList.image_sizes` 等 property |
| 原始片段 | `@property`<br>`def image_sizes(self): ...` |
| 语法意义 | 调用者写 `obj.image_sizes`，而不是 `obj.image_sizes()` |
| 实际作用 | DD3D structure 暴露派生 metadata |
| 不懂的断点 | 在 class 中找到 method，却不理解为何调用处没有括号 |

核心 BEVFormer inference 路径没有自定义 property，因此属于 Level B/C 边界知识。

## 6. Level C：遇到再查即可

### C1. 自定义 `__call__`

| 字段 | 内容 |
| --- | --- |
| Concept | 普通 Python class 定义 `__call__` 后，其实例可像函数一样调用 |
| File Path | `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` |
| Class / Function | pipeline transforms 的 `__call__` |
| 原始片段 | `def __call__(self, results):` |
| 语法意义 | `transform(results)` 会执行 `transform.__call__(results)` |
| 实际作用 | OpenMMLab data pipeline 把多个 transform object 串联起来 |
| 不懂的断点 | 看不懂 config 中 pipeline object 为何可以像函数使用 |

注意：模型 `module(...)` 更重要的是 PyTorch `nn.Module.__call__`，不要把它与普通 pipeline 自定义 `__call__` 混为一谈。

### C2. `staticmethod`

| 字段 | 内容 |
| --- | --- |
| Concept | class namespace 内不接收 `self` 的函数 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` |
| Class / Function | `BEVFormerEncoder.get_reference_points` |
| 原始片段 | `@staticmethod`<br>`def get_reference_points(H, W, ...):` |
| 语法意义 | 可通过 class 或 instance 调用，但函数不依赖 instance state |
| 实际作用 | 根据显式 H/W/Z 参数生成 2D/3D reference points |
| 不懂的断点 | 会寻找不存在的 `self`，或误以为它会自动读取 encoder 配置 |

### C3. Autograd `Function`

| 字段 | 内容 |
| --- | --- |
| Concept | `torch.autograd.Function` 用静态 `forward/backward` 包装自定义算子 |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py` |
| Class / Function | `MultiScaleDeformableAttnFunction_fp32` |
| 原始片段 | `class ... (Function):`<br>`@staticmethod`<br>`def forward(ctx, value, ...):` |
| 语法意义 | 通过 `.apply(...)` 调用，不是普通 `nn.Module.forward` |
| 实际作用 | 将 PyTorch Tensor 交给 MMCV `_ext.ms_deform_attn_forward` CUDA extension |
| 不懂的断点 | 会寻找 module object，无法理解 `.apply()`、`ctx` 和 custom symbolic/plugin 边界 |

### C4. Generator-based context manager

| 字段 | 内容 |
| --- | --- |
| Concept | `@contextmanager` 将含 `yield` 的函数变为 context manager |
| File Path | `tools/deployment/bevformer_stateful_onnx_wrapper.py` |
| Class / Function | deployment override helpers |
| 原始片段 | `@contextmanager`<br>`def ...():`<br>`try: yield`<br>`finally: restore()` |
| 语法意义 | `yield` 前是 enter，`yield` 后/finally 是 exit |
| 实际作用 | 临时替换 SCA/transformer forward，退出后恢复官方行为 |
| 不懂的断点 | 无法审计 deployment-only 修改是否会泄漏到普通 inference |

## 7. PyTorch Tensor 操作速查：只记 layout 变化

| 操作 | 当前仓库例子 | 阅读时写法 |
| --- | --- | --- |
| `flatten(2)` | `bev_pos.flatten(2)` | `[B,C,H,W] → [B,C,HW]` |
| `permute(2,0,1)` | BEV pos | `[B,C,HW] → [HW,B,C]` |
| `unsqueeze(1)` | BEV query | `[N,C] → [N,1,C]` |
| `repeat(1,bs,1)` | BEV query | 为 batch 复制数据 |
| `reshape(bs*2,...)` | Temporal queue | queue 维折入 effective batch |
| `stack([prev,current],1)` | Encoder | 新建 history/current queue 维 |
| `cat(features,2)` | Transformer | 拼接各 level flattened spatial values |
| `transpose(1,2)` | Affine theta | 交换 matrix 的最后两个语义维 |
| `contiguous()` | Temporal weights | materialize permutation 后的连续布局 |

不要只写 shape，必须写 layout 含义。例如：

```text
[2500,1,256] = [Nbev,B,C]
[1,2500,256] = [B,Nbev,C]
```

两者元素总数相同，但不是同一个模块 contract。

## 8. 真实调用链中的 Python / PyTorch / MMCV 机制

目标调用链：

```text
model(...)
↓
nn.Module.__call__()
↓
forward(...)
↓
forward_test(...)
↓
simple_test(...)
↓
pts_bbox_head(...)
↓
transformer(...)
```

### 8.1 `model(...)`

```text
object: 构建完成的 BEVFormer detector instance；分布式包装时外层可能是 MMDataParallel
class: projects/.../detectors/bevformer.py::BEVFormer
机制: Python callable syntax + PyTorch Module
```

调用处通常传入：

```python
model(return_loss=False, img=img, img_metas=img_metas)
```

这里 `model` 不是普通函数，而是 `BEVFormer` object。

### 8.2 `nn.Module.__call__()`

```text
定义位置: 当前环境 torch.nn.Module（不在本仓库）
机制: PyTorch
```

它负责：

```text
pre-forward hooks
forward(...)
post-forward hooks
autocast/tracing/module machinery
```

然后动态分派到 `BEVFormer.forward()`。

### 8.3 `BEVFormer.forward(...)`

文件与定义：

```text
projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
class BEVFormer
def forward(self, return_loss=True, **kwargs)
```

机制分类：BEVFormer 自定义 dispatch，使用 Python `if` 和 `**kwargs`。

```python
if return_loss:
    return self.forward_train(**kwargs)
else:
    return self.forward_test(**kwargs)
```

Inference 必须传 `return_loss=False`，否则会进入训练链路。

### 8.4 `forward_test(...)`

对象仍是同一个 `BEVFormer` instance。该 method：

```text
验证 img_metas 外层 list
检查 scene_token
必要时 reset prev_bev
计算 can_bus translation/rotation delta
调用 simple_test
保存 new_prev_bev、absolute position、absolute angle
```

机制分类：BEVFormer 自定义 temporal state machine，主要使用 Python dict/list/indexing/deepcopy。

关键状态：

```python
self.prev_frame_info = {
    'prev_bev': None,
    'scene_token': None,
    'prev_pos': 0,
    'prev_angle': 0,
}
```

### 8.5 `simple_test(...)`

定义：

```text
projects/.../detectors/bevformer.py
BEVFormer.simple_test
```

它先调用：

```python
img_feats = self.extract_feat(...)
```

再调用 `simple_test_pts`，最后用 `zip` 把 batch detection 放进结果 dict。

机制分类：BEVFormer orchestration + 继承的 backbone/neck modules。

### 8.6 `pts_bbox_head(...)`

`self.pts_bbox_head` 是配置和 registry 构建出的 `BEVFormerHead` object：

```text
projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
class BEVFormerHead(DETRHead)
```

因为它继承 `nn.Module`，所以：

```python
self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
```

会通过 `nn.Module.__call__` 进入 `BEVFormerHead.forward()`。

机制分类：OpenMMLab registry 构建 + PyTorch call dispatch + BEVFormer custom forward。

### 8.7 `transformer(...)`

`self.transformer` 由父类/Builder 根据 config 创建，实际 class 为：

```text
projects/mmdet3d_plugin/bevformer/modules/transformer.py
class PerceptionTransformer(BaseModule)
```

`BEVFormerHead.forward()` 中：

```python
outputs = self.transformer(
    mlvl_feats,
    bev_queries,
    object_query_embeds,
    ...,
    prev_bev=prev_bev)
```

再次经过 `nn.Module.__call__`，进入 `PerceptionTransformer.forward()`。该 method 先调用 `get_bev_features()`/encoder，再调用 decoder。

机制分类：MMCV `BaseModule`/registry + PyTorch module dispatch + BEVFormer transformer 实现。

### 8.8 调用链机制总表

| Step | Object/Class | Method location | Mechanism |
| --- | --- | --- | --- |
| `model(...)` | `BEVFormer` instance | caller | Python callable syntax |
| `nn.Module.__call__` | inherited PyTorch method | installed PyTorch | PyTorch hooks + forward dispatch |
| `forward` | `BEVFormer` | `detectors/bevformer.py` | BEVFormer train/test dispatch |
| `forward_test` | same detector | `detectors/bevformer.py` | BEVFormer temporal state |
| `simple_test` | same detector | `detectors/bevformer.py` | BEVFormer test orchestration |
| `pts_bbox_head(...)` | `BEVFormerHead` | `dense_heads/bevformer_head.py` | Registry-built PyTorch module |
| `transformer(...)` | `PerceptionTransformer` | `modules/transformer.py` | Registry-built MMCV BaseModule |
| `encoder(...)` | `BEVFormerEncoder` | `modules/encoder.py` | MMCV TransformerLayerSequence |
| `attention(...)` | Temporal/Spatial objects | corresponding module files | Registry-built attention modules |

## 9. OpenMMLab 最难阅读的五类机制

### 9.1 `**kwargs` 使用统计与路线

核心扫描范围：

```text
functions defining **kwargs: 39
calls forwarding **kwargs:   41
```

高价值位置：

| File | Function | 传递内容 |
| --- | --- | --- |
| `detectors/bevformer.py` | `forward`, `forward_test` | train/test inputs、rescale 等 |
| `dense_heads/bevformer_head.py` | `__init__` | DETRHead 父类构建参数 |
| `modules/transformer.py` | `get_bev_features`, `forward` | `img_metas` 等 runtime metadata |
| `modules/encoder.py` | encoder/layer `forward` | reference points、mask、motion、metadata |
| `modules/custom_base_transformer_layer.py` | `forward` | 通用 attention 参数转发 |
| `tools/deployment/bevformer_stateful_onnx_wrapper.py` | deployment forwards | 保持官方 API contract |

阅读方法：在每层旁边写一个“kwargs 当前 key 集合”，尤其追踪 `img_metas`。

### 9.2 Decorators

核心范围共 56 个 decorator，主要分四类：

| 类别 | 示例 | 作用 |
| --- | --- | --- |
| Registry | `@DETECTORS.register_module()` | class 名登记到配置构建系统 |
| Precision | `@auto_fp16`, `@force_fp32` | 包装 method 并转换指定参数 dtype |
| Autograd | `@staticmethod`, `@custom_fwd`, `@custom_bwd` | 自定义 CUDA autograd Function |
| Context | `@contextmanager` | deployment patch 的安装/恢复 |

Decorator 等价于“定义完成后再包装”：

```python
@decorator
def f(...): ...

# 概念上约等于
f = decorator(f)
```

### 9.3 Inheritance

核心范围有 28 个显式继承 class。最重要的继承关系：

```text
BEVFormer
  └─ MVXTwoStageDetector

BEVFormerHead
  └─ DETRHead

PerceptionTransformer
  └─ mmcv.runner.BaseModule

BEVFormerEncoder
  └─ TransformerLayerSequence

BEVFormerLayer
  └─ MyCustomBaseTransformerLayer

TemporalSelfAttention / SpatialCrossAttention / CustomMSDA
  └─ BaseModule

BEVFormerStatefulONNXWrapper
  └─ torch.nn.Module
```

看到当前文件没有定义的 attribute/method 时，沿继承树向上找，不要立即认为代码不完整。

### 9.4 `super()`

核心范围共 26 次。最关键位置：

```text
BEVFormer.__init__
BEVFormerHead.__init__
PerceptionTransformer.__init__
BEVFormerEncoder.__init__
BEVFormerLayer.__init__
DetectionTransformerDecoder.__init__
TemporalSelfAttention.__init__
SpatialCrossAttention.__init__
```

`super()` 的主要作用不是“形式要求”，而是让父类完成：

```text
nn.Module internal initialization
submodule/parameter registration
common transformer layers
DETR loss/assigner/sampler
BaseModule init_cfg
```

漏读 `super()` 会漏掉一半以上对象状态来源。

### 9.5 Registry 与 `build_from_cfg`

注册：

```python
@DETECTORS.register_module()
class BEVFormer(...):
```

含义：把字符串 `BEVFormer` 映射到 Python class。

配置：

```python
dict(type='BEVFormer', ...)
```

构建概念：

```text
cfg['type'] = 'BEVFormer'
        ↓ Registry 查表
BEVFormer class
        ↓ class(**remaining_cfg)
BEVFormer object
```

仓库中显式 `build_from_cfg` 位置：

| File | Function | Registry |
| --- | --- | --- |
| `projects/mmdet3d_plugin/datasets/builder.py` | dataset build | `DATASETS` |
| `projects/mmdet3d_plugin/datasets/samplers/sampler.py` | `build_sampler` | `SAMPLER` |
| `projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py` | custom hook build | `HOOKS` |

Transformer/attention 还通过 MMCV helper 间接调用 registry/build_from_cfg：

```text
build_transformer_layer_sequence(encoder)
build_attention(deformable_attention)
build_bbox_coder(bbox_coder)
```

这些 helper 的实现主要位于安装的 MMCV/MMDetection，而不在当前 BEVFormer 文件中。

## 10. 如何实际阅读一个函数

以 `PerceptionTransformer.get_bev_features()` 为例，不要逐字翻译。按下面顺序：

1. 写出输入 Tensor contract。
2. 圈出 `self.xxx`，确认它们来自哪个 `__init__` 或父类。
3. 对每个 `reshape/permute/stack/cat` 写出前后 layout。
4. 对每个 `**kwargs['key']` 向上追踪 key 的来源。
5. 对每个 `module(...)` 确认 object 的实际 class，再打开其 `forward()`。
6. 对 `if prev_bev is not None` 等状态分支分别画路径。
7. 最后再看具体数学公式。

示例笔记：

```text
bev_queries
[2500,256]
  unsqueeze/repeat
[2500,B,256]

bev_pos
[B,256,50,50]
  flatten/permute
[2500,B,256]

feat
[B,Ncam,C,H,W]
  flatten/permute
[Ncam,B,HW,C]
```

## 11. 最小学习计划

### Day 1：Object 与调用入口

只学：

```text
class
object
self
__init__
method
attribute
```

练习文件：`detectors/bevformer.py`。

目标：能够解释 `model.prev_frame_info` 为什么能跨 frame 保存状态。

### Day 2：Inheritance、`super()`、`nn.Module.__call__`

只学：

```text
inheritance
super()
nn.Module
module(...)
forward(...)
```

练习：从 `self.pts_bbox_head(...)` 找到 `BEVFormerHead.forward()`；从 `self.transformer(...)` 找到 `PerceptionTransformer.forward()`。

目标：不再寻找模型 class 中不存在的自定义 `__call__`。

### Day 3：容器、参数与 metadata

只学：

```text
list / tuple / dict
indexing / slicing
positional / keyword / default argument
*args / **kwargs
deepcopy
```

练习：手工画出 `img_metas[0][0]['can_bus']` 的嵌套结构，并追踪 `img_metas` 从 detector 到 encoder。

目标：能够独立解释 `forward_test` 的 scene reset 和 motion delta。

### Day 4：Tensor Shape 与 Layout

只学：

```text
shape / dtype / device
reshape / view
permute / transpose
unsqueeze / squeeze
cat / stack
contiguous
```

练习文件：`transformer.py::get_bev_features`。

目标：独立写出 BEV query、FPN value 和 `prev_bev` 的每一步 layout。

### Day 5：OpenMMLab 机制

只学：

```text
decorator
Registry
register_module
build_from_cfg
BaseModule
auto_fp16 / force_fp32
```

练习：从 config 中的 `type='BEVFormer'` 追到 class，从 attention config 追到 `SpatialCrossAttention` object。

目标：看见字符串配置时知道它最终对应哪个 Python class。

### Day 6：Inference 主链

完整追踪：

```text
model(...)
→ forward
→ forward_test
→ simple_test
→ simple_test_pts
→ BEVFormerHead.forward
→ PerceptionTransformer.forward
→ get_bev_features
→ BEVFormerEncoder.forward
```

目标：每一步都能回答 object、class、method file、主要 Tensor shape。

### Day 7：Attention 与 Deployment

只读：

```text
TemporalSelfAttention.forward
SpatialCrossAttention.forward
MultiScaleDeformableAttnFunction_fp32.apply
BEVFormerStatefulONNXWrapper.forward
deployment_export_overrides
```

重点学习：

```text
with/context manager
try/finally
hasattr/getattr
custom autograd Function
detach/cpu/numpy
Path
```

目标：区分官方 inference、deployment-only override 和 ONNX custom-op boundary。

## 12. 学完后的自测标准

如果能不看答案回答下面问题，就已经具备独立阅读当前工程的基础：

1. 为什么 `model(...)` 会进入 `BEVFormer.forward()`？
2. 为什么 `self.pts_bbox_head` 在 `BEVFormer` 当前文件中没有创建细节？
3. `img_metas[0][0]['can_bus']` 每一层索引分别是什么？
4. `prev_bev` 何时是 `[N,B,C]`，何时转换为 `[B,N,C]`？
5. `stack` 与 `cat` 在 temporal queue 中有什么区别？
6. `**kwargs` 中的 `img_metas` 从哪一层传到 encoder？
7. `@ATTENTION.register_module()` 为什么使 config 能构建 class？
8. `@force_fp32` 为什么可能改变函数入口 Tensor dtype？
9. deployment context 为什么必须在异常退出时恢复？
10. 为什么 ONNX wrapper 应通过 `module(...)` 而不是直接调用 `.forward()`？

达到这个标准后，再继续学习完整 Python 语言的收益已经不高；应把主要时间放在真实调用链、Tensor layout 和 config-to-object 映射上。
