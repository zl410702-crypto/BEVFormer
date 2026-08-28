# 从 BEVFormer 源码反向设计 Python 学习知识树

## 1. 结论先行

当前阅读障碍不是“缺少完整 Python 知识”，而是以下几条项目路径上的知识依赖没有连起来：

```text
名字引用对象
→ 对象具有 attribute/method
→ class 创建对象
→ inheritance 补充当前文件没有定义的行为
→ object(...) 触发 callable / __call__
→ nn.Module.__call__ 分派到 forward
```

```text
容器与逐层索引
→ function parameter
→ keyword argument
→ dict packing/unpacking
→ **kwargs 跨 detector/head/transformer/encoder 传递 metadata
```

```text
Tensor shape
→ indexing/slicing
→ reshape/view
→ permute/transpose
→ contiguous
→ BEVFormer 中 [N,B,C] 与 [B,N,C] 的 layout contract
```

因此学习顺序由依赖关系决定，而不是从 Python 语法百科第一页开始。

## 2. 扫描范围和方法

重点扫描：

```text
projects/mmdet3d_plugin/bevformer/**/*.py
tools/deployment/**/*.py
tools/export_bevformer_stateful_onnx.py
```

为确认特殊方法和配置构建机制，还查看了：

```text
projects/mmdet3d_plugin/datasets/**/*.py
projects/mmdet3d_plugin/models/utils/**/*.py
projects/mmdet3d_plugin/dd3d/structures/**/*.py
```

核心 32 个 BEVFormer/deployment 文件的统计：

| 真实机制 | 数量 | 结论 |
| --- | ---: | --- |
| 带 `**kwargs` 的函数 | 39 | 参数流是 P0 障碍 |
| 调用时展开 `**kwargs` | 41 | 必须会追踪参数来源 |
| Decorator | 56 | Registry/精度/context/autograd 高频 |
| 有父类的 class | 28 | 当前文件只能看到一部分行为 |
| `super()` | 26 | 父类初始化不可跳过 |
| List comprehension | 46 | batch/frame/layer 数据组织高频 |
| Dict comprehension | 7 | audit 和筛选报告常见 |
| `assert` | 41 | shape/config contract 高频 |
| `raise` | 47 | deployment 采用 fail-closed |
| `try` block | 10 | override 恢复与兼容处理 |

最终筛出 21 个核心学习节点：P0 9 个、P1 8 个、P2 4 个。P3 只列出暂时无需系统学习的内容。

## 3. 当前源码暴露出的能力缺口

### 3.1 不是算法问题，而是“对象身份”丢失

典型代码：

```python
self.pts_bbox_head(...)
self.transformer(...)
self.encoder(...)
```

障碍通常不是不知道 head/transformer 的算法，而是不知道：

```text
self 是谁？
pts_bbox_head 是在哪里赋值的？
它是什么 class？
为什么加括号就能执行？
最终进入哪个 forward？
```

### 3.2 参数经过多层转发后“凭空出现”

典型代码：

```python
return self.forward_test(**kwargs)
...
kwargs['img_metas']
```

如果没有掌握 dict、keyword argument 和 packing/unpacking，就无法从 detector 追到 encoder。

### 3.3 嵌套容器没有画结构

典型代码：

```python
img_metas[0][0]['can_bus']
```

读者往往试图“一眼读完”，但正确方式是逐层写类型：

```text
img_metas                    list: test augmentations
img_metas[0]                 list: batch samples
img_metas[0][0]              dict: first sample metadata
img_metas[0][0]['can_bus']   array: ego state
```

### 3.4 Tensor shape 能看懂，layout 变化看不懂

```python
bev_query = bev_query.permute(1, 0, 2)
prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(...)
```

只知道 `reshape` 的名字不够，必须追踪每一维语义。

### 3.5 OpenMMLab 隐藏了对象创建过程

```python
@DETECTORS.register_module()
class BEVFormer(...):
```

配置文件里只有字符串 `type='BEVFormer'`，运行时却得到 Python object。理解这一点需要先掌握 class/object/function argument/decorator，然后才能学习 Registry。

## 4. Python Knowledge Dependency Graph

```text
Root：独立阅读当前 BEVFormer inference / deployment 源码
│
├─ 1. 数据与引用
│  ├─ name / assignment
│  ├─ object / type / identity
│  ├─ reference
│  ├─ mutable vs immutable
│  └─ shallow copy / deepcopy
│
├─ 2. 容器与访问
│  ├─ list / tuple
│  ├─ dict / key / value
│  ├─ set
│  ├─ indexing / slicing / ellipsis
│  ├─ nested containers
│  └─ list.append / dict.setdefault
│
├─ 3. 函数与参数流
│  ├─ function / local scope / return
│  ├─ parameter vs argument
│  ├─ positional / keyword / default argument
│  ├─ iterable unpacking
│  ├─ *args
│  └─ **kwargs packing / unpacking
│
├─ 4. Object-Oriented Python
│  ├─ class / instance
│  ├─ attribute / method
│  ├─ self / method binding
│  ├─ __init__
│  ├─ inheritance / override
│  └─ super / method resolution
│
├─ 5. Python 调用协议
│  ├─ callable object
│  ├─ __call__
│  ├─ __getitem__
│  └─ PyTorch nn.Module.__call__ → forward
│
├─ 6. Tensor 表达
│  ├─ attribute/method call prerequisites
│  ├─ shape / dtype / device
│  ├─ indexing / slicing
│  ├─ reshape / view
│  ├─ permute / transpose
│  ├─ unsqueeze / squeeze
│  ├─ cat / stack
│  └─ contiguous / detach / cpu / numpy
│
├─ 7. 高频工程表达
│  ├─ for / if
│  ├─ enumerate / zip / range
│  ├─ list/dict comprehension
│  ├─ isinstance / hasattr / getattr
│  └─ lambda / closure / default capture
│
└─ 8. 工程控制机制
   ├─ module / package / import / relative import
   ├─ exception / raise / assert
   ├─ with / context manager / try-finally
   ├─ decorator
   └─ MMCV Registry / build_from_cfg / precision decorators
```

树中的先后关系是强依赖。例如在理解 `nn.Module.__call__` 之前，必须先理解 object、method、inheritance 和 callable；在理解 `**kwargs` 之前，必须先理解 function、parameter、dict 和 keyword argument。

## 5. P0：现在不懂就无法继续读源码

### P0-1 `[Python Language]` Name、object、reference、mutation

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])`<br>`img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']` |

这段代码同时存在“保存旧值”和“原地修改当前对象”。如果不知道名字只是引用对象，就无法理解为什么要 deepcopy。

**Prerequisites**

```text
variable/name
assignment
basic object/type
identity
mutable vs immutable
```

**Python 本身是什么**

```python
a = [1, 2]
b = a
b.append(3)
print(a)  # [1, 2, 3]
```

赋值通常没有复制 object；`a` 和 `b` 引用同一个 list。

**心智模型**

```text
a ─┐
   ├──> list object [1, 2]
b ─┘
```

**常见误区**

- 误以为 `b = a` 创建独立副本。
- 误以为所有 `x -= y` 都创建新对象；NumPy/Tensor slicing 可能原地修改底层数据。
- 把名字、对象和类型混成同一概念。

**回到项目**

`tmp_pos` 必须保存修改前 absolute position；随后 metadata 中 `can_bus[:3]` 被改为 relative translation。若只是普通引用，保存的 previous position 可能一起变化。

---

### P0-2 `[Python Language]` List、tuple、dict 与 nested access

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward_test` |
| Representative Code | `img_metas[0][0]['scene_token']`<br>`img_metas[0][0]['can_bus'][:3]` |

**Prerequisites**

```text
object/type
integer/string
sequence vs mapping
index
key/value
```

**Python 本身是什么**

```python
records = [[{'name': 'frame0', 'motion': [1.0, 2.0]}]]
motion = records[0][0]['motion']
```

`list` 按整数位置访问；`dict` 按 key 访问；tuple 常表达固定返回结构。

**心智模型**

```text
outer list
  [0] → batch list
          [0] → metadata dict
                  ['can_bus'] → array
```

**常见误区**

- 一眼读取多层表达式，不逐层判断类型。
- 认为所有方括号都是 Tensor indexing。
- 不理解 `return a, b` 实际返回 tuple。

**回到项目**

`img_metas` 在 test-time 有 augmentation 和 batch 两层 list；最内层才是 sample metadata dict。读 `forward_test` 时必须保留这三层结构。

---

### P0-3 `[Python Language]` Indexing、slicing、ellipsis

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead` |
| Function | `forward` |
| Representative Code | `tmp[..., 0:2] += reference[..., 0:2]`<br>`tmp[..., 4:5] += reference[..., 2:3]` |

**Prerequisites**

```text
list/sequence
zero-based index
half-open range
nested dimensions
```

**Python 本身是什么**

```python
x = [10, 20, 30, 40]
x[1]    # 20
x[1:3]  # [20, 30], end 3 is excluded
```

Tensor 的 `...` 表示省略前面的若干维。

**心智模型**

```text
0:2 = indices 0 and 1
4:5 = index 4 but keep a length-1 dimension
... = all unspecified leading dimensions
```

**常见误区**

- 把 slice end 当作包含端点。
- 认为 `x[..., 4]` 与 `x[..., 4:5]` shape 相同。
- 忽略 `+=` 是原地更新目标 slice。

**回到项目**

这些 slice 分别选 bbox 的 xy/z 相关 channel，并保持 broadcast 所需维度。错误理解会直接导致 bbox layout 阅读失败。

---

### P0-4 `[Python Language]` Function、parameter、argument、return

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `simple_test_pts` |
| Representative Code | `def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):`<br>`return outs['bev_embed'], bbox_results` |

**Prerequisites**

```text
name/object
assignment
expression
local scope
```

**Python 本身是什么**

```python
def add(x, y=1):       # x/y are parameters; y has a default
    return x + y

v = add(3, y=2)        # 3 and 2 are arguments
```

**心智模型**

```text
call arguments
    ↓ bind to parameters
local function scope
    ↓ return object(s)
caller receives result
```

**常见误区**

- 混淆 parameter（定义处）与 argument（调用处）。
- 不知道 `return a, b` 是 tuple，可用两个名字解包。
- 忽略 default argument，误以为调用缺参数。

**回到项目**

`prev_bev`、`rescale` 是可选参数；返回值是 `(bev_embed, bbox_results)`，由上层 `new_prev_bev, bbox_pts = ...` 解包。

---

### P0-5 `[Python Language]` Positional/keyword/default 与 `*args/**kwargs`

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `forward` |
| Representative Code | `def forward(self, return_loss=True, **kwargs):`<br>`return self.forward_test(**kwargs)` |

**Prerequisites**

```text
function
parameter/argument
dict/key/value
iterable unpacking
```

**Python 本身是什么**

```python
def show(a, b=0, *args, **kwargs):
    return a, b, args, kwargs

options = {'b': 2, 'mode': 'test'}
show(1, 3, 4, 5, mode='train')
show(1, **options)
```

`*args` 是额外位置参数 tuple；`**kwargs` 是额外 keyword dict。调用位置的 `**options` 将 dict 展开为关键字参数。

**心智模型**

```text
kwargs dict {'img': ..., 'img_metas': ...}
       ↓ ** unpack
forward_test(img=..., img_metas=...)
```

**常见误区**

- 认为 `kwargs` 是 BEVFormer 特殊对象；它只是普通 dict。
- 只看当前函数签名，不追踪 dict 中有哪些 key。
- 混淆定义处 `**kwargs`（packing）和调用处 `**kwargs`（unpacking）。

**回到项目**

`BEVFormer.forward` 根据 `return_loss` 分流，但 img/img_metas 等输入都由 `kwargs` 转发。Transformer/encoder 中 `kwargs['img_metas']` 的来源要沿此参数流向上追踪。

---

### P0-6 `[Python Language]` Class、instance、attribute、method、`self`、`__init__`

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class | `BEVFormer` |
| Function | `__init__` |
| Representative Code | `class BEVFormer(MVXTwoStageDetector):`<br>`self.prev_frame_info = {'prev_bev': None, ...}` |

**Prerequisites**

```text
name/object/reference
function
scope
dict
```

**Python 本身是什么**

```python
class Counter:
    def __init__(self, start=0):
        self.value = start

    def increment(self):
        self.value += 1

c = Counter(3)
c.increment()
```

`self` 是调用该 method 的 instance；`self.value` 是 instance attribute。

**心智模型**

```text
Counter class --instantiate--> c object
                             ├─ value attribute
                             └─ increment bound method
```

**常见误区**

- 把 class 和 instance 当成同一对象。
- 认为 `self` 是关键字；它只是约定俗成的第一个参数名。
- 只在当前 method 搜索 attribute 来源，不查看 `__init__`。

**回到项目**

同一个 `BEVFormer` instance 在连续 frame 中保存 `prev_frame_info`。`self.pts_bbox_head` 等成员部分来自父类初始化，而不是当前文件直接赋值。

---

### P0-7 `[Python Language]` Inheritance、override、`super()`

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` |
| Class | `BEVFormerHead(DETRHead)` |
| Function | `__init__` |
| Representative Code | `class BEVFormerHead(DETRHead):`<br>`super(BEVFormerHead, self).__init__(*args, transformer=transformer, **kwargs)` |

**Prerequisites**

```text
class/instance
attribute/method
__init__
function arguments
```

**Python 本身是什么**

```python
class Base:
    def __init__(self):
        self.base_value = 1

class Child(Base):
    def __init__(self):
        super().__init__()
        self.child_value = 2
```

子类继承父类 attribute/method；同名 method 是 override；`super()` 按 method resolution order 调父类。

**心智模型**

```text
BEVFormerHead object
   ├─ child-defined behavior
   └─ inherited DETRHead behavior/state
```

**常见误区**

- 当前文件找不到 `self.num_query` 就认为代码缺失。
- 把 `super()` 当成可省略样板；对 `nn.Module` 子类，父类初始化负责 parameter/submodule 注册。
- 不区分 override 与新增 method。

**回到项目**

`BEVFormerHead` 的 assigner、sampler、loss、query 数量和 transformer 构建大量来自 `DETRHead`。阅读 head 必须同时意识到父类状态存在。

---

### P0-8 `[Python Language]` Callable、method binding、`__call__`

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` |
| Class / Function | `BEVFormer.simple_test_pts` |
| Representative Code | `outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)` |

**Prerequisites**

```text
class/instance
attribute lookup
method/self
inheritance
```

**Python 本身是什么**

```python
class Scale:
    def __call__(self, x):
        return x * 2

s = Scale()
s(3)  # calls s.__call__(3)
```

访问 `obj.method` 会得到绑定到 `obj` 的 bound method，因此调用时不显式传 self。

**心智模型**

```text
self.pts_bbox_head
      ↓ BEVFormerHead object (callable through nn.Module)
object(...)
      ↓ inherited nn.Module.__call__
BEVFormerHead.forward(...)
```

**常见误区**

- 在 `BEVFormerHead` 文件中找自定义 `__call__`，找不到就中断。
- 认为 `pts_bbox_head(...)` 是普通函数。
- 直接调用 `.forward()`，忽略 PyTorch hooks/autocast/tracing。

**回到项目**

`BEVFormerHead`、`PerceptionTransformer`、encoder layer 都是 PyTorch module object。加括号会经过继承的 `nn.Module.__call__()` 再进入实际 class 的 `forward()`。

---

### P0-9 `[PyTorch]` Tensor layout transformation

该节点来源于项目，但主体前置仍是 Python object、method call、indexing。

**为什么项目需要它**

| Field | Value |
| --- | --- |
| File Path | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` |
| Class | `PerceptionTransformer` |
| Function | `get_bev_features` |
| Representative Code | `bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)`<br>`bev_pos = bev_pos.flatten(2).permute(2, 0, 1)` |

**Prerequisites**

```text
object/attribute/method
tuple
indexing/slicing
integer dimensions
function arguments
```

**Python/PyTorch 最小示例**

```python
import torch
x = torch.zeros(2, 3)
y = x.unsqueeze(1)       # [2, 1, 3]
z = y.permute(1, 0, 2)   # [1, 2, 3]
```

**心智模型**

```text
Tensor = storage + shape + stride + dtype + device

reshape/view:       reinterpret dimensions
permute/transpose:  reorder dimension meanings
contiguous:         materialize contiguous storage order
```

**常见误区**

- 只比较元素总数，不记录维度语义。
- 把 `[N,B,C]` 与 `[B,N,C]` 当成相同 contract。
- 认为 `permute` 会改变数值内容，而忽略它主要改变 stride/layout。
- 不理解 `stack` 新增维度，`cat` 使用已有维度。

**回到项目**

必须能持续追踪：

```text
bev_query: [2500,256] → [2500,B,256] → [B,2500,256]
prev_bev:  [2500,B,256] ↔ [B,2500,256]
features:  [B,Ncam,C,H,W] → [Ncam,B,HW,C]
```

否则 Temporal/Spatial attention 和 ONNX/TensorRT boundary 都无法阅读。

## 6. P1：当前工程高频，近期必须掌握

### P1-1 `[Python Language]` `for`、`if`、`range`、`enumerate`、`zip`

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/modules/transformer.py
Class: PerceptionTransformer
Function: get_bev_features
Code: for lvl, feat in enumerate(mlvl_feats):
```

**Prerequisites**：iterable、list/tuple、assignment、boolean expression。

**Pure Python**

```python
for index, value in enumerate(['a', 'b']):
    print(index, value)

for left, right in zip([1, 2], [3, 4]):
    print(left, right)
```

**Mental Model**：`enumerate` 产生 `(index,value)`；`zip` 产生同位置元素 tuple；`range(n)` 产生 0 到 n-1。

**误区**：认为 `zip` 会自动检查两个序列长度相同；它默认在最短序列处停止。

**Back to Project**：`lvl` 既用于选择 level embedding，也用于记录 spatial shape；`zip(bbox_list,bbox_pts)` 将 batch result 一一组合。

---

### P1-2 `[Python Language]` Comprehension

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
Class: BEVFormer
Function: obtain_history_bev
Code: img_metas = [each[i] for each in img_metas_list]
```

**Prerequisites**：list/dict、for、if、expression、local name。

**Pure Python**

```python
squares = [x * x for x in range(4)]
large = {name: value for name, value in scores.items() if value > 0.5}
```

**Mental Model**：先把 comprehension 展开成普通 for，再读 expression 和过滤条件。

**误区**：嵌套 comprehension 一眼读完；忽略 expression 在最前、循环在后。

**Back to Project**：从 `[batch][time]` metadata 中抽取统一 timestamp；audit 中 dict comprehension 只保留失败项。

---

### P1-3 `[Python Language]` Mutable container methods：`append` / `setdefault`

**Project Code**

```text
File: tools/export_bevformer_stateful_onnx.py
Function: descendants
Code: consumers.setdefault(name, []).append(index)
```

**Prerequisites**：dict、key/value、list、reference/mutability、method call、default argument。

**Pure Python**

```python
groups = {}
groups.setdefault('camera', []).append(0)
groups.setdefault('camera', []).append(1)
# {'camera': [0, 1]}
```

**Mental Model**

```text
dict lookup key
 ├─ exists → return existing list
 └─ missing → store and return new []
returned list.append(index)
```

**误区**：认为 `setdefault` 每次都会覆盖现有值；忽略 `append` 原地修改 list。

**Back to Project**：为每个 ONNX tensor name 累积所有 consumer node index，用于输入依赖图遍历。

---

### P1-4 `[Python Standard Library]` `copy.deepcopy`

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py
Class: MyCustomBaseTransformerLayer
Function: __init__
Code: attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
```

**Prerequisites**：reference、mutable nested object、import/module、function call。

**Pure Python**

```python
from copy import deepcopy
a = {'items': [1]}
b = deepcopy(a)
b['items'].append(2)
# a is unchanged
```

**Mental Model**：deepcopy 递归创建独立嵌套对象图，而不只复制最外层容器。

**误区**：把 `list.copy()`、浅 copy 和 deepcopy 当成一样；对 Tensor deepcopy 的成本没有意识。

**Back to Project**：每个 attention/FFN 配置必须独立，避免修改某层配置时污染其他层；temporal metadata 也必须保存修改前值。

---

### P1-5 `[Python Language]` `isinstance`、`hasattr`、`getattr`

**Project Code**

```text
File: tools/deployment/bevformer_onnx_wrapper.py
Function: deployment_export_overrides
Code: had_symbolic = hasattr(function_class, 'symbolic')
Code: original_symbolic = getattr(function_class, 'symbolic', None)
```

**Prerequisites**：object/type/class、attribute lookup、inheritance、default value。

**Pure Python**

```python
class Item:
    value = 3

x = Item()
isinstance(x, Item)              # True
hasattr(x, 'value')              # True
getattr(x, 'missing', None)      # None
```

**Mental Model**：runtime inspection；attribute 名本身是字符串数据。

**误区**：`isinstance` 只匹配精确 class；实际上也匹配子类。`getattr` 不是 dict lookup。

**Back to Project**：deployment override 只 patch 特定 module 类型，并记录原 symbolic 是否存在，以便精确恢复。

---

### P1-6 `[Python Language + Standard Library]` `with`、context manager、`try/finally`

**Project Code**

```text
File: tools/deployment/bevformer_stateful_onnx_wrapper.py
Function: static_spatial_cross_attention_override
Code: try: yield
      finally: child.forward = original
```

**Prerequisites**：function、exception、object protocol、scope、reference、assignment。

**Pure Python**

```python
from contextlib import contextmanager

@contextmanager
def temporary(state):
    old = state['value']
    state['value'] = 'temporary'
    try:
        yield
    finally:
        state['value'] = old
```

**Mental Model**

```text
with enter: install temporary state
     body:  use it
with exit:  restore, including exception path
```

**误区**：认为异常时 finally 不运行；认为 `yield` 等于普通 return；忽略多个 context manager 退出顺序相反。

**Back to Project**：static SCA、stateful BEV path 和 symbolic 只在 export context 中生效，普通 inference 必须恢复。

---

### P1-7 `[Python Language]` Decorator

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
Class: BEVFormer
Code: @DETECTORS.register_module()
Code: @auto_fp16(apply_to=('img'))
```

**Prerequisites**：function as object、assignment、scope、call/return、closure（理解复杂 decorator 时）。

**Pure Python**

```python
def log_calls(func):
    def wrapped(*args, **kwargs):
        print('calling')
        return func(*args, **kwargs)
    return wrapped

@log_calls
def add(a, b):
    return a + b
```

概念上 `add = log_calls(add)`。

**Mental Model**：decorator 接收原 function/class，返回替代 object 或执行登记副作用。

**误区**：以为 decorator 只是注释；忽略它可能改变参数 dtype、签名或注册全局状态。

**Back to Project**：Registry decorator 使配置字符串能找到 class；precision decorator 在进入 method 前处理 Tensor dtype。

---

### P1-8 `[Python Language]` Module/package/import/relative import

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/modules/transformer.py
Code: from mmcv.runner.base_module import BaseModule
Code: from .temporal_self_attention import TemporalSelfAttention
```

**Prerequisites**：name、file/path、namespace、attribute。

**Pure Python**

```python
import math
from pathlib import Path
from .helper import function  # inside a package
```

**Mental Model**

```text
module file executes once
→ creates namespace
→ import binds selected object names in current namespace
```

**误区**：认为所有 import 都对应仓库文件；`mmcv`/`torch` 来自环境。不了解相对导入的 `.` 指当前 package。

**Back to Project**：先区分行为来自当前 BEVFormer、安装的 PyTorch、MMCV 还是 MMDetection，才能知道下一步去哪里查源码。

## 7. P2：出现时查询即可

### P2-1 `[Python Language]` `lambda` 与 closure capture

**Project Code**

```text
File: tools/golden_sequence_recorder.py
Function: hook registration
Code: lambda m, i, o, index=index: ...
```

**Prerequisites**：function object、parameter/default、scope、closure、callback。

**Pure Python**

```python
callbacks = []
for i in range(3):
    callbacks.append(lambda i=i: i)
```

**Mental Model**：lambda 是匿名 function；`i=i` 在创建 callback 时冻结当前值。

**误区**：写成 `lambda: i` 后所有 callback 都读取循环结束后的同一个 i。

**Back to Project**：Golden hooks 必须把各 layer index 固定下来。核心 inference 32 文件中没有 lambda，因此无需提前深入。

---

### P2-2 `[Python Language]` `staticmethod` 与 autograd `Function.apply`

**Project Code**

```text
File: projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
Class: MultiScaleDeformableAttnFunction_fp32
Code: @staticmethod
      def forward(ctx, value, ...)
```

**Prerequisites**：class/method/self、decorator、function arguments、return。

**Pure Python**

```python
class Math:
    @staticmethod
    def twice(x):
        return x * 2
```

**Mental Model**：function 放在 class namespace，但不绑定 instance/self。PyTorch custom autograd Function 使用 `.apply()` 协调 static forward/backward。

**误区**：把它当普通 `nn.Module.forward`；寻找不存在的 module instance state。

**Back to Project**：MSDA Python wrapper 最终通过 `.apply()` 调 MMCV CUDA extension；这也是 ONNX symbolic/plugin 边界。

---

### P2-3 `[Python Language]` `property` 与特殊方法 `__getitem__`

**Project Code**

```text
File: projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
Class: CustomNuScenesDataset
Function: __getitem__

File: projects/mmdet3d_plugin/dd3d/structures/image_list.py
Code: @property
```

**Prerequisites**：class/object/method、decorator、indexing syntax。

**Pure Python**

```python
class Items:
    def __getitem__(self, index):
        return index * 10

    @property
    def size(self):
        return 3

x = Items()
x[2]    # x.__getitem__(2)
x.size  # property method without parentheses
```

**Mental Model**：Python 语法操作可通过特殊方法协议委托给 object。

**误区**：看见 `dataset[index]` 只当 list；看见 `obj.size` 不知道背后运行 method。

**Back to Project**：dataset indexing 触发数据准备；property 主要在 DD3D structures，非主 inference 热点。

---

### P2-4 `[Python Standard Library]` `pathlib.Path`

**Project Code**

```text
File: tools/export_bevformer_stateful_onnx.py
Function: module defaults / main
Code: DEFAULT_OUTPUT = Path('artifacts/...onnx')
Code: args.sequence_root / 'sequence.json'
```

**Prerequisites**：object/method、operator overloading、string、filesystem path。

**Pure Python**

```python
from pathlib import Path
root = Path('artifacts')
model = root / 'model.onnx'
model.parent.mkdir(parents=True, exist_ok=True)
```

**Mental Model**：Path 是路径 object；`/` 被 class 重载为路径拼接。

**误区**：把 `/` 当除法；混用 Path 和 string method；忽略相对路径依赖 cwd。

**Back to Project**：Golden、ONNX、audit JSON 与 fixture 路径都由 Path 组合。

## 8. P3：暂时不需要系统学习

当前目标是 inference/deployment 阅读，下列内容暂不值得投入系统学习时间：

- Metaclass 的完整实现细节。Registry 会使用相关机制，但先理解输入输出即可。
- Descriptor protocol 的完整理论。先会使用 property/bound method 即可。
- Async/await、threading、multiprocessing 内部机制。
- Python GUI/Web 框架。
- 高级 generator pipeline；只需理解 contextmanager 中的单次 `yield`。
- 完整 autograd backward 数学；当前部署阅读主要关注 forward/custom-op boundary。
- Python packaging 发布流程；只需分辨当前仓库 package 与环境依赖。

## 9. 严格区分五个知识层级

### `[Python Language]`

```text
name/object/reference
list/tuple/dict/set
function/argument/*args/**kwargs
class/self/inheritance/super
callable/__call__/__getitem__
for/if/comprehension
decorator/exception/with protocol
```

### `[Python Standard Library]`

```text
copy.deepcopy
pathlib.Path
contextlib.contextmanager
collections.Counter/deque
```

### `[PyTorch]`

```text
Tensor
nn.Module
nn.Module.__call__
forward
no_grad/eval/cuda
reshape/permute/stack/cat/contiguous
autograd.Function.apply
```

### `[MMCV/MMDetection]`

```text
Registry/build_from_cfg
BaseModule
TransformerLayerSequence
auto_fp16/force_fp32
build_attention/build_transformer_layer_sequence
DETRHead/MVXTwoStageDetector
```

### `[BEVFormer]`

```text
BEVFormer
prev_frame_info / prev_bev
BEVFormerHead
PerceptionTransformer
BEVFormerEncoder
TemporalSelfAttention
SpatialCrossAttention
deployment-only wrappers
```

看到一行不懂时，先标记它属于哪层。Python 语法问题不要用 BEVFormer 算法解释，MMCV builder 问题也不要在 PyTorch Tensor API 中寻找答案。

## 10. Registry / build_from_cfg 的依赖分解

这是 OpenMMLab 最大障碍，但不应该第一个学。

依赖顺序：

```text
dict
↓
function keyword arguments
↓
class is an object
↓
class(**config) creates instance
↓
decorator can register class object
↓
Registry maps string → class object
↓
build_from_cfg extracts type and calls class(**remaining_cfg)
```

项目实例：

```python
@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    ...
```

配置概念：

```python
dict(type='SpatialCrossAttention', embed_dims=256, num_cams=6)
```

构建概念：

```text
'SpatialCrossAttention'
       ↓ Registry lookup
SpatialCrossAttention class object
       ↓ **remaining_config
SpatialCrossAttention instance
```

仓库显式 `build_from_cfg`：

```text
projects/mmdet3d_plugin/datasets/builder.py
projects/mmdet3d_plugin/datasets/samplers/sampler.py
projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py
```

BEVFormer attention/transformer 更多通过 MMCV helper 间接构建：

```text
build_attention
build_transformer_layer_sequence
build_bbox_coder
```

## 11. 由依赖关系生成的主学习路线

### Stage 1：名字、对象、引用、容器

为什么最先学：class、function、Tensor 都建立在 object/reference 上；metadata 又以嵌套容器组织。

学习节点：

```text
name / assignment
object / type / identity
reference
mutable / immutable
list / tuple / dict
index / key
nested container
```

项目练习（只解释 Python，不解释算法）：

```python
self.prev_frame_info = {
    'prev_bev': None,
    'scene_token': None,
    'prev_pos': 0,
    'prev_angle': 0,
}
```

```python
img_metas[0][0]['can_bus'][:3]
```

```python
new_prev_bev, bbox_results = self.simple_test(...)
```

完成标准：能逐层说出 object 类型、是否 mutable、名字是否引用同一对象。

### Stage 2：函数、参数、返回值

依赖 Stage 1：参数和返回值本质是名字对对象的绑定。

学习节点：

```text
function
local scope
parameter vs argument
positional / keyword / default
return / tuple unpacking
```

项目练习：

```python
def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
```

```python
new_prev_bev, bbox_pts = self.simple_test_pts(
    img_feats, img_metas, prev_bev, rescale=rescale)
```

完成标准：能将每个 call argument 映射到定义处 parameter。

### Stage 3：参数 packing/unpacking

依赖 Stage 2 + dict/tuple。

学习节点：

```text
iterable unpacking
*args
**kwargs
definition packing
call-site unpacking
```

项目练习：

```python
def forward(self, return_loss=True, **kwargs):
    return self.forward_test(**kwargs)
```

```python
super(BEVFormerHead, self).__init__(
    *args, transformer=transformer, **kwargs)
```

```python
self.encoder(..., prev_bev=prev_bev, shift=shift, **kwargs)
```

完成标准：能手写某一层 `kwargs` 可能包含的 key，并继续追踪 `img_metas`。

### Stage 4：Class/object/self/method

依赖 Stage 1 + Stage 2。

学习节点：

```text
class / instance
attribute / method
self
method binding
__init__
```

项目练习：

```python
class BEVFormer(MVXTwoStageDetector):
```

```python
self.video_test_mode = video_test_mode
self.prev_frame_info = {...}
```

```python
self.pts_bbox_head(...)
```

完成标准：能区分 class object、model instance、bound method 和 instance attribute。

### Stage 5：Inheritance、super、callable、nn.Module

依赖 Stage 4 + Stage 3。

学习节点：

```text
inheritance / override
super / MRO basics
callable / __call__
nn.Module.__call__ / forward
```

项目练习：

```python
class BEVFormerHead(DETRHead):
    ...
    super(BEVFormerHead, self).__init__(*args, **kwargs)
```

```python
outputs = self.transformer(...)
```

```python
output = layer(...)
```

完成标准：能从任意 `module(...)` 找到 object 的实际 class 和对应 `forward()`。

### Stage 6：Tensor layout reading

依赖 indexing + method call + tuple。

学习节点：

```text
shape / dtype / device
reshape / view
permute / transpose
unsqueeze / squeeze
cat / stack
contiguous
detach / cpu / numpy
```

项目练习：

```python
bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
```

```python
prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
    bs * 2, len_bev, -1)
```

```python
attention_weights = attention_weights.permute(...).reshape(...).contiguous()
```

完成标准：每行都能写出 before/after shape 和 layout 名称。

### Stage 7：高频工程表达

依赖容器 + function。

学习节点：

```text
for / if / enumerate / zip / range
comprehension
append / setdefault
isinstance / hasattr / getattr
deepcopy
```

项目练习：

```python
for lvl, feat in enumerate(mlvl_feats):
```

```python
consumers.setdefault(name, []).append(index)
```

```python
if isinstance(child, SpatialCrossAttention):
```

完成标准：能把一行表达式展开成多行基础操作，并说明是否修改原对象。

### Stage 8：工程控制机制与 OpenMMLab

依赖前面全部阶段，尤其 function object、inheritance 和 exception。

学习节点：

```text
module/package/import
exception / raise / assert
with / context manager / try-finally
decorator / closure basics
Registry / build_from_cfg
auto_fp16 / force_fp32
```

项目练习：

```python
@DETECTORS.register_module()
class BEVFormer(...):
```

```python
with deployment_export_overrides(self.model), \
        static_spatial_cross_attention_override(self.model):
```

```python
try:
    yield
finally:
    child.forward = original
```

完成标准：能解释 config 如何变成 object，以及 deployment override 为什么不改变普通 inference。

## 12. 项目调用链综合练习

在完成 Stage 1–8 后，完整解释：

```text
model(...)
↓ Python callable syntax
nn.Module.__call__
↓ PyTorch hook/dispatch
BEVFormer.forward
↓ return_loss + **kwargs
BEVFormer.forward_test
↓ nested metadata + temporal dict state
BEVFormer.simple_test
↓ module attributes
BEVFormerHead.forward
↓ Registry-built module call
PerceptionTransformer.forward
↓ get_bev_features / encoder / decoder
```

每一步都必须写：

```text
object name
actual class
method file
arguments
return type/structure
属于 Python / PyTorch / MMCV / BEVFormer 哪一层
```

## 13. 最终源码阅读自测

### 基础对象与容器

1. `a = b` 是否复制 object？什么时候需要 deepcopy？
2. 逐层解释 `img_metas[0][0]['can_bus'][-1]`。
3. `return new_prev_bev, bbox_list` 返回什么类型？
4. `tmp[..., 4]` 与 `tmp[..., 4:5]` shape 有何区别？

### 参数流

5. 定义处和调用处的 `**kwargs` 分别做 packing 还是 unpacking？
6. `kwargs['img_metas']` 最早从哪里传入？
7. 为什么 positional argument 与 keyword argument 混用时仍能正确绑定？

### 对象与调用

8. `self.pts_bbox_head` 是什么 object，由哪个 class 创建？
9. 为什么当前 class 找不到的 attribute 可能来自父类？
10. `self.transformer(...)` 为什么进入 `PerceptionTransformer.forward()`？
11. 为什么不建议直接调用 `module.forward(...)`？

### Tensor

12. 写出 BEV query 从 `[N,C]` 到 `[N,B,C]` 再到 `[B,N,C]` 的操作。
13. `stack` 和 `cat` 在 temporal queue 中分别做什么？
14. `permute` 后为什么可能需要 `contiguous()`？
15. 为什么 `[2500,1,256]` 与 `[1,2500,256]` 不能自动视为相同 boundary？

### 工程机制

16. `@DETECTORS.register_module()` 是注释还是可执行机制？
17. Registry 如何把 `type='BEVFormer'` 变为 object？
18. `with ... try/finally` 如何保证 deployment patch 异常时也恢复？
19. `hasattr/getattr` 在 symbolic patch 中为什么必要？
20. 如何判断一行代码属于 Python、PyTorch、MMCV 还是 BEVFormer？

能独立回答 16 项以上，并能从 `model(...)` 追到 encoder，说明已经达到“可独立阅读 inference/deployment 源码”的目标。此时无需先补完整 Python 百科，应继续通过真实源码维护熟练度。

## 14. P0 / P1 / P2 汇总

### P0（9）

1. Name/object/reference/mutation
2. List/tuple/dict/nested access
3. Indexing/slicing/ellipsis
4. Function/parameter/argument/return
5. Positional/keyword/default/`*args`/`**kwargs`
6. Class/instance/attribute/method/self/`__init__`
7. Inheritance/override/`super()`
8. Callable/method binding/`__call__`
9. Tensor shape/layout transformations

### P1（8）

1. `for`/`if`/`range`/`enumerate`/`zip`
2. List/dict comprehension
3. Mutable container methods：`append`/`setdefault`
4. `copy.deepcopy`
5. `isinstance`/`hasattr`/`getattr`
6. `with`/context manager/`try-finally`
7. Decorator
8. Module/package/import/relative import

### P2（4）

1. Lambda/closure/default capture
2. `staticmethod`/autograd `Function.apply`
3. Property/`__getitem__`
4. `pathlib.Path`

## 15. 不明显但必须补齐的前置基础

这些基础在源码里不一定以显眼关键字出现，但没有它们无法理解上层机制：

| 前置基础 | 为什么必须补 |
| --- | --- |
| Name 与 object 的区别 | 理解 assignment、attribute、parameter binding、module tree |
| Reference 与 identity | 理解原地 metadata 修改、hook patch、deepcopy |
| Mutable vs immutable | 理解 list/dict/Tensor 更新和 context restore |
| Local scope | 理解 closure、decorator、wrapper 临时变量 |
| Function 也是 object | 理解 decorator、forward monkey patch、callback |
| Bound method | 理解 `self` 为什么自动传入及 `types.MethodType` |
| Iterable/unpacking | 理解 `*args`、tuple return、for/zip |
| Boolean truth/condition | 理解 `if prev_bev is not None`、gate、validation |
| Attribute lookup/MRO basics | 理解父类提供的 head/module 状态 |
| Exception propagation | 理解 fail-closed export 和 finally 恢复 |

它们应在对应 Stage 中学习，不需要单独按传统教材扩展为完整章节。
