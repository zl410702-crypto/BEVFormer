# Python for AI Source Reading

## 使用说明

本文只讲 Python。本仓库中的 BEVFormer、PyTorch、MMCV 用法放在独立文档 `docs/bevformer_python_usage_index.md`。

目标不是学完 Python，而是获得阅读、判断和小幅修改 AI 工程源码所需的语言能力。知识顺序按依赖关系排列：后面的章节默认掌握前面的对象、容器、函数和 class 模型。

优先级：

- **P0**：不掌握就无法稳定阅读主调用链。
- **P1**：工程高频，近期掌握。
- **P2**：先认识，遇到时查询。

边界标签：

- `[Python Language]`：Python 语言机制。
- `[Python Standard Library]`：随 Python 提供的库。
- `[PyTorch]`、`[MMCV/MMDetection]`、`[BEVFormer]`：第三方库或项目层，不在本文展开。

---

# 第一部分：Python 最基础的运行模型

## 1. Name、object、type、assignment 与 reference `[Python Language]` `P0`

### 1. 这是什么

Python 中，name（名字）通过 assignment（赋值）绑定到 object（对象）。对象有 type、identity 和 value。多个名字可以引用同一个对象。

`is` 比较是否为同一个对象；`==` 比较值是否相等。Mutable 对象可以原地改变，immutable 对象不能原地改变其值。

### 2. 为什么 Python 需要它

名字让程序能够引用对象；共享引用避免每次赋值都复制大型对象；identity 和 mutability 决定一次修改会不会从另一个名字看到。

### 3. 最简单的例子

```python
a = [1, 2]
b = a
c = [1, 2]

print(type(a).__name__)
print(a is b)
print(a == c)
print(a is c)

b.append(3)
print(a)

x = 10
y = x
y = 20
print(x, y)
```

### 4. 输出结果

```text
list
True
True
False
[1, 2, 3]
10 20
```

### 5. 逐行解释

- `a = [1, 2]`：创建 list object，把名字 `a` 绑定到它。
- `b = a`：让 `b` 引用同一个 list，没有复制。
- `c = [1, 2]`：创建另一个值相等的 list。
- `a is b`：identity 相同。
- `a == c`：值相等。
- `b.append(3)`：原地修改 mutable list，因此通过 `a` 也能看到。
- int 是 immutable；`y = 20` 让 `y` 改绑到新 int，不改变 `x` 引用的 10。

### 6. 心智模型

```text
a ─┐
   ├──> list object [1, 2]
b ─┘

c ─────> another list object [1, 2]

x ─────> int object 10
y ─────> int object 20
```

### 7. 【学习重点】

1. Python name 不是“装值的盒子”，而是对象引用。
2. `b = a` 默认不复制对象。
3. `is` 比较 identity，`==` 比较 value。
4. List/dict/set 通常 mutable；int/float/bool/str/tuple 通常 immutable。
5. 原地修改共享对象会从所有引用看到。

### 8. 常见误区

- 用 `is` 比较数字或字符串值。
- 认为重新赋值与原地修改相同。
- 认为函数参数会自动复制对象。

### 9. 【掌握标准】

看到 `b = a`、`x += y` 或 `obj.method()` 时，能先判断对象是否共享、是否 mutable、操作是原地修改还是重新绑定。

### 10. 小练习

1. 预测：`a=[1]; b=a; b[0]=9; print(a)`。
2. 判断：`a=[1]; b=[1]` 中 `a == b` 和 `a is b`。
3. 画出三个 name 引用同一个 dict 的 ASCII 图。

---

# 第二部分：基本数据类型

## 2. int、float、bool、str 与 None `[Python Language]` `P0`

### 1. 这是什么

`int` 表示整数，`float` 表示浮点数，`bool` 表示真假，`str` 表示文本。`None` 是唯一的“无值/不存在”单例对象，其 type 是 `NoneType`。

### 2. 为什么 Python 需要它

它们表达计数、数值、条件、名称和可选状态。`None` 能区分“尚未提供对象”与合法数值 0、空 list 等。

### 3. 最简单的例子

```python
count = 3
score = 0.75
enabled = True
name = 'camera'
history = None

print(type(count).__name__)
print(type(score).__name__)
print(enabled and score > 0.5)
print(name.upper())
print(history is None)
```

### 4. 输出结果

```text
int
float
True
CAMERA
True
```

### 5. 逐行解释

- 每个 literal 创建相应类型对象。
- `and` 组合条件。
- `name.upper()` 调用 str method，返回新字符串。
- `history is None` 检查对象是否就是唯一的 `None`。

### 6. 心智模型

```text
None = 一个专门表示“没有对象”的单例

optional value
 ├─ real object
 └─ None
```

### 7. 【学习重点】

1. `bool` 只有 `True/False`，但条件表达式也会进行 truth testing。
2. `None` 不等于 0、False、空字符串或空 list。
3. 检查 None 使用 `is None` / `is not None`。
4. String immutable，method 通常返回新 string。
5. Float 运算存在舍入误差。

### 8. 常见误区

- 写 `x == None` 而不是 `x is None`。
- 把 `if x:` 与 `if x is not None:` 当成完全相同。
- 期待浮点数逐位精确。

### 9. 【掌握标准】

能够解释 `if previous is not None:` 与 `if previous:` 的差异，并选择正确写法表示可选对象。

### 10. 小练习

1. 比较 `None`、0、False、`[]` 是否相等以及 truth value。
2. 预测 `bool('False')` 的结果。
3. 写一个变量表示“尚无历史数据”。

---

# 第三部分：容器

## 3. list、tuple、dict 与 set `[Python Language]` `P0`

### 1. 这是什么

- `list`：有序、可变序列。
- `tuple`：有序、通常不可变的固定结构。
- `dict`：key 到 value 的可变映射。
- `set`：无重复元素的可变集合。

### 2. 为什么 Python 需要它

容器把多个对象组织成 batch、配置、metadata、返回结构或唯一成员集合。

### 3. 最简单的例子

```python
items = [1, 2]
items.append(3)
items.extend([4, 5])

shape = (1, 6, 256)

meta = {'scene': 'A', 'frame': 0}
print(meta['scene'])
print(meta.get('missing', 'default'))
meta.setdefault('tokens', []).append('t0')

names = {'image', 'bev', 'image'}

print(items)
print(shape)
print(list(meta.items()))
print(sorted(names))
```

### 4. 输出结果

```text
A
default
[1, 2, 3, 4, 5]
(1, 6, 256)
[('scene', 'A'), ('frame', 0), ('tokens', ['t0'])]
['bev', 'image']
```

### 5. 逐行解释

- `append(3)` 把一个对象作为一个元素加入 list。
- `extend([4,5])` 遍历输入，把两个元素分别加入。
- Tuple 用圆括号表示固定 shape。
- `meta[key]` 缺 key 会抛 `KeyError`；`get` 可返回默认值。
- `setdefault` 在 key 缺失时保存默认 list，再返回该 list供 `append`。
- Set 自动去重；示例排序只是为了稳定显示。

### 6. 心智模型

```text
list:  index 0 → object
       index 1 → object

dict:  key 'scene' → value 'A'
       key 'frame' → value 0

set:   membership, no duplicate entries
```

### 7. 【学习重点】

1. List、dict、set 是 mutable；tuple 通常作为不可变结构使用。
2. `append` 加一个对象；`extend` 加入 iterable 的每个元素。
3. Dict 使用 key/value；掌握 `[]`、`get`、`items`、`keys`、`values`、`setdefault`。
4. `**kwargs` 本质与 dict 密切相关。
5. 容器可以嵌套并共享内部 mutable 对象。

### 8. 常见误区

- `items.append([3,4])` 与 `items.extend([3,4])` 混淆。
- 认为 tuple 内所有引用对象都绝对不能变化；tuple 本身不可改，但内部可引用 mutable 对象。
- 依赖 set 的展示顺序。
- 把空 `{}` 当 set；它是 dict，空 set 要写 `set()`。

### 9. 【掌握标准】

能看到任意嵌套表达式时逐层写出 list/tuple/dict；能选择 `append`、`extend`、`get`、`setdefault`。

### 10. 小练习

1. 预测 `[1].append([2,3])` 和 `[1].extend([2,3])`。
2. 把 `{'a': [1]}` 中的 list 追加 2。
3. 用 set 找出 `['a','b','a']` 的唯一值。

---

# 第四部分：Index 与 Slice

## 4. Indexing、negative indexing、slicing 与 nested indexing `[Python Language]` `P0`

### 1. 这是什么

Indexing 用一个位置或 key 取对象；slicing 用 `start:end:step` 取序列的一段。索引从 0 开始，负数从末尾计数，slice 的 end 不包含。

### 2. 为什么 Python 需要它

它允许读取序列局部、最后一个元素、嵌套 metadata 或按步长抽样。

### 3. 最简单的例子

```python
values = [10, 20, 30, 40, 50]
record = [[{'motion': [1.0, 2.0, 3.0]}]]

print(values[0])
print(values[-1])
print(values[1:4])
print(values[::2])
print(record[0][0]['motion'][-1])
```

### 4. 输出结果

```text
10
50
[20, 30, 40]
[10, 30, 50]
3.0
```

### 5. 逐行解释

- `[0]` 取第一个元素。
- `[-1]` 取最后一个元素。
- `[1:4]` 取 1、2、3，不含 4。
- `[::2]` 省略 start/end，每隔两个取一次。
- Nested access 每一对方括号都对上一层结果继续访问。

### 6. 心智模型

```text
record
  [0] → list
          [0] → dict
                  ['motion'] → list
                                  [-1] → 3.0
```

### 7. 【学习重点】

1. Index 从 0 开始。
2. Slice end 永远不包含。
3. `x[4]` 返回一个元素；`x[4:5]` 返回长度 1 的序列。
4. 多层索引必须逐层判断对象类型。
5. NumPy/PyTorch 也使用 `[]`，但具体 view/copy 和维度规则属于第三方库。

### 8. 常见误区

- 把 `[1:4]` 读成四个元素。
- 忘记负索引。
- 把 dict key access 与 list index 混成同一机制。

### 9. 【掌握标准】

能在不运行代码的情况下预测普通 list slice，并把嵌套 access 展开成逐层临时变量。

### 10. 小练习

1. 预测 `list(range(6))[1:-1:2]`。
2. 把 `data[0][1]['x']` 展开成三行。
3. 解释 `x[-2:]`。

---

# 第五部分：控制流

## 5. if/elif/else、for/while、break/continue 与 range `[Python Language]` `P1`

### 1. 这是什么

条件语句选择路径；循环重复执行；`break` 结束当前循环；`continue` 跳过本次；`range` 生成整数序列。

### 2. 为什么 Python 需要它

程序需要根据状态分支，并遍历 frame、camera、layer 或配置项。

### 3. 最简单的例子

```python
total = 0
for i in range(6):
    if i == 4:
        break
    if i % 2 == 0:
        continue
    total += i

count = 2
while count > 0:
    count -= 1

print(total)
print(count)
```

### 4. 输出结果

```text
4
0
```

### 5. 逐行解释

- `range(6)` 产生 0..5。
- i=0、2 时 continue；i=1、3 加入 total；i=4 时 break。
- while 在条件为真时重复。

### 6. 心智模型

```text
condition?
 ├─ true  → branch/body
 └─ false → next branch/exit
```

### 7. 【学习重点】

1. Python 用缩进确定代码块。
2. `if x is not None` 与 `if x` 不相同。
3. `range(n)` 不包含 n。
4. break/continue 只影响当前最近一层循环。
5. 遍历 object 时要知道它的 iterable 元素类型。

### 8. 常见误区

- 忽略缩进。
- 在嵌套循环中误判 break 的范围。
- 修改正在遍历的 list。

### 9. 【掌握标准】

能手工走完短循环、预测每个分支，并解释循环变量每次是什么对象。

### 10. 小练习

1. 预测 `for i in range(1,5,2)` 的 i。
2. 写一个跳过 None 的循环。
3. 把简单 for 改成 while。

---

# 第六部分：函数

## 6. def、function、parameter、argument、return 与 scope `[Python Language]` `P0`

### 1. 这是什么

`def` 创建 function object。Parameter 是定义处的名字；argument 是调用时传入的对象。Return 把对象交回调用者。Scope 决定名字在哪些代码区域可见。

### 2. 为什么 Python 需要它

函数封装重复逻辑、定义输入输出 contract，并建立局部名字空间。

### 3. 最简单的例子

```python
factor = 10

def scale(value, amount=2):
    result = value * amount
    return result, amount

output, used = scale(3, amount=4)
print(output)
print(used)
print(factor)
```

### 4. 输出结果

```text
12
4
10
```

### 5. 逐行解释

- `scale` 绑定到 function object。
- `value`、`amount` 是 parameters，amount 默认 2。
- 3 是 positional argument，4 通过 keyword `amount` 传入。
- `result` 是 local name。
- `return result, amount` 返回 tuple。
- 调用者用 tuple unpacking 绑定两个名字。

### 6. 心智模型

```text
arguments
   ↓ parameter binding
function local scope
   ↓ return object (possibly tuple)
caller
```

### 7. 【学习重点】

1. 区分 parameter 与 argument。
2. 位置参数按顺序，关键字参数按名字。
3. Default 只在调用未提供该参数时使用。
4. `return a, b` 实际返回 tuple。
5. Function 是 object，可赋值、传递、包装。
6. Local name 通常在函数外不可见。

### 8. 常见误区

- 使用 mutable default，如 `def f(items=[])`，导致多次调用共享对象。
- 认为传参自动复制 object。
- 忽略函数可能返回 None（没有显式 return）。

### 9. 【掌握标准】

看到函数定义和调用，能逐项完成 argument-to-parameter binding，判断默认值和返回结构。

### 10. 小练习

1. 预测 `scale(5)`。
2. 把 `a,b = func()` 写成先接收 tuple 再索引。
3. 判断一个 local name 在函数外是否存在。

---

# 第七部分：`*` 与 `**`

## 7. `*args`、`**kwargs`、packing 与 unpacking `[Python Language]` `P0`

### 1. 这是什么

定义处的 `*args` 把额外 positional arguments 打包成 tuple；`**kwargs` 把额外 keyword arguments 打包成 dict。调用处的 `*iterable` 和 `**mapping` 做相反的 unpacking。

### 2. 为什么 Python 需要它

它允许 wrapper、继承层和通用框架转发当前层不需要逐个列出的参数。

### 3. 最简单的例子

```python
def foo(a, b=0, *args, **kwargs):
    print(a)
    print(b)
    print(args)
    print(kwargs)

foo(1, 2, 3, 4, mode='test')

values = [10, 20]
options = {'mode': 'eval'}
foo(*values, **options)
```

### 4. 输出结果

```text
1
2
(3, 4)
{'mode': 'test'}
10
20
()
{'mode': 'eval'}
```

### 5. 逐行解释

- 第一次调用：1→a，2→b，3/4→args tuple，mode→kwargs dict。
- `*values` 展开成两个 positional arguments。
- `**options` 展开成 `mode='eval'`。
- 第二次没有额外位置参数，所以 args 是空 tuple。

### 6. 心智模型

```text
definition:
extra positional ──pack──> args tuple
extra keywords   ──pack──> kwargs dict

call:
list/tuple ──* unpack──> positional arguments
dict       ──** unpack─> keyword arguments
```

### 7. 【学习重点】

1. `args` 是 tuple，`kwargs` 是 dict；名字可变，星号语义才关键。
2. 定义处是 packing，调用处是 unpacking。
3. `**kwargs` 中每个 key 必须是可用的参数名字符串。
4. 多层转发时必须追踪当前 kwargs 的 key 集合。
5. `*` 也可用于普通 iterable unpacking，但不要与乘法混淆。

### 8. 常见误区

- 认为 kwargs 是框架专有类型。
- 不区分定义处和调用处。
- 同一个 keyword 既显式传又在 `**options` 中传，导致 duplicate argument error。

### 9. 【掌握标准】

能够手工写出一次复杂调用后 a、b、args、kwargs 的精确内容；能把 `func(*values, **options)` 展开成普通调用。

### 10. 小练习

1. 预测 `foo(1, mode='x')`。
2. 把 `foo(*[1,2,3])` 展开。
3. 写一个 wrapper，接收并原样转发 `*args, **kwargs`。

---

# 第八部分：Class 与 Object

## 8. class、instance、self、attribute、method 与 `__init__` `[Python Language]` `P0`

### 1. 这是什么

Class 描述一类对象的状态和行为；调用 class 创建 instance。Attribute 是对象上的名字；method 是定义在 class 中、调用时绑定 instance 的函数。`__init__` 初始化新 instance。

### 2. 为什么 Python 需要它

它把相关状态与行为组合，允许多个独立 instance 共享 class 行为但保存各自状态。

### 3. 最简单的例子

```python
class Car:
    def __init__(self, speed):
        self.speed = speed

    def run(self):
        return self.speed

car = Car(10)
print(car.run())
print(car.speed)
```

### 4. 输出结果

```text
10
10
```

### 5. 逐行解释

- `class Car` 创建 class object。
- `Car(10)` 创建 Car instance，并调用 `__init__`。
- `self` 指当前 instance；它不是 Python keyword，只是强约定。
- `self.speed = speed` 保存 instance attribute。
- `car.run()` 获取 bound method；Python 自动把 car 作为 self。

### 6. 心智模型

```text
Car class
    ↓ instantiate
car object
├── speed = 10
└── run()  (behavior from Car class)
```

理解模型：

```text
car.run()
≈ Car.run(car)
```

这是用于阅读的基础模型，不展开 descriptor 的全部细节。

### 7. 【学习重点】

1. Class 与 instance 必须区分。
2. `self` 表示当前 instance，不是关键字。
3. `self.xxx` 是 attribute lookup/assignment。
4. `obj.method()` 调用时 self 自动绑定。
5. `__init__` 初始化对象，不等同于创建对象的全部底层过程。
6. 不同 instance 有独立 instance attributes。

### 8. 常见误区

- 调用 `car.run(car)`，重复传 self。
- 在 class 中找不到 instance runtime attribute 就认为不存在。
- 把 class attribute 与 instance attribute 混淆。

### 9. 【掌握标准】

看到 `obj.foo(x)` 能立即判断：obj 是 object；foo 经 attribute lookup 得到 bound method；调用时 self 对应 obj，x 绑定下一个 parameter。

### 10. 小练习

1. 创建两个 Car，验证 speed 相互独立。
2. 用 `Car.run(car)` 调用并比较结果。
3. 标出示例中的 class object、instance、attribute、method。

---

# 第九部分：Inheritance

## 9. Parent、Child、inheritance、override 与 `super()` `[Python Language]` `P0`

### 1. 这是什么

Child class 可继承 Parent 的行为。Child 可直接使用未重新定义的方法，也可 override 同名方法。`super()` 从当前 class 的继承顺序继续调用父类实现。

### 2. 为什么 Python 需要它

它复用共同实现，让子类只描述差异，并允许框架定义统一接口。

### 3. 最简单的例子

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return 'sound'

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)

    def speak(self):
        return 'woof'

    def describe(self):
        return self.name

dog = Dog('Milo')
print(dog.describe())
print(dog.speak())
```

### 4. 输出结果

```text
Milo
woof
```

### 5. 逐行解释

- `Dog(Animal)` 声明继承。
- Dog 没定义 name 的保存逻辑，`super().__init__` 调 Animal 初始化。
- `describe` 使用父类初始化的 attribute。
- Dog 的 `speak` override 父类版本。

### 6. 心智模型

```text
Dog instance
  lookup method/attribute
      ↓ Dog class
      ↓ Animal class
      ↓ object
```

### 7. 【学习重点】

1. 当前 class 找不到 method/attribute 不代表不存在，要检查父类。
2. 子类可继承、增加或 override 行为。
3. `super()` 沿 MRO 继续，不只是“固定调用某个名字的父类”。
4. 父类 `__init__` 常负责关键状态初始化。
5. 多继承只需先知道 MRO 存在，暂不深入算法。

### 8. 常见误区

- 忘记父类初始化。
- 认为 override 会修改父类本身。
- 只在当前文件搜索所有 attribute。

### 9. 【掌握标准】

看到 `class Child(Parent)` 和 `super()`，能画出基本继承链，判断 method 是继承还是 override，并向父类寻找缺失状态。

### 10. 小练习

1. 删除 Dog.speak，预测输出。
2. 给 Animal 增加方法，让 Dog 直接调用。
3. 用 `Dog.__mro__` 查看查找顺序。

---

# 第十部分：Attribute Lookup 与 Method Binding

## 10. `obj.xxx` 与 bound method `[Python Language]` `P0`

### 1. 这是什么

`obj.xxx` 触发 attribute lookup。基础阅读模型是先看 instance，再看其 class，然后沿 parent classes 查找。Class 中的普通 function 经 instance 访问时成为 bound method。

### 2. 为什么 Python 需要它

它让对象同时使用 instance state 和 class/shared behavior，并支撑继承。

### 3. 最简单的例子

```python
class Reader:
    language = 'Python'

    def show(self, topic):
        return f'{self.language}: {topic}'

r = Reader()
method = r.show
print(r.language)
print(method('objects'))
print(method.__self__ is r)
```

### 4. 输出结果

```text
Python
Python: objects
True
```

### 5. 逐行解释

- `language` 在 class 上。
- `r.language` 经 instance lookup 找到 class attribute。
- `r.show` 返回绑定到 r 的 method object。
- 保存 method 后调用，self 仍是 r。

### 6. 心智模型

```text
r.show
  ↓ lookup Reader.show function
  ↓ bind r as self
bound method
```

### 7. 【学习重点】

1. `obj.xxx` 是 lookup，不保证 xxx 存在于当前源码文件。
2. 查找可继续到 class 和 parent class。
3. Bound method 已保存 self。
4. Attribute 可在 runtime 被替换；工程 patch 常利用这一点。
5. 不需要先深入 descriptor protocol。

### 8. 常见误区

- 把 `obj.method` 与 `Class.method` 当成相同对象。
- 只搜索文本 `self.xxx =`，忽略父类或 class attribute。
- 不理解保存 bound method 后仍绑定原 object。

### 9. 【掌握标准】

能解释 `obj.foo` 的可能来源，并区分 attribute value、unbound function 与 bound method。

### 10. 小练习

1. 输出 `Reader.show` 和 `r.show` 的类型描述。
2. 将 `r.language` 设为 `'C++'`，预测 lookup 结果。
3. 保存 `method=r.show` 后再调用。

---

# 第十一部分：Callable 与 `__call__`

## 11. Callable object `[Python Language]` `P0`

### 1. 这是什么

定义 `__call__` 的对象可以使用 `object(...)` 语法调用。普通 function 也是 callable object。

### 2. 为什么 Python 需要它

它让带状态对象提供函数式接口，适合 transform、模型、loss 和 callback。

### 3. 最简单的例子

```python
class Scale:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

s = Scale(2)
print(s(3))
print(callable(s))
```

### 4. 输出结果

```text
6
True
```

### 5. 逐行解释

- `s` 是 Scale instance，不是普通 function。
- `s(3)` 调用 `s.__call__(3)`。
- `__call__` 使用 instance state factor。

### 6. 心智模型

```text
s(3)
 ↓ callable protocol
s.__call__(3)
 ↓
6
```

### 7. 【学习重点】

1. `object(...)` 不一定是普通 function，也可能是 class 或 callable instance。
2. 先确认 object 的 actual class，再找 `__call__` 来源。
3. `__call__` 也可能继承自父类。
4. PyTorch `nn.Module` 利用 Python callable protocol，但具体 hook/forward 流程属于 PyTorch。

### 8. 常见误区

- 看见括号就认定左边是函数。
- 在子类文件中找不到 `__call__` 就认为无法调用。
- 把 class 调用（实例化）与 instance `__call__` 混淆。

### 9. 【掌握标准】

看到 `thing(x)` 能列出三种可能：function、class、callable instance，并知道如何用 `type(thing)` 和继承链确认。

### 10. 小练习

1. 给 Scale 增加调用次数状态。
2. 比较 `Scale(2)` 和 `s(3)` 各自调用什么机制。
3. 打印 `type(s)` 与 `callable(s)`。

---

# 第十二部分：工程高频循环工具

## 12. `enumerate` 与 `zip` `[Python Language]` `P1`

### 1. 这是什么

`enumerate(iterable)` 同时产生 index 和 value；`zip(a,b,...)` 按位置组合多个 iterable。

### 2. 为什么 Python 需要它

避免手工维护 index，并让多个对应序列并行遍历。

### 3. 最简单的例子

```python
names = ['front', 'rear']
scores = [0.9, 0.8]

for index, name in enumerate(names):
    print(index, name)

for name, score in zip(names, scores):
    print(name, score)
```

### 4. 输出结果

```text
0 front
1 rear
front 0.9
rear 0.8
```

### 5. 逐行解释

- Enumerate 每次给 `(index,value)` tuple，再解包。
- Zip 每次给 `(name,score)` tuple。

### 6. 心智模型

```text
enumerate: value → (position, value)
zip:       a[i], b[i] → (a[i], b[i])
```

### 7. 【学习重点】

1. 默认 index 从 0 开始，可指定 start。
2. Zip 默认在最短 iterable 结束。
3. Loop target 常使用 tuple unpacking。
4. Zip 不自动验证两个输入长度一致。

### 8. 常见误区

- 认为 enumerate 修改原 list。
- 认为 zip 返回完整 list；Python 3 中它返回 lazy iterator。

### 9. 【掌握标准】

能把 enumerate/zip 循环展开成 index-based 普通循环，并预测每次绑定值。

### 10. 小练习

1. 使用 `enumerate(names, start=1)`。
2. Zip 两个不同长度 list，观察结果。
3. 将 zip 循环写成 `range(len(...))` 版本。

---

# 第十三部分：Comprehension

## 13. List、dict、set comprehension `[Python Language]` `P1`

### 1. 这是什么

Comprehension 用表达式、for 和可选 if 简洁构造容器。

### 2. 为什么 Python 需要它

它把“遍历→转换/过滤→收集”表达为一个容器构造式。

### 3. 最简单的例子

```python
nums = [1, 2, 3, 4]

doubled = [x * 2 for x in nums]
mapping = {x: x * 2 for x in nums if x % 2 == 0}
remainders = {x % 2 for x in nums}

print(doubled)
print(mapping)
print(remainders == {0, 1})
```

等价普通 for：

```python
doubled_long = []
for x in nums:
    doubled_long.append(x * 2)

mapping_long = {}
for x in nums:
    if x % 2 == 0:
        mapping_long[x] = x * 2

remainders_long = set()
for x in nums:
    remainders_long.add(x % 2)
```

### 4. 输出结果

```text
[2, 4, 6, 8]
{2: 4, 4: 8}
True
```

### 5. 逐行解释

- 最前面是要收集的 expression。
- `for x in nums` 提供每个 x。
- Dict comprehension 使用 `key: value`。
- `if` 只保留满足条件的元素。

### 6. 心智模型

```text
source iterable
 → for binding
 → optional filter
 → output expression
 → new container
```

### 7. 【学习重点】

1. 遇到复杂 comprehension，先展开成普通 for。
2. Comprehension 创建新容器，但其中元素可能仍是共享对象引用。
3. 区分 list `[]`、dict `{k:v}` 和 set `{expr}`。
4. 不要为了简短写三层难读 comprehension。

### 8. 常见误区

- 从左到右机械读，忽略 for 是数据来源。
- 把 set comprehension 当 dict。
- 认为新 list 意味内部对象也 deep copied。

### 9. 【掌握标准】

能把任意一层 comprehension 展开为普通循环，也能将简单循环压缩回来。

### 10. 小练习

1. 展开 `[x for x in nums if x>2]`。
2. 写一个 value 为字符串长度的 dict comprehension。
3. 判断 `{x for x in [1,1,2]}` 输出。

---

# 第十四部分：对象复制

## 14. Reference、shallow copy 与 `deepcopy` `[Python Language + Standard Library]` `P1`

### 1. 这是什么

Assignment 共享对象；shallow copy 创建新外层容器但复用内部对象；deepcopy 尝试递归复制整个对象图。

### 2. 为什么 Python 需要它

程序有时需要共享数据，有时需要独立修改嵌套配置或 metadata。

### 3. 最简单的例子

```python
from copy import copy, deepcopy

original = {'values': [1, 2]}
alias = original
shallow = copy(original)
deep = deepcopy(original)

shallow['values'].append(3)
deep['values'].append(4)

print(original)
print(alias is original)
print(shallow is original)
print(shallow['values'] is original['values'])
print(deep)
```

### 4. 输出结果

```text
{'values': [1, 2, 3]}
True
False
True
{'values': [1, 2, 4]}
```

### 5. 逐行解释

- alias 与 original 是同一 dict。
- shallow 是新 dict，但内部 values list 共享。
- deep 的 dict 和内部 list 都独立。
- 修改 shallow 内部 list 因共享而影响 original。

### 6. 心智模型

```text
original ──> dict ──> inner list
alias    ────┘          ↑
shallow  ──> new dict ──┘
deep     ──> new dict ──> new inner list
```

### 7. 【学习重点】

1. Assignment、shallow copy、deepcopy 是三种不同操作。
2. 浅 copy 只复制最外层。
3. Deepcopy 可能成本高，也不保证所有外部资源都适合复制。
4. 先判断需要共享还是隔离，再选择复制策略。

### 8. 常见误区

- 看到新 dict 就认为内部完全独立。
- 对大型模型/Tensor 随意 deepcopy。
- 用 deepcopy 掩盖不清楚的数据所有权。

### 9. 【掌握标准】

能为一层和两层嵌套容器画引用图，并预测三种复制方式后的修改影响。

### 10. 小练习

1. 改 shallow 的顶层 key，判断 original 是否变化。
2. 使用 `list(original_list)` 测试嵌套 list。
3. 解释何时只需要 shallow copy。

---

# 第十五部分：模块与包

## 15. `.py` file、module、package、namespace 与 import `[Python Language]` `P1`

### 1. 这是什么

Python 文件可作为 module；包含多个 module 的目录可构成 package。每个 module 有自己的 namespace。Import 执行/加载 module，并在当前 namespace 绑定名字。

### 2. 为什么 Python 需要它

大型工程必须拆文件、复用实现并控制名字来源。

### 3. 最简单的例子

假设 `tools.py`：

```python
value = 3

def double(x):
    return x * 2
```

`main.py`：

```python
import tools
from tools import double

print(tools.value)
print(double(4))
```

Package 内还可写：

```python
from .helper import function
```

### 4. 输出结果

```text
3
8
```

### 5. 逐行解释

- `import tools` 绑定 module object，可用 `tools.value`。
- `from tools import double` 直接绑定 function object。
- 相对 import 的 `.` 表示当前 package。

### 6. 心智模型

```text
module file executes
      ↓
module namespace object
├── value
└── double
      ↓ import binds selected names
current namespace
```

### 7. 【学习重点】

1. Import 的对象可能来自当前仓库，也可能来自环境 site-packages。
2. Module 是 namespace，不只是文件文本。
3. Relative import 必须在 package context 中理解。
4. 不深入 packaging 发布；先会定位定义来源。
5. 同名 import 可能覆盖当前名字。

### 8. 常见误区

- 在仓库里寻找所有第三方 import。
- 直接运行使用 relative import 的内部文件导致 import error。
- 误以为 import 每次都从头执行 module。

### 9. 【掌握标准】

看到 import 能判断绑定的是 module 还是其中对象，并能定位到仓库文件或环境依赖。

### 10. 小练习

1. 比较 `import math` 与 `from math import sqrt`。
2. 创建两个小 module 并导入 function。
3. 解释 `.helper` 的点代表什么。

---

# 第十六部分：异常

## 16. try、except、else、finally、raise 与 assert `[Python Language]` `P1`

### 1. 这是什么

Exception 表示不能按正常路径继续。`try/except` 捕获；`else` 在无异常时运行；`finally` 无论如何执行；`raise` 主动抛出；`assert` 检查开发期假设。

### 2. 为什么 Python 需要它

错误必须显式传播或处理，资源和临时状态必须可靠清理。

### 3. 最简单的例子

```python
def parse_positive(text):
    try:
        value = int(text)
    except ValueError:
        print('invalid')
        return None
    else:
        if value <= 0:
            raise ValueError('must be positive')
        return value
    finally:
        print('cleanup')

print(parse_positive('3'))
print(parse_positive('bad'))
```

### 4. 输出结果

```text
cleanup
3
invalid
cleanup
None
```

### 5. 逐行解释

- int 转换失败产生 ValueError，被 except 捕获。
- 无异常才进入 else。
- finally 在 return 和 exception 路径都执行。
- `raise` 主动报告非法值。

### 6. 心智模型

```text
try
 ├─ exception → matching except
 └─ success   → else
          ↓
       finally always
```

### 7. 【学习重点】

1. Finally 特别适合 cleanup/restore。
2. 捕获具体 exception，不要无条件吞掉所有错误。
3. Raise 保留 fail-fast/fail-closed 行为。
4. Assert 不应代替用户输入验证；优化模式可移除 assert。
5. 阅读 traceback 要从最后的异常类型和调用链开始。

### 8. 常见误区

- `except Exception: pass` 隐藏根因。
- 认为 return 会跳过 finally。
- 用 assert 做必须永久执行的安全检查。

### 9. 【掌握标准】

能预测 try 的四种路径，解释 finally 为什么用于恢复临时状态，并区分 raise 与 assert。

### 10. 小练习

1. 在 try 中主动 raise，观察 finally。
2. 增加一个只在成功时打印的 else。
3. 用明确 ValueError 替代一个输入 assert。

---

# 第十七部分：with 与 Context Manager

## 17. `with`、`__enter__`、`__exit__` 与 `contextmanager` `[Python Language + Standard Library]` `P1`

### 1. 这是什么

Context manager 定义进入和退出代码块时的成对行为。Class 可实现 `__enter__/__exit__`；`contextlib.contextmanager` 可用 generator function 和一次 `yield` 创建它。

### 2. 为什么 Python 需要它

文件、锁、精度模式和临时 patch 必须在正常或异常退出时都清理。

### 3. 最简单的例子

```python
from contextlib import contextmanager

@contextmanager
def temporary(config):
    old = config['mode']
    config['mode'] = 'temporary'
    try:
        yield config
    finally:
        config['mode'] = old

config = {'mode': 'normal'}
with temporary(config) as active:
    print(active['mode'])
print(config['mode'])
```

Class protocol 的概念形式：

```python
class Resource:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False
```

### 4. 输出结果

```text
temporary
normal
```

### 5. 逐行解释

- Decorator 将 generator function 转为 context manager factory。
- Yield 前是 enter 逻辑。
- Yield 的值绑定给 `as active`。
- Yield 后/finally 是 exit cleanup。
- `with open(...)` 使用同一协议自动关闭文件。

### 6. 心智模型

```text
enter
  ↓
with body
  ↓ normal or exception
exit / finally
```

### 7. 【学习重点】

1. With 的核心价值是保证 exit。
2. `__exit__` 是否返回 True 会影响异常是否被抑制。
3. `contextmanager` 中 yield 只用于把控制权交给 with body。
4. 多个 context manager 按进入顺序进入、反序退出。
5. 临时全局/对象修改必须有恢复路径。

### 8. 常见误区

- 把 yield 当普通 return。
- 在 yield 前修改状态，却没有 finally。
- 认为 with 只用于文件。

### 9. 【掌握标准】

看到 `with manager():` 能找到 enter/exit 来源，并检查异常路径是否恢复资源或状态。

### 10. 小练习

1. 在 with body 中 raise，验证 mode 仍恢复。
2. 使用 `with open` 写入再读取一个临时文件。
3. 手写简单 `__enter__/__exit__` class。

---

# 第十八部分：Decorator

## 18. Function decorator 与 decorator arguments `[Python Language]` `P1`

### 1. 这是什么

Function 本身是 object。Decorator 接收 function/class，并返回替代对象或执行登记。`@decorator` 概念上是一次重新赋值。

### 2. 为什么 Python 需要它

它把日志、验证、注册、精度转换等横切行为放到主体函数外，并保留简洁声明方式。

### 3. 最简单的例子

```python
def announce(label):
    def decorator(func):
        def wrapped(*args, **kwargs):
            print(label)
            return func(*args, **kwargs)
        return wrapped
    return decorator

@announce('calling add')
def add(a, b):
    return a + b

print(add(2, 3))
```

### 4. 输出结果

```text
calling add
5
```

### 5. 逐行解释

- `announce('calling add')` 返回 decorator。
- Decorator 接收原 add function。
- Wrapped closure 保存 label 和原 func。
- 定义完成后概念上执行 `add = announce(...)(add)`。

### 6. 心智模型

```text
original function
      ↓ decorator
wrapped/replacement function
      ↓ rebound to original name
```

### 7. 【学习重点】

1. Decorator 是可执行 Python，不是注释。
2. Function/class 都可被 decorator 接收。
3. Decorator with arguments 多一层“创建 decorator”的调用。
4. Wrapper 常用 `*args/**kwargs` 保持通用参数流。
5. Closure 让 wrapper 记住外层 label/func。

### 8. 常见误区

- 从主体函数第一行开始读，忽略 decorator 已改变调用行为。
- 混淆 decorator 执行时机与被装饰函数调用时机。
- Wrapper 不返回原函数结果。

### 9. 【掌握标准】

能把 `@decorator` 和 `@factory(option)` 分别展开成显式赋值，并说明 wrapper 捕获了哪些对象。

### 10. 小练习

1. 去掉 `@`，手工写 `add = announce(...)(add)`。
2. 写一个打印参数的 decorator。
3. 预测 decorator 定义时和函数调用时各打印什么。

---

# 第十九部分：常见特殊方法

## 19. `__init__`、`__call__`、`__getitem__`、`__len__`、`__iter__`、`__repr__` `[Python Language]` `P2`

### 1. 这是什么

双下划线特殊方法让自定义 object 参与 Python 语法协议，如初始化、调用、索引、长度、迭代和显示。

### 2. 为什么 Python 需要它

统一语法可以操作内置类型和用户类型，而不用每个 class 发明不同 API。

### 3. 最简单的例子

```python
class Batch:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return f'Batch({self.values!r})'

    def __call__(self):
        return sum(self.values)

b = Batch([1, 2, 3])
print(b[1])
print(len(b))
print(list(b))
print(b)
print(b())
```

### 4. 输出结果

```text
2
3
[1, 2, 3]
Batch([1, 2, 3])
6
```

### 5. 逐行解释

- `b[1]` 委托 `__getitem__`。
- `len(b)` 委托 `__len__`。
- For/list conversion 使用 iteration protocol。
- `print(b)` 使用 `__repr__`（若无更合适的 `__str__`）。
- `b()` 使用 `__call__`。

### 6. 心智模型

```text
syntax          protocol method
obj[index]  →  __getitem__
len(obj)    →  __len__
for x in obj→  __iter__
obj(...)    →  __call__
repr(obj)   →  __repr__
```

### 7. 【学习重点】

1. 熟悉语法可能调用用户 class 的方法。
2. 遇到 dataset[index] 要找 `__getitem__`。
3. 遇到 object(...) 要确认 `__call__` 来源。
4. 只学阅读所需协议，不必背所有 dunder methods。

### 8. 常见误区

- 把 dataset 当普通 list。
- 手工直接调用所有 dunder method，而不用自然语法。
- 认为 `__iter__` 一定返回 list。

### 9. 【掌握标准】

看到六种语法时能立即指出对应特殊方法，并能在 class/父类中定位实现。

### 10. 小练习

1. 给 Batch 支持负索引（内部 list 已支持）。
2. 删除 `__iter__`，观察 Python 是否还能借助 `__getitem__` 迭代。
3. 修改 `__repr__` 输出。

---

# 第二十部分：其他高频机制

## 20. isinstance、hasattr/getattr、property、staticmethod/classmethod、lambda/closure `[Python Language]` `P1/P2`

### 1. 这是什么

- `isinstance`：检查 object 是否是某 class 或子类的 instance。
- `hasattr/getattr`：动态检查/读取 attribute。
- `property`：把 method 暴露为 attribute-style access。
- `staticmethod`：class namespace 中不绑定 self 的 function。
- `classmethod`：绑定 class object 为 `cls`。
- `lambda`：匿名 function expression。
- Closure：内部函数记住外层 scope 的对象。

### 2. 为什么 Python 需要它

它们支持 runtime inspection、简洁只读属性、class-level construction、callback 和 wrapper。

### 3. 最简单的例子

```python
class Item:
    kind = 'base'

    def __init__(self, value):
        self.value = value

    @property
    def doubled(self):
        return self.value * 2

    @staticmethod
    def valid(value):
        return value >= 0

    @classmethod
    def zero(cls):
        return cls(0)

item = Item.zero()
get_value = lambda obj: obj.value

def make_adder(amount):
    def add(value):
        return value + amount
    return add

add_three = make_adder(3)

print(isinstance(item, Item))
print(hasattr(item, 'value'))
print(getattr(item, 'missing', None))
print(item.doubled)
print(Item.valid(-1))
print(get_value(item))
print(add_three(4))
```

### 4. 输出结果

```text
True
True
None
0
False
0
7
```

### 5. 逐行解释

- `zero` 接收 cls 并创建当前 class instance。
- Property 调用 method，但调用处没有括号。
- Getattr 第三个参数是 attribute 不存在时的默认值。
- Lambda 创建 function object。
- `add` closure 记住 amount=3。

### 6. 心智模型

```text
runtime object
 ├─ type check: isinstance
 ├─ attribute exists: hasattr
 └─ dynamic lookup: getattr

outer scope amount
       ↓ captured by
closure function add
```

### 7. 【学习重点】

1. `isinstance` 匹配子类，不只是 exact type。
2. `getattr(obj,name,default)` 的 name 是字符串。
3. Property 调用处没有括号。
4. Staticmethod 不接收 self；classmethod 接收 cls。
5. Lambda 只适合短表达式。
6. Closure 捕获名字/对象；循环 callback 要注意 late binding。

### 8. 常见误区

- 用 `type(x) is Base` 代替需要支持子类的 `isinstance`。
- 把 getattr 当 dict access。
- 给 property 加括号。
- 在循环中写 `lambda: i`，所有回调最后看到同一个 i。

### 9. 【掌握标准】

能识别动态 attribute 操作；区分 instance/static/class method；解释简单 closure 捕获的对象。

### 10. 小练习

1. 用 getattr 读取存在和不存在的 attribute。
2. 写一个 classmethod 创建指定默认值对象。
3. 修复循环 lambda late-binding：使用 `lambda i=i: i`。

---

# 学习路线与边界

## 推荐顺序

```text
1 Name/Object/Reference
→ 2 Basic Types
→ 3 Containers
→ 4 Index/Slice
→ 5 Control Flow
→ 6 Functions
→ 7 *args/**kwargs
→ 8 Class/Object
→ 9 Inheritance
→ 10 Attribute Lookup/Binding
→ 11 Callable
→ 12–14 Engineering Expressions and Copy
→ 15–18 Modules/Exceptions/Context/Decorators
→ 19–20 Special Methods and Runtime Introspection
```

## Python 与第三方库边界

```text
[Python Language]
callable protocol / __call__
        ↓ used by
[PyTorch]
nn.Module.__call__ → hooks → forward
```

```text
[Python Language]
decorator
        ↓ used by
[MMCV/MMDetection]
@DETECTORS.register_module(), @auto_fp16
```

```text
[Python Language]
class / inheritance / dict / **kwargs
        ↓ used by
[BEVFormer]
detector/head/transformer/stateful metadata
```

本文到 Python 层为止。第三方库和项目中的真实位置请使用 `docs/bevformer_python_usage_index.md`。
