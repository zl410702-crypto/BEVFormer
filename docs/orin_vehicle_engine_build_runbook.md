# BEVFormer 实车 Orin FP32 Engine 构建操作手册

本文用于在无法使用 Codex 时，独立完成实车 Orin 上的环境确认、
ResNet50 对照构建和完整 BEVFormer FP32 Engine 构建。

## 1. 安全边界

- 只在车辆静止、进入公司规定的维护状态后操作。
- 不在驾驶、自动驾驶、充电或关键业务运行期间构建 Engine。
- 不自行发送未知 PHM 指令，不猜测 watchdog/心跳控制协议。
- 不覆盖已有正式 ONNX、Plugin 或 Engine。
- 本阶段只构建 FP32，不启用 FP16、INT8 或性能优化。
- 如果出现 SSH 失联、系统冻结或重启，同一命令不要连续重试。
- 是否停止 onboard 服务必须遵守实车维护 SOP；没有明确许可时不要停止。

## 2. 本次实验要回答的问题

台架 Orin 上已经观察到：

```text
ResNet50 parser PASS
→ cuDNN tactic profiling
→ 系统卡死/重启

完整 BEVFormer parser PASS
→ 12 个 MSDA Plugin 创建成功
→ tactic profiling
→ 系统卡死/重启
```

实车测试用于区分：

```text
实车 ResNet50 PASS
→ 台架硬件、镜像或运行状态更可疑

实车 ResNet50也卡死
→ DriveOS / TensorRT / cuDNN 构建路径更可疑

实车 ResNet50 PASS，完整 BEVFormer FAIL
→ 故障范围缩小到完整网络特有结构或规模
```

## 3. 所需文件

PC 侧交付目录：

```text
/data/bevformer/orin/
├── bin/build_bevformer_fp32_engine_plugin_count_control
├── lib/libms_deformable_attention_plugin.so
└── models/
    ├── bevformer_tiny_resnet50_c5_opset13.onnx
    └── bevformer_tiny_stateful_fp32_host_affine_trt84_pad_compat.onnx
```

已知 SHA256：

```text
973115941c7b448d2061bb1163b7415463918baa694378ea71428b7494195c6d  bin/build_bevformer_fp32_engine_plugin_count_control
b2fe3f21a117fdc0909a2e36be7e3bbd218b58e676a8e3ec4e0e62dcfe120b00  lib/libms_deformable_attention_plugin.so
28e315fd22c745cf7d3a7e891384a24e56d915ec42c3664f779a7f5507890118  models/bevformer_tiny_resnet50_c5_opset13.onnx
fdc28ca37a43938d0a3a395d1a9b3a61e48d4549c0a94c101f4b602d49ef9ff2  models/bevformer_tiny_stateful_fp32_host_affine_trt84_pad_compat.onnx
```

将这些文件复制到实车独立目录，例如：

```text
/var/pony/immutable_data/eric/orin/
```

复制方式应使用公司允许的文件同步流程。不要覆盖其他人的正式目录。

## 4. 上车后的文件检查

```bash
cd /var/pony/immutable_data/eric/orin

ls -lh \
  bin/build_bevformer_fp32_engine_plugin_count_control \
  lib/libms_deformable_attention_plugin.so \
  models/bevformer_tiny_resnet50_c5_opset13.onnx \
  models/bevformer_tiny_stateful_fp32_host_affine_trt84_pad_compat.onnx
```

检查哈希：

```bash
sha256sum \
  bin/build_bevformer_fp32_engine_plugin_count_control \
  lib/libms_deformable_attention_plugin.so \
  models/bevformer_tiny_resnet50_c5_opset13.onnx \
  models/bevformer_tiny_stateful_fp32_host_affine_trt84_pad_compat.onnx
```

检查 ARM64 和动态库：

```bash
file bin/build_bevformer_fp32_engine_plugin_count_control
file lib/libms_deformable_attention_plugin.so
ldd lib/libms_deformable_attention_plugin.so | grep 'not found'
```

预期：

```text
builder 和 Plugin 均为 ARM aarch64
ldd 不输出 not found
```

## 5. 环境检查

```bash
cat /etc/nvidia/version-*
cat /etc/os-release
dpkg-query -W libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libcudnn8
```

当前目标基线：

```text
DriveOS:  6.0.5.1
Ubuntu:   20.04.5
CUDA:     11.4
cuDNN:    8.4.1
TensorRT: 8.4.12
```

如果实车版本不同，先记录，不要直接将结果与台架作严格 A/B 对比。

检查空间和服务状态：

```bash
df -h /var/pony/immutable_data
systemctl --user status onboard_daemon.service --no-pager
```

不要为了释放资源自行停止实车服务。先遵守现场维护流程。

## 6. 为什么先构建 ResNet50

这个 ONNX 是从正式 BEVFormer 图中原样提取的真实 Backbone：

```text
输入:  images [1,6,3,480,800] FP32
内部:  Squeeze → [6,3,480,800]，六相机作为 batch
输出:  C5 [6,2048,15,25] FP32
Conv:  53
MSDA:  0
```

它保留正式模型权重，但不含 FPN、Transformer、MSDA Plugin 和检测头。

## 7. 构建 ResNet50 对照 Engine

执行：

```bash
cd /var/pony/immutable_data/eric/orin

./bin/build_bevformer_fp32_engine_plugin_count_control \
  models/bevformer_tiny_resnet50_c5_opset13.onnx \
  lib/libms_deformable_attention_plugin.so \
  models/bevformer_tiny_resnet50_c5_trt8412_fp32.engine \
  --workspace-mib=512 \
  --tactic-sources=reduced \
  --min-timing-iterations=1 \
  --avg-timing-iterations=1 \
  --expected-plugin-layers=0 \
  2>&1 | tee resnet50_engine_build.log
```

参数含义：

```text
workspace-mib=512          限制 TensorRT workspace
tactic-sources=reduced     仅保留 cuBLAS 和 cuDNN 来源
min/avg timing=1           降低 tactic 重复测速次数
expected-plugin-layers=0   ResNet50 不包含 MSDA Plugin
```

正常过程至少应出现：

```text
Parser: PASS
PluginV2 layers: 0
Workspace MiB: 512
...
FP32 engine: PASS
```

构建结束后检查：

```bash
ls -lh models/bevformer_tiny_resnet50_c5_trt8412_fp32.engine
sha256sum models/bevformer_tiny_resnet50_c5_trt8412_fp32.engine
```

如果只有 `Parser: PASS` 而没有 Engine 文件，不能写 Build PASS。

## 8. ResNet50 结果判断

### 8.1 构建成功

记录：

```text
ResNet50 parser: PASS
ResNet50 engine build: PASS
Engine size:
Build duration:
SHA256:
```

然后才能继续完整 BEVFormer。

### 8.2 普通错误退出

保存：

```bash
cp resnet50_engine_build.log resnet50_engine_build.failed.log
tail -100 resnet50_engine_build.failed.log
```

不要随意修改 ONNX。先根据第一个明确 TensorRT ERROR 分析。

### 8.3 系统冻结或重启

重启后记录：

```bash
journalctl --list-boots | tail -5
journalctl -b -1 -k --no-pager \
  | grep -iE 'NVRM|Xid|fault|watchdog|oom|panic' \
  | tail -100
```

随后停止测试。不要马上重复同一命令。

## 9. 构建完整 BEVFormer FP32 Engine

只有 ResNet50 构建成功后执行：

```bash
cd /var/pony/immutable_data/eric/orin

./bin/build_bevformer_fp32_engine_plugin_count_control \
  models/bevformer_tiny_stateful_fp32_host_affine_trt84_pad_compat.onnx \
  lib/libms_deformable_attention_plugin.so \
  models/bevformer_tiny_stateful_orin_trt8412_fp32_host_affine.engine \
  --workspace-mib=512 \
  --tactic-sources=reduced \
  --min-timing-iterations=1 \
  --avg-timing-iterations=1 \
  2>&1 | tee bevformer_engine_build.log
```

注意：完整 BEVFormer **不要**传：

```text
--expected-plugin-layers=0
```

builder 默认要求 12 个 Plugin Layer。预期：

```text
Parser: PASS
PluginV2 layers: 12
...
FP32 engine: PASS
```

检查产物：

```bash
ls -lh models/bevformer_tiny_stateful_orin_trt8412_fp32_host_affine.engine
sha256sum models/bevformer_tiny_stateful_orin_trt8412_fp32_host_affine.engine
```

## 10. 结果记录模板

```text
Vehicle/Orin ID:
Date:

DriveOS:
TensorRT:
CUDA:
cuDNN:

onboard service state:
available RAM before build:
GPU temperature before build:

ResNet50 parser: PASS / FAIL
ResNet50 PluginV2 layers: 0 / other
ResNet50 build: PASS / FAIL / SYSTEM RESET
ResNet50 engine size:
ResNet50 engine SHA256:
ResNet50 last log line:

BEVFormer parser: PASS / FAIL / NOT RUN
BEVFormer PluginV2 layers: 12 / other
BEVFormer build: PASS / FAIL / SYSTEM RESET / NOT RUN
BEVFormer engine size:
BEVFormer engine SHA256:
BEVFormer last log line:

Previous-boot kernel errors:
Conclusion:
```

## 11. 当前已知事实

```text
PC TRT 8.5.3 完整 BEVFormer FP32 engine build: PASS
PC engine 不能部署到 Orin
台架 ResNet50 ONNX parser: PASS
台架 ResNet50 PluginV2 layer count: 0（正确）
台架 ResNet50 tactic profiling: 系统卡死/重启
台架完整 BEVFormer parser: PASS
台架完整 BEVFormer MSDA Plugin layers: 12
台架完整 BEVFormer tactic profiling: 系统卡死/重启
```

因此实车实验首先是环境/硬件 A/B 验证，不是正式上车功能测试。

## 12. 禁止事项

- 不使用 PC 生成的 `.engine` 部署 Orin。
- 不覆盖实车已有正式 Engine。
- 不在本任务中启用 FP16 或 INT8。
- 不修改 MSDA Plugin 数学实现。
- 不把 parser PASS 当成 engine build PASS。
- 不因为一次系统重启而反复执行相同构建。
- 不在缺少维护许可时停止 onboard、PHM 或 watchdog。
- 不在实车运动或安全关键模块运行期间开展构建实验。
