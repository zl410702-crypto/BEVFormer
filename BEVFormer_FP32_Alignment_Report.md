# BEVFormer-tiny 独立模型 FP32 对齐结果

## 基线与输入

- 历史 Golden 保留于 `golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3/`，采集时 cuDNN TF32 开启；本轮没有覆盖。
- 独立 FP32 Reference 保存于 `golden_tensors_fp32/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3/`。采集程序 `tools/inference_only/capture_fp32_reference.py` 沿用 `tools/run_golden_sequence.py` 和 `tools/golden_sequence_recorder.py` 的原始三帧推理与 Hook，只在进程内关闭 cuDNN/矩阵乘法 TF32，并记录 Backbone 真实输入。`capture_settings.json` 记录 Python、PyTorch、CUDA、cuDNN、OpenMMLab 版本、GPU、Checkpoint 哈希和三个输入哈希。目录已存在时采集程序拒绝覆盖。
- 旧环境：Python 3.8.20，PyTorch 1.9.1+cu111，CUDA 11.1，cuDNN 8005，MMCV 1.4.0，MMDetection 2.14.0，MMDetection3D 0.17.1，MMSegmentation 0.14.1。
- 新环境：PyTorch 2.8.0+cu128，RTX 4060 Laptop。两环境均使用 `model.eval()`、FP32、禁用 Autocast/TF32、cuDNN deterministic、关闭 benchmark，以及同一 643 项 Checkpoint；校验程序先核对 Checkpoint 和逐帧输入 SHA256。
- 新 Reference 的 Frame0/1 图像与现有 `artifacts/bevformer_tiny_stateful_trt84_runtime/` fixture 逐元素相同；Frame0/1 的 CAN、`lidar2img` 亦逐元素相同。Frame2 图像直接由原 Golden 推理路径的 Backbone pre-hook 捕获。

## 数值结果

以下均为新环境独立模型相对 **TF32 关闭 Reference** 的 MaxAE。三帧连续运行，`prev_bev` 由独立模型上一帧输出反馈。

| Tensor | Frame0 | Frame1 | Frame2 |
|---|---:|---:|---:|
| Backbone C5 | 6.79e-6 | 6.56e-6 | 8.23e-6 |
| FPN | 2.77e-5 | 2.48e-5 | 2.10e-5 |
| TSA Layer0 输出 | 7.21e-6 | 1.43e-5 | 1.69e-5 |
| SCA Layer0 输出 | 1.55e-5 | 2.72e-5 | 2.29e-5 |
| 最终 BEV | 2.03e-5 | 2.52e-5 | 2.39e-5 |
| 分类 logits | 3.17e-4 | 4.63e-5 | 7.13e-5 |
| 回归输出 | 9.92e-5 | 8.17e-5 | 5.58e-5 |
| 解码后 3D boxes | 1.86e-4 | 7.84e-5 | 4.58e-5 |
| 解码后 scores | 3.90e-6 | 2.65e-6 | 2.98e-6 |
| 解码后 labels | 0 | 0 | 0 |

同一独立模型在 PyTorch 1.9 中运行，C5/FPN 对原模型均为 **0**；Frame0 BEV、分类 logits、回归输出的 MaxAE 分别为 8.57e-6、2.29e-5、4.20e-5。PyTorch 2.8 的额外误差主要来自跨版本浮点计算。TSA/SCA 的同版本小误差来自原 CUDA Attention 与 `grid_sample` 实现的数值差异；三帧结果没有出现持续放大的分叉。

使用 Reference `prev_bev` 的单步对照也已运行，结果见 `BEVFormer_FP32_Reference_History_Metrics.json`。反馈模式结果见 `BEVFormer_FP32_Alignment_Metrics.json`，内含所有已采集关键 Tensor 的 MaxAE、MAE、MSE。

## 实现边界与结论

- 模型计算入口 `tools/inference_only/model.py::BEVFormerTiny.forward` 保持原始 643 项 `state_dict` 的 `strict=True` 加载；纯 PyTorch Attention 位于 `tools/inference_only/deformable_attention.py`，运行时不加载 MMCV、MMDetection、MMDetection3D 或 MMSegmentation。
- 新增 `tools/inference_only/detection.py::decode_detections` 实现原 `NMSFreeCoder.decode_single`、`BEVFormerHead.get_bboxes` 的推理解码语义，不改变模型计算。用原始 raw logits/boxes 解码时，三帧 labels 完全一致，boxes 最大差异 4.77e-7。
- 独立模型完成真实单帧、三帧 Stateful 反馈及检测解码的 FP32 对齐。该结论基于上述实测误差，不声称跨环境逐 bit 相等，也未通过放宽已有 Golden 阈值获得。
- 本阶段没有执行 ModelOpt Calibration、ONNX 导出或 TensorRT 构建。进入 PTQ 前应明确量化评估使用此 TF32 关闭 Reference，并保留历史 TF32 Golden 作为独立历史记录。

## 复现

在旧环境运行 `python tools/inference_only/capture_fp32_reference.py`；输出目录存在时会拒绝覆盖。在新环境运行 `python -m tools.inference_only.validate_fp32_reference`。检测解码测试为 `python -m unittest tests.test_inference_only_detection`。
