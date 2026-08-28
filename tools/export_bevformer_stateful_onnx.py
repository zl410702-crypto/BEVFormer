#!/usr/bin/env python
"""Export and audit a stateful single-frame BEVFormer Tiny ONNX model.

本脚本不是单纯调用 ``torch.onnx.export``，而是按以下顺序建立部署基线：

1. 从三帧 Golden sequence 中读取 Frame 1，并使用 Frame 0 ``final_bev``
   作为真实 ``prev_bev``；
2. 复现官方 ``forward_test`` 对 ``can_bus`` translation/rotation delta 的
   处理，并在 host 侧计算 BEV shift；
3. 在导出前验证 Stateful wrapper 的 Frame 1 输出与 PyTorch Golden 一致；
4. 验证 ``prev_bev`` 确实影响输出，并验证统一接口的 Frame 0 gate 能复现
   官方 ``prev_bev=None`` 语义；
5. 导出 opset 13 ONNX，安装 deployment-only symbolic/forward overrides；
6. 将可证明为常量的 Slice axes canonicalize 为 TensorRT 8.4 可解析形式；
7. 审计 graph interface、12 个 MSDeformableAttention custom nodes、输入依赖
   和 alignment 子图，并写出同名 ``*_audit.json``。

默认采用 fail-closed 策略：任何 Golden validation failure 都会阻止导出。
``--allow-validation-failure`` 只用于生成结构调试模型，不能作为部署通过依据。
"""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import onnx
import torch

from deployment.bevformer_onnx_wrapper import (
    BEVFormerONNXWrapper, CUSTOM_DOMAIN, CUSTOM_OP_TYPE,
    deployment_export_overrides)
from deployment.bevformer_stateful_onnx_wrapper import (
    BEVFormerStatefulONNXWrapper)
from deployment.onnx_slice_compat import canonicalize_static_slice_axes_file
from run_golden_sample import (DEFAULT_CHECKPOINT, DEFAULT_CONFIG,
                               pipeline_image_tensor, prepare_golden_sample,
                               validate_pipeline_item)
from tensor_compare import compare_arrays


DEFAULT_FRAME1_TOKEN = '3950bd41f74548429c0f7700ff3d8269'
DEFAULT_SEQUENCE = Path(
    'golden_tensors/fcbccedd61424f1b85dcbf8f897f9754_3e8750f3')
DEFAULT_OUTPUT = Path('artifacts/bevformer_tiny_stateful_opset13.onnx')


def parse_args():
    """定义 Golden、模型权重、导出文件和可选 runtime fixture 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=DEFAULT_FRAME1_TOKEN)
    parser.add_argument('--sequence-root', type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--runtime-fixture-dir', type=Path,
        help='Optionally save Frame 0/1 FP32 TensorRT inputs and references.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--allow-validation-failure', action='store_true',
        help=('Export a structural debug candidate after Golden validation '
              'fails; disabled by default and not deployment-ready.'))
    return parser.parse_args()


def metrics(expected, actual):
    """比较同 shape 数组；shape 不同立即失败，禁止隐式 transpose。"""
    result = compare_arrays(expected, actual)
    if result is None:
        raise ValueError('Shape mismatch: {} != {}'.format(
            expected.shape, actual.shape))
    return result


def prepare_runtime_meta(meta, current_absolute, previous_absolute):
    """按官方 test-time 逻辑生成 Frame t 相对 Frame t-1 的 motion metadata。

    ``can_bus[:3]`` 是相对 translation，``can_bus[-1]`` 是相对 rotation。
    差值必须先在 metadata 原有的 float64 域计算，再在构造模型输入时转
    float32；提前 cast 会引入与 Golden 不一致的 rounding。
    """
    runtime = copy.deepcopy(meta)
    # Match BEVFormer.forward_test: subtract in the metadata's float64 domain,
    # then cast when creating the model input tensor.
    can_bus = np.asarray(current_absolute, dtype=np.float64).copy()
    previous = np.asarray(previous_absolute, dtype=np.float64)
    can_bus[:3] -= previous[:3]
    can_bus[-1] -= previous[-1]
    runtime['can_bus'] = can_bus
    return runtime


def first_frame_runtime_meta(meta, absolute):
    """构造首帧 metadata：没有历史帧，因此 translation/rotation delta 为 0。"""
    runtime = copy.deepcopy(meta)
    can_bus = np.asarray(absolute, dtype=np.float64).copy()
    can_bus[:3] = 0.0
    can_bus[-1] = 0.0
    runtime['can_bus'] = can_bus
    return runtime


def compute_shift(can_bus, model):
    """在 host 侧复现 ``PerceptionTransformer`` 的 BEV shift 公式。

    输入使用官方 delta can_bus，输出顺序为 ``[shift_x, shift_y]``。这里保留
    NumPy float64 中间计算，再显式转换成模型要求的 float32，避免 exporter
    自行重建 motion 几何公式。
    """
    head = model.pts_bbox_head
    transformer = head.transformer
    delta_x = np.asarray([can_bus[0]], dtype=np.float64)
    delta_y = np.asarray([can_bus[1]], dtype=np.float64)
    ego_angle = np.asarray([can_bus[-2] / np.pi * 180.0], dtype=np.float64)
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180.0
    bev_angle = ego_angle - translation_angle
    grid_y = head.real_h / head.bev_h
    grid_x = head.real_w / head.bev_w
    shift_y = translation_length * np.cos(
        bev_angle / 180.0 * np.pi) / grid_y / head.bev_h
    shift_x = translation_length * np.sin(
        bev_angle / 180.0 * np.pi) / grid_x / head.bev_w
    shift = np.stack((shift_x * transformer.use_shift,
                      shift_y * transformer.use_shift), axis=-1)
    return torch.from_numpy(shift.astype(np.float32))


def tensor_inputs(meta, model):
    """将一帧 metadata 转成 wrapper 的三个显式 CUDA runtime inputs。"""
    can_bus = torch.from_numpy(
        np.asarray(meta['can_bus'], dtype=np.float32)).unsqueeze(0).cuda()
    shift = compute_shift(meta['can_bus'], model).cuda()
    lidar2img = torch.from_numpy(
        np.asarray(meta['lidar2img'], dtype=np.float32)).unsqueeze(0).cuda()
    return can_bus, shift, lidar2img


def golden_outputs(root, frame):
    """按 ONNX 输出顺序读取指定 Golden frame 的检测与 BEV reference。"""
    base = root / frame
    return tuple(np.load(str(path), allow_pickle=False) for path in (
        base / 'detection/cls_scores.npy',
        base / 'detection/bbox_preds.npy',
        base / 'bev/final_bev.npy'))


def compare_triplet(expected, actual):
    """比较 ``cls/bbox/bev`` 三个正式 graph outputs。"""
    names = ('all_cls_scores', 'all_bbox_preds', 'bev_embed')
    report = {}
    for name, reference, output in zip(names, expected, actual):
        report[name] = metrics(reference, output.detach().cpu().numpy())
        report[name]['shape'] = list(reference.shape)
    return report


def save_runtime_frame(root, frame, inputs, outputs):
    """为 C++ TensorRT runner 保存确定性的连续 FP32 raw bindings。

    每个 binding 写为无 header 的 ``.bin``，shape/dtype/path 写入同目录
    ``metadata.json``。这里强制 contiguous FP32，避免 C++ runtime 猜测 layout
    或依赖 NumPy stride。
    """
    frame_root = root / frame
    frame_root.mkdir(parents=True, exist_ok=True)
    metadata = {'inputs': {}, 'outputs': {}}
    for group, tensors in (('inputs', inputs), ('outputs', outputs)):
        for name, tensor in tensors.items():
            array = (tensor.detach().cpu().numpy() if torch.is_tensor(tensor)
                     else np.asarray(tensor))
            array = np.ascontiguousarray(array.astype(np.float32, copy=False))
            path = frame_root / '{}.bin'.format(name)
            array.tofile(str(path))
            metadata[group][name] = {
                'shape': list(array.shape), 'dtype': 'float32',
                'path': path.name}
    with (frame_root / 'metadata.json').open('w') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')


def assert_close_report(report, atol=1e-5, allow_failure=False):
    """执行 export gate；默认任一输出 MaxAE 超限就拒绝继续导出。"""
    failed = {name: values for name, values in report.items()
              if values['MaxAE'] > atol}
    if failed:
        if not allow_failure:
            raise RuntimeError('Wrapper validation failed: {}'.format(failed))
        print('WARNING: exporting structural candidate after validation '
              'failure: {}'.format(failed))


def value_description(value):
    """把 ONNX ValueInfo 转成便于写入 audit JSON 的名称/shape/dtype。"""
    tensor = value.type.tensor_type
    dims = []
    for dimension in tensor.shape.dim:
        if dimension.HasField('dim_value'):
            dims.append(dimension.dim_value)
        elif dimension.HasField('dim_param'):
            dims.append(dimension.dim_param)
        else:
            dims.append('?')
    return {'name': value.name, 'shape': dims,
            'dtype': onnx.TensorProto.DataType.Name(tensor.elem_type)}


def descendants(graph, source):
    """沿 ONNX consumer 方向遍历，用于证明 graph input 确实影响输出。"""
    consumers = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
    seen_values = {source}
    seen_nodes = set()
    queue = deque([source])
    while queue:
        value = queue.popleft()
        for index in consumers.get(value, []):
            if index in seen_nodes:
                continue
            seen_nodes.add(index)
            for output in graph.node[index].output:
                if output not in seen_values:
                    seen_values.add(output)
                    queue.append(output)
    return seen_values, seen_nodes, consumers.get(source, [])


def audit_onnx(path):
    """对导出模型执行 schema、依赖关系和 alignment 结构审计。

    审计重点不是只检查文件可加载，而是确认：

    * graph interface 与 Stateful runtime contract 完全一致；
    * 12 个 ``bevformer::MSDeformableAttention`` 均为五输入且保留
      ``im2col_step=64``；
    * 每个 runtime input 都能到达至少一个正式 graph output；
    * rotation/alignment 路径没有在 trace 时被裁掉；
    * ONNX checker 通过，并尽量完成 shape inference。
    """
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    counts = Counter((node.domain or 'ai.onnx', node.op_type)
                     for node in model.graph.node)
    custom_nodes = [node for node in model.graph.node
                    if node.domain == CUSTOM_DOMAIN and
                    node.op_type == CUSTOM_OP_TYPE]
    if len(custom_nodes) != 12:
        raise RuntimeError('Expected 12 custom nodes, found {}'.format(
            len(custom_nodes)))
    invalid_custom = [node.name for node in custom_nodes
                      if len(node.input) != 5 or
                      not any(attribute.name == 'im2col_step' and
                              attribute.i == 64 for attribute in node.attribute)]
    if invalid_custom:
        raise RuntimeError('Invalid custom node schema: {}'.format(
            invalid_custom))
    expected_inputs = {'images', 'prev_bev', 'can_bus', 'shift',
                       'lidar2img', 'use_prev_bev'}
    inputs = [value_description(value) for value in model.graph.input]
    outputs = [value_description(value) for value in model.graph.output]
    if {item['name'] for item in inputs} != expected_inputs:
        raise RuntimeError('Unexpected graph inputs: {}'.format(inputs))
    dependencies = {}
    output_names = {value.name for value in model.graph.output}
    for name in expected_inputs:
        values, nodes, direct = descendants(model.graph, name)
        dependencies[name] = {
            'direct_consumer_indices': direct,
            'node_count': len(nodes),
            'reaches_graph_output': bool(values & output_names),
            'custom_node_count_downstream': sum(
                1 for index in nodes
                if model.graph.node[index].domain == CUSTOM_DOMAIN),
        }
        if not dependencies[name]['reaches_graph_output']:
            raise RuntimeError('Input is not connected to output: {}'.format(
                name))
    alignment_ops = {name: counts[('ai.onnx', name)] for name in (
        'Sin', 'Cos', 'MatMul', 'Round', 'Clip', 'GatherElements', 'Gather',
        'Reshape', 'Transpose') if counts[('ai.onnx', name)]}
    if not alignment_ops.get('Sin') or not alignment_ops.get('Cos'):
        raise RuntimeError('Dynamic rotation trigonometry is absent')
    shape_inference = {'status': 'PASS'}
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        shape_inference['value_info_count'] = len(inferred.graph.value_info)
    except Exception as error:
        shape_inference = {'status': 'PARTIAL/FAIL',
                           'reason': '{}: {}'.format(type(error).__name__, error)}
    return {
        'file': str(path), 'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'ir_version': model.ir_version,
        'producer': '{} {}'.format(model.producer_name, model.producer_version),
        'opsets': [{'domain': item.domain or 'ai.onnx',
                    'version': item.version} for item in model.opset_import],
        'node_count': len(model.graph.node), 'inputs': inputs,
        'outputs': outputs,
        'custom_ops': {'domain': CUSTOM_DOMAIN, 'op_type': CUSTOM_OP_TYPE,
                       'count': len(custom_nodes), 'input_count': 5,
                       'im2col_step': 64},
        'input_dependencies': dependencies,
        'alignment_standard_ops': alignment_ops,
        'grid_sample_node_count': sum(
            count for (domain, op), count in counts.items()
            if 'gridsample' in op.lower() or 'grid_sampler' in op.lower()),
        'shape_inference': shape_inference,
        'onnx_checker': 'PASS',
    }


def print_report(title, report):
    """以稳定顺序打印三个模型输出的数值比较摘要。"""
    print('\n{}'.format(title))
    print('=' * 64)
    for name in ('bev_embed', 'all_cls_scores', 'all_bbox_preds'):
        values = report[name]
        print('{} shape={} MAE={} MaxAE={} RMSE={} Cosine={}'.format(
            name, values['shape'], values['MAE'], values['MaxAE'],
            values['RMSE'], values['Cosine Similarity']))


def main():
    args = parse_args()

    # sequence.json 是帧连续性与 absolute can_bus 的可信来源。Stateful export
    # 必须使用真实 Frame 1，不能拿 Frame 0 或三个独立 sample 代替。
    sequence = json.loads((args.sequence_root / 'sequence.json').read_text())
    frames = sequence['frames']
    if len(frames) < 2 or frames[1]['sample_token'] != args.token:
        raise ValueError('Sequence Frame 1 does not match --token')
    context = prepare_golden_sample(
        args.token, args.config, args.checkpoint, args.seed)
    model = context['model'].module

    # 复现官方 forward_test 的状态准备：Frame 1 metadata 改为相对运动，
    # Frame 0 encoder 输出直接作为 Frame 1 的 stored previous-BEV state。
    frame1_meta = prepare_runtime_meta(
        context['metas'], frames[1]['can_bus_absolute'],
        frames[0]['can_bus_absolute'])
    frame1_images = context['image_tensor'].unsqueeze(0).cuda()
    frame1_prev = torch.from_numpy(np.load(
        str(args.sequence_root / 'frame_000/bev/final_bev.npy'),
        allow_pickle=False)).cuda()
    can_bus1, shift1, lidar2img1 = tensor_inputs(frame1_meta, model)
    gate1 = torch.ones(1, dtype=torch.float32, device='cuda')
    wrapper1 = BEVFormerStatefulONNXWrapper(model, frame1_meta).cuda().eval()

    # Gate 1：真实历史路径。此验证必须在 torch.onnx.export 前通过，否则导出
    # 出来的图即使 checker PASS，也不能作为 Stateful Golden model。
    with torch.no_grad():
        outputs1 = wrapper1(frame1_images, frame1_prev, can_bus1, shift1,
                            lidar2img1, gate1)
    frame1_report = compare_triplet(
        golden_outputs(args.sequence_root, 'frame_001'), outputs1)
    print_report('PyTorch Stateful Wrapper vs Golden Frame 1', frame1_report)
    assert_close_report(
        frame1_report, allow_failure=args.allow_validation_failure)

    # Sensitivity test 防止 prev_bev 只出现在 graph input 列表中，却没有真正
    # 参与输出计算。相同 Frame 1 输入下，real/zero history 必须产生非零差异。
    zero_prev = torch.zeros_like(frame1_prev)
    with torch.no_grad():
        zero_outputs = wrapper1(frame1_images, zero_prev, can_bus1, shift1,
                                lidar2img1, gate1)
    sensitivity = compare_triplet(
        tuple(output.detach().cpu().numpy() for output in outputs1),
        zero_outputs)
    if all(values['MaxAE'] == 0.0 for values in sensitivity.values()):
        raise RuntimeError('prev_bev sensitivity check produced no difference')
    print_report('Frame 1 real prev_bev vs zero prev_bev sensitivity', sensitivity)

    # Gate 0：ONNX 不能把 prev_bev 设为 None，因此首帧统一使用 zero tensor +
    # use_prev_bev=0。这里将其与官方 use_prev_bev=False/prev_bev=None 路径比较，
    # 证明统一 runtime interface 没有改变首帧语义。
    frame0_info = context['dataset'].data_infos[context['index'] - 1]
    if frame0_info['token'] != frames[0]['sample_token']:
        raise ValueError('Frame 0 is not immediately before Frame 1')
    frame0_data = context['dataset'][context['index'] - 1]
    _, frame0_pipeline_meta = validate_pipeline_item(
        frame0_data, frame0_info, frame0_info['token'])
    frame0_meta = first_frame_runtime_meta(
        frame0_pipeline_meta, frames[0]['can_bus_absolute'])
    frame0_images = pipeline_image_tensor(frame0_data).unsqueeze(0).cuda()
    can_bus0, shift0, lidar2img0 = tensor_inputs(frame0_meta, model)
    gate0 = torch.zeros(1, dtype=torch.float32, device='cuda')
    official0 = BEVFormerONNXWrapper(
        model, frame0_meta, use_prev_bev=False).cuda().eval()
    wrapper0 = BEVFormerStatefulONNXWrapper(model, frame0_meta).cuda().eval()
    with torch.no_grad():
        official0_outputs = official0(frame0_images)
        stateful0_outputs = wrapper0(
            frame0_images, torch.zeros_like(frame1_prev), can_bus0, shift0,
            lidar2img0, gate0)
    frame0_report = compare_triplet(
        tuple(output.detach().cpu().numpy() for output in official0_outputs),
        stateful0_outputs)
    print_report('Frame 0 gate=0 vs official prev_bev=None', frame0_report)
    assert_close_report(
        frame0_report, allow_failure=args.allow_validation_failure)

    # 可选生成 C++ runner fixture。正式导出不依赖这些文件；它们仅用于后续
    # TensorRT Frame 0/1 runtime 与 PyTorch outputs 的可重复比较。
    if args.runtime_fixture_dir:
        save_runtime_frame(args.runtime_fixture_dir, 'frame_000', {
            'images': frame0_images,
            'prev_bev': torch.zeros_like(frame1_prev),
            'can_bus': can_bus0, 'shift': shift0,
            'lidar2img': lidar2img0, 'use_prev_bev': gate0,
        }, dict(zip(
            ('all_cls_scores', 'all_bbox_preds', 'bev_embed'),
            stateful0_outputs)))
        save_runtime_frame(args.runtime_fixture_dir, 'frame_001', {
            'images': frame1_images, 'prev_bev': frame1_prev,
            'can_bus': can_bus1, 'shift': shift1,
            'lidar2img': lidar2img1, 'use_prev_bev': gate1,
        }, dict(zip(
            ('all_cls_scores', 'all_bbox_preds', 'bev_embed'), outputs1)))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # deployment_export_overrides() 只在此 context 内临时安装 ONNX symbolic 和
    # deployment-only forward，包括 Stateful temporal path、static-Nq SCA 及
    # MSDeformableAttention custom op。退出 context 后必须恢复普通 PyTorch 行为。
    # do_constant_folding=False 用于保留已审计的部署图结构，避免 exporter 在此
    # 阶段重写 custom-op 周边 contract。
    with torch.no_grad(), deployment_export_overrides():
        torch.onnx.export(
            wrapper1,
            (frame1_images, frame1_prev, can_bus1, shift1, lidar2img1, gate1),
            str(args.output), opset_version=13,
            input_names=['images', 'prev_bev', 'can_bus', 'shift',
                         'lidar2img', 'use_prev_bev'],
            output_names=['all_cls_scores', 'all_bbox_preds', 'bev_embed'],
            do_constant_folding=False, verbose=False)
    # PyTorch 1.9 可能把固定 Slice axes 导成 Constant→Unsqueeze 子图，而 TRT
    # 8.4 要求 axes 是 build-time weights。兼容 pass 只处理可严格静态求值的
    # axes；真正 runtime-dependent 的情况会 fail closed/skip，绝不硬编码。
    slice_compat = canonicalize_static_slice_axes_file(args.output)

    # Audit 在保存成功后立即执行，使 ONNX 文件与旁路 audit JSON 始终对应。
    audit = audit_onnx(args.output)
    audit['slice_axes_compat'] = slice_compat
    audit['pytorch_validation'] = {
        'frame1_wrapper_vs_golden': frame1_report,
        'frame1_real_vs_zero_prev_sensitivity': sensitivity,
        'frame0_gate0_vs_none': frame0_report,
    }
    audit['runtime_semantics'] = {
        'prev_bev': {'shape': [2500, 1, 256], 'layout': 'Nbev,B,C'},
        'bev_embed': {'shape': [2500, 1, 256], 'layout': 'Nbev,B,C'},
        'can_bus': ('18 float32 values after BEVFormer.forward_test delta '
                    'processing; index -1 rotation delta is degrees'),
        'shift': ('[shift_x, shift_y] computed by the reference host formula'),
        'lidar2img': 'per-frame camera projection matrices',
        'use_prev_bev': 'float32 scalar gate: exactly 0 or 1',
    }
    # Audit JSON 记录文件 hash、接口、custom-op contract、输入依赖、Golden
    # metrics 和 motion 语义，是后续 ONNX/TRT build 的机器可读交接物。
    audit_path = args.output.with_name(args.output.stem + '_audit.json')
    with audit_path.open('w') as stream:
        json.dump(audit, stream, indent=2)
        stream.write('\n')
    print('\nStateful ONNX Audit')
    print('=' * 64)
    print(json.dumps(audit, indent=2))
    print('Audit JSON: {}'.format(audit_path))


if __name__ == '__main__':
    main()
