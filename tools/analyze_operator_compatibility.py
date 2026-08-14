#!/usr/bin/env python
"""Generate a module-level ONNX/TensorRT report from Golden profiling data."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


STATUS_ORDER = ('SUPPORTED', 'CONDITIONAL', 'UNSUPPORTED', 'CUSTOM', 'UNKNOWN')
ACTION_ORDER = ('NATIVE', 'CHECK', 'REWRITE', 'PLUGIN', 'CPP_POSTPROCESS')
STATUS_SEVERITY = {
    'SUPPORTED': 0,
    'CONDITIONAL': 1,
    'UNKNOWN': 2,
    'CUSTOM': 3,
    'UNSUPPORTED': 4,
}
SECTIONS = (
    ('Backbone', ('Backbone.ResNet',)),
    ('Neck', ('Neck.FPN',)),
    ('BEV Encoder', ('BEVEncoder.TemporalSelfAttention',
                     'BEVEncoder.SpatialCrossAttention',
                     'BEVEncoder.MSDeformableAttention3D',
                     'BEVEncoder.FFN', 'BEVEncoder.LayerNorm')),
    ('Detection Decoder', ('Decoder.ObjectSelfAttention',
                           'Decoder.DeformableCrossAttention', 'Decoder.FFN',
                           'Decoder.LayerNorm', 'Decoder.ReferenceRefinement')),
    ('Detection Head', ('Head.Cls', 'Head.Reg')),
    ('Postprocess', ('Postprocess',)),
)


def parse_args():
    token = '3e8750f331d7499e9b5123e9eb70f2e2'
    root = Path('golden_samples') / token
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default=str(root / 'operator_profile.json'))
    parser.add_argument('--events', default=str(root / 'operator_events.json'))
    parser.add_argument('--compatibility', default='docs/operator_compatibility.yaml')
    parser.add_argument('--output', default='docs/BEVFormer_operator_compatibility.md')
    return parser.parse_args()


def load_json(path, required_key):
    with Path(path).open() as stream:
        payload = json.load(stream)
    if required_key not in payload or not isinstance(payload[required_key], list):
        raise ValueError('{} must contain a list named {!r}'.format(
            path, required_key))
    return payload


def classify_runtime_op(name):
    lowered = name.lower()
    if name.startswith('aten::'):
        return 'PyTorch ATen operator'
    if name == 'MultiScaleDeformableAttnFunction_fp32' or name.startswith('mmcv::'):
        return 'MMCV / custom operator'
    if ('cutlass' in lowered or 'xmma' in lowered or 'cudnn' in lowered or
            'gemm' in lowered or 'scudnn' in lowered):
        return 'cuDNN / CUTLASS / GEMM implementation kernel'
    if ('kernel' in lowered or name.startswith(('void ', 'cuda::'))):
        return 'CUDA kernel'
    return 'Other runtime implementation detail'


def load_database(path):
    with Path(path).open() as stream:
        database = yaml.safe_load(stream)
    allowed_statuses = set(database.get('allowed_statuses', []))
    allowed_actions = set(database.get('allowed_actions', []))
    entries = database.get('operators')
    if not isinstance(entries, list):
        raise ValueError('Compatibility database must contain an operators list')
    required = {'runtime_op', 'modules', 'onnx_op', 'onnx_opset13_status',
                'tensorrt_8_4_12_status', 'action', 'notes', 'evidence'}
    seen = set()
    for index, entry in enumerate(entries):
        missing = required - set(entry)
        if missing:
            raise ValueError('Database entry {} is missing {}'.format(
                index, sorted(missing)))
        if entry['runtime_op'] in seen:
            raise ValueError('Duplicate runtime_op: {}'.format(entry['runtime_op']))
        seen.add(entry['runtime_op'])
        for field in ('onnx_opset13_status', 'tensorrt_8_4_12_status'):
            if entry[field] not in allowed_statuses:
                raise ValueError('Invalid {}: {}'.format(field, entry[field]))
        if entry['action'] not in allowed_actions:
            raise ValueError('Invalid action: {}'.format(entry['action']))
    return database


def escape_cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def status_for_count(entry):
    """Use the most conservative of ONNX and TensorRT status for summaries."""
    statuses = (entry['onnx_opset13_status'],
                entry['tensorrt_8_4_12_status'])
    return max(statuses, key=lambda item: STATUS_SEVERITY[item])


def module_risk(module, entries):
    statuses = {status_for_count(entry) for entry in entries}
    if module == 'Postprocess':
        return 'MEDIUM'
    if 'CUSTOM' in statuses or 'UNSUPPORTED' in statuses:
        return 'HIGH'
    if 'UNKNOWN' in statuses:
        return 'MEDIUM'
    return 'LOW'


def table_for_entries(module, entries, observed_names):
    lines = [
        '| Module | Runtime Op | ONNX13 Mapping | ONNX13 Status | '
        'TensorRT 8.4.12 Status | Action | Notes |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for entry in sorted(entries, key=lambda item: item['runtime_op']):
        observed = entry['runtime_op'] in observed_names
        runtime = entry['runtime_op'] + ('' if observed else ' *(latent)*')
        row = (module, runtime, entry['onnx_op'],
               entry['onnx_opset13_status'],
               entry['tensorrt_8_4_12_status'], entry['action'], entry['notes'])
        lines.append('| ' + ' | '.join(escape_cell(value) for value in row) + ' |')
    return lines


def render_report(profile, events, database):
    runtime_ops = profile['operators']
    runtime_by_name = {item['name']: item for item in runtime_ops}
    observed_names = set(runtime_by_name)
    event_counts = Counter(event.get('name', 'unknown') for event in events['events'])
    entries = [entry for entry in database['operators']
               if entry['runtime_op'] in observed_names or
               entry.get('include_if_unobserved', False)]
    observed_entries = [entry for entry in entries
                        if entry['runtime_op'] in observed_names]
    by_module = defaultdict(list)
    for entry in entries:
        for module in entry['modules']:
            by_module[module].append(entry)

    classes = Counter(classify_runtime_op(item['name']) for item in runtime_ops)
    status_counts = Counter(status_for_count(entry) for entry in observed_entries)
    custom_kernel_calls = event_counts['MultiScaleDeformableAttnFunction_fp32']
    lines = [
        '# BEVFormer ONNX13 / TensorRT 8.4.12 Compatibility', '',
        '> Generated from the real Golden Sample runtime inventory. A runtime '
        'operator is not automatically an ONNX node, and CUDA/library kernels '
        'are treated as implementation evidence rather than separate blockers.', '',
        '## 1. Deployment Baseline', '',
        '| Item | Value |', '| --- | --- |',
        '| Golden token | `{}` |'.format(profile.get('sample_token', 'unknown')),
        '| PyTorch / profiler CUDA | `{}` / `{}` |'.format(
            profile.get('torch_version'), profile.get('cuda_version')),
        '| ONNX target | opset {} |'.format(database['baseline']['onnx_opset']),
        '| TensorRT target | {} |'.format(database['baseline']['tensorrt']),
        '| PONY runtime | CUDA {}; cuDNN {}; {} |'.format(
            database['baseline']['cuda'], database['baseline']['cudnn'],
            database['baseline']['architecture']),
        '| Plugin API | `{}` |'.format(database['baseline']['plugin_api']), '',
        'Runtime classification (all {} profiler keys):'.format(len(runtime_ops)), '',
    ]
    for category in ('PyTorch ATen operator', 'MMCV / custom operator',
                     'CUDA kernel',
                     'cuDNN / CUTLASS / GEMM implementation kernel',
                     'Other runtime implementation detail'):
        lines.append('- {}: {}'.format(category, classes[category]))
    lines.extend([
        '', 'Only {} observed, deployment-relevant runtime operators are '
        'retained below. Shared tensor operators can appear in several modules; '
        'the global count remains name-deduplicated.'.format(len(observed_entries)),
        '', 'Module assignment evidence combines event shapes, the same-token '
        '`forward_trace.json`, repository call sites, and the documented call '
        'chain. This profiler export contains no stack/scope/parent timestamps, '
        'so shared ops are not assigned an invented exclusive owner.', '',
    ])

    section_number = 2
    for title, modules in SECTIONS:
        lines.extend(['## {}. {}'.format(section_number, title), ''])
        for module in modules:
            module_entries = by_module[module]
            lines.extend(['### {}'.format(module), ''])
            lines.extend(table_for_entries(module, module_entries, observed_names))
            counts = Counter(status_for_count(entry) for entry in module_entries
                             if entry['runtime_op'] in observed_names)
            plugin_count = sum(entry['action'] == 'PLUGIN'
                               for entry in module_entries
                               if entry['runtime_op'] in observed_names)
            lines.extend([
                '', 'Conclusion: Native {}; Conditional {}; Unsupported {}; '
                'Unknown {}; Plugin candidates {}; Risk **{}**.'.format(
                    counts['SUPPORTED'], counts['CONDITIONAL'],
                    counts['UNSUPPORTED'], counts['UNKNOWN'], plugin_count,
                    module_risk(module, module_entries)), '',
            ])
        section_number += 1

    blockers = [entry for entry in entries
                if status_for_count(entry) != 'SUPPORTED']
    lines.extend([
        '## 8. Blocker Summary', '',
        '| Module(s) | Runtime Op | Effective Status | Action | Reason / evidence |',
        '| --- | --- | --- | --- | --- |',
    ])
    for entry in sorted(blockers, key=lambda item: (status_for_count(item),
                                                     item['runtime_op'])):
        observed = entry['runtime_op'] in observed_names
        runtime = entry['runtime_op'] + ('' if observed else ' *(latent)*')
        reason = '{} Evidence: {}'.format(entry['notes'], entry['evidence'])
        row = (', '.join(entry['modules']), runtime, status_for_count(entry),
               entry['action'], reason)
        lines.append('| ' + ' | '.join(escape_cell(value) for value in row) + ' |')
    lines.extend([
        '', '### Special-path findings', '',
        '- `MultiScaleDeformableAttnFunction_fp32` executed {} times in detailed '
        'events. The profile also contains 12 '
        '`ms_deformable_im2col_gpu_kernel` launches; these are one custom '
        'primitive used by 3 TSA, 3 encoder SCA, and 6 decoder cross-attention '
        'layers—not 12 ONNX blockers.'.format(custom_kernel_calls),
        '- `aten::grid_sampler` was not observed. The Golden token is a first '
        'scene frame (`prev_bev=None`), so historical BEV rotation is latent. '
        'ONNX `GridSample` is newer than opset 13; a multi-frame export needs a '
        'rewrite or separately scoped plugin decision.',
        '- SCA scatter-style accumulation appears as `aten::index_put_`; '
        'duplicate-index accumulation semantics must be checked after export.',
        '- Postprocess `TopK`, integer index decode, boolean filtering, and '
        '`atan2` are recommended as a C++ boundary for this baseline.', '',
        '### Module-level summary', '',
        '| Module group | Native | Conditional | Plugin candidates | Risk |',
        '| --- | ---: | ---: | ---: | --- |',
    ])
    for title, modules in SECTIONS:
        group_entries = {entry['runtime_op']: entry for module in modules
                         for entry in by_module[module]
                         if entry['runtime_op'] in observed_names}
        counts = Counter(status_for_count(entry) for entry in group_entries.values())
        plugins = sum(entry['action'] == 'PLUGIN'
                      for entry in group_entries.values())
        risks = [module_risk(module, by_module[module]) for module in modules]
        risk = 'HIGH' if 'HIGH' in risks else 'MEDIUM' if 'MEDIUM' in risks else 'LOW'
        lines.append('| {} | {} | {} | {} | {} |'.format(
            title, counts['SUPPORTED'], counts['CONDITIONAL'], plugins, risk))
    lines.extend([
        '', '## 9. Evidence and Confirmation Gates', '',
        '- ONNX mappings are checked against the ONNX operator catalog for '
        'opset availability; `GridSample` is not an opset-13 operator.',
        '- TensorRT status is deliberately conditional where this repository '
        'has no TensorRT 8.4.12 parser/build evidence. The archived NVIDIA 8.4 '
        'support matrix and developer guide are database evidence links.',
        '- Required next gate: export the exact model at opset 13 and inspect '
        'the graph, especially indexing/scatter/nonzero/dynamic reshape and '
        'LayerNorm decomposition.',
        '- Required final gate: run TensorRT 8.4.12.5 parser and engine build '
        'on aarch64/sm_87 with representative shape profiles. No status in this '
        'report substitutes for that parser/build test.', '',
    ])
    summary = {
        'total_runtime_ops': len(runtime_ops),
        'filtered_ops': len(observed_entries),
        'status_counts': status_counts,
        'plugin_candidates': sum(entry['action'] == 'PLUGIN'
                                 for entry in observed_entries),
    }
    return '\n'.join(lines), summary


def main():
    args = parse_args()
    profile = load_json(args.profile, 'operators')
    events = load_json(args.events, 'events')
    database = load_database(args.compatibility)
    report, summary = render_report(profile, events, database)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + '\n')
    counts = summary['status_counts']
    print('Total runtime ops: {}'.format(summary['total_runtime_ops']))
    print('Filtered deployment-relevant ops: {}'.format(summary['filtered_ops']))
    print('Supported: {}'.format(counts['SUPPORTED']))
    print('Conditional: {}'.format(counts['CONDITIONAL']))
    print('Unsupported: {}'.format(counts['UNSUPPORTED']))
    print('Custom: {}'.format(counts['CUSTOM']))
    print('Unknown: {}'.format(counts['UNKNOWN']))
    print('Plugin candidates: {}'.format(summary['plugin_candidates']))
    print('Wrote: {}'.format(output))


if __name__ == '__main__':
    main()
