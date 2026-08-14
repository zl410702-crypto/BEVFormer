import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / 'profile_golden_sample_ops.py'
SPEC = importlib.util.spec_from_file_location('profile_golden_sample_ops', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeEvent:
    key = 'aten::reshape'
    count = 3
    self_cpu_time_total = 12
    cpu_time_total = 14
    self_cuda_time_total = 8
    # Deliberately omit cuda_time_total to exercise PyTorch compatibility.


def test_aggregate_operators_allows_missing_timing_fields():
    operators = MODULE.aggregate_operators([FakeEvent()])
    assert operators == [{
        'name': 'aten::reshape',
        'calls': 3,
        'self_cpu_time_total': 12.0,
        'cpu_time_total': 14.0,
        'self_cuda_time_total': 8.0,
        'cuda_time_total': None,
    }]


def test_potential_risk_operators_uses_names_only():
    operators = [
        {'name': 'aten::conv2d'},
        {'name': 'aten::grid_sampler'},
        {'name': 'mmcv::deform_attn'},
        {'name': 'cuda::kernel'},
        {'name': 'autograd::engine::evaluate_function'},
    ]
    assert MODULE.potential_risk_operators(operators) == [
        'aten::grid_sampler',
        'autograd::engine::evaluate_function',
        'mmcv::deform_attn',
    ]


def test_check_golden_reference_pass_and_fail(tmp_path):
    boxes = np.arange(9, dtype=np.float32).reshape(1, 9)
    scores = np.asarray([0.5], dtype=np.float32)
    labels = np.asarray([2], dtype=np.int64)
    reference = tmp_path / 'detections.npz'
    np.savez(str(reference), boxes_3d=boxes, scores_3d=scores,
             labels_3d=labels)

    passed = MODULE.check_golden_reference(reference, boxes, scores, labels)
    assert passed['pass']
    assert passed['boxes_3d_max_abs_error'] == 0.0
    failed = MODULE.check_golden_reference(
        reference, boxes + 1, scores, labels + 1)
    assert not failed['pass']
    assert failed['boxes_3d_max_abs_error'] == 1.0
    assert not failed['labels_3d_exact_match']


def test_write_reports(tmp_path):
    operators = MODULE.aggregate_operators([FakeEvent()])
    paths = MODULE.write_reports(
        tmp_path, 'token', operators,
        [{'name': 'aten::reshape', 'input_shapes': [[1, 2]],
          'cpu_time': 1.0, 'cuda_time': 2.0}])

    with paths[0].open() as stream:
        assert json.load(stream)['operators'] == operators
    with paths[1].open() as stream:
        rows = list(csv.DictReader(stream))
        assert rows[0]['name'] == 'aten::reshape'
        assert rows[0]['calls'] == '3'
    with paths[2].open() as stream:
        assert json.load(stream)['events'][0]['input_shapes'] == [[1, 2]]
