import importlib.util
import sys
from pathlib import Path

import torch


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    'trace_bevformer_forward', TOOLS / 'trace_bevformer_forward.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_describe_tensor():
    description = MODULE.describe(torch.zeros(1, 2, 3))
    assert description['shape'] == [1, 2, 3]
    assert description['dtype'] == 'torch.float32'


def test_recorder_captures_shape_before_in_place_change():
    class Squeeze:
        def forward(self, value):
            value.squeeze_()
            return value

    module = Squeeze()
    recorder = MODULE.TraceRecorder()
    recorder.wrap(module, 'forward', 'squeeze')
    output = module.forward(torch.zeros(1, 2, 3))

    assert recorder.events[0]['inputs'][0]['shape'] == [1, 2, 3]
    assert recorder.events[0]['output']['shape'] == [2, 3]
    assert list(output.shape) == [2, 3]
