import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / 'tools' /
          'analyze_operator_compatibility.py')
SPEC = importlib.util.spec_from_file_location(
    'analyze_operator_compatibility', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_classify_runtime_op_filters_implementation_layers():
    assert MODULE.classify_runtime_op('aten::matmul') == (
        'PyTorch ATen operator')
    assert MODULE.classify_runtime_op(
        'MultiScaleDeformableAttnFunction_fp32') == 'MMCV / custom operator'
    assert MODULE.classify_runtime_op('cutlass::Kernel') == (
        'cuDNN / CUTLASS / GEMM implementation kernel')
    assert MODULE.classify_runtime_op(
        'void ms_deformable_im2col_gpu_kernel<float>()') == 'CUDA kernel'
    assert MODULE.classify_runtime_op(
        'void at::native::unrolled_elementwise_kernel()') == 'CUDA kernel'


def test_effective_status_preserves_explicit_unsupported():
    entry = {
        'onnx_opset13_status': 'UNSUPPORTED',
        'tensorrt_8_4_12_status': 'UNKNOWN',
    }
    assert MODULE.status_for_count(entry) == 'UNSUPPORTED'


def test_repository_database_is_valid():
    database = MODULE.load_database(
        Path(__file__).parents[1] / 'docs' / 'operator_compatibility.yaml')
    names = {entry['runtime_op'] for entry in database['operators']}
    assert 'MultiScaleDeformableAttnFunction_fp32' in names
    assert 'aten::grid_sampler' in names
    assert 'aten::atan2' in names
