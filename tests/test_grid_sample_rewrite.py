import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from deployment.grid_sample_rewrite import (  # noqa: E402
    bev_grid_sample_nearest, bev_rotate_nearest,
    bev_rotate_nearest_tensor)


@pytest.mark.parametrize('angle', [0.0, -1.0353196, 17.25])
def test_tensor_angle_rotate_matches_torchvision(angle):
    input_tensor = torch.arange(3 * 50 * 50, dtype=torch.float32).reshape(
        3, 50, 50)
    expected = rotate(input_tensor, angle, center=[100, 100])
    actual = bev_rotate_nearest_tensor(
        input_tensor, torch.tensor([angle]), center=[100, 100])
    torch.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def reference(input_tensor, grid):
    return F.grid_sample(input_tensor, grid, mode='nearest',
                         padding_mode='zeros', align_corners=False)


def assert_matches(input_tensor, grid):
    expected = reference(input_tensor, grid)
    actual = bev_grid_sample_nearest(input_tensor, grid)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def normalized(pixel, size):
    return (2.0 * pixel + 1.0) / size - 1.0


def test_random_small_tensor():
    torch.manual_seed(7)
    assert_matches(torch.randn(1, 1, 4, 5),
                   torch.empty(1, 7, 6, 2).uniform_(-1.5, 1.5))


def test_integer_pixel_sampling_positions():
    grid = torch.tensor([[[[normalized(0, 4), normalized(0, 3)],
                           [normalized(3, 4), normalized(2, 3)]]]])
    assert_matches(torch.arange(12.0).reshape(1, 1, 3, 4), grid)


def test_subpixel_sampling_uses_nearest():
    grid = torch.tensor([[[[normalized(1.49, 4), normalized(1.49, 3)],
                           [normalized(1.51, 4), normalized(1.51, 3)]]]])
    assert_matches(torch.arange(12.0).reshape(1, 1, 3, 4), grid)


@pytest.mark.parametrize('grid', [
    [[[[normalized(0, 4), normalized(1, 3)]]]],       # left boundary
    [[[[normalized(3, 4), normalized(1, 3)]]]],       # right boundary
    [[[[normalized(2, 4), normalized(0, 3)]]]],       # top boundary
    [[[[normalized(2, 4), normalized(2, 3)]]]],       # bottom boundary
])
def test_image_boundaries(grid):
    assert_matches(torch.arange(12.0).reshape(1, 1, 3, 4),
                   torch.tensor(grid))


def test_completely_out_of_bounds():
    grid = torch.tensor([[[[-3.0, 0.0], [3.0, 0.0]],
                           [[0.0, -3.0], [0.0, 3.0]]]])
    assert_matches(torch.ones(1, 1, 3, 4), grid)


@pytest.mark.parametrize('coordinate', [-1.0 - 1e-6, -1.0,
                                         -1.0 + 1e-6])
def test_normalized_coordinate_near_minus_one(coordinate):
    assert_matches(torch.arange(12.0).reshape(1, 1, 3, 4),
                   torch.tensor([[[[coordinate, 0.0]]]]))


@pytest.mark.parametrize('coordinate', [1.0 - 1e-6, 1.0, 1.0 + 1e-6])
def test_normalized_coordinate_near_plus_one(coordinate):
    assert_matches(torch.arange(12.0).reshape(1, 1, 3, 4),
                   torch.tensor([[[[coordinate, 0.0]]]]))


def test_multiple_channels():
    torch.manual_seed(11)
    assert_matches(torch.randn(1, 4, 5, 6),
                   torch.empty(1, 3, 7, 2).uniform_(-1.2, 1.2))


def test_multiple_batches():
    torch.manual_seed(13)
    assert_matches(torch.randn(3, 2, 5, 6),
                   torch.empty(3, 4, 7, 2).uniform_(-1.2, 1.2))


def test_tiny_temporal_rotation_matches_torchvision():
    torch.manual_seed(17)
    input_tensor = torch.randn(4, 50, 50)
    angle = -1.0353196091174368
    expected = rotate(input_tensor, angle, center=[100, 100])
    actual = bev_rotate_nearest(input_tensor, angle, center=[100, 100])
    torch.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_rejects_unsupported_layout():
    with pytest.raises(ValueError):
        bev_grid_sample_nearest(torch.ones(3, 4, 5),
                                torch.ones(1, 2, 2, 2))


def test_real_temporal_fixture():
    fixture_path = (Path(__file__).parents[1] / 'golden_samples' /
                    '3950bd41f74548429c0f7700ff3d8269' /
                    'temporal_grid_sample_reference.pt')
    if not fixture_path.is_file():
        pytest.skip('CUDA-captured real temporal fixture is not available')
    fixture = torch.load(str(fixture_path), map_location='cpu')
    actual = bev_grid_sample_nearest(fixture['prev_bev'], fixture['grid'])
    torch.testing.assert_allclose(actual, fixture['reference_output'],
                                  rtol=0.0, atol=1e-5)


def test_onnx_export_status_is_not_run_without_onnx(tmp_path):
    script = TOOLS / 'validate_grid_sample_rewrite.py'
    spec = importlib.util.spec_from_file_location('validate_rewrite', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if importlib.util.find_spec('onnx') is not None:
        pytest.skip('This test documents the no-onnx reference environment')
    fixture = {'prev_bev': torch.ones(1, 1, 2, 2),
               'grid': torch.zeros(1, 1, 1, 2)}
    result = module.export_and_check(fixture, tmp_path / 'rewrite.onnx')
    assert result['status'] == 'NOT RUN'
    assert result['remaining'] == 'NOT CHECKED'
