import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from mmcv.parallel import DataContainer
"""
cd /home/eric/workspace/bevformer/BEVFormer

    export PYTHONPATH=$(pwd):$PYTHONPATH
    export CUDA_HOME=/usr/local/cuda-11.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    python tools/run_golden_sample.py \
    --token 3e8750f331d7499e9b5123e9eb70f2e2

    实际 checkpoint 位于 /data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth，已作为入口默认值，也
    可通过 --checkpoint 覆盖。
"""


SCRIPT = Path(__file__).parents[1] / 'tools' / 'run_golden_sample.py'
SPEC = importlib.util.spec_from_file_location('run_golden_sample', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_find_sample_index():
    infos = [{'token': 'other'}, {'token': 'golden'}]
    assert MODULE.find_sample_index(infos, 'golden') == 1
    with pytest.raises(ValueError):
        MODULE.find_sample_index(infos, 'missing')


def test_import_plugin_uses_configured_package():
    with mock.patch.object(MODULE.importlib, 'import_module') as import_module:
        MODULE.import_plugin({'plugin': True,
                              'plugin_dir': 'projects/mmdet3d_plugin/'},
                             'unused.py')
    import_module.assert_called_once_with('projects.mmdet3d_plugin')


def test_pipeline_image_tensor_unwraps_test_augmentation():
    image = torch.zeros(6, 3, 4, 5)
    data = {'img': [DataContainer(image)]}
    assert MODULE.pipeline_image_tensor(data) is image


def test_save_reference_round_trip(tmp_path):
    boxes = np.arange(18, dtype=np.float32).reshape(2, 9)
    scores = np.asarray([0.75, 0.25], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    npz_path, json_path, summary = MODULE.save_reference(
        tmp_path, 'golden', 123, boxes, scores, labels,
        ['car', 'truck'], {'camera_names': ['camera'] * 6})

    saved = np.load(str(npz_path))
    np.testing.assert_array_equal(saved['boxes_3d'], boxes)
    np.testing.assert_array_equal(saved['scores_3d'], scores)
    np.testing.assert_array_equal(saved['labels_3d'], labels)
    assert saved['sample_token'].item() == 'golden'
    assert saved['timestamp'].item() == 123
    assert json_path.is_file()
    assert summary['detection_count'] == 2
