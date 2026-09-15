import json
import importlib.util
from pathlib import Path

import numpy as np
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / 'fluxvla/engines/runners'
    '/serving/debug_capture.py')
_SPEC = importlib.util.spec_from_file_location('remote_debug_capture',
                                               _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DebugCaptureWriter = _MODULE.DebugCaptureWriter


def test_debug_capture_saves_raw_model_inputs_and_actions(tmp_path):
    writer = DebugCaptureWriter(str(tmp_path), every=1, max_captures=1)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 255
    request_dir = writer.start(
        0, {
            'qpos': np.arange(16, dtype=np.float32),
            'cam_front': image,
            'task_description': 'task one',
        }, 'private')

    assert request_dir is not None
    writer.write_mapping(request_dir, 'model_inputs', {
        'states': torch.ones(1, 16),
        'images': [torch.zeros(1, 3, 4, 4)],
    })
    writer.write_array(request_dir, 'normalized_actions',
                       torch.zeros(1, 2, 16))
    writer.finish(request_dir, 0.25)

    metadata = json.loads((request_dir / 'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['raw_observation']['values']['task_description'] == \
        'task one'
    assert metadata['model_inputs']['arrays']['states']['shape'] == [1, 16]
    assert (request_dir / 'raw_observation/cam_front.png').is_file()
    assert np.load(request_dir / 'raw_observation/qpos.npy').shape == (16, )
    assert writer.start(1, {}, '') is None


def test_debug_capture_interval(tmp_path):
    writer = DebugCaptureWriter(str(tmp_path), every=2, max_captures=2)
    assert writer.start(1, {}, '') is None
    assert writer.start(2, {}, '') is not None
    assert writer.start(3, {}, '') is None
    assert writer.start(4, {}, '') is not None
    assert writer.start(6, {}, '') is None
