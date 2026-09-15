# Copyright 2026 Limx Dynamics

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
DATASET_SUFFIX = '20260519_dual_franka_teleop'
VIDEO_KEYS = [
    'observation.images.cam_front',
    'observation.images.cam_wrist_left',
    'observation.images.cam_wrist_right',
]

CONFIGS = (
    ('gr00tn17/gr00tn17_qwen3vl_2b_franka_dual_eepose_full_finetune.py',
     'GrootN17VLA', 40),
    ('gr00tn15/gr00tn15_eagle_3b_franka_dual_eepose_full_finetune.py',
     'LlavaVLA', 32),
    ('pi0/pi0_paligemma_franka_dual_eepose_full_finetune.py',
     'PI0FlowMatching', 50),
    ('pi05/pi05_paligemma_franka_dual_eepose_20260519_full_finetune.py',
     'PI05FlowMatching', 50),
    ('fastwam/fastwam_franka_dual_eepose_full_finetune.py', 'FastWAMVLA', 32),
    ('smolvla/smolvla_franka_dual_eepose_finetune.py', 'SmolVLAFlowMatching',
     50),
    ('cosmos3/cosmos3edge_franka_dual_eepose_full_finetune.py',
     'Cosmos3FlowMatching', 32),
    ('dit4dit/dit4dit_franka_dual_eepose_full_finetune.py', 'DiT4DiTVLA', 16),
)


def _source_dataset(cfg):
    datasets = cfg.train_dataloader.dataset.datasets
    return datasets[0] if isinstance(datasets, (list, tuple)) else datasets


def _process_inputs(source):
    return next(transform for transform in source.transforms
                if transform.type == 'ProcessParquetInputs')


@pytest.mark.parametrize(('relative_path', 'model_type', 'horizon'), CONFIGS)
def test_franka_task1_train_and_inference_contract(relative_path, model_type,
                                                   horizon):
    cfg = Config.fromfile(ROOT / 'configs' / relative_path)
    wrapper = cfg.train_dataloader.dataset
    source = _source_dataset(cfg)

    assert cfg.model.type == model_type
    assert wrapper.statistic_name == 'private'
    assert source.action_key == 'observation.eepose'
    assert source.action_window_size == horizon
    assert not (source.get('require_full_window', False)
                and source.get('supervise_terminal_padding', False))
    assert str(source.data_root_path).endswith(DATASET_SUFFIX)
    assert _process_inputs(source).video_keys == VIDEO_KEYS

    assert cfg.inference.type == 'FrankaInferenceRunner'
    assert cfg.inference.action_mode == 'cartesian'
    assert tuple(cfg.inference.active_arms) == ('left', 'right')
    assert cfg.inference.state_dim == 16
    assert cfg.inference.action_chunk == horizon
    assert cfg.inference.denormalize_action.action_dim == 16
    assert cfg.inference.operator.type == 'FrankaDualOperator'
    assert cfg.inference.operator.command_mode == 'cartesian'
    assert cfg.inference.dataset.img_keys == [
        'cam_front', 'cam_wrist_left', 'cam_wrist_right'
    ]


@pytest.mark.parametrize(
    'relative_path',
    [item[0] for item in CONFIGS if not item[0].startswith('gr00tn17/')],
)
def test_new_task1_configs_compute_absolute_franka_statistics(relative_path):
    cfg = Config.fromfile(ROOT / 'configs' / relative_path)
    dataset = cfg.train_dataloader.dataset

    assert dataset.auto_compute_statistics == dict(profile='franka-eepose')
    assert dataset.name_mappings == {
        'observation.eepose': ['proprio', 'action']
    }


def test_fastwam_reinitializes_only_shape_dependent_input_output_layers():
    cfg = Config.fromfile(
        ROOT / 'configs/fastwam/fastwam_franka_dual_eepose_full_finetune.py')

    assert cfg.model.proprio_dim == 16
    assert cfg.model.vla_head.action_dit_config.action_dim == 16
    assert cfg.model.vla_head.video_dit_config.action_dim == 16
    assert cfg.model.pretrained_skip_prefixes == [
        'vla_head.mot.mixtures.action.action_encoder.',
        'vla_head.mot.mixtures.action.head.',
        'vla_head.proprio_encoder.',
    ]

    source = _source_dataset(cfg)
    train_resize = next(t for t in source.transforms
                        if t.type == 'ResizeImages')
    inference_resize = next(t for t in cfg.inference.dataset.transforms
                            if t.type == 'ResizeImages')
    assert (train_resize.height, train_resize.width) == (256, 256)
    assert (inference_resize.height, inference_resize.width) == (256, 256)

    # top_bottom_pair produces H=1.5*256, W=256. Wan VAE downsamples by 16
    # before the video DiT applies a 2x2 patch.
    assert (int(1.5 * train_resize.height) // 16) % 2 == 0
    assert (train_resize.width // 16) % 2 == 0


def test_dit4dit_adds_singleton_state_history_axis():
    cfg = Config.fromfile(
        ROOT / 'configs/dit4dit/dit4dit_franka_dual_eepose_full_finetune.py')
    source = _source_dataset(cfg)
    train_prepare = next(t for t in source.transforms
                         if t.type == 'PrepareStateActionTargets')
    inference_prepare = next(t for t in cfg.inference.dataset.transforms
                             if t.type == 'PrepareStateActionTargets')

    assert train_prepare.state_history_length == 1
    assert train_prepare.action_horizon == 16
    assert train_prepare.valid_action_dim == 16
    assert train_prepare.dtype == 'float16'
    assert inference_prepare.state_history_length == 1
    assert inference_prepare.dtype == 'float32'


def test_pi0_uses_its_matching_local_tokenizer(monkeypatch):
    checkpoint_root = '/mnt/data/cpfs/users/leo/checkpoints'
    monkeypatch.setenv('FLUXVLA_CHECKPOINT_ROOT', checkpoint_root)
    cfg = Config.fromfile(
        ROOT / 'configs/pi0/pi0_paligemma_franka_dual_eepose_full_finetune.py')

    assert cfg.model.pretrained_name_or_path == (
        f'{checkpoint_root}/pi0_base/model.safetensors')
    assert cfg.runner.tokenizer.model_path == f'{checkpoint_root}/pi0_base'
    assert cfg.runner.keep_params_fp32 is True
    assert cfg.runner.sharding_strategy == 'global-shard-grad-op'


def test_model_agnostic_remote_client_uses_safe_common_horizon(monkeypatch):
    monkeypatch.setenv('FRANKA_ZMQ_HOST', 'model-dsw')
    monkeypatch.setenv('FRANKA_ZMQ_PORT', '3333')
    cfg = Config.fromfile(
        ROOT / 'configs/franka/franka_dual_eepose_task1_remote_inference.py')

    assert cfg.inference.type == 'FrankaInferenceRunner'
    assert cfg.inference.remote_inference.server_host == 'model-dsw'
    assert cfg.inference.remote_inference.server_port == 3333
    assert cfg.inference.execute_horizon == 16
    assert cfg.inference.action_mode == 'cartesian'
    assert cfg.inference.operator.type == 'FrankaDualOperator'
