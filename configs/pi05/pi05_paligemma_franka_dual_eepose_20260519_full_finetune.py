# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PI0.5 Task-1 recipe pinned to the 20260519 dual-Franka dataset."""

import os

_base_ = ['./pi05_paligemma_franka_dual_eepose_full_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_TOKENIZER_PATH = os.path.join(_CHECKPOINT_ROOT, 'pi05_base')

train_dataloader = dict(
    dataset=dict(
        statistic_name='private',
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATASET_ROOT,
                action_key='observation.eepose',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.eepose',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
                        ],
                        video_keys=[
                            'observation.images.cam_front',
                            'observation.images.cam_wrist_left',
                            'observation.images.cam_wrist_right',
                        ],
                        name_mappings={
                            'observation.eepose': ['states'],
                            'actions': ['actions'],
                        }),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=None,
                        state_dim=None,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        output_dtype='float32'),
                    dict(type='PreparePromptWithState'),
                    dict(
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_TOKENIZER_PATH)),
                    dict(type='PadStatesAndActions', model_action_dim=32),
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ],
                action_window_size=50,
                window_start_idx=0,
                supervise_terminal_padding=True,
            ),
        ],
    ), )

runner = dict(
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_TOKENIZER_PATH,
    ), )

inference = dict(task_suite_name='private', state_dim=16)
