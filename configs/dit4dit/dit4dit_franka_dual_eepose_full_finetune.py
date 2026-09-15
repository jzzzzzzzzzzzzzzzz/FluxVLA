# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DiT4DiT fine-tuning for Task 1 on dual-Franka absolute EE poses."""

import os

_base_ = ['./dit4dit_robocasa_full_data_full_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_STATISTIC_NAME = 'private'
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')
_COSMOS_BASE_MODEL = os.path.join(_CHECKPOINT_ROOT, 'Cosmos-Predict2.5-2B')
_COSMOS_TOKENIZER = dict(
    type='PretrainedTokenizer',
    model_path=_COSMOS_BASE_MODEL + '/tokenizer',
    model_max_length=512)

model = dict(
    vlm_backbone=dict(base_model=_COSMOS_BASE_MODEL),
    vla_head=dict(ori_action_dim=16),
)
inference_model = dict(
    vlm_backbone=dict(base_model=_COSMOS_BASE_MODEL),
    vla_head=dict(ori_action_dim=16),
)

train_dataloader = dict(
    per_device_num_workers=1,
    dataset=dict(
        _delete_=True,
        type='DistributedRepeatingDataset',
        auto_compute_statistics=dict(profile='franka-eepose'),
        name_mappings={'observation.eepose': ['proprio', 'action']},
        statistic_keys=['observation.eepose', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        shuffle=False,
        reshuffle_each_epoch=False,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATASET_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='observation.eepose',
                use_delta=False,
                window_start_idx=0,
                action_window_size=16,
                frame_window_size=9,
                frame_sample_stride=2,
                require_full_window=True,
                action_dtype=None,
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
                        type='ProcessCosmos25Prompt',
                        tokenizer=_COSMOS_TOKENIZER,
                        input_key='task_description',
                        remove_input_key=True),
                    dict(
                        type='ResizeImages',
                        height=224,
                        width=224,
                        backend='torch',
                        scale_divisor=255.0,
                        output_layout='flattened_chw'),
                    dict(
                        type='PrepareVideo',
                        num_views=3,
                        frame_window_size=9,
                        tile_direction='top_bottom_pair',
                        top_view=0,
                        bottom_views=(1, 2),
                        bottom_height_ratio=0.5,
                        combine_view_masks=True),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_key='proprio',
                        action_key='action',
                        state_dim=64,
                        action_dim=32,
                        state_norm_type='quantile',
                        action_norm_type='min_max',
                        norm_type='min_max',
                        clip_norm=False,
                        normalization_epsilon=0.0,
                        preserve_input_dtype=True,
                        valid_action_dim=16,
                        mark_all_action_steps_valid=True,
                        output_dtype='float16'),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=16,
                        valid_action_dim=16,
                        dtype='float16'),
                ],
            ),
        ],
    ),
)

runner = dict(
    max_epochs=None,
    max_steps=20_000,
    save_iter_interval=3000,
    max_keep_ckpts=8,
    tokenizer=_COSMOS_TOKENIZER,
)
eval = None
themis = None

inference = dict(
    type='FrankaInferenceRunner',
    seed=7,
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_suite_name=_STATISTIC_NAME,
    task_descriptions={'1': _TASK_DESCRIPTION},
    state_dim=16,
    action_chunk=16,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=16,
    prepare_pose=None,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='ProcessCosmos25Prompt',
                tokenizer=_COSMOS_TOKENIZER,
                input_key='task_description',
                remove_input_key=True),
            dict(
                type='ResizeImages',
                height=224,
                width=224,
                backend='torch',
                scale_divisor=255.0,
                output_layout='flattened_chw'),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5,
                combine_view_masks=True),
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key=None,
                state_dim=64,
                state_norm_type='quantile',
                norm_type='min_max',
                clip_norm=False,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
                output_dtype='float32'),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=16,
                valid_action_dim=16,
                dtype='float32'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='min_max',
        action_dim=16,
        clip_normalized_action=True),
    operator=dict(
        type='FrankaDualOperator',
        image_encoding='rgb8',
        command_mode='cartesian',
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_right_topic='/camera_right_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        puppet_franka_state_right_topic=(
            '/right_arm/franka_state_controller/franka_states'),
        sync_warning_enabled=True,
        cartesian_cmd_left_topic=(
            '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
        cartesian_cmd_right_topic=(
            '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal'))
