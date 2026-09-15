# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Cosmos3-Edge policy tuning for Task 1 on dual-Franka EE poses."""

import os

_base_ = ['./cosmos3edge_aloha_policy_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_COSMOS3_ROOT = os.path.join(_CHECKPOINT_ROOT, 'Cosmos3-Edge')
_WAN22_VAE = os.path.join(_CHECKPOINT_ROOT, 'Wan2.2-TI2V-5B/Wan2.2_VAE.pth')
_STATISTIC_NAME = 'private'
_EMBODIMENT_ID = 13
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')
_TOKENIZER = dict(
    type='PretrainedTokenizer',
    model_path=_COSMOS3_ROOT,
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True)
_ACTION_PROMPT_METADATA = dict(
    append_viewpoint=False,
    conditioning_fps=30.0,
    frame_window_size=33,
    video_height=384,
    video_width=256)

model = dict(
    pretrained_name_or_path=os.path.join(_COSMOS3_ROOT, 'transformer'),
    vision_vae=dict(pretrained_name_or_path=_WAN22_VAE),
    ori_action_dim=16,
)
inference_model = dict(
    pretrained_name_or_path=os.path.join(_COSMOS3_ROOT, 'transformer'),
    ori_action_dim=16,
)

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='DistributedRepeatingDataset',
        auto_compute_statistics=dict(profile='franka-eepose'),
        name_mappings={'observation.eepose': ['proprio', 'action']},
        statistic_keys=['observation.eepose', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_DATASET_ROOT,
            statistic_name=_STATISTIC_NAME,
            action_key='observation.eepose',
            use_delta=False,
            window_start_idx=1,
            action_window_size=32,
            frame_window_size=33,
            require_full_window=True,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=_EMBODIMENT_ID,
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
                dict(type='ResizeImages', height=256, width=256),
                dict(
                    type='AugVideo',
                    rotation_range=0.0,
                    brightness_range=(0.7, 1.3),
                    contrast_range=(0.6, 1.4),
                    crop_scale=(0.95, 0.95),
                    crop_ratio=(1.0, 1.0),
                    prob=1.0,
                    saturation_range=(0.5, 1.5),
                    hue_delta=0.08),
                dict(
                    type='ProcessCosmos3Prompt',
                    tokenizer=_TOKENIZER,
                    max_len=512,
                    cfg_dropout_rate=0.1,
                    action_metadata=_ACTION_PROMPT_METADATA),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=64,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std'),
                dict(
                    type='BuildCosmos3Sequence',
                    raw_action_dim=16,
                    mode='wam',
                    frame_window_size=33,
                    prepend_state_to_action=True,
                    conditioning_fps=30.0),
                dict(
                    type='PrepareVideo',
                    num_views=3,
                    frame_window_size=33,
                    tile_direction='top_bottom_pair',
                    top_view=0,
                    bottom_views=(1, 2),
                    bottom_height_ratio=0.5),
            ],
        ),
    ), )

runner = dict(
    max_epochs=None,
    max_steps=20_000,
    save_iter_interval=3000,
    max_keep_ckpts=8,
    tokenizer=_TOKENIZER,
)
eval = None

inference = dict(
    _delete_=True,
    type='FrankaInferenceRunner',
    seed=7,
    mixed_precision_dtype='bf16',
    task_suite_name=_STATISTIC_NAME,
    task_descriptions={'1': _TASK_DESCRIPTION},
    state_dim=16,
    action_chunk=32,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=20,
    prepare_pose=None,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        embodiment_id=_EMBODIMENT_ID,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=30.0,
                prepend_state_to_action=True),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_TOKENIZER,
                max_len=512,
                cfg_dropout_rate=0.0,
                action_metadata=_ACTION_PROMPT_METADATA,
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks'),
            dict(type='ResizeImages', height=256, width=256),
            dict(type='SimpleNormalizeImages'),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=64,
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std'),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='mean_std',
        action_dim=16),
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
