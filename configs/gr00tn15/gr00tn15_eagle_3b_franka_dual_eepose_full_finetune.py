# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GR00T N1.5 fine-tuning for Task 1 on dual-Franka EE poses."""

import os

_base_ = ['./gr00tn15_eagle_3b_aloha_full_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_BASE_MODEL = os.path.join(_CHECKPOINT_ROOT, 'GR00T-N1.5-3B')
_STATISTIC_NAME = 'private'
_EMBODIMENT_ID = 13
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')
_TOKENIZER_PATH = 'fluxvla/models/third_party_models/eagle2_hg_model'

model = dict(
    pretrained_name_or_path=_BASE_MODEL,
    vla_head=dict(ori_action_dim=16),
)
inference_model = dict(
    pretrained_name_or_path=_BASE_MODEL,
    vla_head=dict(ori_action_dim=16),
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
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATASET_ROOT,
                action_key='observation.eepose',
                use_delta=False,
                window_start_idx=0,
                supervise_terminal_padding=True,
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
                    dict(
                        type='ParquetPrompter',
                        use_conversation=False,
                        add_new_line=True),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_TOKENIZER_PATH)),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375]] * 3,
                        stds=[[58.27148438, 57.02636719, 57.27539062]] * 3),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std'),
                ],
                action_window_size=32,
            ),
        ],
    ), )

runner = dict(
    max_epochs=None,
    max_steps=20_000,
    save_iter_interval=3000,
    max_keep_ckpts=8,
)

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
        embodiment_id=_EMBODIMENT_ID,
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_TOKENIZER_PATH)),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375]] * 3,
                stds=[[58.27148438, 57.02636719, 57.27539062]] * 3),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std'),
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
