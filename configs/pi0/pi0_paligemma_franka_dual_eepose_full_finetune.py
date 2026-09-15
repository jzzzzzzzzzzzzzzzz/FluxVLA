# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PI0 fine-tuning for Task 1 on dual-Franka absolute EE poses."""

import os

_base_ = ['./pi0_paligemma_aloha_full_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_BASE_MODEL = os.path.join(_CHECKPOINT_ROOT, 'pi0_base/model.safetensors')
_STATISTIC_NAME = 'private'
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')
_TOKENIZER_PATH = os.path.join(_CHECKPOINT_ROOT, 'pi0_base')

model = dict(
    pretrained_name_or_path=_BASE_MODEL,
    ori_action_dim=16,
    loss_action_dim=32,
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    vision_backbone=dict(openpi_stem_fp32=True),
)
inference_model = model.copy()

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
                    dict(
                        type='ParquetPrompter',
                        use_conversation=False,
                        add_new_line=True),
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
            ),
        ],
    ), )

runner = dict(
    max_epochs=None,
    max_steps=20_000,
    save_iter_interval=3000,
    max_keep_ckpts=8,
    # ``openpi_fp32_flow`` explicitly disables autocast around the velocity
    # head. Keep its parameters in FP32 as well; otherwise FSDP's default
    # BF16 parameter cast makes the final linear projection fail with a
    # Float/BFloat16 dtype mismatch.
    keep_params_fp32=True,
    sharding_strategy='global-shard-grad-op',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_TOKENIZER_PATH,
    ),
)

inference = dict(
    _delete_=True,
    type='FrankaInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_suite_name=_STATISTIC_NAME,
    task_descriptions={'1': _TASK_DESCRIPTION},
    seed=7,
    state_dim=16,
    action_chunk=50,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=20,
    prepare_pose=None,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                action_dim=None,
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_TOKENIZER_PATH)),
            dict(type='PadStatesAndActions', model_action_dim=32),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='quantile',
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
