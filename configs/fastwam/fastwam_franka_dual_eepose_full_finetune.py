# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FastWAM fine-tuning for Task 1 on dual-Franka absolute EE poses."""

import os

_base_ = ['./fastwam_libero_full_finetune.py']

_DATASET_ROOT = os.environ.get(
    'FRANKA_TASK1_DATA_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_CHECKPOINT_ROOT = os.environ.get(
    'FLUXVLA_CHECKPOINT_ROOT', '/mnt/data/cpfs/mnt/data/liyinhao/checkpoints')
_BASE_MODEL = os.path.join(_CHECKPOINT_ROOT,
                           'fastwam_base_full/fastwam_base_full.safetensors')
_STATISTIC_NAME = 'private'
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')
_TOKENIZER_PATH = os.path.join(_CHECKPOINT_ROOT, 'fastwam_base_full/tokenizer')

# The public FastWAM checkpoint has 7D action I/O and an 8D proprio encoder.
# Reinitialize only those shape-dependent layers for this 16D dual-arm task.
_RESIZED_INPUT_OUTPUT_PREFIXES = [
    'vla_head.mot.mixtures.action.action_encoder.',
    'vla_head.mot.mixtures.action.head.',
    'vla_head.proprio_encoder.',
]
model = dict(
    pretrained_name_or_path=_BASE_MODEL,
    num_views=3,
    proprio_dim=16,
    pretrained_skip_prefixes=_RESIZED_INPUT_OUTPUT_PREFIXES,
    vla_head=dict(
        video_dit_config=dict(action_dim=16),
        action_dit_config=dict(action_dim=16),
    ),
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
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_DATASET_ROOT,
            action_key='observation.eepose',
            use_delta=False,
            window_start_idx=0,
            frame_window_size=9,
            frame_sample_stride=4,
            require_full_window=True,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=13,
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
                    video_backend='torchcodec',
                    name_mappings={
                        'observation.eepose': ['states'],
                        'actions': ['actions'],
                    }),
                dict(
                    type='ResizeImages',
                    # The 3-view top/bottom layout becomes 384x256. After
                    # Wan VAE 16x downsampling its 24x16 latent is divisible
                    # by the video DiT's 2x2 spatial patch.
                    height=256,
                    width=256,
                    backend='torchvision',
                    scale_to_unit_interval=True),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5]),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=16,
                    state_dim=16,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max'),
                dict(
                    type='PrepareVideo',
                    num_views=3,
                    frame_window_size=9,
                    tile_direction='top_bottom_pair'),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_TOKENIZER_PATH),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=(
                        "A video recorded from a robot's point of view "
                        'executing the following instruction: {task}')),
            ],
            action_window_size=32,
        ),
    ), )

runner = dict(
    max_epochs=None,
    max_steps=20_000,
    save_iter_interval=3000,
    max_keep_ckpts=8,
    evaluator=None,
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
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='ResizeImages',
                height=256,
                width=256,
                backend='torchvision',
                scale_to_unit_interval=True),
            dict(
                type='NormalizeImages',
                means=[0.5, 0.5, 0.5],
                stds=[0.5, 0.5, 0.5]),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=16,
                state_dim=16,
                state_key='proprio',
                action_key='action',
                norm_type='min_max'),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair'),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_TOKENIZER_PATH),
                max_len=128,
                use_conversation=False,
                prompt_template=(
                    "A video recorded from a robot's point of view "
                    'executing the following instruction: {task}')),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='min_max',
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
