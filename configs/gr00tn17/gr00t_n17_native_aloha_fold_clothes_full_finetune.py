# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GR00T N1.7 post-training and real-robot inference for ALOHA folding."""

import json

_base_ = ['./gr00t_n17_native_libero_10_full_finetune.py']

_DATASET_ROOT = (
    '/mnt/data/oss/users/sober/fluxthmis-data-realrobot/aloha/fold-clothes/'
    'RealRobot_AgileX_aloha_lerobot_v2')
_DATA_PATHS = [
    f'{_DATASET_ROOT}/20260613_20260613_01_4090_e2e_02',
    f'{_DATASET_ROOT}/20260615_20260615_01_4090_e2e_02',
]
_STATISTIC_NAME = 'private'
_STATISTICS_PATH = 'configs/gr00tn17/statistics/aloha_fold_clothes.json'
_STATS_PAYLOAD = json.load(open(_STATISTICS_PATH, encoding='utf-8'))
_TRANSFORMED_STATS = _STATS_PAYLOAD['norm_stats'][_STATISTIC_NAME]
_ALOHA_STATE_STATS = _TRANSFORMED_STATS['proprio']
_ALOHA_ACTION_STATS = _TRANSFORMED_STATS['action']

# ALOHA has two 6-DoF arms and one gripper coordinate per arm.  Reuse the
# released N1.7 XDOF relative-joint embedding, which is the closest dual-arm
# joint-control embodiment in the checkpoint.
_EMBODIMENT_KEY = 'aloha_dual'
_EMBODIMENT_ID = 28
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_TASK_DESCRIPTION = 'folding clothes'
_DELTA_ACTION_MASK = [True] * 6 + [False] + [True] * 6 + [False]
_JOINT_SIGNS = [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
_GRIPPER_RANGE = (-0.01, 0.08)


def _slice_statistics(statistics, start, end):
    return {
        name: values[start:end]
        for name, values in statistics.items()
        if isinstance(values, list)
    }


_MODALITY_KEYS = [
    'left_arm',
    'left_gripper',
    'right_arm',
    'right_gripper',
]
_RELATIVE_JOINT_ACTION = dict(
    rep='RELATIVE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_ABSOLUTE_GRIPPER_ACTION = dict(
    rep='ABSOLUTE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_N17_MODALITY_CONFIGS = {
    _EMBODIMENT_KEY:
    dict(
        video=dict(
            delta_indices=[0],
            modality_keys=[
                'cam_high',
                'cam_left_wrist',
                'cam_right_wrist',
            ],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=_MODALITY_KEYS,
        ),
        action=dict(
            delta_indices=list(range(40)),
            modality_keys=_MODALITY_KEYS,
            action_configs=[
                _RELATIVE_JOINT_ACTION,
                _ABSOLUTE_GRIPPER_ACTION,
                _RELATIVE_JOINT_ACTION,
                _ABSOLUTE_GRIPPER_ACTION,
            ],
        ),
        language=dict(
            delta_indices=[0],
            modality_keys=['task'],
        ),
    ),
}
_N17_STATISTICS = {
    _EMBODIMENT_KEY:
    dict(
        state=dict(
            left_arm=_slice_statistics(_ALOHA_STATE_STATS, 0, 6),
            left_gripper=_slice_statistics(_ALOHA_STATE_STATS, 6, 7),
            right_arm=_slice_statistics(_ALOHA_STATE_STATS, 7, 13),
            right_gripper=_slice_statistics(_ALOHA_STATE_STATS, 13, 14),
        ),
        action=dict(
            left_arm=_slice_statistics(_ALOHA_ACTION_STATS, 0, 6),
            left_gripper=_slice_statistics(_ALOHA_ACTION_STATS, 6, 7),
            right_arm=_slice_statistics(_ALOHA_ACTION_STATS, 7, 13),
            right_gripper=_slice_statistics(_ALOHA_ACTION_STATS, 13, 14),
        ),
    ),
}
_DATASET_STATISTICS = {
    _STATISTIC_NAME: {
        'states': _ALOHA_STATE_STATS,
        'actions': _ALOHA_ACTION_STATS,
    },
}
_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=_N17_STATISTICS,
    embodiment_id_mapping={_EMBODIMENT_KEY: _EMBODIMENT_ID},
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=True,
    clip_outliers=True,
    use_relative_action=True,
    apply_sincos_state_encoding=False,
    formalize_language=True,
    use_albumentations=True,
    shortest_image_edge=256,
    crop_fraction=0.95,
    image_target_size=(256, 256),
    image_crop_size=(230, 230),
    state_dropout_prob=0.2,
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
)

model = dict(
    embodiment_tag=_EMBODIMENT_KEY,
    processor_kwargs=dict(
        _delete_=True,
        **_PROCESSOR_KWARGS,
    ),
    state_dropout_prob=0.2,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        name_mappings={
            'observation.state': ['states'],
            'action': ['actions'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=_DATASET_STATISTICS,
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATA_PATHS,
                statistic_name=_STATISTIC_NAME,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=_EMBODIMENT_ID,
                        parquet_keys=[
                            'observation.state',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist',
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                    ),
                    dict(
                        type='JointSignTransform',
                        signs=_JOINT_SIGNS,
                    ),
                    dict(
                        type='OpenPIAlohaGripperCoordinates',
                        gripper_input_range=_GRIPPER_RANGE,
                    ),
                    dict(
                        type='RelativeActions',
                        mask=_DELTA_ACTION_MASK,
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_key='states',
                        action_key='actions',
                        state_dim=132,
                        action_dim=132,
                        norm_type='quantile',
                        clip_norm=True,
                        normalization_epsilon=0.0,
                        preserve_input_dtype=True,
                    ),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=40,
                        valid_action_dim=14,
                        state_dropout_prob=0.2,
                    ),
                    dict(
                        type='GrootN17ImageAugmentation',
                        embodiment_tag=_EMBODIMENT_KEY,
                        image_key='images',
                        output_image_key='images',
                        train_mode=True,
                        processor_kwargs=_PROCESSOR_KWARGS,
                    ),
                    dict(
                        type='QWen2VLImageTransform',
                        img_key='images',
                        size=dict(
                            shortest_edge=65536,
                            longest_edge=16777216,
                        ),
                        patch_size=16,
                        temporal_patch_size=2,
                        merge_size=2,
                        image_mean=[0.5, 0.5, 0.5],
                        image_std=[0.5, 0.5, 0.5],
                        to_tensor=True,
                    ),
                    dict(
                        type='ProcessPromptsWithImage',
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_QWEN_TOKENIZER_PATH,
                            padding_side='left',
                            trust_remote_code=False,
                        ),
                        max_len=256,
                        add_system=False,
                        add_assistant_stub=False,
                        task_pos='after_images',
                        image_tag_template='',
                        img_start='<|vision_start|>',
                        img_end='<|vision_end|>',
                        img_context_token='<|image_pad|>',
                        img_tokens_source='from_image_grid_thw',
                        image_grid_thw_key='image_grid_thw',
                        image_merge_size=2,
                        padding_side='left',
                        use_eos_as_pad=False,
                        truncate=False,
                        lowercase_task_description=True,
                        strip_task_punctuation=True,
                        attention_mask_dtype='int64',
                        output_keys=[
                            'lang_tokens',
                            'lang_masks',
                            'images',
                            'image_grid_thw',
                            'states',
                            'actions',
                            'action_masks',
                            'embodiment_ids',
                        ],
                    ),
                ],
                action_window_size=40,
                require_full_window=True,
            ),
        ],
    ),
)

# Epoch-based scheduling keeps the intended three passes over the 640 episodes
# independent of whether DLC supplies one or two 8-GPU nodes.
runner = dict(
    max_steps=None,
    max_epochs=3,
    grad_accumulation_steps=1,
    save_iter_interval=1000,
    save_epoch_interval=1,
    max_keep_ckpts=3,
    collator=dict(
        type='DictCollator',
        keys=[
            'lang_tokens',
            'lang_masks',
            'images',
            'image_grid_thw',
            'states',
            'actions',
            'action_masks',
            'embodiment_ids',
        ],
    ),
    metric=dict(
        active_trackers=('jsonl', 'wandb'),
        grad_accumulation_steps=1,
    ),
    lr_scheduler=dict(
        _delete_=True,
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    sharding_strategy='shard-grad-op',
    keep_params_fp32=True,
)

# This is a real-robot recipe; do not run the inherited LIBERO simulator eval.
eval = None

inference = dict(
    type='AlohaInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_descriptions={'1': _TASK_DESCRIPTION},
    seed=7,
    async_execution=False,
    execute_horizon=40,
    publish_rate=30,
    # Median first-frame qpos over all 640 demonstrations.
    prepare_pose=(
        [
            -0.3281394839,
            1.1748427153,
            -0.8211952448,
            -1.0307831764,
            0.9545297027,
            1.7270476818,
            0.0795999989,
        ],
        [
            0.2969155312,
            1.1419432163,
            -0.9499132633,
            0.9057128429,
            1.0749399662,
            -1.4709379673,
            0.0691599995,
        ],
    ),
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        embodiment_id=_EMBODIMENT_ID,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='JointSignTransform',
                signs=_JOINT_SIGNS,
                action_key=None,
            ),
            dict(
                type='OpenPIAlohaGripperCoordinates',
                gripper_input_range=_GRIPPER_RANGE,
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_key='states',
                action_key=None,
                state_dim=132,
                norm_type='quantile',
                clip_norm=True,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=14,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag=_EMBODIMENT_KEY,
                image_key='images',
                output_image_key='images',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='images',
                size=dict(
                    shortest_edge=65536,
                    longest_edge=16777216,
                ),
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                to_tensor=True,
            ),
            dict(
                type='ProcessPromptsWithImage',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_QWEN_TOKENIZER_PATH,
                    padding_side='left',
                    trust_remote_code=False,
                ),
                max_len=256,
                add_system=False,
                add_assistant_stub=False,
                task_pos='after_images',
                image_tag_template='',
                img_start='<|vision_start|>',
                img_end='<|vision_end|>',
                img_context_token='<|image_pad|>',
                img_tokens_source='from_image_grid_thw',
                image_grid_thw_key='image_grid_thw',
                image_merge_size=2,
                padding_side='left',
                use_eos_as_pad=False,
                truncate=False,
                lowercase_task_description=True,
                strip_task_punctuation=True,
                attention_mask_dtype='int64',
                output_keys=[
                    'lang_tokens',
                    'lang_masks',
                    'images',
                    'image_grid_thw',
                    'states',
                    'embodiment_ids',
                ],
            ),
        ],
    ),
    denormalize_action=dict(
        type='OpenPIAlohaActionPostprocess',
        action_dim=14,
        adapt_to_pi=True,
        use_delta_joint_actions=True,
        gripper_input_range=_GRIPPER_RANGE,
        gripper_output_range=_GRIPPER_RANGE,
    ),
    gripper_threshold=-0.0055,
    gripper_closed_value=-0.01,
    action_chunk=40,
    operator=dict(
        type='AlohaOperator',
        image_encoding='rgb8',
        img_front_topic='/camera_h/color/image_raw',
        img_left_topic='/camera_l/color/image_raw',
        img_right_topic='/camera_r/color/image_raw',
        img_front_depth_topic='/camera_h/depth/image_raw',
        img_left_depth_topic='/camera_l/depth/image_raw',
        img_right_depth_topic='/camera_r/depth/image_raw',
        puppet_arm_left_cmd_topic='/master/joint_left',
        puppet_arm_right_cmd_topic='/master/joint_right',
        puppet_arm_left_topic='/puppet/joint_left',
        puppet_arm_right_topic='/puppet/joint_right',
        robot_base_topic='/odom_raw',
        robot_base_cmd_topic='/cmd_vel',
    ),
)
