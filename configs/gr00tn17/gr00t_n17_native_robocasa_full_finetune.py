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
"""Native GR00T N1.7 RoboCasa full-data training and evaluation config."""

import json

_DATASET_ROOT = './datasets/robocasa_lerobot_V2.1'
_STATISTIC_NAME = 'robocasa_gr1_tabletop_native'
_N17_INIT_CKPT = './checkpoints/GR00T-N1.7-3B'
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_STATISTICS_PATH = ('configs/gr00tn17/statistics/robocasa_gr1_tabletop.json')
# MMEngine configs only allow their own ``read_base`` context manager, so the
# small checked-in metadata file is loaded as one expression here.
_N17_STATISTICS = json.load(open(_STATISTICS_PATH, encoding='utf-8'))

_QWEN3_VL_CONFIG = dict(
    architectures=['Qwen3VLForConditionalGeneration'],
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
    tie_word_embeddings=False,
    text_config=dict(
        model_type='qwen3_vl_text',
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters=dict(
            rope_type='default',
            rope_theta=5000000.0,
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
        ),
    ),
    vision_config=dict(
        model_type='qwen3_vl_vision',
        depth=24,
        hidden_size=1024,
        hidden_act='gelu_pytorch_tanh',
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17],
        initializer_range=0.02,
    ),
)
_ACTIVE_TRACKERS = ('jsonl', )

_TASK_NAMES = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]
_DATA_PATHS = [f'{_DATASET_ROOT}/{task_name}' for task_name in _TASK_NAMES]

# Converted parquet columns use left arm, left hand, right arm, right hand,
# waist. N1.7 uses left arm, right arm, left hand, right hand, waist.
_N17_DIMENSION_PERMUTATION = (
    list(range(0, 7)) + list(range(13, 20)) + list(range(7, 13)) +
    list(range(20, 29)))
_MODALITY_KEYS = [
    'left_arm',
    'right_arm',
    'left_hand',
    'right_hand',
    'waist',
]
_RELATIVE_JOINT_ACTION = dict(
    rep='RELATIVE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_ABSOLUTE_JOINT_ACTION = dict(
    rep='ABSOLUTE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_N17_MODALITY_CONFIGS = dict(
    robocasa_gr1_tabletop=dict(
        video=dict(
            delta_indices=[0],
            modality_keys=['ego_view_bg_crop_pad_res256_freq20'],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=_MODALITY_KEYS,
            sin_cos_embedding_keys=_MODALITY_KEYS,
        ),
        action=dict(
            delta_indices=list(range(8)),
            modality_keys=_MODALITY_KEYS,
            action_configs=[
                _RELATIVE_JOINT_ACTION,
                _RELATIVE_JOINT_ACTION,
                _RELATIVE_JOINT_ACTION,
                _RELATIVE_JOINT_ACTION,
                _ABSOLUTE_JOINT_ACTION,
            ],
        ),
        language=dict(
            delta_indices=[0],
            modality_keys=['task'],
        ),
    ))


def _flatten_group_statistics(statistics, groups):
    fields = ('min', 'max', 'mean', 'std', 'q01', 'q99')
    return {
        field:
        [value for group in groups for value in statistics[group][field]]
        for field in fields
    }


def _build_horizon_action_statistics(statistics, horizon=8):
    """Build official N1.7 mixed relative/absolute action bounds."""
    result = {}
    for field, absolute_field in (('min', 'q01'), ('max', 'q99')):
        rows = []
        for step in range(horizon):
            row = []
            for group in _MODALITY_KEYS:
                if group == 'waist':
                    values = statistics['action'][group][absolute_field]
                else:
                    values = statistics['relative_action'][group][field][step]
                row.extend(values)
            rows.append(row)
        result[field] = rows
    return result


_ROBOCASA_N17_STATS = _N17_STATISTICS['robocasa_gr1_tabletop']
_FLAT_STATE_STATS = _flatten_group_statistics(_ROBOCASA_N17_STATS['state'],
                                              _MODALITY_KEYS)
_FLAT_ACTION_STATS = _build_horizon_action_statistics(_ROBOCASA_N17_STATS)
_DATASET_STATISTICS = {
    _STATISTIC_NAME: {
        'states': _FLAT_STATE_STATS,
        'actions': _FLAT_ACTION_STATS,
    }
}

_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=_N17_STATISTICS,
    embodiment_id_mapping=dict(robocasa_gr1_tabletop=10),
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=True,
    clip_outliers=True,
    use_relative_action=True,
    # The modality lists sin/cos-capable keys, but the official N1.7 fine-tune
    # defaults and released RoboCasa processor keep this global switch off.
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
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    embodiment_tag='ROBOCASA_GR1_TABLETOP',
    processor_kwargs=_PROCESSOR_KWARGS,
    state_dropout_prob=0.2,
    load_metadata=True,
    qwen3_runtime='compat_457',
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='GrootN17Qwen3Backbone',
        model_config=_QWEN3_VL_CONFIG,
        select_layer=16,
        reproject_vision=False,
        use_flash_attention=True,
        load_bf16=False,
        qwen3_runtime='compat_457',
    ),
    vla_head=dict(type='GrootN17ActionHead'),
)

train_dataloader = dict(
    # 16 GPUs x 32 samples/GPU = the official global batch size of 512.
    per_device_batch_size=32,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
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
                        embodiment_id=10,
                        parquet_keys=[
                            'observation.state',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
                        ],
                        video_keys=['observation.images.ego_view'],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                    ),
                    # N1.5 and N1.7 use the same GR1 group order here. Keep
                    # only the existing bridge's reorder operation enabled.
                    dict(
                        type='RobocasaGR1N15Bridge',
                        apply_state_sincos=False,
                        reorder_action_stats=False,
                    ),
                    dict(
                        type='RelativeActions',
                        mask=[True] * 26 + [False] * 3,
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_key='states',
                        action_key='actions',
                        state_dim=132,
                        action_dim=132,
                        norm_type='quantile',
                        state_norm_type='quantile',
                        action_norm_type='min_max',
                        clip_norm=True,
                        normalization_epsilon=0.0,
                        preserve_input_dtype=True,
                    ),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=40,
                        valid_action_dim=29,
                        state_dropout_prob=0.2,
                    ),
                    dict(
                        type='GrootN17ImageAugmentation',
                        embodiment_tag='ROBOCASA_GR1_TABLETOP',
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
                        max_len=180,
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
                action_window_size=8,
                require_full_window=True,
            ),
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=60000,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-5,
    ),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    sampler=None,
    save_iter_interval=2000,
    save_epoch_interval=1,
    max_keep_ckpts=5,
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
        type='VLAMetric',
        active_trackers=_ACTIVE_TRACKERS,
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    seed=42,
    sharding_strategy='no-shard',
    change_key_name=False,
)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='groot_n17_native',
    task_list=[
        f'gr1_unified/{task_name}_GR1ArmsAndWaistFourierHands_Env'
        for task_name in _TASK_NAMES
    ],
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=20,
    seed=7,
    unnorm_key=_STATISTIC_NAME,
    action_order='n17',
    denormalize_action_chunk=True,
    save_video=False,
    deterministic_env=True,
    deterministic_action_sampling=True,
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=256,
                normalize=False,
                embodiment_id=10,
            ),
            dict(
                type='RobocasaGR1N15Bridge',
                apply_state_sincos=False,
                reorder_action_stats=False,
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
                statistics_key='stats',
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=29,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag='ROBOCASA_GR1_TABLETOP',
                image_key='pixel_values',
                output_image_key='pixel_values',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='pixel_values',
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
                max_len=180,
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
                    'pixel_values',
                    'img_masks',
                    'image_grid_thw',
                    'states',
                    'embodiment_ids',
                    'replay_img',
                ],
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='min_max',
        action_dim=29,
        delta_action_mask=[True] * 26 + [False] * 3,
        state_permutation=_N17_DIMENSION_PERMUTATION,
        normalize_gripper_action=False,
        invert_gripper_action=False,
    ),
)
