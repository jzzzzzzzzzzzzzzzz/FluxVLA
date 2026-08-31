# Copyright 2026 Limx Dynamics

import numpy as np
from mmengine import Config

from fluxvla.transforms.normalize import (DenormalizeDeltaAction,
                                          NormalizeStatesAndActions)
from fluxvla.transforms.robocasa_transforms import RobocasaGR1N15Bridge
from fluxvla.transforms.transform_actions import RelativeActions

_CONFIG = 'configs/gr00tn17/gr00t_n17_native_robocasa_full_finetune.py'


def test_n17_horizon_normalization_and_delta_denormalization_round_trip():
    cfg = Config.fromfile(_CONFIG)
    permutation = np.asarray(cfg._N17_DIMENSION_PERMUTATION)
    inverse_permutation = np.argsort(permutation)
    stats = cfg._DATASET_STATISTICS[cfg._STATISTIC_NAME]
    action_low = np.asarray(stats.actions.min, dtype=np.float32)
    action_high = np.asarray(stats.actions.max, dtype=np.float32)

    target_state = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    source_state = target_state[inverse_permutation]
    target_delta_or_absolute = (action_low + action_high) / 2.0
    target_actions = target_delta_or_absolute.copy()
    target_actions[:, :26] += target_state[:26]
    source_actions = target_actions[:, inverse_permutation]

    sample = {
        'states': source_state,
        'actions': source_actions,
        'stats': stats.to_dict(),
    }
    sample = RobocasaGR1N15Bridge(
        apply_state_sincos=False,
        reorder_action_stats=False,
    )(
        sample)
    sample = RelativeActions(mask=[True] * 26 + [False] * 3)(sample)
    sample = NormalizeStatesAndActions(
        state_key='states',
        action_key='actions',
        state_dim=132,
        action_dim=132,
        state_norm_type='quantile',
        action_norm_type='min_max',
        clip_norm=True,
        normalization_epsilon=0.0,
        preserve_input_dtype=True,
    )(
        sample)

    denormalizer = DenormalizeDeltaAction(
        norm_stats={cfg._STATISTIC_NAME: {
            'action': stats.actions.to_dict()
        }},
        statistic_name=cfg._STATISTIC_NAME,
        norm_type='min_max',
        action_dim=29,
        delta_action_mask=[True] * 26 + [False] * 3,
        state_permutation=permutation.tolist(),
        normalize_gripper_action=False,
        invert_gripper_action=False,
    )
    restored = denormalizer({
        'action': sample['actions'],
        'state': source_state,
    })

    assert sample['actions'].shape == (8, 132)
    np.testing.assert_allclose(sample['actions'][:, :29], 0.0, atol=2e-6)
    np.testing.assert_allclose(restored, target_actions, atol=2e-6)
