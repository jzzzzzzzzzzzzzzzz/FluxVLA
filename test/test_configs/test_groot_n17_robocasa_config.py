# Copyright 2026 Limx Dynamics

from mmengine import Config

_BASE_CONFIG = ('configs/gr00tn17/gr00t_n17_native_robocasa_full_finetune.py')
_BALANCED_CONFIG = (
    'configs/gr00tn17/gr00t_n17_native_robocasa_balanced_full_finetune.py')


def test_groot_n17_robocasa_training_contract():
    cfg = Config.fromfile(_BASE_CONFIG)
    dataset = cfg.train_dataloader.dataset
    parquet = dataset.datasets[0]
    stats = cfg._DATASET_STATISTICS[cfg._STATISTIC_NAME]

    assert dataset.type == 'DistributedRepeatingDataset'
    assert len(parquet.data_root_path) == 24
    assert parquet.action_window_size == 8
    assert parquet.require_full_window is True
    assert len(stats.states.q01) == 29
    assert len(stats.states.q99) == 29
    assert len(stats.actions.min) == 8
    assert all(len(row) == 29 for row in stats.actions.min)
    assert len(stats.actions.max) == 8
    assert all(len(row) == 29 for row in stats.actions.max)

    transform_types = [item.type for item in parquet.transforms]
    assert transform_types[:4] == [
        'ProcessParquetInputs',
        'RobocasaGR1N15Bridge',
        'RelativeActions',
        'NormalizeStatesAndActions',
    ]
    bridge = parquet.transforms[1]
    assert bridge.apply_state_sincos is False
    assert bridge.reorder_action_stats is False
    assert cfg.runner.max_steps == 60_000
    assert cfg.runner.keep_params_fp32 is True
    assert cfg.runner.seed == 42
    assert cfg.runner.sharding_strategy == 'no-shard'


def test_groot_n17_robocasa_balanced_variant_changes_only_wrapper():
    base = Config.fromfile(_BASE_CONFIG)
    balanced = Config.fromfile(_BALANCED_CONFIG)

    assert base.train_dataloader.dataset.type == 'DistributedRepeatingDataset'
    assert balanced.train_dataloader.dataset.type == (
        'DistributedBalancedRepeatingDataset')
    assert balanced.train_dataloader.dataset.datasets == (
        base.train_dataloader.dataset.datasets)
    assert balanced.model == base.model
    assert balanced.runner == base.runner
    assert balanced.eval == base.eval


def test_groot_n17_robocasa_eval_contract():
    cfg = Config.fromfile(_BASE_CONFIG)

    assert cfg.eval.action_order == 'n17'
    assert cfg.eval.eval_chunk_size == 8
    assert cfg.eval.denormalize_action_chunk is True
    assert cfg.eval.denormalize_action.type == 'DenormalizeDeltaAction'
    assert cfg.eval.denormalize_action.action_dim == 29
    assert cfg.eval.denormalize_action.delta_action_mask == ([True] * 26 +
                                                             [False] * 3)
    assert len(cfg.eval.denormalize_action.state_permutation) == 29
