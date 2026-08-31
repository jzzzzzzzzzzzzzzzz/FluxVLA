# Copyright 2026 Limx Dynamics
"""Task-balanced sampling variant of native GR00T N1.7 RoboCasa."""

_base_ = ['./gr00t_n17_native_robocasa_full_finetune.py']

train_dataloader = dict(
    dataset=dict(type='DistributedBalancedRepeatingDataset'), )
