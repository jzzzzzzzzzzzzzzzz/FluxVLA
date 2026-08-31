# GR00T N1.7 RoboCasa 训练与评测执行计划

## 目标

在当前最新 N1.7 分支上补齐 RoboCasa 24-task 全量训练与评测链路，优先使用 FluxVLA 已有 runner、dataset wrapper 和组合式 transforms，不恢复已经在 review 中删除的大型 N1.7 state/action codec。

首要目标是尽快获得一组端到端可训练、可评测的结果；采样器作为后续单变量实验比较。

## 已确认资源

- 目标仓库：`/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-rebase-2f51d7a`
- 数据：`/mnt/data/cpfs/mnt/data/yiming/fluxvla/datasets/robocasa_lerobot_V2.1`
- Base 权重：`/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-3B`
- 历史已训练 checkpoint：`/mnt/data/cpfs/mnt/data/yiming/fluxvla-dev/work_dirs/groot_n17_robocasa_24x1000/checkpoint-60000`
- 参考统计量：`/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-robocasa-pr/configs/gr00tn17/statistics/robocasa_gr1_tabletop.json`

## 数据契约

- 24 个任务，每个任务 1000 episodes。
- parquet flat state/action 顺序：`left_arm, left_hand, right_arm, right_hand, waist`。
- N1.7 模态顺序：`left_arm, right_arm, left_hand, right_hand, waist`。
- state/action 有效维度均为 29，模型最大维度为 132。
- 数据 action horizon 为 8，模型 action horizon 为 40。
- arm/hand 使用相对当前 state 的动作；waist 保持绝对动作。
- state 使用 q01/q99；relative arm/hand action 使用 horizon-dependent min/max；absolute waist 使用普通 action q01/q99 作为 min/max。

## 实现方案

训练 transform 链：

1. `ProcessParquetInputs`
2. 复用 `RobocasaGR1N15Bridge`（关闭 state sin/cos 和 statistics 重排）完成 flat-vector 顺序转换
3. `RelativeActions`
4. `NormalizeStatesAndActions`
5. `PrepareStateActionTargets`
6. `GrootN17ImageAugmentation`
7. `QWen2VLImageTransform`
8. `ProcessPromptsWithImage`
9. `DictCollator`

评测复用相同的 state order、normalization、图像和 prompt 处理。`RobocasaEvalRunner` 增加 `action_order='n17'` 和整 chunk 反归一化能力；`DenormalizeDeltaAction` 支持 raw state permutation。

## 采样实验顺序

### A. 首个闭环：DistributedRepeatingDataset

- 将 24 个 task roots 组成一个 multi-root `ParquetDataset`。
- 所有有效 frame 组成全局样本集合，因此任务概率随有效 frame 数变化。
- 优先完成数据样本、forward、短训、checkpoint 和评测闭环。

### B. 单变量对照：DistributedBalancedRepeatingDataset

- 保持模型、transform、统计量、batch、seed、优化器和训练步数不变。
- wrapper 自动将 multi-root dataset 的每个 root 识别为一个 source。
- 不设置 `sampling_weights`，使用每个 source 一次的 balanced cycling。

### C. 延后项：N1.7 sharded sampler

只有当 A/B 均明显弱于历史 sharded 结果，且排除训练 dtype、batch、eval seed 等因素后，再迁移旧的 official-style sampler。

## 默认训练口径

- max steps：60000（smoke 使用覆盖值）
- global batch：512
- per-device batch：根据实际 GPU 数量确定
- gradient accumulation：用于保持 global batch 512
- train/data seed：42
- learning rate：`1e-4`
- warmup ratio：`0.05`
- weight decay：`1e-5`
- state dropout：`0.2`
- BF16 autocast + `keep_params_fp32=True`
- 首选 `no-shard`；若显存不足再验证 `shard-grad-op`

## 验证门槛

1. Config 可以由 MMEngine 正常加载。
2. permutation、relative action、horizon stats 和 normalize/denormalize round-trip 单测通过。
3. 真实 parquet 样本输出 shape：state `[1,132]`，action `[40,132]`，mask 与前 8×29 有效区一致。
4. Base checkpoint 严格加载，完成 forward/predict smoke。
5. 单步训练后 AdamW state 为 FP32，loss 有限。
6. 历史 checkpoint 若与最新模型结构兼容，完成单任务 RoboCasa eval smoke。
7. Repeating 获得首个训练/评测结果后，再运行 balanced 单变量对照。

## 评测口径

- 首轮 smoke：1 个 task、1 至 2 个 trials。
- 第一组正式可复现结果：24 tasks × 20 trials，eval seed 7。
- 如需对齐官方未固定 seed 的报告，再补无固定 eval seed 或更多 trials。

## 暂不执行

- 不直接迁移旧 `GrootN17StateActionTransform`。
- 不在 pipeline 打通前迁移 sharded sampler。
- 不在缺少 parity/smoke 证据时直接启动 60k 全量训练。
