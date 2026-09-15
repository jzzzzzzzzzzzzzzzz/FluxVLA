# Franka Task 1: shuttlecock-bucket handoff

Task prompt:

> The right arm picks up the shuttlecock bucket, hands it to the left arm,
> and places it on the plate.

The task uses three RGB views and a 16D absolute Cartesian state/action:

```text
[left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw, left_gripper,
 right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw,
 right_gripper]
```

## Config matrix

| Model | Training and GPU-serving config | Horizon | Normalization |
| --- | --- | ---: | --- |
| GR00T N1.7 | `configs/gr00tn17/gr00tn17_qwen3vl_2b_franka_dual_eepose_full_finetune.py` | 40 | quantile |
| GR00T N1.5 | `configs/gr00tn15/gr00tn15_eagle_3b_franka_dual_eepose_full_finetune.py` | 32 | mean/std |
| PI0 | `configs/pi0/pi0_paligemma_franka_dual_eepose_full_finetune.py` | 50 | quantile |
| PI0.5 | `configs/pi05/pi05_paligemma_franka_dual_eepose_20260519_full_finetune.py` | 50 | quantile |
| FastWAM | `configs/fastwam/fastwam_franka_dual_eepose_full_finetune.py` | 32 | min/max |
| SmolVLA | `configs/smolvla/smolvla_franka_dual_eepose_finetune.py` | 50 | mean/std |
| Cosmos3-Edge | `configs/cosmos3/cosmos3edge_franka_dual_eepose_full_finetune.py` | 32 | mean/std |
| DiT4DiT | `configs/dit4dit/dit4dit_franka_dual_eepose_full_finetune.py` | 16 | quantile state, min/max action |

## 每个模型相对原配置的修改

这些修改已经落在上表的配置文件中，不需要在启动 DLC 前手工再改一遍。

| 模型 | 已完成的关键修改 |
| --- | --- |
| GR00T N1.7 | 保留已验证的 native processor；将 16D 绝对 EE pose 拆成 `7+1+7+1` 四个 modality，动作设为 `ABSOLUTE/NON_EEF`，horizon=40；加入三相机、Task-1 prompt、Franka 推理段以及可配置的模型/数据路径。 |
| GR00T N1.5 | 基于 Eagle-3B ALOHA 配置，改为三相机、`observation.eepose` 绝对动作、`ori_action_dim=16`、horizon=32 和 mean/std 自动统计；加入双 Franka Cartesian 推理段。 |
| PI0 | 基于 PI0 ALOHA 配置，设 `ori_action_dim=16`、`loss_action_dim=32`，16D 数据 pad 到 32D，horizon=50、quantile 统计；模型和 tokenizer 都使用 `pi0_base`。 |
| PI0.5 | 在已有双 Franka 配置上增加 20260519 专用 overlay，固定三相机与数据根目录；保持 16D→32D padding、horizon=50、quantile 和 PI0.5 tokenizer/EMA 训练路径。 |
| FastWAM | 改为三视角、16D proprio/action、horizon=32 和 min/max；只跳过并重置 checkpoint 中尺寸相关的 action encoder、action head、proprio encoder，其余预训练参数继续加载。 |
| SmolVLA | 改为三相机、`ori_action_dim=16`、32D state/action padding、horizon=50 和 mean/std；tokenizer/VLM 资源指向本地 `SmolVLM2-500M-Video-Instruct`。 |
| Cosmos3-Edge | 改为三相机 tiled video、16D 原始动作、64D state/action padding、horizon=32 和 mean/std；接入本地 transformer、Cosmos tokenizer 与 Wan2.2 VAE。 |
| DiT4DiT | 改为三相机 9-frame video、16D 原始动作、32D action/64D state、horizon=16；state 用 quantile、action 用 min/max；加载 Cosmos-Predict2.5 backbone，DiT action head 从头训练。 |

`FRANKA_TASK1_DATA_ROOT` changes the single-dataset root for all new configs.
The default is the 20260519, 99-episode dataset. For a multi-root run, replace
the relevant `ParquetDataset.data_root_path` with the list in the data-audit
section below.

Except for GR00T N1.7, training computes transformed statistics before model
construction and saves them in the work directory. N1.7 currently contains
the checked 20260519 quantiles inline; recompute both its flat dataset values
and split native-processor values before changing its data mixture.

## Statistics rule

Use one pooled set of statistics for the exact data mixture used by one run,
not one statistics file per dated directory. Do not treat statistics as a
global task artifact: action horizon, terminal padding and normalization type
differ across these models. Each work directory therefore keeps its own
`dataset_statistics.json`, and GPU serving must use the file beside that
run's checkpoint.

## Dataset audit

The following roots were checked using front-camera frames from the beginning,
handoff and placement phases of multiple episodes, together with 16D EE-pose
and gripper trajectories. They are directly compatible 30 FPS Task-1 data:

```python
TASK1_DATA_ROOTS = [
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260511_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260512_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260513_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260520_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260521_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260522_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260525_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260526_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260527_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260528_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260602_dual_franka_teleop',
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260603_dual_franka_teleop',
]
```

Together these contain 1,732 episodes and 1,324,602 frames. At global batch
128, one pass is approximately 10,000 optimizer steps.

Do not add these roots directly:

- `20260422`: the video is Task 1, but it is 15 FPS and `observation.eepose`
  is 14D without gripper widths. Convert/resample it first; the current config
  expects 30 FPS and 16D.
- `20260515`: episodes 132--140 are missing the complete
  `observation.images.cam_wrist_right` video files. Do not use this root until
  those nine videos are restored or the corresponding episodes are removed
  consistently from parquet and metadata.
- `20260612`: mislabeled; the video shows the three-plate chick-transfer task.
- `20260623` and `20260624`: mislabeled; the video shows opening a backpack
  and moving the chick to a plate.
- The undated dataset at the parent root is a byte-identical duplicate of
  `20260519`; including both double-weights those 99 episodes.
- All later dated roots show other tasks in the sampled videos.

## Adopted training datasets and budgets

GR00T N1.7 is not included in this training round. The main pool deliberately
uses five visually audited and video-complete Task-1 roots instead of every
compatible root:

```text
20260511:  81 episodes,  41,329 frames
20260512: 101 episodes,  84,890 frames
20260513: 141 episodes, 121,321 frames
20260519:  99 episodes,  77,327 frames
20260520: 105 episodes,  77,404 frames
total:    527 episodes, 402,271 frames
```

At an actual global batch of 128, one pass through this pool is approximately
3,143 optimizer steps. SmolVLA additionally uses 20260521 and 20260522 because
it is substantially cheaper to train: 810 episodes and 612,428 frames, or
approximately 4,785 optimizer steps per pass.

| Model | Episodes | Passes | `max_steps` | Save every | Reason |
| --- | ---: | ---: | ---: | ---: | --- |
| GR00T N1.5 | 527 | about 5 | 15,000 | 3,000 | full VLA adaptation |
| PI0 | 527 | about 4 | 12,000 | 3,000 | one pass beyond the PI0.5 baseline |
| PI0.5 | 527 | about 3 | 9,000 | 3,000 | follows the successful three-epoch experience |
| FastWAM | 527 | about 5 | 15,000 | 3,000 | resized action/proprio I/O, bounded for runtime |
| SmolVLA | 810 | about 5 | 24,000 | 4,000 | small model, so use a larger pool |
| Cosmos3-Edge | 527 | about 6 | 18,000 | 3,000 | action policy is reinitialized |
| DiT4DiT | 527 | about 8 | 24,000 | 3,000 | action head starts from scratch |

Scale the step budget inversely with actual global batch:
`adjusted_steps = table_steps * 128 / actual_global_batch`. Select the best
real-robot checkpoint rather than assuming the last checkpoint wins.

## Checkpoint audit

All model configs can use one checkpoint root:

```bash
export FLUXVLA_CHECKPOINT_ROOT=/mnt/data/cpfs/users/leo/checkpoints
```

The required files were checked for nonzero size. SmolVLA and Cosmos3-Edge
were additionally opened through `safetensors`, and their tokenizers were
loaded in offline mode.

| Model | Config fields and asset below checkpoint root | Status |
| --- | --- | --- |
| GR00T N1.7 | `model.model_path=GR00T-N1.7-3B` | ready, 6.5 GB |
| GR00T N1.5 | `model.pretrained_name_or_path=GR00T-N1.5-3B` | ready, 5.1 GB |
| PI0 | `model.pretrained_name_or_path=pi0_base/model.safetensors`; tokenizer `pi0_base` | ready, 14 GB |
| PI0.5 | `model.pretrained_name_or_path=pi05_base/model.safetensors`; tokenizer `pi05_base` | ready, 14 GB |
| FastWAM | `model.pretrained_name_or_path=fastwam_base_full/fastwam_base_full.safetensors` | ready, 24 GB |
| SmolVLA | `model.pretrained_name_or_path=smolvla_base/model.safetensors`; tokenizer/backbone `SmolVLM2-500M-Video-Instruct` | ready |
| Cosmos3-Edge | `model.pretrained_name_or_path=Cosmos3-Edge/transformer`; tokenizer `Cosmos3-Edge`; VAE `Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | ready |
| DiT4DiT | `model.vlm_backbone.base_model=Cosmos-Predict2.5-2B` | base ready; action head starts from scratch |

`RLinf-Pi05-LIBERO-SFT`, `pi05_libero` and the PI0.5 LIBERO work directory
are simulation-finetuned weights and are not Task-1 initializers. Likewise,
`dit4dit-model` contains evaluation artifacts but no usable model tensor.

### DSW image environment

The DSW image contains a separate `fluxvla_task1` Conda environment so these
dependencies do not modify the existing `fluxvla` environment. It was cloned
from `fluxvla` and then prepared with:

- PyTorch `2.8.0+cu128`
- TorchCodec `0.7.0`, with an actual 30 FPS Task-1 video decode verified
- the repository-pinned Diffusers `0.37.0.dev0`, including
  `Cosmos2_5_PredictBasePipeline`
- Transformers `5.3.0` and PyAV `14.2.0`

Before saving the DSW image, this command verifies the two dependencies that
previously blocked FastWAM and DiT4DiT:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fluxvla_task1
python -c 'import torchcodec, diffusers; from torchcodec.decoders import VideoDecoder; from diffusers import Cosmos2_5_PredictBasePipeline; print(torchcodec.__version__, diffusers.__version__)'
```

Save this DSW as a new image and select that image explicitly for every DLC
worker. CPFS shares the repository and datasets, but does not share the Conda
environment under `/root/miniconda3`.

## Ready-to-run DLC training

The command below implements the adopted 527/810-episode plan. Set `MODEL` to
one of the seven names shown below; GR00T N1.7 is intentionally not present.
The `case` block specifies every model-specific config, data-pool override,
work directory, initialization checkpoint, step budget and save interval.

The commands use microbatch 8 and gradient accumulation 1. On 16 GPUs this is
global batch 128. If DLC has a different GPU count, adjust accumulation so
`8 * GPU_COUNT * GRAD_ACCUM = 128` before comparing models.

| `MODEL` | `max_steps` | Save every | Initialization note |
| --- | ---: | ---: | --- |
| `gr00tn15` | 15,000 | 3,000 | 527 episodes, about 5 passes |
| `pi0` | 12,000 | 3,000 | 527 episodes, about 4 passes |
| `pi05` | 9,000 | 3,000 | 527 episodes, about 3 passes |
| `fastwam` | 15,000 | 3,000 | 527 episodes, about 5 passes |
| `smolvla` | 24,000 | 4,000 | 810 episodes, about 5 passes |
| `cosmos3` | 18,000 | 3,000 | 527 episodes, about 6 passes |
| `dit4dit` | 24,000 | 3,000 | 527 episodes, about 8 passes |

For example, change only the first line to `MODEL=pi0`, `MODEL=cosmos3`, and
so on:

```bash
MODEL=pi05 /bin/bash -lc '
set -euo pipefail

CODE_ROOT=/mnt/data/cpfs/mnt/data/yiming/fluxvla-latest-main
CKPT_ROOT=/mnt/data/cpfs/users/leo/checkpoints
DATASET_ROOT=/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1
COMMON_DATA_ROOTS="[$DATASET_ROOT/20260511_dual_franka_teleop,$DATASET_ROOT/20260512_dual_franka_teleop,$DATASET_ROOT/20260513_dual_franka_teleop,$DATASET_ROOT/20260519_dual_franka_teleop,$DATASET_ROOT/20260520_dual_franka_teleop]"
SMOL_DATA_ROOTS="[$DATASET_ROOT/20260511_dual_franka_teleop,$DATASET_ROOT/20260512_dual_franka_teleop,$DATASET_ROOT/20260513_dual_franka_teleop,$DATASET_ROOT/20260519_dual_franka_teleop,$DATASET_ROOT/20260520_dual_franka_teleop,$DATASET_ROOT/20260521_dual_franka_teleop,$DATASET_ROOT/20260522_dual_franka_teleop]"
MICRO_BATCH=8
GRAD_ACCUM=1

case "${MODEL:?Set MODEL}" in
  gr00tn15)
    CONFIG=configs/gr00tn15/gr00tn15_eagle_3b_franka_dual_eepose_full_finetune.py
    RUN_NAME=gr00tn15_franka_task1_527eps
    MAX_STEPS=15000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.0.data_root_path
    MODEL_OPTIONS=(model.pretrained_name_or_path="$CKPT_ROOT/GR00T-N1.5-3B")
    ;;
  pi0)
    CONFIG=configs/pi0/pi0_paligemma_franka_dual_eepose_full_finetune.py
    RUN_NAME=pi0_franka_task1_527eps
    MAX_STEPS=12000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.0.data_root_path
    MODEL_OPTIONS=(model.pretrained_name_or_path="$CKPT_ROOT/pi0_base/model.safetensors")
    ;;
  pi05)
    CONFIG=configs/pi05/pi05_paligemma_franka_dual_eepose_20260519_full_finetune.py
    RUN_NAME=pi05_franka_task1_527eps
    MAX_STEPS=9000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.0.data_root_path
    MODEL_OPTIONS=(model.pretrained_name_or_path="$CKPT_ROOT/pi05_base/model.safetensors")
    ;;
  fastwam)
    CONFIG=configs/fastwam/fastwam_franka_dual_eepose_full_finetune.py
    RUN_NAME=fastwam_franka_task1_527eps
    MAX_STEPS=15000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.data_root_path
    MODEL_OPTIONS=(model.pretrained_name_or_path="$CKPT_ROOT/fastwam_base_full/fastwam_base_full.safetensors")
    ;;
  smolvla)
    CONFIG=configs/smolvla/smolvla_franka_dual_eepose_finetune.py
    RUN_NAME=smolvla_franka_task1_810eps
    MAX_STEPS=24000
    SAVE_INTERVAL=4000
    DATA_ROOTS="$SMOL_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.0.data_root_path
    MODEL_OPTIONS=(model.pretrained_name_or_path="$CKPT_ROOT/smolvla_base/model.safetensors")
    ;;
  cosmos3)
    CONFIG=configs/cosmos3/cosmos3edge_franka_dual_eepose_full_finetune.py
    RUN_NAME=cosmos3edge_franka_task1_527eps
    MAX_STEPS=18000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.data_root_path
    MODEL_OPTIONS=(
      model.pretrained_name_or_path="$CKPT_ROOT/Cosmos3-Edge/transformer"
      model.vision_vae.pretrained_name_or_path="$CKPT_ROOT/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )
    ;;
  dit4dit)
    CONFIG=configs/dit4dit/dit4dit_franka_dual_eepose_full_finetune.py
    RUN_NAME=dit4dit_franka_task1_527eps
    MAX_STEPS=24000
    SAVE_INTERVAL=3000
    DATA_ROOTS="$COMMON_DATA_ROOTS"
    DATA_CFG_KEY=train_dataloader.dataset.datasets.0.data_root_path
    MODEL_OPTIONS=(model.vlm_backbone.base_model="$CKPT_ROOT/Cosmos-Predict2.5-2B")
    ;;
  *)
    echo "Unknown MODEL=$MODEL" >&2
    exit 2
    ;;
esac

cd "$CODE_ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fluxvla_task1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR=/tmp/numba_cache
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONHASHSEED=7
export FLUXVLA_CHECKPOINT_ROOT="$CKPT_ROOT"

WORK_DIR="$CODE_ROOT/work_dirs/franka_task1/${RUN_NAME}_mb${MICRO_BATCH}_ga${GRAD_ACCUM}"

exec bash scripts/train.sh \
  "$CONFIG" \
  "$WORK_DIR" \
  --cfg-options \
  train_dataloader.per_device_batch_size="$MICRO_BATCH" \
  runner.grad_accumulation_steps="$GRAD_ACCUM" \
  runner.max_steps="$MAX_STEPS" \
  runner.save_iter_interval="$SAVE_INTERVAL" \
  runner.max_keep_ckpts=8 \
  "$DATA_CFG_KEY=$DATA_ROOTS" \
  "${MODEL_OPTIONS[@]}"
'
```

Exact invocations are therefore:

```bash
MODEL=gr00tn15  # GR00T N1.5
MODEL=pi0       # PI0
MODEL=pi05      # PI0.5
MODEL=fastwam   # FastWAM
MODEL=smolvla   # SmolVLA
MODEL=cosmos3   # Cosmos3-Edge
MODEL=dit4dit   # DiT4DiT
```

These lines select the matching branch in the complete launcher; they are not
standalone shell commands. Run one model at a time by replacing the first line
of the launcher.

## ZMQ deployment

After copying the work directory (including `dataset_statistics.json`) to the
OSS path shared with the model-side DSW, start the GPU server with the same
training config. Do not run `scripts/inference.py` on the model DSW; that is
the ROS/ZMQ client and does not load the checkpoint:

```bash
cd /mnt/data/cpfs/mnt/data/yiming/fluxvla-latest-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fluxvla_task1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

# Save the exact decoded request, transformed model tensors, normalized
# predictions, and denormalized robot commands for the first 30 requests.
export FRANKA_DEBUG_CAPTURE_DIR=/mnt/data/cpfs/mnt/data/yiming/franka_task1_capture
export FRANKA_DEBUG_CAPTURE_EVERY=1
export FRANKA_DEBUG_CAPTURE_MAX=30

python -m fluxvla.engines.runners.serving.serve \
  --config configs/pi0/pi0_paligemma_franka_dual_eepose_full_finetune.py \
  --ckpt-path /oss/path/work_dir/checkpoints/model.safetensors \
  --host 0.0.0.0 \
  --port 3333 \
  --device cuda:0 \
  --dtype bf16
```

On the robot, use the model-agnostic client. The default execute horizon is
16, which is safe for all eight models; set `FRANKA_EXECUTE_HORIZON=20` for
the models with horizons of at least 20 if desired.

```bash
cd /path/to/fluxvla-latest-main
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export NO_ALBUMENTATIONS_UPDATE=1
export FRANKA_ZMQ_HOST=<model-dsw-ip-or-tunnel-host>
export FRANKA_ZMQ_PORT=3333
python scripts/inference.py \
  --config configs/franka/franka_dual_eepose_task1_remote_inference.py
```

If an SSH tunnel maps robot-local port 5556 to model-DSW port 3333, create the
tunnel first and use the local endpoint instead:

```bash
ssh -N -L 5556:127.0.0.1:3333 <user>@<model-dsw-host>

export FRANKA_ZMQ_HOST=127.0.0.1
export FRANKA_ZMQ_PORT=5556
python scripts/inference.py \
  --config configs/franka/franka_dual_eepose_task1_remote_inference.py
```

`FRANKA_ZMQ_HOST` must be a plain hostname/IP; do not paste Markdown such as
`[127.0.0.1](http://127.0.0.1)`. Different ports are valid only when a tunnel
or port-forward explicitly maps 5556 to 3333.

The server performs preprocessing and denormalization using the copied
`dataset_statistics.json`; the robot client only captures observations,
sends them over ZMQ, and executes the returned 16D Cartesian actions.

## Compare real-robot inputs with training data

The server-side capture is the authoritative input: it records the request
after ZMQ/JPEG decoding and the tensors immediately before `predict_action`.
After a short non-contact or low-speed run, stop the robot and run:

```bash
python tools/compare_franka_capture.py \
  --capture-dir /mnt/data/cpfs/mnt/data/yiming/franka_task1_capture \
  --stats /oss/path/work_dir/dataset_statistics.json \
  --dataset-root \
    /mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260511_dual_franka_teleop \
    /mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260512_dual_franka_teleop \
    /mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260513_dual_franka_teleop \
    /mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop \
    /mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_franka_dual_lerobotv2.1/20260520_dual_franka_teleop
```

The output contains `comparison.md`, `comparison.json`, and
`camera_montage.png`. Check, in order:

1. `Prompt exact match` is true.
2. All raw states are 16D in
   `[left xyz qx qy qz qw gripper, right xyz qx qy qz qw gripper]` order.
3. Quaternion norms are near 1, grippers are in metres, and few dimensions
   fall outside the training min/max range.
4. The montage shows front/left-wrist/right-wrist in the correct slots and no
   RGB/BGR swap warning.
5. `model_inputs` and both action stages exist. Missing `denormalized_actions`
   means the server failed before returning a robot command.

Always pass the `dataset_statistics.json` beside the exact checkpoint work
directory. A statistics file copied from a different run can produce valid
shapes but severely wrong Cartesian commands.
