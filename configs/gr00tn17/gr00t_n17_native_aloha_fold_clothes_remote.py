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
"""Remote ZMQ inference client for GR00T N1.7 ALOHA clothes folding.

GPU server:

    python -m fluxvla.engines.runners.serving.serve \
        --config configs/gr00tn17/gr00t_n17_native_aloha_fold_clothes_full_finetune.py \
        --ckpt-path /path/to/checkpoint.safetensors \
        --host 127.0.0.1 --port 3333 \
        --device cuda:0 --dtype bf16 --dataset-key inference

Robot-side SSH tunnel:

    ssh -NT -L 5555:127.0.0.1:3333 user@gpu-server

Robot-side client:

    python scripts/inference.py \
        --config configs/gr00tn17/gr00t_n17_native_aloha_fold_clothes_remote.py
"""

_base_ = ['./gr00t_n17_native_aloha_fold_clothes_full_finetune.py']

inference = dict(
    remote_inference=dict(
        server_host='127.0.0.1',
        server_port=5555,
        timeout_s=30.0,
        serializer='msgpack',
        compress=True,
        enable_profiling=True,
    ),
    task_suite_name='private',
)
