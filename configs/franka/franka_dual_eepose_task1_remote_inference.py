# Copyright 2026 Limx Dynamics
"""Model-agnostic ZMQ robot client for dual-Franka Task 1.

The GPU server owns model preprocessing and action denormalization, so this
robot-side config can be reused with every Task-1 model config.
"""

import os

_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')

inference = dict(
    type='FrankaInferenceRunner',
    remote_inference=dict(
        server_host=os.environ.get('FRANKA_ZMQ_HOST', '127.0.0.1'),
        server_port=int(os.environ.get('FRANKA_ZMQ_PORT', '5555')),
        timeout_s=float(os.environ.get('FRANKA_ZMQ_TIMEOUT_S', '120.0')),
        serializer='msgpack',
        compress=True,
        enable_profiling=True),
    seed=7,
    task_suite_name='private',
    task_descriptions={'1': _TASK_DESCRIPTION},
    state_dim=16,
    # All supported servers return at most their native horizon. Execute 16
    # actions so the same client also works with DiT4DiT's 16-step policy.
    action_chunk=50,
    execute_horizon=int(os.environ.get('FRANKA_EXECUTE_HORIZON', '16')),
    publish_rate=30,
    max_publish_step=10000,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    prepare_pose=None,
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
