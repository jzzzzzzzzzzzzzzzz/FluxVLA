# Single-arm joint-position client. The server must return absolute 8D
# [q1..q7, gripper_width] actions. Topics must match the existing robot stack.
# Robot command sending is disabled until explicitly enabled for testing.
inference = dict(
    type='FrankaInferenceRunner',
    action_mode='joint',
    active_arms=('left', ),
    state_dim=8,
    camera_names=['cam_front', 'cam_wrist_left'],
    task_suite_name='private',
    task_descriptions={'1': 'Follow the reference joint trajectory.'},
    action_chunk=50,
    publish_rate=50,
    async_execution=True,
    execute_horizon=10,
    disable_puppet_arm=True,
    remote_inference=dict(
        server_host='127.0.0.1',
        server_port=5555,
        timeout_s=30.0,
        serializer='msgpack',
        compress=True),
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='tracking',
        num_stitch=6,
        # Illustrative experiment limits; replace with controller limits.
        max_velocity=0.25,
        max_acceleration=1.0,
        max_jerk=5.0),
    operator=dict(
        type='FrankaOperator',
        command_mode='joint',
        image_encoding='rgb8',
        img_front_topic='/camera_front/color/image_raw',
        img_left_topic='/camera_left/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        joint_cmd_topic=(
            '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal'))
