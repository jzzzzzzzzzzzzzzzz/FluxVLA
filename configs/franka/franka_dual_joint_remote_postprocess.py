_base_ = './franka_joint_remote_postprocess.py'

# The dual-arm server must return two ordered 8D action blocks.
inference = dict(
    active_arms=('left', 'right'),
    state_dim=16,
    camera_names=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
    operator=dict(
        _delete_=True,
        type='FrankaDualOperator',
        command_mode='joint',
        image_encoding='rgb8',
        img_front_topic='/camera_front/color/image_raw',
        img_left_topic='/camera_left/color/image_raw',
        img_right_topic='/camera_right/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        puppet_franka_state_right_topic=(
            '/right_arm/franka_state_controller/franka_states'),
        joint_cmd_left_topic=(
            '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
        joint_cmd_right_topic=(
            '/right_arm/ruckig_joint_impedance_controller/target_joint_state'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal'))
