from __future__ import annotations

import time
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import stretch_package.stretch_movement.move_body as _move_body_mod
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.mode_controller import ModeController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.jointstate_subscriber import JointStateSubscriber
from stretch_package.stretch_state.odom_subscriber import OdomSubscriber
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose2D, Pose3D
from utils.recursive_config import Config
from utils.robot_utils.global_parameters import *


try:
    NAV_BACKEND = Config()["navigation"]["backend"]
except Exception:
    NAV_BACKEND = "funmap"

_NAV_ACTION_NAMES = {"funmap": "/move_base", "nav2": "navigate_to_pose"}

_move_body_mod.DEFAULT_NAV_ACTION = _NAV_ACTION_NAMES.get(NAV_BACKEND, "/move_base")


def spin_until_complete(node: Node) -> None:
    """
    Spin node until movement is done.
    
    Args:
        node (Node): ROS2 node to spin
    """
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)
    node.done = False


def get_odom() -> Odometry:
    """
    Returns the robot's current odometry.
    
    Returns:
        Odometry: The current odometry of the robot.
    """
    odom_node = OdomSubscriber()
    spin_until_complete(odom_node)
    odom = odom_node.odom
    odom_node.destroy_node()
    return odom


def get_joint_states() -> JointState:
    """
    Return the robot's current joint states.
    
    Returns:
        JointState: The current joint states of the robot.
    """
    joint_state_node = JointStateSubscriber()
    spin_until_complete(joint_state_node)
    joint_state = joint_state_node.jointstate
    joint_state_node.destroy_node()
    return joint_state


def move_body(node: BaseController, pose: Pose2D) -> bool:
    """
    Move the robot to a specified position and orientation in the world frame.
    
    Args:
        node (BaseController): ROS2 node to control the robot's base
        pose (Pose2D): Target position and orientation to go to
    
    Returns:
        bool: Whether the movement was successful
    """
    goal_pos = np.array([pose.coordinates[0], pose.coordinates[1]])

    mode_node = None
    if NAV_BACKEND == "nav2":
        mode_node = ModeController()
        mode_node.switch_to_navigation_mode()

    node.send_goal(round(float(pose.coordinates[0]), 3),round(float(pose.coordinates[1]), 3),round(float(pose.direction()[0]), 3), round(float(pose.direction()[1]), 3))
    spin_until_complete(node)

    if mode_node is not None:
        mode_node.switch_to_position_mode()
        mode_node.destroy_node()

    if NAV_BACKEND == "nav2":
        if node.success:
            print(f"Reached goal position: {goal_pos}.")
            return True
        print("Failed to reach goal position.")
        return False

    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])

    if np.allclose(current_pos, goal_pos, atol=POS_TOL):
        print(f"Reached goal position: {goal_pos}.")
        return True

    print("Failed to reach goal position.")
    return False


def turn_body(node: JointPoseController, pose: Pose2D, transform_node: FrameTransformer, grasp: bool= True, small: bool = False) -> None:
    """
    Turn the robot to a specified orientation.
    
    Args:
        node (JointPoseController): ROS2 node to control the robot's base
        pose (Pose2D): Target orientation to turn towards
        transform_node (FrameTransformer): ROS2 node to transform frames
        grasp (bool): Whether grasping after turning is necessary (turn pi/2 further)
        small (bool): Whether to turn by a smaller angle
    """
    # Get current position and yaw from TF map->base_link so heading math is in map frame.
    try:
        transform = transform_node.get_tf_matrix("map", "base_link")
        if transform is None:
            print("Failed to lookup transform.")
            return
        current_pos = np.array([transform[0, 3], transform[1, 3], transform[2, 3]])
        current_dir = np.arctan2(transform[1, 0], transform[0, 0])

    except Exception:
        print("Failed to lookup transform.")
        return

    goal_pos = pose.coordinates
    goal_dir = np.arctan2(goal_pos[1]-current_pos[1], goal_pos[0]-current_pos[0])
    
    # ===== DEBUG BLOCK START =====
    print("\n[TURN_DEBUG] ------------------------------")
    print(f"[TURN_DEBUG] grasp={grasp} small={small}")
    print(f"[TURN_DEBUG] current_pos_xy(map)=({current_pos[0]:.3f}, {current_pos[1]:.3f})")
    print(f"[TURN_DEBUG] goal_pos_xy(assumed same frame)=({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
    print(f"[TURN_DEBUG] current_yaw_deg={np.degrees(current_dir):.2f}")
    print(f"[TURN_DEBUG] goal_yaw_deg={np.degrees(goal_dir):.2f}")

    raw_delta = goal_dir - current_dir
    print(f"[TURN_DEBUG] raw_delta_deg={np.degrees(raw_delta):.2f}")

    # Debugging the angle normalization to ensure it's correct
    buggy_norm = raw_delta + np.pi % (2 * np.pi) - np.pi
    # Correct wrapping
    correct_norm = (raw_delta + np.pi) % (2 * np.pi) - np.pi

    print(f"[TURN_DEBUG] buggy_norm_deg={np.degrees(buggy_norm):.2f}")
    print(f"[TURN_DEBUG] correct_norm_deg={np.degrees(correct_norm):.2f}")

    # Useful to see if target is already nearly aligned
    print(f"[TURN_DEBUG] abs_correct_norm_deg={abs(np.degrees(correct_norm)):.2f}")
    print("[TURN_DEBUG] ------------------------------\n")
    # ===== DEBUG BLOCK END =====
    
    if grasp: # Note: Turn pi/2 further to grasp
        if small:
            turn_dir = goal_dir - current_dir + np.pi/1.85
        else:
            turn_dir = goal_dir - current_dir + np.pi/2.0
        print(f"TURN DIR (GRASP): {np.degrees(turn_dir):.2f} degrees")
    else:
        turn_dir = goal_dir - current_dir
    
    norm_turn_dir = (turn_dir + np.pi) % (2*np.pi) - np.pi
    turn_value = {'rotate_mobile_base': norm_turn_dir}
    print(f"Turning by {np.degrees(norm_turn_dir):.2f} degrees")
    node.send_joint_pose(turn_value)
    spin_until_complete(node)
    
    # ===== POST-TURN DEBUG =====
    try:
        transform_after = transform_node.get_tf_matrix("map", "base_link")
        if transform_after is None:
            print("Failed to lookup transform for post-turn debug.")
            return
        after_yaw = np.arctan2(transform_after[1, 0], transform_after[0, 0])
    except Exception:
        print("Failed to lookup transform for post-turn debug.")
        return
    residual = (goal_dir - after_yaw + np.pi) % (2 * np.pi) - np.pi

    print("\n[TURN_DEBUG_POST] -------------------------")
    print(f"[TURN_DEBUG_POST] after_yaw_deg={np.degrees(after_yaw):.2f}")
    print(f"[TURN_DEBUG_POST] residual_to_goal_deg={np.degrees(residual):.2f}")
    print("[TURN_DEBUG_POST] -------------------------\n")

def unstow_arm(node: JointPoseController, pose: Pose3D, yaw: float = np.pi/2, pitch: float = 0.0, roll: float = 0.0, lift: float = 0.0) -> None:
    """
    Put the arm in the "unstow" position.
    
    Args:
        node (JointPoseController): ROS2 node to move arm into a certain pose
        pose (Pose3D): Target position of object
        yaw (float): Yaw angle for the gripper
        lift (float): Lift reduction for the gripper
    """
    unstow_pos = {'joint_lift': pose.coordinates[2]-lift, 'joint_wrist_yaw': yaw, 'joint_wrist_pitch': pitch, 'joint_wrist_roll': roll}
    node.send_joint_pose(unstow_pos)
    spin_until_complete(node)
    
def adjust_door(node: JointPoseController, pose: Pose3D, yaw: float = np.pi/2, pitch: float = 0.0, roll: float = 0.0, lift: float = 0.0) -> None:
    """
    Put the arm in the "unstow" position.
    
    Args:
        node (JointPoseController): ROS2 node to move arm into a certain pose
        pose (Pose3D): Target position of object
        yaw (float): Yaw angle for the gripper
        lift (float): Lift reduction for the gripper
    """
    unstow_pos = {'joint_lift': pose.coordinates[2]-lift,'joint_wrist_pitch': pitch, 'joint_wrist_roll': roll}
    node.send_joint_pose(unstow_pos)
    spin_until_complete(node)


def stow_arm(node: StowArmController) -> None:
    """
    Put the arm in stowed position.
    
    Args:
        node (StowArmController): ROS2 node to stow the arm
    """
    node.send_stow_request()
    rclpy.spin_until_future_complete(node, node.future)
    time.sleep(2)
    
    
def carry_arm(node: JointPoseController) -> None:
    """
    Put the arm into carry position.
    
    Args:
        node (JointPoseController): ROS2 node to move arm into a certain pose
    """
    # Lift arm
    joint_states = get_joint_states()
    # get the joint_lift value
    joint_lift = joint_states.position[joint_states.name.index('joint_lift')]
    carry_pos_1 = {'joint_lift': joint_lift + 0.05, 'joint_wrist_pitch': INIT_WRIST_PITCH}
    node.send_joint_pose(carry_pos_1)
    spin_until_complete(node)
    # Retract arm
    carry_pos_2 = {'wrist_extension': MIN_ARM_POS}
    node.send_joint_pose(carry_pos_2)
    spin_until_complete(node)
    time.sleep(1)
    # Go into carry position
    carry_pos_3 = {'joint_lift': INIT_LIFT_POS, 'joint_wrist_yaw': 2.5, 'joint_wrist_roll': INIT_WRIST_ROLL}
    node.send_joint_pose(carry_pos_3)
    spin_until_complete(node)


def set_gripper(node: JointPoseController, gripper_open: bool | float) -> None:
    """
    Set the gripper openness. Maximum is 0.22 and minimum 0.00.
    
    Args:
        node (JointPoseController): ROS2 node to move arm into a certain pose
        gripper_open (bool | float): can be float in [0.0, 1.0], False (=0.0) or True (=1.0)
    """
    fraction = float(gripper_open) * 0.22 # Note: 0.22 is the maximum gripper aperture
    gripper_pos = {'gripper_aperture': fraction}
    node.send_joint_pose(gripper_pos)
    spin_until_complete(node)
    

def move_arm(node: JointPositionController, pose: Pose3D, roll: bool = True, lower: float = 0.03) -> None:
    """
    Move the arm to a specified location in the world frame.
    
    Args:
        node (JointPositionController): ROS2 node to move arm to a certain position
        pose (Pose3D): Target position and orientation to go to
        roll (bool): Whether to calculate roll for gripper (needs rotation matrix).
    """
    pos = np.array([round(pose.coordinates[0], 3),round(pose.coordinates[1], 3), round(pose.coordinates[2]-lower, 3)])
    dir = np.array([round(pose.direction()[0], 3),round(pose.direction()[1], 3), round(pose.direction()[2], 3)])
    if not roll:
        rot = None
    else:
        rot = pose.rot_matrix

    joint_states = get_joint_states()
    node.send_joint_pos(pos, dir, joint_states, rot)
    spin_until_complete(node)


def move_head(node: HeadJointController, pose: Pose3D, z_fix: float = 0.0, tilt_bool: bool = True) -> None:
    """
    Gaze at target position with head camera.
    
    Args:
        node (HeadJointController): ROS2 node to control the head
        pose (Pose3D): Target position to gaze at
        z_fix (float, optional): Z-axis offset for the target position. Defaults to 0.0.
        tilt_bool (bool, optional): Whether to tilt the head or not. Defaults to True.
    """
    pos = np.array([round(pose.coordinates[0], 3),round(pose.coordinates[1], 3), round(pose.coordinates[2]+z_fix, 3)])
    node.send_joint_pose(pos, tilt_bool=tilt_bool)
    spin_until_complete(node)
    
    
def look_ahead(node: JointPoseController) -> None:
    """
    Look ahead with head camera.
    
    Args:
        node (JointPoseController): ROS2 node to control the head
    """
    ahead_pos = {'joint_head_pan': -np.pi/2, 'joint_head_tilt': 0.0}
    node.send_joint_pose(ahead_pos)
    spin_until_complete(node)


def move_body_test(node: BaseController, pose: Pose2D) -> bool:
    """
    Move the robot to a specified position and orientation in the world frame.
    
    Args:
        node (BaseController): ROS2 node to control the robot's base
        pose (Pose2D): Target position and orientation to go to
    
    Returns:
        bool: Whether the movement was successful
    """
    goal_pos = np.array([pose[0], pose[1]])
    node.send_goal(round(float(pose[0]), 3),round(float(pose[1]), 3),round(float(pose[2]), 3), round(float(pose[3]), 3), round(float(pose[4]), 3))
    spin_until_complete(node)

    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])   

    if np.allclose(current_pos, goal_pos, atol=POS_TOL):
        print(f"Reached goal position: {goal_pos}.")
        return True

    print("Failed to reach goal position.")
    return False


def yaw_from_quaternion(qx, qy, qz, qw):
    """Convert quaternion to yaw (radians)."""
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )

def turn_body_test(node: JointPoseController, pose: list, grasp: bool=True, full_rotation: bool=True) -> None:
    """
    Turn the robot to a specified orientation.
    
    Args:
        node (JointPoseController): ROS2 node to control the robot's base
        pose (list): [x, y, qx, qy, qz, qw]
        grasp (bool): If True, add pi/2 extra rotation
        full_rotation (bool): If True, rotate fully to the given yaw (no shortest path)
    """
    odom = get_odom()
    cq = odom.pose.pose.orientation

    current_yaw = yaw_from_quaternion(cq.x, cq.y, cq.z, cq.w)

    goal_qx, goal_qy, goal_qz, goal_qw = pose[2], pose[3], pose[4], pose[5]
    goal_yaw = yaw_from_quaternion(goal_qx, goal_qy, goal_qz, goal_qw)

    # compute yaw difference
    turn_dir = goal_yaw - current_yaw
    # if grasp:
    #     turn_dir += np.pi / 2.0

    if not full_rotation:
        # normalize to shortest path
        turn_dir = (turn_dir + np.pi) % (2*np.pi) - np.pi

    turn_value = {'rotate_mobile_base': turn_dir}
    print(f"Turning by {np.degrees(turn_dir):.2f} degrees")
    node.send_joint_pose(turn_value)
    spin_until_complete(node)