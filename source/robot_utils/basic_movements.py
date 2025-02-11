"""
This file contains very basic motions to be accessed by more complex movements.
"""

from __future__ import annotations

import time
import numpy as np

import rclpy
from rclpy.node import Node

from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.odom_subscriber import OdomSubscriber
from stretch_package.stretch_state.jointstate_subscriber import JointStateSubscriber

from utils.coordinates import Pose2D, Pose3D, pose_distanced
from utils.robot_utils.global_parameters import *


def spin_until_complete(node: Node):
    # Spin until movement is done
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)
    node.done = False
  

def get_odom():
    """
    Returns the robot's current odometry.
    """
    # Get current odometry
    odom_node = OdomSubscriber()
    spin_until_complete(odom_node)
    odom = odom_node.odom
    odom_node.destroy_node()
    return odom


def get_joint_states():
    """
    Return the robot's current joint states.
    """
    # Get current joint states
    joint_state_node = JointStateSubscriber()
    spin_until_complete(joint_state_node)
    joint_state = joint_state_node.jointstate
    joint_state_node.destroy_node()
    return joint_state


def move_body(node: BaseController, pose: Pose2D) -> bool:
    """
    Move the robot to a specified position and orientation in the world frame.
    :param node: BaseController ROS2 node
    :param pose: target position and orientation to go to
    :return: bool whether movement was successful
    """
    # Convert goal position to numpy array
    goal_pos = np.array([pose.coordinates[0], pose.coordinates[1]])
    # Send goal position and orientation to robot
    node.send_goal(round(pose.coordinates[0], 3),round(pose.coordinates[1], 3),round(pose.direction()[0], 3), round(pose.direction()[1], 3))
    spin_until_complete(node)
    # Get current positition
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])   
    # Robot has reached target position and orientation
    if np.allclose(current_pos, goal_pos, atol=POS_TOL):
        print(f"Reached goal position: {goal_pos}.")
        return True
    # Robot failed to reach target position and orientation
    print("Failed to reach goal position.")
    return False


def turn_body(node: JointPoseController, pose: Pose2D):
    """
    Turn the robot to a specified orientation.
    :param node: JointPoseController ROS2 node
    :param pose: target orientation to turn to
    :return: bool whether movement was successful
    """
    # Get current position and orientation
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
    current_dir = np.array([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
    current_dir = np.arctan2(current_dir[2], current_dir[3]) * 2.0
    # Convert goal position and orientation
    goal_pos = pose.coordinates
    goal_dir = np.arctan2(goal_pos[1]-current_pos[1], goal_pos[0]-current_pos[0])
    print(f"current dir: {current_dir}, goal dir: {goal_dir}")
    # Calculate turn direction to goal position
    turn_dir = goal_dir - current_dir + np.pi/2.0
    norm_turn_dir = turn_dir + np.pi % (2*np.pi) - np.pi
    print(f"turn by: {norm_turn_dir}")
    # Send goal orientation to robot
    turn_value = {'rotate_mobile_base': norm_turn_dir}
    node.send_joint_pose(turn_value)
    spin_until_complete(node)


def unstow_arm(node: JointPoseController, pose: Pose3D) -> None:
    """
    Put the arm in the "unstow" position.
    :param node: JointPoseController ROS2 node
    """
    # Unstow arm
    unstow_pos = {'joint_lift': pose.coordinates[2], 'joint_wrist_yaw': np.pi/2, 'joint_wrist_pitch': 0.0, 'joint_wrist_roll': 0.0}
    node.send_joint_pose(unstow_pos)
    spin_until_complete(node)


def carry_arm(node: JointPoseController) -> None:
    """
    Put the arm into carry position.
    :param node: JointPoseController ROS2 node
    """
    # Lift arm
    carry_pos_1 = {'joint_lift': MAX_LIFT_POS, 'joint_wrist_pitch': 0.0}
    node.send_joint_pose(carry_pos_1)
    spin_until_complete(node)
    # Retract arm
    carry_pos_2 = {'wrist_extension': MIN_ARM_POS}
    node.send_joint_pose(carry_pos_2)
    spin_until_complete(node)
    # Move backwards
    carry_pos_3 = {'translate_mobile_base': -0.2}
    node.send_joint_pose(carry_pos_3)
    spin_until_complete(node)
    # Go into carry position
    carry_pos_4 = {'joint_lift': MAX_LIFT_POS // 3, 'wrist_yaw': 2.5, 'wrist_roll': 0.0}
    node.send_joint_pose(carry_pos_4)
    spin_until_complete(node)


def stow_arm(node: StowArmController) -> None:
    """
    Put the arm in stowed position.
    :param node: StowArmController ROS2 node
    """
    # Stow arm
    node.send_stow_request()
    rclpy.spin_until_future_complete(node, node.future)
    time.sleep(2)


def set_gripper(node: JointPoseController, gripper_open: bool | float) -> None:
    """
    Set the gripper openness. Maximum is 0.22 and minimum 0.05.
    :param node: JointPoseController ROS2 node
    :param gripper_open: can be float in [0.0, 1.0], False (=0.0) or True (=1.0)
    """
    # Get gripper openness (scale accordingly)
    fraction = float(gripper_open) * (0.22-0.05) + 0.05
    # Set gripper openness
    gripper_pos = {'gripper_aperture': fraction}
    node.send_joint_pose(gripper_pos)
    spin_until_complete(node)


def move_arm(node: JointPositionController, pose: Pose3D) -> None:
    """
    Move the arm to a specified location in the world frame.
    :param node: JointPositionController ROS2 node
    :param pose: pose to move arm to, specified both coordinates and rotation
    """
    # Convert position to numpy array
    pos = np.array([round(pose.coordinates[0], 3),round(pose.coordinates[1], 3), round(pose.coordinates[2]-0.03, 3)])
    dir = np.array([round(pose.direction()[0], 3),round(pose.direction()[1], 3), round(pose.direction()[2], 3)])
    # Get current joint states
    joint_states = get_joint_states()
    # Move gripper to pose
    node.send_joint_pos(pos, dir, joint_states)
    spin_until_complete(node)


def move_arm_distanced(node: JointPositionController, pose: Pose3D, distance: float) -> Pose3D:
    """
    Move the arm to a specified pose, but with a distance offset in the viewing direction.
    :param node: JointPositionController ROS2 node
    :param pose: theoretical pose (if distance = 0)
    :param distance: distance offset from the theoretical pose
    :return: pose with offset in viewing direction (but same height)
    """
    # Calculate distanced position
    pose_dist = pose_distanced(pose, distance)
    pose_dist.coordinates[2] = pose.coordinates[2]
    move_arm(node, pose_dist)
    return pose_dist


def gaze(node: HeadJointController, target: Pose3D) -> None:
    """
    Gaze at target position with head camera.
    :param node: HeadJointController ROS2 node
    :param target: target position [x, y, z] to gaze at
    """
    # Convert target to numpy array
    pos = np.array([round(target.coordinates[0], 3),round(target.coordinates[1], 3), round(target.coordinates[2]-0.01, 3)])
    # Gaze at object
    node.send_joint_pose(pos)
    spin_until_complete(node)
