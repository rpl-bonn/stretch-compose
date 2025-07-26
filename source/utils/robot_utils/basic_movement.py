from __future__ import annotations

import time
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.jointstate_subscriber import JointStateSubscriber
from stretch_package.stretch_state.odom_subscriber import OdomSubscriber
from utils.coordinates import Pose2D, Pose3D
from utils.robot_utils.global_parameters import *


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
    node.send_goal(round(float(pose.coordinates[0]), 3),round(float(pose.coordinates[1]), 3),round(float(pose.direction()[0]), 3), round(float(pose.direction()[1]), 3))
    spin_until_complete(node)

    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])   

    if np.allclose(current_pos, goal_pos, atol=POS_TOL):
        print(f"Reached goal position: {goal_pos}.")
        return True

    print("Failed to reach goal position.")
    return False


def turn_body(node: JointPoseController, pose: Pose2D, grasp: bool= True) -> None:
    """
    Turn the robot to a specified orientation.
    
    Args:
        node (JointPoseController): ROS2 node to control the robot's base
        pose (Pose2D): Target orientation to turn towards
        grasp (bool): Whether grasping after turning is necessary (turn pi/2 further)
    """
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
    current_dir = np.array([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
    current_dir = np.arctan2(current_dir[2], current_dir[3]) * 2.0

    goal_pos = pose.coordinates
    goal_dir = np.arctan2(goal_pos[1]-current_pos[1], goal_pos[0]-current_pos[0])
    
    if grasp: # Note: Turn pi/2 further to grasp
        turn_dir = goal_dir - current_dir + np.pi/2.0
    else:
        turn_dir = goal_dir - current_dir
    
    norm_turn_dir = turn_dir + np.pi % (2*np.pi) - np.pi
    turn_value = {'rotate_mobile_base': norm_turn_dir}
    node.send_joint_pose(turn_value)
    spin_until_complete(node)


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
