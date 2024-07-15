"""
This file contains very basic motions to be accessed by more complex movements.
"""

from __future__ import annotations

import time
import numpy as np
import PyKDL

from robot import HelloRobot
from utils.coordinates import Pose2D, Pose3D, pose_distanced
from global_parameters import *


def move_body(hello_robot: HelloRobot, pose: Pose2D, timeout: int = 10) -> bool:
    """
    This is the most basic movement function. It tells the robot to go to a specified position and orientation.
    :param hello_robot: StretchClient robot controller
    :param pose: target with [x, y, theta] as position and orientation to go to
    :param timeout: seconds after which the movement is considered as failed
    :return: bool whether movement was successful
    """
    # Convert target pose to numpy array
    xyt_goal = pose.pose_as_array()
    # Make sure theta is within [-pi, pi]
    xyt_goal[2] = (xyt_goal[2] + np.pi) % (2 * np.pi) - np.pi
    # Take start time for timeout measurements
    start_time = time.time()
    # Navigation until goal reached or timeout occurs
    while time.time() - start_time < timeout:
        hello_robot.robot.nav.navigate_to(xyt_goal, blocking=False)
        # Get current robot position and orientation
        xyt_curr = hello_robot.robot.nav.get_base_pose()
        print("The robot currently locates at " + str(xyt_curr))
        # Robot has reached target position and orientation
        if np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL) and np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL):
            print("The robot is finally at " + str(xyt_goal))
            return True
        time.sleep(1)
    # Robot failed to reach target position and orientation
    print("Failed to reach the goal within the timeout period")
    return False


def unstow_arm(hello_robot: HelloRobot) -> None:
    """
    Put the arm in the "unstow" position.
    :param hello_robot: StretchClient robot controller
    """
    # Unstow the arm
    hello_robot.move_to_position(lift_pos=MAX_LIFT_POS)
    time.sleep(1)
    hello_robot.move_to_position(arm_pos=MIN_ARM_POS+0.1, wrist_yaw=0, wrist_pitch=0, wrist_roll=0)
    time.sleep(1)


def carry_arm(hello_robot: HelloRobot) -> None:
    """
    Put the arm into carry position.
    :param hello_robot: StretchClient robot controller
    """
    # Carry object
    hello_robot.move_to_position(lift_pos=MAX_LIFT_POS)
    time.sleep(1)
    hello_robot.move_to_position(arm_pos=MIN_ARM_POS)
    time.sleep(1)
    hello_robot.move_to_position(wrist_yaw=2.5)
    time.sleep(1)
    hello_robot.move_to_position(lift_pos=MAX_LIFT_POS // 2)
    time.sleep(1)


def stow_arm(robot: HelloRobot, gripper_open: bool = False) -> None:
    """
    Put the arm in stowed position.
    :param robot: StretchClient robot controller
    :param gripper_open: Gripper closed
    """
    # Stow the arm
    robot.move_to_position(arm_pos=INIT_ARM_POS, gripper_pos=gripper_open)
    time.sleep(1)
    robot.move_to_position(lift_pos=INIT_LIFT_POS,
                           wrist_yaw=INIT_WRIST_YAW, wrist_pitch=INIT_WRIST_PITCH, wrist_roll=INIT_WRIST_ROLL)
    time.sleep(1)


def set_gripper(hello_robot: HelloRobot, gripper_open: bool | float) -> None:
    """
    Set the gripper openness.
    :param hello_robot: StretchClient robot controller
    :param gripper_open: can be float in [0.0, 1.0], False (=0.0) or True (=1.0)
    """
    # Get gripper openness
    fraction = float(gripper_open)
    assert 0.0 <= fraction <= 1.0, "Gripper openness should be between 0.0 and 1.0."
    # Set gripper openness
    hello_robot.move_to_position(gripper_pos=gripper_open)
    time.sleep(1)


def move_arm(hello_robot: HelloRobot, pose: Pose3D, base_node: str, gripper_node: str, move_mode: int = 1) -> None:
    """
    Moves the arm to a specified location relative to the body frame.
    :param hello_robot: StretchClient robot controller
    :param pose: pose to move arm to, specified both coordinates and rotation
    :param base_node: top_camera node
    :param gripper_node: gripper_mid node
    :param move_mode: determines whether to move lift first or everything simultaneously
    """
    # Get joint transform
    transform, _, _ = hello_robot.get_joint_transform(base_node, gripper_node)
    # Determine transformed destination frame
    dest_frame = PyKDL.Frame(pose.rot_matrix, PyKDL.Vector(pose.coordinates[0], pose.coordinates[1], pose.coordinates[2]))
    transformed_frame = transform * dest_frame
    transformed_frame.p[2] -= 0.2
    # Move gripper to pose
    hello_robot.move_to_pose(
        [transformed_frame.p[0], transformed_frame.p[1], transformed_frame.p[2]],
        [transformed_frame.M.GetRPY()[0], transformed_frame.M.GetRPY()[1], transformed_frame.M.GetRPY()[2]],
        move_mode
    )


def move_arm_distanced(hello_robot: HelloRobot, pose: Pose3D, distance: float, base_node: str, gripper_node: str) -> Pose3D:
    """
    Move the arm to a specified pose, but with a distance offset in the viewing direction.
    :param hello_robot: StretchClient robot controller
    :param pose: theoretical pose (if distance = 0)
    :param distance: distance in m that the actual final pose is offset from the pose by
    :param base_node: top_camera node
    :param gripper_node: gripper_mid node
    :return: pose with offset in viewing direction
    """
    # Calculate distanced position and move arm to it
    result_pos = pose_distanced(pose, distance)
    move_arm(hello_robot, result_pos, base_node, gripper_node)
    return result_pos


def gaze(hello_robot: HelloRobot, target: Pose3D) -> None:
    """
    Gaze at target position with head camera.
    :param hello_robot: StretchClient robot controller
    :param target: target position [x, y, z] to gaze at
    """
    # Get head camera position
    camera_xyz = hello_robot.robot.head.get_pose()[:3, 3]
    # Convert target pose to numpy array
    target_xyz = np.asarray(target.coordinates)
    # Calculate direction to target
    vector = camera_xyz - target_xyz
    # Compute head tilt
    tilt = -np.arctan2(vector[2], np.linalg.norm(vector[:2]))
    hello_robot.move_to_position(head_tilt=tilt)
