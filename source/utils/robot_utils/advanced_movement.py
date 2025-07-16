from __future__ import annotations

import time
import numpy as np
import open3d as o3d

from utils.robot_utils.global_parameters import *
from utils.robot_utils.basic_perception import check_object_distance
from utils.robot_utils.basic_movement import *
from scipy.spatial.transform import Rotation
from utils import vis
from utils.coordinates import Pose2D, Pose3D, from_a_to_b_distanced, pose_distanced, get_door_opening_poses
from utils.graspnet_interface import predict_full_grasp
from gpd.gpd_client_api import predict_full_grasp as gpd_predict_full_grasp
from utils.importer import PointCloud
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.time import convert_time
from scripts.my_robot_scripts.graspnet_testing import visualize_grasps
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer

# Config and Paths
config = Config()
IMG_DIR = config.get_subpath("images")


def move_body_distanced(node: BaseController, end_pose: Pose2D, distance: float, sleep: bool = True) -> None:
    """
    Move to a location, with a certain distance offset. The distance is in the direction of the current pose.
    :param node: BaseController ROS2 node
    :param end_pose: final pose to walk towards (minus the distance)
    :param distance: distance offset to final pose
    :param sleep: whether to sleep in between movements for safety
    """
    sleep_multiplier = 1 if sleep else 0
    # Get start pose of robot
    odom = get_odom()
    current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
    current_dir = np.arctan2(odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) * 2.0
    start_pose = Pose2D(current_pos, current_dir)
    # Calculate destination pose
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)
    # Move robot
    move_body(node, destination_pose)
    time.sleep(1 * sleep_multiplier)
    
    
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
    move_arm(node, pose_dist, lower=0.0)
    return pose_dist


def positional_grab(position_node: JointPositionController, pose_node: JointPoseController, pose: Pose3D, distance_start: float, distance_end: float, widths) -> None:
    """
    Grab something at a specified position. The gripper will first move towards the pose, but offset by distance_start
    in the opposite direction of the viewing direction. Then it will move towards the specified pose offset by distance_end.
    So in essence, the direction of the pose specifies the axis along which it moves.
    :param position_node: JointPositionController ROS2 node
    :param pose_node: JointPoseController ROS2 node
    :param pose: pose to grab
    :param distance_start: distance from which to start grab
    :param distance_end: distance at which to end grab
    :param already_gripping: whether to NOT open up the gripper in the beginning
    """
    # Move arm to grab start position
    move_arm_distanced(position_node, pose, distance_start)
    set_gripper(pose_node, True)
    # Move arm to grab end position and grab
    move_arm_distanced(position_node, pose, distance_end)
    # Approach slowly when slightly inaccurate
    distance_finetuned = check_object_distance(pose_node)
    move_arm_distanced(position_node, pose, distance_finetuned)
    set_gripper(pose_node, (widths-0.06)/0.22)
    # Move arm back to grab start position
    carry_arm(pose_node)
    
     
def pull_door(pos_node: JointPositionController, pose_node: JointPoseController, transform_node: FrameTransformer, handle_center: Pose3D, hinge_center: Pose3D, opens_to: str) -> None:
    """
    Execute a pulling motion for doors.
    This function moves the arm in front of the door handle, closes the gripper, and pulls the door open.

    Args:
        pose_node (JointPoseController): ROS2 node for joint pose control
        transform_node (FrameTransformer): ROS2 node for frame transformation
        handle_center (Pose3D): 3D position of door handle
        hinge_center (Pose3D): 3D position of door hinge
        opens_to (str): Direction in which the door opens
    """
    # Open gripper and move arm towards door handle
    set_gripper(pose_node, 0.4)
    pull_pose_start = {'wrist_extension': (0.35, 40.0)}
    pose_node.send_joint_pose(pull_pose_start)
    spin_until_complete(pose_node)
    # Close gripper to grab handle
    #set_gripper(pose_node, 0.0)
    dist = np.linalg.norm(handle_center.coordinates[0] - hinge_center.coordinates[0])
    print("dist: ", dist)
    poses = get_door_opening_poses(hinge_center, dist, opens_to)
    
    for pose in poses:
        # Move arm into position
        move_arm(pos_node, pose)
        spin_until_complete(pose_node)
        

def pull_drawer(pose_node: JointPoseController):
    """
    Exectue a pulling motion (e.g. for drawers).
    This function moves the arm in front of the drawer handle, opens the gripper, moves the arm to the handle, closes the gripper,
    and pulls the drawer open.

    Args:
        pose_node (JointPoseController): ROS2 node for joint pose control
        transform_node (FrameTransformer): ROS2 node for frame transformation
        handle_center (Pose3D): position of drawer handle in 3D space

    Returns:
        Pose3D: end pose of the gripper after pulling
    """
    # Open gripper and move arm towards drawer handle
    set_gripper(pose_node, 0.4)
    pull_pose_start = {'wrist_extension': (0.5, 40.0)}
    pose_node.send_joint_pose(pull_pose_start)
    spin_until_complete(pose_node)
    # Close gripper to grab handle and pull drawer
    set_gripper(pose_node, -0.01)
    pull_pose_end = {'wrist_extension': 0.05}
    pose_node.send_joint_pose(pull_pose_end)
    spin_until_complete(pose_node)
    time.sleep(2.0)


def push(pose_node: JointPoseController, height: float) -> None:
    """
    Executes a pushing motion (e.g. for drawers).
    :param position_node: JointPositionController ROS2 node
    :param pose_node: JointPoseController ROS2 node 
    :param start_pose: pose of knob in 3D space
    :param end_pose: pose of knob when drawer closed
    :param start_distance: how far from the button to start push
    :param end_distance: how far to push
    """
    # Move arm in front of the drawer
    push_pose = {'wrist_extension': 0.0, 'joint_lift': height, 'joint_wrist_pitch': 0.0}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    # Push drawer
    push_pose = {'wrist_extension': (0.49, 50.0)}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)
    time.sleep(1.0)
    # Retract arm after pushing
    push_pose = {'wrist_extension': 0.1}
    pose_node.send_joint_pose(push_pose)
    spin_until_complete(pose_node)


def adapt_body(best_pose: Pose3D, best_grasp: Pose3D) -> Pose3D:
    # Get extra offset from body-grasp distance
    print(f"body-grasp dist: {np.linalg.norm(best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates)}")
    extra_offset = 0.6-np.linalg.norm(best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates)
    # Get direction
    direction = best_pose.to_dimension(2).coordinates - best_grasp.to_dimension(2).coordinates
    unit_direction = direction / np.linalg.norm(direction)
    # Add offset to direction
    body_pose_distanced = best_pose.copy()
    body_pose_distanced.coordinates = body_pose_distanced.coordinates + np.append(unit_direction, 0.0) * extra_offset
    return body_pose_distanced


def adapt_grasp(tf_node: FrameTransformer, grasp_pose: np.ndarray, min_height: float = 0.0) -> np.ndarray:
    """
    Mirrors the grasp pose if the grasp is from behind.

    Args:
        tf_node (FrameTransformer): ROS2 node for frame transformation
        grasp_pose (np.ndarray): 3D pose of the selected grasp

    Returns:
        np.ndarray: adapted grasp pose
    """
    # get height of the grasp pose ndarray
    if grasp_pose[2, 3] < min_height:
        grasp_pose[2, 3] = min_height
    tf = tf_node.get_tf_matrix("base_link", "map")
    tf_in_base = tf @ grasp_pose
    rotation = tf_in_base[:3, :3].copy()
    euler_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
    if 0 < euler_angles[2] < 180:  # if yaw angle is between 0 and 180 degrees, mirror it
        euler_angles[2] = -euler_angles[2]
    
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    tf_in_base[:3, :3] = rotation.copy()
    new_grasp_pose = np.linalg.inv(tf) @ tf_in_base   
    
    return new_grasp_pose


def filter_grasps(tf_node: FrameTransformer, tf_matrices: np.ndarray, widths: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters grasps based on their yaw angle.

    Args:
        tf_node (FrameTransformer): ROS2 node for frame transformation
        tf_matrices (np.ndarray): transformation matrices of the grasps
        widths (np.ndarray): widths of the grasps
        scores (np.ndarray): scores of the grasps

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: filtered transformation matrices, widths, and scores
    """
    filtered_tf_matrices = []
    filtered_widths = []
    filtered_scores = []
    
    tf = tf_node.get_tf_matrix("base_link", "map")
    
    for tf_matrix, width, score in zip(tf_matrices, widths, scores):
        # Transform grasp pose into robot frame
        tf_in_base = tf @ tf_matrix
        euler_angles = Rotation.from_matrix(tf_in_base[:3, :3]).as_euler('xyz', degrees=True)
        if -100 <= euler_angles[2] <= 100:  # filter out grasps with a yaw angle < -100 degrees
            filtered_tf_matrices.append(tf_matrix)
            filtered_widths.append(width)
            filtered_scores.append(score)
    
    return np.array(filtered_tf_matrices), np.array(filtered_widths), np.array(filtered_scores)


def adapt_grasp_orientation(grasp_pose: Pose3D, body_pose: Pose3D, target_pose: Pose3D) -> Pose3D:
    body_to_target_direction = target_pose.coordinates - body_pose.coordinates
    body_to_target_direction_normalized = body_to_target_direction / np.linalg.norm(body_to_target_direction)
    current_grasp_direction = grasp_pose.direction() 
    
    # if grasp direction deviates by more than 45 degrees from the body-to-target direction around the z-axis, mirror it around this axis
    if np.dot(body_to_target_direction_normalized, current_grasp_direction) < np.cos(np.radians(45)):
        rotation_matrix = Rotation.from_euler("z", 180, degrees=True).as_matrix()
        adjusted_grasp_pose = Pose3D(grasp_pose.coordinates.copy(), rotation_matrix @ grasp_pose.rot_matrix)
    else:
        adjusted_grasp_pose = Pose3D(grasp_pose.coordinates.copy(), grasp_pose.rot_matrix.copy())
    
    return adjusted_grasp_pose


def optimize_joints(target: Pose3D, tf_matrices: np.ndarray, widths: np.ndarray, grasp_scores: np.ndarray, body_scores: list[tuple[Pose3D, float]], 
                    lambda_body: float = 0.5, lambda_alignment: float = 1.0, temperature: float = 0.2,) -> tuple[Pose3D, Pose3D, float]: 
    if not body_scores:
        raise ValueError("body_scores is empty; at least one pose is required.")
    
    nr_grasps = tf_matrices.shape[0]
    nr_poses = len(body_scores)
    # matrix is nr_grasps x nr_poses 
    grasp_scores = grasp_scores.reshape((nr_grasps, 1))
    grasp_score_mat = np.tile(grasp_scores, (1, nr_poses))
    pose_scores = np.asarray([score for (_, score) in body_scores]).reshape((1, nr_poses))
    pose_score_mat = np.tile(pose_scores, (nr_grasps, 1))
    
    grasp_directions_np = np.stack([Pose3D.from_matrix(tf).direction() for tf in tf_matrices], axis=0)
    body_coordinates_np = np.stack([pose.coordinates for (pose, _) in body_scores], axis=1)
    body_to_targets_np = target.coordinates.reshape((3, 1)) - body_coordinates_np
    body_coordinates_norm = body_to_targets_np / np.linalg.norm(body_to_targets_np, axis=0, keepdims=True)

    alignment_mat = grasp_directions_np @ body_coordinates_norm
    alignment_mat_tanh = np.tanh(alignment_mat * temperature)
    joint_matrix = (grasp_score_mat + lambda_body * pose_score_mat + lambda_alignment * alignment_mat_tanh)

    grasp_argmax, pose_argmax = np.unravel_index(np.argmax(joint_matrix), joint_matrix.shape)
    print(f"{grasp_argmax=}", f"{pose_argmax=}")
    
    best_grasp = Pose3D.from_matrix(tf_matrices[grasp_argmax])
    best_pose = body_scores[pose_argmax][0]
    best_width = widths[grasp_argmax]  
    return best_grasp, best_pose, best_width


def find_new_grasp_dynamically(
    pos_node: JointPositionController,
    pose_node: JointPoseController,
    tf_node: FrameTransformer,
    body_pose: Pose3D,
    distance_start: float,
    distance_end: float,
    config: Config,
    pcd_obj,
    pcd_env,
) -> None:
    """
    Adaptive grasp based on collecting a new point cloud close to the supposed grasp position allowing it to adjust for
    some drift in localization. This method specifically creates a new point cloud at the supposed position, calculates
    the transformation from original PCD to the new dynamically collected PCD, and transforms the grasp accordingly.
    :param hello_robot: StretchClient robot controller
    :param pose: supposed grasp position
    :param distance_start: start of the grabbing motion
    :param distance_end: end of the grabbing motion
    :param config:
    :param nr_captures: number of captures to create PCD
    :param offset: increments in which to move the camera when collecting new PCD
    :param tolerance: tolerance for maximum possible drift
    :param degrees: whether increments is in degrees (True) or radians (False)
    """
    # Remove lowest points from object to prevent low grasps
    points = np.asarray(pcd_obj.points)
    pcd_filtered = o3d.geometry.PointCloud()
    # cut off min 3cm from the bottom
    obj_height= np.max(points[:, 2]) - np.min(points[:, 2])
    min_height = np.min(points[:, 2]) + 0.03
    #pcd_filtered.points = o3d.utility.Vector3dVector(points[points[:, 2] > np.min(points[:, 2]) + max(obj_height/3, 0.02)] )
    #pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd_obj.colors)[points[:, 2] > np.min(points[:, 2]) + max(obj_height/3, 0.02)])
    
    # Downsample point clouds
    pcd_obj = pcd_obj.voxel_down_sample(0.005)
    pcd_env = pcd_env.voxel_down_sample(0.02)
    
    try:
        # Get grasp pose
        start_time = time.time_ns()
        tf_matrices, widths, scores = gpd_predict_full_grasp(pcd_obj, pcd_env, config, vis_block=True)
        end_time = time.time_ns()
        minutes, seconds = convert_time(end_time - start_time)
        print(f"\nGrasp prediction RUNTIME: {minutes}min {seconds}s\n")
        
        tf_matrices, widths, scores = filter_grasps(tf_node, tf_matrices, widths, scores)
        visualize_grasps(pcd_obj, pcd_env, tf_matrices, widths, scores, "filtered_grasps")

        final_grasp = adapt_grasp(tf_node, tf_matrices[0], min_height)
        visualize_grasps(pcd_obj, pcd_env, [final_grasp], [widths[0]], [scores[0]], "final_grasp")
        final_grasp = Pose3D(final_grasp[:3, 3], final_grasp[:3, :3])
    except Exception as e:
        print(f"Error: Failed predicting grasp. {e}")  
    try:
        positional_grab(pos_node, pose_node, final_grasp, distance_start, distance_end, widths[0])
    except Exception as e:
        print(f"Error: Failed grabbing object. {e}")


def look_into_drawer(pose_node: JointPoseController, handle_pose: Pose3D):
    """
    Move the arm into a position above the drawer to look into it.
    This function moves the arm to a position above the drawer, with a z-offset of 0.3m and a wrist pitch of -45 degrees.

    Args:
        pose_node (JointPoseController): ROS2 node to move arm into a certain pose
        handle_pose (Pose3D): 3D position of drawer handle
    """
    height = handle_pose.coordinates[2] + 0.3
    gripper_pose = {'wrist_extension': 0.01 , 'gripper_aperture': 1.0}
    pose_node.send_joint_pose(gripper_pose)
    spin_until_complete(pose_node)
    gripper_pose = {'joint_lift': height, 'wrist_extension': 0.2 , 'joint_wrist_pitch': -np.pi/4, 'joint_wrist_roll': 0.0}
    pose_node.send_joint_pose(gripper_pose)
    spin_until_complete(pose_node)
    
    
def move_in_front_of(
    stow_node: StowArmController, base_node: BaseController, head_node: HeadJointController, pose_node: JointPoseController, 
    body_pose: Pose3D, target_center: Pose3D, yaw: float, pitch: float, roll: float, lift: float, stow: bool = True, grasp: bool = False
) -> None:
    """
    Move and turn the robot in front of a target object.
    This function stows the arm, moves the robot's base to a specified position,
    and turns the robot to face the target object, either with the head or the gripper.

    Args:
        stow_node (StowArmController): ROS2 node to stow the arm
        base_node (BaseController): ROS2 node to control the robot's base
        head_node (HeadJointController): ROS2 node to control the robot's head
        pose_node (JointPoseController): ROS2 node to move the arm to a specified pose
        body_pose (Pose3D): 3D pose where the robot should move to
        target_center (Pose3D): 3D pose of the target to turn to
        yaw (float): Gripper yaw angle
        pitch (float): Gripper pitch angle
        roll (float): Gripper roll angle
        lift (float): Gripper lift adjustment
        stow (bool, optional): Whether the robot stows at the beginning. Defaults to True.
        grasp (bool, optional): Whether the robot wants to grasp an object. Defaults to False.
    """
    if stow:
        stow_arm(stow_node)
    move_body(base_node, body_pose.to_dimension(2))
    turn_body(pose_node, target_center.to_dimension(2), grasp=grasp)
    if grasp:
        look_ahead(pose_node)
        unstow_arm(pose_node, target_center, yaw=yaw, pitch=pitch, roll=roll, lift=lift)
    else:
        move_head(head_node, target_center, tilt_bool=True)
        time.sleep(1)
    
        
def drive_home(base_node: BaseController, pose_node: JointPoseController, stow_node: StowArmController,):
    stow_arm(stow_node)
    move_body(base_node, Pose2D(np.array([0.0, 0.0])))
    turn_body(pose_node, Pose2D(np.array([1.0, 0.0])), grasp=False)