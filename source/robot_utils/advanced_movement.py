from __future__ import annotations

import copy
import time
import numpy as np
import math
import open3d as o3d

from robot import HelloRobot
from global_parameters import *
from robot_utils.basic_movements import move_arm_distanced, move_body, set_gripper
from robot_utils.frame_transformer import (GRAPH_SEED_FRAME_NAME, VISUAL_SEED_FRAME_NAME, FrameTransformerSingleton)
from robot_utils.video import (GRIPPER_DEPTH, get_d_pictures, point_cloud_from_camera_captures)
from scipy.spatial.transform import Rotation
from utils import vis
from utils.coordinates import (Pose2D, Pose3D, from_a_to_b_distanced, spherical_angle_views_from_target)
from utils.graspnet_interface import predict_partial_grasp
from utils.importer import PointCloud
from utils.point_clouds import icp
from utils.recursive_config import Config
from utils.user_input import confirm_coordinates

frame_transformer = FrameTransformerSingleton()


def rotate(hello_robot: HelloRobot, end_pose: Pose2D) -> None:
    """
    Rotate in position to rotation of end_pose.
    :param hello_robot: StretchClient robot controller
    :param end_pose: will rotate to match end_pose.rot_matrix
    """
    # Get start pose of robot
    start_pose = Pose2D.from_array(hello_robot.robot.nav.get_base_pose())
    # Calculate rotation
    start = start_pose.as_ndarray()
    end = end_pose.as_ndarray()
    path = end - start
    yaw = math.atan2(path[1], path[0])
    # Rotate body in position
    rotated_pose = Pose2D(start_pose.coordinates, yaw)
    move_body(hello_robot, rotated_pose)


def rotate_and_move_distanced(hello_robot: HelloRobot, end_pose: Pose2D, distance: float, sleep: bool = True) -> None:
    """
    First rotate, then move to a location with a certain distance offset in the direction of the current pose.
    :param hello_robot: StretchClient robot controller
    :param end_pose: final pose to walk towards (minus the distance)
    :param distance: distance offset to final pose
    :param sleep: whether to sleep in between movements for safety
    """
    sleep_multiplier = 1 if sleep else 0
    # Get start pose of robot
    start_pose = Pose2D.from_array(hello_robot.robot.nav.get_base_pose())
    # Calculate destination pose
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)
    # Confirm movement action
    confirm = confirm_coordinates(start_pose, end_pose, destination_pose, distance)
    assert confirm, "Aborting due to negative confirmation!"
    # Rotate and move robot to destination pose
    time.sleep(1 * sleep_multiplier)
    rotate(hello_robot, end_pose)
    time.sleep(1 * sleep_multiplier)
    move_body(hello_robot, destination_pose)
    time.sleep(1 * sleep_multiplier)


def move_body_distanced(hello_robot: HelloRobot, end_pose: Pose2D, distance: float, sleep: bool = True) -> None:
    """
    Move to a location, with a certain distance offset. The distance is in the direction of the current pose.
    :param hello_robot: StretchClient robot controller
    :param end_pose: final pose to walk towards (minus the distance)
    :param distance: distance offset to final pose
    :param sleep: whether to sleep in between movements for safety
    """
    sleep_multiplier = 1 if sleep else 0
    # Get start pose of robot
    start_pose = Pose2D.from_array(hello_robot.robot.nav.get_base_pose())
    # Calculate destination pose
    destination_pose = from_a_to_b_distanced(start_pose, end_pose, distance)
    # Move robot
    move_body(hello_robot, destination_pose)
    time.sleep(1 * sleep_multiplier)


def positional_grab(
        hello_robot: HelloRobot, pose: Pose3D, distance_start: float, distance_end: float, already_gripping: bool = False
) -> None:
    """
    Grab something at a specified position. The gripper will first move towards the pose, but offset by distance_start
    in the opposite direction of the viewing direction. Then it will move towards the specified pose offset by distance_end.
    So in essence, the direction of the pose specifies the axis along which it moves.
    :param hello_robot: StretchClient robot controller
    :param pose: pose to grab
    :param distance_start: distance from which to start grab
    :param distance_end: distance at which to end grab
    :param already_gripping: whether to NOT open up the gripper in the beginning
    """
    # Move arm to grab start position
    move_arm_distanced(hello_robot, pose, distance_start, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    set_gripper(hello_robot, not already_gripping)
    # Move arm to grab end position and grab
    move_arm_distanced(hello_robot, pose, distance_end, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    set_gripper(hello_robot, False)
    # Move arm back to grab start position
    move_arm_distanced(hello_robot, pose, distance_start, TOP_CAMERA_NODE, GRIPPER_MID_NODE)


def pull(
        hello_robot: HelloRobot, pose: Pose3D, start_distance: float, mid_distance: float, end_distance: float
) -> (Pose3D, Pose3D):
    """
    Executes a pulling motion (e.g. for drawers).
    :param hello_robot: StretchClient robot controller
    :param pose: pose of knob in 3D space
    :param start_distance: how far from the knob to start grab
    :param mid_distance: how far to go before grabbing
    :param end_distance: how far to pull
    """
    # Move arm before handle and open gripper
    move_arm_distanced(hello_robot, pose, start_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    set_gripper(hello_robot, True)
    # Move arm towards handle and close gripper to grab
    move_arm_distanced(hello_robot, pose, mid_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    set_gripper(hello_robot, False)
    # Pull drawer and save poses before and after pulling
    start_pos, start_quat = hello_robot.robot.manip.get_ee_pose()
    pull_start = Pose3D(start_pos, start_quat)
    move_arm_distanced(hello_robot, pose, end_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    end_pos, end_quat = hello_robot.robot.manip.get_ee_pose()
    pull_end = Pose3D(end_pos, end_quat)
    # Release handle after opening drawer
    set_gripper(hello_robot, True)
    move_arm_distanced(hello_robot, pull_end, start_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    return pull_start, pull_end


def push(hello_robot: HelloRobot, start_pose: Pose3D, end_pose: Pose3D, start_distance: float, end_distance: float) -> None:
    """
    Executes a pushing motion (e.g. for drawers).
    :param hello_robot: StretchClient robot controller
    :param start_pose: pose of knob in 3D space
    :param end_pose: pose of knob when drawer closed
    :param start_distance: how far from the button to start push
    :param end_distance: how far to push
    """
    # Move arm before handle
    move_arm_distanced(hello_robot, start_pose, start_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    # Close gripper
    set_gripper(hello_robot, False)
    # Push drawer
    move_arm_distanced(hello_robot, end_pose, end_distance, TOP_CAMERA_NODE, GRIPPER_MID_NODE)


def adapt_grasp(body_pose: Pose3D, grasp_pose: Pose3D):
    # Transform grasp pose into body frame
    grasp_in_body = body_pose.inverse() @ grasp_pose
    grasp_pose_new = Pose3D(grasp_pose.coordinates.copy(), grasp_pose.rot_matrix.copy())
    # Calculate top direction of the grasp pose
    top_dir = grasp_in_body.rot_matrix @ np.array([0, 0, 1])
    # If x<0 or z<-0.2
    if top_dir[0] < 0 or top_dir[2] < -0.2:
        # Rotate 180 degree around x-axis
        roll_matrix = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        grasp_pose_new.rot_matrix = grasp_pose_new.rot_matrix @ roll_matrix
    return grasp_pose_new


def collect_dynamic_point_cloud(
    hello_robot: HelloRobot,
    start_pose: Pose3D,
    target_pose: Pose3D,
    nr_captures: int = 4,
    offset: float = 10,
    degrees: bool = True,
) -> PointCloud:
    """
    Collect a point cloud of an object in front of the gripper.
    The poses are distributed spherically around the target_pose with a certain offset looking at the target.
    :param hello_robot: StretchClient robot controller
    :param start_pose: Pose3D describing the start position
    :param target_pose: Pose3D describing the target position (center of sphere)
    :param nr_captures: number of poses calculated on the circle (equidistant on it)
    :param offset: offset from target to circle as an angle seen from the center
    :param degrees: whether offset is given in degrees
    :return: a point cloud stitched together from all views
    """
    # Calculate poses for the gripper
    if nr_captures > 0:
        angled_view_poses = spherical_angle_views_from_target(start_pose, target_pose, nr_captures, offset, degrees)
    else:
        angled_view_poses = [start_pose]
    # Take depth images of the target
    depth_images = []
    for angled_pose in angled_view_poses:
        move_arm_distanced(hello_robot, angled_pose, 0.0, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
        depth_images.extend(get_d_pictures([GRIPPER_DEPTH]))  # TODO
    # Calculate point cloud from images
    pcd_odom = point_cloud_from_camera_captures(depth_images)  # TODO
    return pcd_odom


# TODO
def dynamically_refined_grasp_renew_grasp(
    hello_robot: HelloRobot,
    pose: Pose3D,
    item_cloud: PointCloud,
    distance_start: float,
    distance_end: float,
    nr_captures: int = 4,
    offset: float = 20,
    degrees: bool = True,
    drift_threshold: float = 0.03,
    icp_multiplier: int = 5,
    vis_block: bool = False,
) -> None:
    """
    Adaptive grasp based on collecting a new point cloud close to the supposed grasp position allowing it to adjust for
    some drift in localization. This method specifically creates a new point cloud at the supposed position, calculates
    the transformation from original PCD to the new dynamically collected PCD, and transforms the grasp accordingly.
    :param hello_robot: StretchClient robot controller
    :param pose: supposed grasp position
    :param item_cloud: cloud of the item to grab
    :param distance_start: start of the grabbing motion
    :param distance_end: end of the grabbing motion
    :param nr_captures: number of captures to create PCD
    :param offset: increments in which to move the camera when collecting new PCD
    :param degrees: whether increments is in degrees (True) or radians (False)
    :param drift_threshold: threshold of ICP alignment
    :param icp_multiplier: how much of the point cloud around the original grasp is used for ICP alignment, in radius
    as multiple of drift_threshold
    :param vis_block: whether to visualize ICP alignment
    """
    result_pose = move_arm_distanced(hello_robot, pose, distance_start, TOP_CAMERA_NODE, GRIPPER_MID_NODE)

    # get point cloud in ODOM frame
    dynamic_pcd_odom = collect_dynamic_point_cloud(hello_robot, result_pose, pose, nr_captures, offset, degrees)
    # transform to seed frame to match with originalF PCD
    seed_tform_odom = frame_transformer.transform_matrix(ODOM_FRAME_NAME, VISUAL_SEED_FRAME_NAME)
    # now in same frame as ordinary point cloud
    dynamic_pcd = dynamic_pcd_odom.transform(seed_tform_odom)

    original_grasp_coordinates = pose.as_ndarray()
    dynamic_points = np.array(dynamic_pcd.points)
    points_within = np.where(
        np.linalg.norm(dynamic_points - original_grasp_coordinates, axis=1) < icp_multiplier * drift_threshold
    )[0]
    selected_dynamic_pcd = dynamic_pcd.select_by_index(points_within)
    # voxel_size = 0.008  # for example, 0.005 unit length voxel size
    # selected_dynamic_pcd = selected_dynamic_pcd.voxel_down_sample(voxel_size)

    # perform ICP
    if not selected_dynamic_pcd.has_normals():
        selected_dynamic_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if not item_cloud.has_normals():
        item_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # calculate the transformation (drift) from the original cloud (seed) to the new dynamically created cloud
    if vis_block:
        vis.show_two_geometries_colored(selected_dynamic_pcd, item_cloud)
    dynamic_tform_seed = icp(selected_dynamic_pcd, item_cloud, drift_threshold, max_iteration=100, point_to_point=True)

    euler = Rotation.from_matrix(dynamic_tform_seed.copy()[:3, :3]).as_euler("xyz", degrees=True)
    print(euler)

    # offset original grasp pose by similar amount
    item_cloud = item_cloud.transform(dynamic_tform_seed)
    if vis_block:
        vis.show_two_geometries_colored(selected_dynamic_pcd, item_cloud)
    pose = Pose3D.from_matrix(dynamic_tform_seed) @ pose
    return positional_grab(hello_robot, pose, distance_start, distance_end)


# TODO
def dynamically_refined_grasp_find_new_grasp(
    hello_robot: HelloRobot,
    pose: Pose3D,
    distance_start: float,
    distance_end: float,
    config: Config,
    nr_captures: int = 3,
    offset: float = 10,
    tolerance: float = 0.1,
    degrees: bool = True,
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
    start_pose = move_arm_distanced(hello_robot, pose, distance_start, TOP_CAMERA_NODE, GRIPPER_MID_NODE)
    pcd_body = collect_dynamic_point_cloud(hello_robot, start_pose, pose, nr_captures, offset, degrees)

    # transform into coordinate of starting pose
    seed_tform_body = frame_transformer.transform_matrix(BODY_FRAME_NAME, GRAPH_SEED_FRAME_NAME)
    pose_tform_seed = start_pose.as_matrix()
    pose_tform_body = pose_tform_seed @ seed_tform_body

    pcd_pose = pcd_body.transform(pose_tform_body)
    pose_pose = copy.deepcopy(pose)
    pose_pose.transform(pose_tform_seed)

    # get new gripper pose
    tf_matrix, _ = predict_partial_grasp(pcd_pose, pose_pose, tolerance, config, robot.logger)
    new_pose = Pose3D.from_matrix(tf_matrix)

    return positional_grab(hello_robot, new_pose, distance_start, distance_end)
