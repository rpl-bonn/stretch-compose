from __future__ import annotations

import copy
import cv2
import numpy as np
import open3d as o3d
import os
import time

from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D, get_circle_points, get_arc_view_poses
from utils.importer import PointCloud
from utils.mask3D_interface import get_coordinates_from_item
from utils import recursive_config
from utils.robot_utils.basic_movement import move_arm
from utils.robot_utils.basic_perception import get_depth_picture, get_rgb_picture
from utils.time import convert_time
from utils.zero_shot_object_detection import get_cloud_from_gripper_detection, yolo_detect_object, sam_detect_object


def add_coordinate_system(
    cloud: PointCloud, color: tuple[int, int, int], ground_coordinate: np.ndarray = None,
    e1: np.ndarray = None, e2: np.ndarray = None, e3: np.ndarray = None, e_relative_to_ground: bool = True, size: int = 1
) -> PointCloud:
    """
    Given a point cloud, add a coordinate system to it.
    It adds three axes, where x has the lowest distance between spheres, y the second lowest, and z the highest.
    :param cloud: original point cloud
    :param color: color of the coordinate system to add
    :param ground_coordinate: center of the coordinate system
    :param e1: end of x-axis
    :param e2: end of y-axis
    :param e3: end of z-axis
    :param e_relative_to_ground: whether e is specified relative to the ground coordinate or not
    :param size: size multiplier for axes
    :return: point cloud including new axes
    """
    nrx, nry, nrz = 40 * size, 20 * size, 5 * size
    if ground_coordinate is None:
        ground_coordinate = np.asarray([0, 0, 0])
    if e1 is None:
        e1 = np.asarray([1, 0, 0])
    if e2 is None:
        e2 = np.asarray([0, 1, 0])
    if e3 is None:
        e3 = np.asarray([0, 0, 1])

    if not e_relative_to_ground:
        e1 = e1 - ground_coordinate
        e2 = e2 - ground_coordinate
        e3 = e3 - ground_coordinate

    e1 = e1 * size
    e2 = e2 * size
    e3 = e3 * size

    # x
    x_vector = np.linspace(0, 1, nrx).reshape((nrx, 1))
    full_x_vector = x_vector * np.tile(e1, (nrx, 1))
    # y
    y_vector = np.linspace(0, 1, nry).reshape((nry, 1))
    full_y_vector = y_vector * np.tile(e2, (nry, 1))
    # z
    z_vector = np.linspace(0, 1, nrz).reshape((nrz, 1))
    full_z_vector = z_vector * np.tile(e3, (nrz, 1))

    full_vector = np.vstack([full_x_vector, full_y_vector, full_z_vector])
    ground_coordinate = np.tile(ground_coordinate, (full_vector.shape[0], 1))
    full_vector = full_vector + ground_coordinate

    color_vector = np.asarray(color)
    color_vector = np.tile(color_vector, (full_vector.shape[0], 1))
    
    # Add the new points to the cloud
    points = np.asarray(cloud.points)
    new_points = np.vstack([points, full_vector])
    new_cloud_points = o3d.utility.Vector3dVector(new_points)
    cloud.points = new_cloud_points

    colors = np.asarray(cloud.colors)
    new_colors = np.vstack([colors, color_vector])
    new_cloud_colors = o3d.utility.Vector3dVector(new_colors)
    cloud.colors = new_cloud_colors
    return cloud


def visualize_environment(item_cloud, environment_cloud, body_pose_distanced: Pose3D):
    item_cloud.paint_uniform_color([1, 0, 1])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(body_pose_distanced.coordinates)
    o3d.visualization.draw_geometries([item_cloud, environment_cloud, sphere])


def body_planning(
    env_cloud: PointCloud,
    target_pose: Pose3D,
    resolution: int = 16,
    nr_circles: int = 3,
    floor_height_thresh: float = -0.1,
    body_height: float = 1.20,
    min_distance: float = 0.75,
    max_distance: float = 1,
    lambda_distance: float = 0.5,
    n_best: int = 1,
    vis_block: bool = False,
) -> list[tuple[Pose3D, float]]:
    """
    Plans a position for the robot to go to given a cloud *without* the item to be
    grasped, as well as point to be grasped.
    :param env_cloud: the point cloud *without* the item
    :param target_pose: target pose for grasping
    :param floor_height_thresh: z value under which to cut floor
    :param body_height: height of robot body
    :param min_distance: minimum distance from object
    :param max_distance: max distance from object
    :param lambda_distance: trade-off between distance to obstacles and distance to target, higher
    lam, more emphasis on distance to target
    lam, more emphasis on direction of grasp
    :param n_best: number of positions to return
    :param vis_block: whether to visualize the position
    :return: list of viable coordinates ranked by score
    """
    target = target_pose.as_ndarray()
    
    # delete floor from point cloud, so it doesn't interfere with the SDF
    points = np.asarray(env_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > floor_height_thresh
    index = np.where(points_bool)[0]
    pc_no_ground = env_cloud.select_by_index(index)

    # get points radiating outwards from target coordinate
    circle_points = get_circle_points(
        resolution=resolution,
        nr_circles=nr_circles,
        start_radius=min_distance,
        end_radius=max_distance,
        return_cartesian=True,
    )
    ## get center of radiating circle
    target_at_body_height = target.copy()
    target_at_body_height[-1] = body_height
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    ## add the radiating circle center to the points to elevate them
    circle_points = circle_points + target_at_body_height
    ## filter every point that is outside the scanned scene
    circle_points_bool = (min_points+0.2 <= circle_points) & (circle_points <= max_points-0.2)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool].reshape((-1, 3))

    # transform point cloud to mesh to calculate SDF from
    ball_sizes = (0.02, 0.011, 0.005)
    ball_sizes = o3d.utility.DoubleVector(ball_sizes)
    mesh_no_ground = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pc_no_ground, radii=ball_sizes)
    mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_no_ground_legacy)

    # of the filtered points, cast ray from target to point to see if there are collisions
    ray_directions = filtered_circle_points - target
    rays_starts = np.tile(target, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 1 # 3
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    
    points_no_ground_np = np.asarray(pc_no_ground.points)
    valid_points = []
    for point in filtered_circle_points:
        # Check horizontal proximity to obstacles (layer check)
        horizontal_distances = np.linalg.norm(points_no_ground_np[:, :2] - point[:2], axis=1)
        layer_clearance = np.sum(horizontal_distances < 0.2) <= 2
        
        # Check vertical clearance (above and below)
        #vertical_distances = np.abs(points[:, 2] - point[2])
        #vertical_clearance = np.all(vertical_distances > 0.2)
        
        if layer_clearance: #and vertical_clearance:
            valid_points.append(point)
            
    if np.array(valid_points).size == 0:
        print("No valid points after clearance checks.")
    else:
        print(f"{np.array(valid_points).size} valid points found.")
            
    filtered_circle_points = np.array(valid_points)
    circle_tensors = o3d.core.Tensor(filtered_circle_points, dtype=o3d.core.Dtype.Float32)
    
    # calculate the best body positions
    ## calculate SDF distances
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    ## calculate distance to target
    target = target.reshape((1, 1, 3))
    target_distances = filtered_circle_points - target
    target_distances = target_distances.squeeze()
    target_distances = np.linalg.norm(target_distances, ord=2, axis=-1)
    ## get the top n coordinates
    scores = distances - lambda_distance * target_distances
    # Flatten the array and get the indices that would sort it in descending order
    flat_indices = np.argsort(-scores.flatten())
    top_n_indices = np.unravel_index(flat_indices[:n_best], scores.shape)
    top_n_coordinates = filtered_circle_points[top_n_indices]
    top_n_scores = scores[top_n_indices]

    if vis_block:
        # draw the entries in the cloud
        x = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        x.translate(target.reshape(3, 1))
        x.paint_uniform_color((1, 0, 0))
        y = copy.deepcopy(env_cloud)
        y = add_coordinate_system(y, (1, 0, 0), (0, 0, 0))
        drawable_geometries = [x, y]
        for idx, coordinate in enumerate(top_n_coordinates, 1):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere.translate(coordinate)
            color = np.asarray([0, 1, 0]) * (idx / (2 * n_best) + 0.5)
            sphere.paint_uniform_color(color)
            drawable_geometries.append(sphere)
        o3d.visualization.draw_geometries(drawable_geometries)

    poses = []
    for score, coord in zip(top_n_scores, top_n_coordinates):
        pose = Pose3D(coord)
        pose.set_rot_from_direction(target - coord)
        poses.append((pose, score))
    return poses


def body_planning_front(
    env_cloud: PointCloud,
    target: np.ndarray,
    furniture_normal: np.ndarray,
    floor_height_thresh: float = 0,
    body_height: float = 0.45,
    min_target_distance: float = 0.75,
    max_target_distance: float = 1,
    min_obstacle_distance: float = 0.5,
    n: int = 4,
    vis_block: bool = False
) -> Pose3D:
    """
    Plans a frontal position for the robot.
    :param env_cloud: the point cloud *without* the item
    :param target: target coordinates for grasping
    :param furniture_normal: normal of the front face
    :param floor_height_thresh: z value under which to cut floor
    :param body_height: height of robot body
    :param min_distance: minimum distance from object
    :param max_distance: max distance from object
    :param lam: trade-off between distance to obstacles and distance to target, higher
    lam, more emphasis on distance to target
    :param n_best: number of positions to return
    :param vis_block: whether to visualize the position
    :return: list of viable coordinates ranked by score
    """
    start_time = time.time_ns()
    # Delete floor from point cloud, so it doesn't interfere with the SDF
    points = np.asarray(env_cloud.points)
    min_points = np.min(points, axis=0)
    max_points = np.max(points, axis=0)
    points_bool = points[:, 2] > floor_height_thresh
    index = np.where(points_bool)[0]
    pc_no_ground = env_cloud.select_by_index(index)

    # get points radiating outwards from target coordinate
    circle_points = get_circle_points(
        resolution=32,
        nr_circles=2,
        start_radius=min_target_distance,
        end_radius=max_target_distance,
        return_cartesian=True,
    )
    ## get center of radiating circle
    target_at_body_height = target.copy()
    target_at_body_height[-1] = body_height
    target_at_body_height = target_at_body_height.reshape((1, 1, 3))
    ## add the radiating circle center to the points to elevate them
    circle_points = circle_points + target_at_body_height
    ## filter every point that is outside the scanned scene
    circle_points_bool = (min_points <= circle_points) & (circle_points <= max_points)
    circle_points_bool = np.all(circle_points_bool, axis=2)
    filtered_circle_points = circle_points[circle_points_bool]
    filtered_circle_points = filtered_circle_points.reshape((-1, 3))

    # transform point cloud to mesh to calculate SDF from
    pc_no_ground = pc_no_ground.voxel_down_sample(voxel_size=0.01)
    pc_no_ground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    mesh_no_ground, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc_no_ground, depth=6)
    mesh_no_ground_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_no_ground)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_no_ground_legacy)

    # of the filtered points, cast ray from target to point to see if there are collisions
    ray_directions = filtered_circle_points - target
    rays_starts = np.tile(target, (ray_directions.shape[0], 1))
    rays = np.concatenate([rays_starts, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    response = scene.cast_rays(rays_tensor)
    direct_connection_bool = response["t_hit"].numpy() > 1
    filtered_circle_points = filtered_circle_points[direct_connection_bool]
    circle_tensors = o3d.core.Tensor(filtered_circle_points, dtype=o3d.core.Dtype.Float32)

    # calculate the best body positions
    ## calculate SDF distances
    distances = scene.compute_signed_distance(circle_tensors).numpy()
    valid_points = circle_tensors[np.abs(distances) > min_obstacle_distance]
    valid_points_num = valid_points.numpy()

    # select coordinates most aligned with front plane normal
    directions = valid_points_num - target
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    furniture_normal = furniture_normal / np.linalg.norm(furniture_normal)
    cosine_angles = np.dot(directions, furniture_normal)
    most_aligned_point_index = np.argmax(cosine_angles)
    selected_coordinates = valid_points_num[most_aligned_point_index]

    if vis_block:
        # draw the entries in the cloud
        x = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        x.translate(target.reshape(3, 1))
        x.paint_uniform_color((1, 0, 0))
        y = copy.deepcopy(env_cloud)
        y = add_coordinate_system(y, (1, 0, 0), (0, 0, 0))
        drawable_geometries = [x, y]
        for idx, coordinate in enumerate(np.array([selected_coordinates]), 1):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(coordinate)
            color = np.asarray([0, 1, 0]) * (idx / (2 * n) + 0.5)
            sphere.paint_uniform_color(color)
            drawable_geometries.append(sphere)
        o3d.visualization.draw_geometries(drawable_geometries)

    pose = Pose3D(selected_coordinates)
    pose.set_rot_from_direction(target - selected_coordinates)
    end_time = time.time_ns()
    minutes, seconds = convert_time(end_time - start_time)
    print(f"\nBody planning RUNTIME: {minutes}min {seconds}s\n")
    return pose


def get_radius_env_cloud(item_cloud: PointCloud, env_cloud: PointCloud, radius: float) -> PointCloud:
    """
    Given two point clouds, one representing an item, one representing its environment, extract all points from the
    environment cloud that are within a certain radius of the item center.
    :param item_cloud: point cloud of item
    :param env_cloud: point cloud of environment
    :param radius: radius in which to extract points
    """
    center = np.mean(np.asarray(item_cloud.points), axis=0)
    distances = np.linalg.norm(np.asarray(env_cloud.points) - center, ord=2, axis=1)
    env_cloud = env_cloud.select_by_index(np.where(distances < radius)[0].tolist())
    return env_cloud


def icp(
    pcd1: PointCloud, pcd2: PointCloud, threshold: float = 0.2, trans_init: np.ndarray | None = None, 
    max_iteration: int = 1000, point_to_point: bool = False,
) -> np.ndarray:
    """
    Return pcd1_tform_pcd2 via ICP.
    :param pcd1: First point cloud
    :param pcd2: Second point cloud
    :param threshold: threshold for alignment in ICP
    :param trans_init: initial transformation matrix guess (default I_4)
    :param max_iteration: maximum iterations for ICP
    :param point_to_point: whether to use Point2Point (true) or Point2Plane (false) ICP
    :return: pcd1_tform_pcd2
    """
    if trans_init is None:
        trans_init = np.eye(4)

    if point_to_point:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        threshold,
        init=trans_init,
        estimation_method=method,
        criteria=criteria,
    )
    return reg_p2p.transformation


def collect_dynamic_point_cloud(
    obj: str, pos_node: JointPositionController, pose_node: JointPoseController, tf_node: FrameTransformer, start_pose: Pose3D, target_pose: Pose3D, full_env_pcd, offset: float = 20
) -> PointCloud:
    """
    Collect a point cloud of an object in front of the gripper.
    The poses are distributed spherically around the target_pose with a certain offset looking at the target.
    :param node: 
    :param start_pose: Pose3D describing the start position
    :param target_pose: Pose3D describing the target position (center of sphere)
    :param nr_captures: number of poses calculated on the circle (equidistant on it)
    :param offset: offset from target to circle as an angle seen from the center
    :param degrees: whether offset is given in degrees
    :return: a point cloud stitched together from all views
    """
    pcds_down = []
    pcds_masked = []
    pcds_full = []
    voxel_size = 0.01
    
    angled_view_poses = get_arc_view_poses(start_pose, target_pose, offset)

    for i, angled_pose in enumerate(angled_view_poses):
        move_arm(pos_node, angled_pose)
        time.sleep(0.5)
        
        rgb = get_rgb_picture(RGBImageSubscriber, pose_node, "/gripper_camera/color/image_rect_raw", gripper=True, save_block=True)
        image_path = f'/home/ws/data/images/viewpoints/gripper_{i}.png'
        cv2.imwrite(image_path, rgb) 
        get_depth_picture(AlignedDepth2ColorSubscriber, pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True, save_block=True)
       
        try:
            _, dict = yolo_detect_object(obj, "gripper", save_block=True)
            x1, y1, x2, y2 = map(int, dict["box"])
            mask, _, _ = sam_detect_object("gripper", (x1+x2)/2, (y1+y2)/2, i)
        
            pcd_masked = get_cloud_from_gripper_detection(tf_node, mask)
            pcds_masked.append(pcd_masked)
            
            pcd_full = get_cloud_from_gripper_detection(tf_node)
            pcds_full.append(pcd_full)
            #o3d.visualization.draw_geometries([pcd_full, full_env_pcd])
            
            pcd_down = pcd_full.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            pcds_down.append(pcd_down)
        except Exception as e:
            print(f"Failed detecting object in view {i}: {e}")

    # Full registration
    max_corr_coarse = voxel_size * 5
    max_corr_fine = voxel_size * 1.5
    
    pose_graph = full_registration(pcds_down, max_corr_coarse, max_corr_fine)  
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_fine, edge_prune_threshold=0.25, reference_node=len(pcds_down) - 1
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option
    )
    
    # Align single clouds
    obj_points = []
    obj_colors = []
    for point_id in range(len(pcds_masked)):
        pcds_masked[point_id].transform(pose_graph.nodes[point_id].pose)
        obj_points.extend(np.asarray(pcds_masked[point_id].points))
        obj_colors.extend(np.asarray(pcds_masked[point_id].colors))
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(obj_points)
    pcd_obj.colors = o3d.utility.Vector3dVector(obj_colors)

    pcd_obj, _ = pcd_obj.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
    shift = target_pose.coordinates[:2] - np.mean(np.asarray(obj_points), axis=0)[:2]
    shift = np.append(shift, 0.0)
    pcd_obj.translate(shift)
    #o3d.visualization.draw_geometries([pcd_obj])   
    
    env_points = []
    env_colors = []
    for point_id in range(len(pcds_full)):
        pcds_full[point_id].transform(pose_graph.nodes[point_id].pose)
        env_points.extend(np.asarray(pcds_full[point_id].points))
        env_colors.extend(np.asarray(pcds_full[point_id].colors))
    pcd_env = o3d.geometry.PointCloud()
    pcd_env.points = o3d.utility.Vector3dVector(env_points)
    pcd_env.colors = o3d.utility.Vector3dVector(env_colors)
    
    pcd_env, _ = pcd_env.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
    pcd_env.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))
    pcd_env.translate(shift)
    #o3d.visualization.draw_geometries([pcd_env])
    
    # Align env_pcd to full_env_pcd
    # full_env_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))
    # snippet_tform_map = icp(pcd_env, full_env_pcd, point_to_point=True)
    # pcd_env = pcd_env.transform(snippet_tform_map)
    # o3d.visualization.draw_geometries([pcd_env, full_env_pcd])
    # pcd_obj = pcd_obj.transform(snippet_tform_map)
    # o3d.visualization.draw_geometries([pcd_obj, full_env_pcd])
    
    lim_env_cloud = get_radius_env_cloud(pcd_obj, pcd_env, 1.0)
    #o3d.visualization.draw_geometries([pcd_obj, lim_env_cloud, full_env_pcd])
    
    path = f'/home/ws/data/images/viewpoints/'
    o3d.io.write_point_cloud(path+"obj_cloud_vp.ply", pcd_obj)
    o3d.io.write_point_cloud(path+"env_cloud_vp.ply", lim_env_cloud)
    
    return pcd_obj, lim_env_cloud    
    
def pairwise_registration(source, target, max_corr_coarse, max_corr_fine):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    try:
        icp_color = o3d.pipelines.registration.registration_colored_icp(
            source, target, max_corr_fine, icp_fine.transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=50)
        )
    except Exception as e:
        print(f"Colored ICP failed: {e}")
        return icp_fine.transformation
    return icp_color.transformation


def full_registration(pcds, max_corr_coarse, max_corr_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    for source_id in range(len(pcds)):
        for target_id in range(source_id + 1, len(pcds)):
            
            transformation_icp = pairwise_registration(pcds[source_id], pcds[target_id], max_corr_coarse, max_corr_fine)
                
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 uncertain=True))
    return pose_graph


def test():
    ITEM, INDEX = "potted plant", 0
    config = recursive_config.Config()
    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["masked"]
    mask_path = os.path.join(mask_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")

    item_cloud, environment_cloud = get_coordinates_from_item(ITEM, mask_path, pc_path, INDEX)
    x = copy.deepcopy(item_cloud)
    x.paint_uniform_color((1, 0, 0))
    y = copy.deepcopy(environment_cloud)
    y = add_coordinate_system(y, (1, 1, 1), (0, 0, 0))

    end_coordinates = np.mean(np.asarray(item_cloud.points), axis=0)
    end_coordinates = Pose3D(end_coordinates)
    print(end_coordinates)
    robot_target = body_planning(
        environment_cloud,
        end_coordinates,
        min_distance=0.6,
        max_distance=1,
        n_best=10,
        vis_block=True,
    )[0]
    print(robot_target)

    o3d.visualization.draw_geometries([x, y])


if __name__ == "__main__":
    test()
